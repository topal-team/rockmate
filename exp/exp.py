import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import os
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
# from models import *
from LLM import get13Bllama, get3BPhi_2, get7Bllama, get7Bllama_lora, get3BPhi_15, get11Bfalcon, get3Bbloom, get8Bllama, get7Bmistral, get4Bphi3
from rockmate import Rockmate
from rockmate.op_schedule import *
from rockmate.solvers import HILP, CheapSolver, RK_rotor
from rockmate.solvers.main import FastSolver
from rockmate.simulation import Simulator
from datetime import datetime
device = torch.device("cuda")
import pickle
from time import sleep
# from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
import psutil
# from tmp_utils import *
import rkgb
from rkgb.lowlevel.measure import TimerCUDA
from rkgb.core.partitioned import PartitionerBottomToTop, PartitionerSequence
import gc
from transformers import LlamaModel, LlamaConfig

import sys
sys.setrecursionlimit(30000)
timer = TimerCUDA(torch.device("cuda"))
import tracemalloc
tracemalloc.start()
torch.random.manual_seed(0)

def dynamo_ram_trick():
    from rkgb.lowlevel.preprocess_samples import ExampleInputs
    m = torch.nn.Linear(100,100)
    x = ExampleInputs(m, torch.ones(100))
    dynamo_result = torch.export.export(m,  args = tuple(), kwargs=x.dict)

def analyze_mem(rkmod, print_status=False, with_param=True, with_grad=True, details=False):
    md = rkmod.list_solvers[0].md
    mem = {}
    for t in range(md.T):
        for k in md.krange(t):
            mem[t, k] = md.U[t, k].value()
            mem[t, k] += (
                sum(
                    md.Comp[t, k, o].value() * md.overhead[k][o]
                    for o in range(md.nR[k])
                )
                if md.sol(md.sumComp[t, k].value())
                else 0
            )
            mem[t, k] += sum(
                md.mem[i_] * md.delete[t, eidx_d].value() if hasattr(md.delete[t, eidx_d], "value") else md.delete[t, eidx_d]
                for eidx_d, (k_, i_) in enumerate(md.delete_list)
                if k == k_
            )
            act_multiplier = 1/((1-md.param_multiplier) 
                              if isinstance(md.param_multiplier, float) 
                              else (1-md.param_multiplier.value()))
            
            mem[t, k] *= act_multiplier
            if with_param:
                mem[t, k] += md.all_param_mem(t, k, with_multiplier=False).value()

            # for w in range(md.W):
            #     # mem[t,k] += 1*((md.AliveW[t,k,w]+md.PrfW[t,k,w]).value()>0)*md.parameter_size[w]
            #     mem[t,k] += (md.AliveW[t,k,w]+md.PrfW[t,k,w]).value()*md.parameter_size[w]

    max_t, max_k = max(mem, key=mem.get)
    max_i = np.argmax(rkmod.op_sched.save_mem + rkmod.op_sched.interface_mem + rkmod.op_sched.overhead)
    grad_size = 0#max(md.parameter_size)
    optimizer_states_mem = 0#rkmod.op_sched.optimizer_states_size()*rkmod.compiler.storage.gd['optimize_stats']['optimizer_states_size']
    optimizer_states_mem += sum([p.numel()*p.element_size() for p in rkmod.minor_parameters])*rkmod.compiler.storage.gd['optimize_stats']['optimizer_states_size']
    print(
        f"solution peak memory {(max(mem.values())*md.gcd)/1024**2:.0f}MB at {max_t, max_k}"
    )
    print(
        f"op_sched peak memory {(rkmod.op_sched.get_peak_mem(with_interface=True) + optimizer_states_mem +with_grad*grad_size)/1024**2:.0f}MB"
    )
    # return (max_i, max_t, max_k)
    return max(mem.values())*md.gcd, rkmod.op_sched.get_peak_mem(with_interface=True)


def check_correctness(model, sample, budget=1e9, optim=torch.optim.Adam):
    dtype = torch.float64
    model_g = deepcopy(model).to("cuda").to(dtype)
    sample_g = [s.to("cuda").to(dtype) for s in sample]
    model_c = deepcopy(model).to("cpu").to(dtype)
    sample_c = [s.to("cpu").to(dtype) for s in sample]
    optimizer = optim(model_g.parameters())
    def optimize():
        optimizer.step()
    print("gpu model")
    # print(model_g(*sample_g).mean())
    execution(model_g, sample_g, print_mem=False, print_loss=True, optimize_fct=optimize)

    optimizer = optim(model_c.parameters())
    def optimize():
        optimizer.step()
    print("cpu model")
    # print(model_c(*sample_c).mean())
    execution(model_c, sample_c, print_mem=False, print_loss=True, optimize_fct=optimize)

    model_r = Rockmate(model, sample, budget, 
                        solve_sched=False,
                        cpu_optim=optim,
                        gpu_optim=optim,
                        ilp_solver = "PULP_CBC_CMD")
    # prepare_for_offload(model_r)
    model_r.solve_sched(budget)
    model_r.zero_grad()
    sample_r = [s.to("cuda") for s in sample]
    print("rockmate model")    
    execution(model_r, sample_r, print_mem=False, print_loss=True)


def get_sched_stats(rkmod):
    stats = {}
    stats[f"Original module fwd+bwd time"] = sum(kcn.time for kcn in rkmod.rkgb_res.hierarchical_cluster.list_cnodes if kcn.time)
    stats[f"Original module optimize time"] = sum(pnode.mem for pnode in rkmod.rkgb_res.hierarchical_cluster.parameter_nodes)/rkmod.optimize_metrics["gpu_optimize_speed"]
    stats[f"Schedule_total_time"] = sum(step.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_compute_time"] = sum(step.comp_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_offload_time"] = sum(step.ofl_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_prefetch_time"] = sum(step.prf_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_cpu_optimize_time"] = sum(step.opt_ops.time for step in rkmod.op_sched.steps)
    return stats

def Loss(y, labels=None):
    labels = torch.ones(y.shape[0], dtype=torch.long, device=y.device)
    return torch.nn.CrossEntropyLoss()(y, labels)
    return y#.mean()

def execution(model, 
         sample, 
         niters=10, 
         optimize_fct=None,
         print_loss=True,
         print_mem=True,
         zero_grad=True,
          **kwargs):
    torch.random.manual_seed(0)
    print(f"{[s.shape for s in sample]}")
    # def hook_fn(m, args, output):
    #     # print(f"{output.mean()}")
    #     return output
    # for x in model.children():
    #     x.register_forward_hook(hook_fn)
    y = model(*sample, **kwargs)[0]
    # labels = torch.randn(y.shape).softmax(dim=1).to(sample[0].device)
    if print_mem:print(torch.cuda.memory_allocated())
    loss = Loss(y)
    if print_loss:print(f"loss: {loss}")
    loss.backward()
    if print_mem:print(torch.cuda.memory_allocated())
    if optimize_fct:optimize_fct()
    timer.start()
    for i in range(niters):
        if zero_grad:model.zero_grad()
        if print_mem:print(f"Mem before fwd {torch.cuda.memory_allocated()}")
        ys = model(*sample, **kwargs)
        y = ys[0]
        assert y.requires_grad
        loss = Loss(y)
        if print_mem:print(f"Mem after fwd {torch.cuda.memory_allocated()}")
        if print_loss:print(f"loss: {loss}")
        loss.backward()
        # torch.cuda.synchronize()
        # print(f"grad: {model.wte.weight.grad[0,0]}")
        if print_mem:print(f"Mem after bwd {torch.cuda.memory_allocated()}")
        if optimize_fct:optimize_fct()
        # for s in sample:
        #     s.grad = None
        for _y in ys:
            if not isinstance(_y, torch.Tensor):
                continue
            _y.data = torch.empty(0)
            _y.grad = None
            del _y
        # torch.cuda.empty_cache()
    timer.end()
    return timer.elapsed()
    

def exec_rk(model, sample, niters=10, device="cuda", **kwargs):
    torch.random.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    # print(torch.cuda.memory_allocated())
    model.zero_grad()
    sample = [s.to(device) for s in sample]
    mem = torch.cuda.memory_allocated()
    print(f"Memory allocated before exec {mem}")
    time = execution(model, sample, niters, optimize_fct=None, **kwargs)
    peak_mem = torch.cuda.max_memory_allocated() - mem
    return time, peak_mem

def exec_pt(model, sample, optim=torch.optim.Adam, niters=10, device="cuda", profile=None, **kwargs):
    torch.random.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    sample = [s.to(device) for s in sample]

    try: 
        m = model.original_mod
    except AttributeError:
        m = model
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)

    optimizer = optim(model.parameters(), lr=1e-3)
    def optimize():
        optimizer.step()

    optimize() ##To create the optimizer states
    mem = torch.cuda.memory_allocated()
    print(f"Memory allocated before exec {mem}")

    if profile: 
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
            ) as p:
            time = execution(model, sample, niters, optimize_fct=optimize, zero_grad=True, **kwargs)
        p.export_chrome_trace(f"profiles/{profile}.json")
    else:
        time = execution(model, sample, niters, optimize_fct=optimize, zero_grad=True, **kwargs)

    peak_mem = torch.cuda.max_memory_allocated() - mem
    return time, peak_mem

def exp_pt(nlayers=1, 
           batch_size=4, 
           exp_id=None, 
           num_adapters=None, 
           id="7B", 
           optim=torch.optim.Adam, 
           get_model=get7Bllama):
    model, sample = get_model(batch_size, 512, nlayers=nlayers)
    if num_adapters:
        manual_lora(model, target_modules=[f"layers.{i}.self_attn.q_proj" for i in range(nlayers)]
                +[f"layers.{i}.self_attn.k_proj" for i in range(nlayers)]
                +[f"layers.{i}.self_attn.v_proj" for i in range(nlayers)]
                # +[f"layers.{i}.self_attn.o_proj" for i in range(nlayers)]
                # +[f"layers.{i}.mlp.gate_proj" for i in range(nlayers)]
                # +[f"layers.{i}.mlp.up_proj" for i in range(nlayers)]
                # +[f"layers.{i}.mlp.down_proj" for i in range(nlayers)]
                ,
                num_adapters=num_adapters,
                freeze_all=True)
    niters = 5
    exp_stats = {}
    exp_stats["nlayer"] = nlayers
    exp_stats["input_size"] = [s.shape for s in sample]

    exp_stats["model_size"] = sum(p.numel()*p.element_size() for p in model.parameters())
    exp_stats["model_gradient_size"] = sum(p.numel()*p.element_size() for p in model.parameters()
                                           if p.requires_grad)
    # exp_stats["optimize_stats"] = rkmod.compiler.storage.gd["optimize_stats"]
    exp_stats["cpu_optim"] = str(optim)
    exp_stats["gpu_optim"] = str(optim)
    exp_stats["gpu_type"] = torch.cuda.get_device_name()

    time, mem = exec_pt(model.to("cuda"), sample, niters=niters, optim=optim)#, profile="pytorch")
    torch.cuda.synchronize()
    print(time, mem)

    exp_stats["time"] = time/niters
    exp_stats["peak_mem"] = mem
    print(exp_stats)

    if os.path.exists(exp_id):
        with open(exp_id, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    results[nlayers] = exp_stats
    with open(exp_id, "wb") as f:
        pickle.dump(results, f)
    return exp_stats

def exp_rkmod(nlayers=1, batch_size=3, exp_id=None, num_adapters=None, id="7B",
              activation_offload=True, cpu_optimization=True, get_model=get7Bllama,
              dynamic_batch_dim=None, rotor=False, id_batch=False,
              remat=True,):
    model, sample = get_model(batch_size, 512, nlayers=nlayers)
    budget = torch.cuda.get_device_properties(0).total_memory - 2*1024**3#to avoid fragmentation
    niters = 5

    if not remat:
        rkmod = solve_no_remat(model, 
                        sample, 
                        budget=budget,)
    elif not rotor:
        rkmod = solve_rkmod(model, 
                        sample, 
                        budget=budget, 
                        activation_offload=activation_offload,
                        cpu_optimization=cpu_optimization,
                        dynamic_batch_dim=dynamic_batch_dim)
    else:
        rkmod = solve_rockmate(model, 
                        sample, 
                        budget=budget)
    
    print(f"number of HCN: {len(rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs)}")
    print(f"number of HAN: {len(rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HANs)}")
    max_size = max(sum(kdn.mem for kdn in hcn.sub_cluster.parameter_nodes) 
        for hcn in rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs if hcn.sub_cluster)
    print(f"maximum size of subgraph: {max_size}")
    # rkmod.save_to_local(".", "GPT2-12")

    exp_stats = {}
    exp_stats["RAM_0"] = str(psutil.virtual_memory())

    exp_stats["nlayer"] = nlayers
    exp_stats["input_size"] = [s.shape for s in sample]
    exp_stats["max_subgraph_size"] = max(sum(kdn.mem for kdn in hcn.sub_cluster.parameter_nodes) 
    for hcn in rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs if hcn.sub_cluster)

    exp_stats["model_size"] = sum(p.numel()*p.element_size() for p in model.parameters())
    exp_stats["model_gradient_size"] = sum(p.numel()*p.element_size() for p in model.parameters()
                                           if p.requires_grad)
    exp_stats["act_size"] = sum(kdn.mem for kdn in rkmod.rkgb_res.hierarchical_cluster.list_anodes
                                if "grad" not in kdn.name and any(not kcn.is_fwd for kcn in kdn.users_real))
    exp_stats["optimize_stats"] = rkmod.optimize_metrics
    exp_stats["gpu_type"] = torch.cuda.get_device_name()
    exp_stats["budget"] = budget
    

    ### Solve schedule
    # rkmod.preprocess()
    # rkmod.solve_sched(budget, recursive=False)
    if not rotor:
        md = rkmod.list_solvers[0].md
        if md.feasible:
            sample = dynamic_sample(sample, md)
        #     # analyze_mem(rkmod)
        #     pass
        # else:
        #     raise ValueError
            print(md.solving_time)
            exp_stats["solving_time"] = str(md.solving_time)
            exp_stats["ilp_objective"] = str(md.md.objective.value())
            exp_stats["req_w"] = md.req_w.value()
        # mem_ilp, mem_sched = analyze_mem(rkmod)

            stats = get_sched_stats(rkmod)
            exp_stats["sched_stats"] = stats
            print(stats)

            try:
                time, mem = exec_rk(rkmod, sample, niters=niters)
            except Exception as e:
                exp_stats["exception"] = e
                raise e
            torch.cuda.synchronize()
            exp_stats["time"] = time/niters
            exp_stats["peak_mem"] = mem
            exp_stats["date"] = datetime.now().strftime("%x_%H:%M")
    else:
        if rkmod:
            try:
                time, mem = exec_pt(rkmod, sample, niters=niters)
            except Exception as e:
                exp_stats["exception"] = e
                raise e
            torch.cuda.synchronize()
            exp_stats["time"] = time/niters
            exp_stats["peak_mem"] = torch.cuda.max_memory_allocated()
            exp_stats["date"] = datetime.now().strftime("%x_%H:%M")            
    exp_stats["RAM_1"] = str(psutil.virtual_memory())
    del rkmod
    gc.collect()
    print(exp_stats)
    if os.path.exists(exp_id):
        with open(exp_id, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}
    eid = batch_size if id_batch else nlayers
    results[eid] = exp_stats
    with open(exp_id, "wb") as f:
        pickle.dump(results, f)
    return exp_stats
            
def solve_rkmod(model, 
                sample, 
                budget,
                activation_offload=True,
                cpu_optimization=True,
                dynamic_batch_dim=None):
    pat_layers = nlayers+2
    if nlayers>45:
        pat_layers = nlayers//2 +2

    partitioners = [rkgb.partitioned.PartitionerRecognizeRepetitivePattern(
        strict_max_number_of_top_level_nodes=nlayers+4,
        max_number_of_patterns=pat_layers,
        min_percentage_covered_required=0.75)]

    solver = HILP(ilp_solver="PULP_CBC_CMD")
    solver.config.offload = True
    solver.config.solve_only_top_level = True
    list_solvers = [solver]

    if activation_offload:
        # solver.config.activation_offload = False
        list_solvers.append(CheapSolver())
    if not cpu_optimization:
        solver.config.top_solve_kwargs["cpu_optimize"] = False

    rkmod = Rockmate(model, sample, budget, 
                    solve_sched=True, 
                        ilp_solver="PULP_CBC_CMD", 
                        list_solvers=list_solvers,
                        partitioners=partitioners,
                        ilp_time_limit=10*60,
                        minor_offload_size=10*1024**2,
                        dynamic_batch_dim=dynamic_batch_dim
                        )
    return rkmod

       
def solve_rockmate(model, 
                sample, 
                budget):
    partitioners = [rkgb.partitioned.PartitionerSequence(
        sub_partitioner=rkgb.partitioned.Partitioner())]
    model.to(device)
    sample = [s.to(device) for s in sample]
    solver = HILP(ilp_solver="PULP_CBC_CMD")
    solver.config.offload = False
    solver.config.solve_only_top_level = False
    solver.config.nb_total_nodes_top_level = 0
    rk_solver = RK_rotor()
    list_solvers = [solver, rk_solver]

    rkmod = Rockmate(model, sample, budget, 
                    solve_sched=False, 
                        ilp_solver="PULP_CBC_CMD", 
                        list_solvers=list_solvers,
                        partitioners=partitioners,
                        ilp_time_limit=10*60,
                        minor_offload_size=10*1024**2,
                        )
    cluster = rkmod.rkgb_res.hierarchical_cluster
    param_mem = sum(pnode.mem for pnode in cluster.parameter_nodes)
    param_grad_mem = sum(pnode.mem for pnode in cluster.parameter_nodes if pnode.info.requires_grad)
    act_budget = budget - param_mem - (1+rkmod.optimize_metrics["optimizer_states_size"]) * param_grad_mem
    if act_budget<0:
        return False
    print(act_budget)
    rkmod.preprocess()
    rkmod.solve_sched(act_budget)
    rkmod.get_compiled_fct()
    return rkmod

def solve_no_remat(model, 
                sample, 
                budget):
    partitioners = [rkgb.partitioned.PartitionerRecognizeRepetitivePattern(
        strict_max_number_of_top_level_nodes=nlayers+4,
        max_number_of_patterns=nlayers+2,
        min_percentage_covered_required=0.75)]
    solver = HILP(ilp_solver="PULP_CBC_CMD")
    solver.config.offload = True
    solver.config.activation_offload = False
    solver.config.solve_only_top_level = True
    solver.config.add_offload_sched = True

    list_solvers = [solver]

    rkmod = Rockmate(model, sample, budget, 
                    solve_sched=False, 
                        ilp_solver="PULP_CBC_CMD", 
                        list_solvers=list_solvers,
                        partitioners=partitioners,
                        ilp_time_limit=10*60,
                        minor_offload_size=10*1024**2,
                        )
    cluster = rkmod.rkgb_res.hierarchical_cluster
    # param_mem = sum(pnode.mem for pnode in cluster.parameter_nodes)
    # param_grad_mem = sum(pnode.mem for pnode in cluster.parameter_nodes if pnode.info.requires_grad)
    fast = FastSolver(recompute_sched=False)
    rkmod.preprocess(fast)
    rkmod.solve_sched(budget)
    rkmod.get_compiled_fct()
    return rkmod


def dynamic_sample(sample, md):
    if not md.dynamic_batch_size:return sample
    batch_size = int(1/md.req_w.value())
    sample_new = []
    for s in sample:
        sample_new.append(s.repeat(batch_size, 1))
    return sample_new


def exp_zero(nlayers=32, 
               batch_size=4, 
               exp_id=0, 
               get_model=get7Bllama,
               ds_config="ds_config_zero3.json"):
    niters = 5
    import deepspeed
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model,sample = get_model(batch=batch_size, seq_len=512, nlayers=nlayers)

    print(psutil.virtual_memory())
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    model.gradient_checkpointing_enable()
    

    exp_stats = {}
    exp_stats["nlayer"] = nlayers
    exp_stats["input_size"] = [s.shape for s in sample]

    exp_stats["model_size"] = sum(p.numel()*p.element_size() for p in model.parameters())
    exp_stats["model_gradient_size"] = sum(p.numel()*p.element_size() for p in model.parameters()
                                           if p.requires_grad)
    # exp_stats["optimize_stats"] = rkmod.compiler.storage.gd["optimize_stats"]
    exp_stats["gpu_type"] = torch.cuda.get_device_name()
    exp_stats["zero_confnig"] = ds_config
    torch.cuda.synchronize()
    if psutil.virtual_memory().percent > 50:
        exp_stats["exception"] = "OOM"
    else:
        print(psutil.virtual_memory().percent)
        time, mem = exec_pt(model.to("cuda"), sample, niters=niters, optimizer=optimizer, zero_grad=False)
        torch.cuda.synchronize()
        print(time, mem)
        exp_stats["time"] = time/niters
        exp_stats["peak_mem"] = mem
    print(exp_stats)

    if os.path.exists(exp_id):
        with open(exp_id, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    results[nlayers] = exp_stats
    with open(exp_id, "wb") as f:
        pickle.dump(results, f)

if __name__=="__main__":
    # exp_id = datetime.now().strftime('llama-7b')
    # exp_id = "llama-7b-ds"
    # a = torch.empty(2560,1024,1024,device="cuda")
    # a.data = torch.empty(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", nargs="?",  type=str, default="0")
    parser.add_argument("--nlayers", nargs="?",  default=32, type=int)
    parser.add_argument("--num_adapters", nargs="?",  default=0, type=int)
    parser.add_argument("--rk", nargs="?",  default=1, type=int)
    parser.add_argument("--batch_size", nargs="?",  default=4, type=int)
    parser.add_argument("--dtype", nargs="?", default="float32", type=str)
    parser.add_argument("--model", nargs="?", type=str, default="llama7b")
    parser.add_argument("--method", nargs="?", type=str, default="offmate")
    parser.add_argument("--id_batch", nargs="?", type=bool, default=False)

    dtypes = {"float32":torch.float32,
              "bfloat16": torch.bfloat16,
              "float16": torch.float16,
              }
    models = {
        "llama7b": get7Bllama,
        "llama13b": get13Bllama,
        "phi2-3b": get3BPhi_2,
        "phi2-2b": get3BPhi_15,
        "llama7b_lora":get7Bllama_lora,
        "bloom3b": get3Bbloom,
        "falcon11b": get11Bfalcon,
        "llama8b": get8Bllama,
        "mistral7b": get7Bmistral,
        "phi3-4b":get4Bphi3
    }
    kwargs = {
        "offmate":{},
        "offmate_no_act_offload":{"activation_offload":False},
        "offmate_no_cpu_optim":{"cpu_optimization":False},
        "offmate_base":{"cpu_optimization":False, "activation_offload":False},
        "offmate_dynamic_batch":{"dynamic_batch_dim":0},
        "offmate_no_remat":{"remat":False},
        "rockmate":{"rotor":True},
        "zero-3":{"ds_config":"ds_config_zero3.json"},
        "zero-2":{"ds_config":"ds_config_zero2.json"}
    }

    args = parser.parse_args()
    torch.set_default_dtype(dtypes[args.dtype])
    # exp_id = args.exp_id
    exp_id = f"exp_results/{args.method}-{args.model}-{args.dtype}-{args.exp_id}.pkl"
    nlayers = args.nlayers
    num_adapters = args.num_adapters
    # id = args.id if args.id is not None else f"{nlayers}_{num_adapters}"
    rk = args.rk
    batch_size = args.batch_size
    if "mate" in args.method:
        # print("exp on rockmate")
        dynamo_ram_trick()
        exp_rkmod(nlayers=nlayers, 
                  batch_size=batch_size, 
                  exp_id=exp_id, 
                  num_adapters=num_adapters,
                  get_model=models[args.model],
                  id_batch=args.id_batch,
                  **kwargs[args.method])
    elif args.method == "torch":
        exp_pt(nlayers=nlayers, 
               batch_size=batch_size, 
               exp_id=exp_id, 
               num_adapters=num_adapters, 
               get_model=models[args.model])
    elif "zero" in args.method:
        exp_zero(nlayers=nlayers, 
               batch_size=batch_size, 
               exp_id=exp_id, 
               get_model=models[args.model])
    