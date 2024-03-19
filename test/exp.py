import argparse
import os
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
# from models import *
# from models.LLM import *
from rockmate import ORockmate
from rockmate.op_schedule import *
from rockmate.solvers.HILP_pulp_ofl_para import *
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"


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
    optimizer_states_mem = 0#rkmod.op_sched.optimizer_states_size()*rkmod.gd['optimize_stats']['optimizer_states_size']
    optimizer_states_mem += sum([p.numel()*p.element_size() for p in rkmod.minor_parameters])*rkmod.gd['optimize_stats']['optimizer_states_size']
    print(
        f"solution peak memory {(max(mem.values())*md.gcd)/1024**2:.0f}MB at {max_t, max_k}"
    )
    print(
        f"op_sched peak memory {(rkmod.op_sched.get_peak_mem(with_interface=True) + optimizer_states_mem +with_grad*grad_size)/1024**2:.0f}MB"
    )
    # return (max_i, max_t, max_k)
    return max(mem.values())*md.gcd, rkmod.op_sched.get_peak_mem(with_interface=True)


def get7Bllama(batch, seq_len, nlayers=32, dtype=torch.float32):
    #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
    sample = torch.randint(0, 600, [batch, seq_len])
    # Initializing a LLaMA llama-7b style configuration
    configuration = LlamaConfig(num_hidden_layers=nlayers,
                                hidden_size=4096,
                                output_hidden_states=False,
                                output_attentions=False,
                                # use_cache=False
                                )
    # Initializing a model from the llama-7b style configuration
    model = LlamaModel(configuration)#.to(dtype)
    return model, [sample]

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
    exec(model_g, sample_g, print_mem=False, print_loss=True, optimize_fct=optimize)

    optimizer = optim(model_c.parameters())
    def optimize():
        optimizer.step()
    print("cpu model")
    # print(model_c(*sample_c).mean())
    exec(model_c, sample_c, print_mem=False, print_loss=True, optimize_fct=optimize)

    model_r = ORockmate(model, sample, budget, 
                        solve_sched=False,
                        cpu_optim=optim,
                        gpu_optim=optim,
                        ilp_solver = "PULP_CBC_CMD")
    # prepare_for_offload(model_r)
    model_r.solve_sched(budget)
    model_r.zero_grad()
    sample_r = [s.to("cuda") for s in sample]
    print("rockmate model")    
    exec(model_r, sample_r, print_mem=False, print_loss=True)


class LoraLinear(nn.Module):
    def __init__(self, linear, num_adapters=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        u = nn.Parameter(torch.randn(num_adapters, self.linear.weight.shape[0]), requires_grad=True)
        v = nn.Parameter(torch.randn(self.linear.weight.shape[1], num_adapters), requires_grad=True)
        self.register_parameter("u", u)
        self.register_parameter("v", v)

    def forward(self, x):
        res1 = torch.matmul(x, self.v)
        res2 = torch.matmul(res1, self.u)
        y = self.linear(x)
        out = y+res2
        return out

def manual_lora(model:nn.Module, target_modules, num_adapters=10, freeze_all=True):
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False
    for module_name in target_modules:
        module = model.get_submodule(module_name)
        if isinstance(module, nn.Linear):
            new_module = LoraLinear(module, num_adapters=num_adapters)
        else:
            raise TypeError(f"manual lora does not work with {type(module)}")
        # setattr(model, module_name, new_module)

        atoms = module_name.split(".")
        mod: torch.nn.Module = model

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                        "attribute `" + item + "`")
            if getattr(mod, item) == module:
                # setattr(mod, item, new_module)
                mod.add_module(item, new_module)
                break

            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                        "an nn.Module")

def add_sched_stats(stats, rkmod):
    stats[f"Original module iter time"] = sum(kcn.time for kcn in rkmod.rkgb_res.hierarchical_cluster.list_cnodes if kcn.time)
    stats[f"Schedule_total_time"] = sum(step.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_compute_time"] = sum(step.comp_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_offload_time"] = sum(step.ofl_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_prefetch_time"] = sum(step.prf_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_cpu_optimize_time"] = sum(step.opt_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_time_from_waiting_cpu optimize"] = sum((step.time - step.comp_ops.time) for step in rkmod.op_sched.steps if step.time == step.opt_ops.time)
    stats[f"Schedule_time_from_waiting_offload"] = sum((step.time - step.max2nd()) for step in rkmod.op_sched.steps if step.time == step.ofl_ops.time)
    stats[f"Schedule_time_from_waiting_prefetch"] = sum((step.time - step.max2nd()) for step in rkmod.op_sched.steps if step.time == step.prf_ops.time)

def Loss(y, labels=None):
    return y.mean()
    return torch.nn.CrossEntropyLoss()(y, labels)

def exec(model, 
         sample, 
         niters=10, 
         optimize_fct=None,
         print_loss=True,
         print_mem=False,
         zero_grad=True,
          **kwargs):
    torch.random.manual_seed(0)
    def hook_fn(m, args, output):
        # print(f"{output.mean()}")
        return output
    for x in model.children():
        x.register_forward_hook(hook_fn)
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
        if print_mem:print(torch.cuda.memory_allocated())
        if zero_grad:model.zero_grad()
        y = model(*sample, **kwargs)[0]
        assert y.requires_grad
        loss = Loss(y)
        if print_loss:print(f"loss: {loss}")
        loss.backward()
        # torch.cuda.synchronize()
        # print(f"grad: {model.wte.weight.grad[0,0]}")
        if optimize_fct:optimize_fct()
        # for s in sample:
        #     s.grad = None
        y.data = torch.empty(0)
        y.grad = None
        del y
        torch.cuda.empty_cache()
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
    time = exec(model, sample, niters, optimize_fct=None, **kwargs)
    peak_mem = torch.cuda.max_memory_allocated() - mem
    return time, peak_mem

def exec_pt(model, sample, optim=torch.optim.Adam, niters=10, device="cuda", **kwargs):
    torch.random.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    sample = [s.to(device) for s in sample]
    optimizer = optim(model.parameters())
    def optimize():
        optimizer.step()
    mem = torch.cuda.memory_allocated()
    time = exec(model, sample, niters, optimize_fct=optimize, zero_grad=True, **kwargs)
    peak_mem = torch.cuda.max_memory_allocated() - mem
    return time, peak_mem

def exp_pt(nlayers=1, batch_size=3, exp_id=None, num_adapters=None, id="7B", optim=torch.optim.Adam):
    model, sample = get7Bllama(batch_size, 512, nlayers=nlayers)
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
    # exp_stats["optimize_stats"] = rkmod.gd["optimize_stats"]
    exp_stats["cpu_optim"] = str(optim)
    exp_stats["gpu_optim"] = str(optim)
    exp_stats["gpu_type"] = torch.cuda.get_device_name()

    time, mem = exec_pt(model.to("cuda"), sample, niters=niters, optim=optim)
    torch.cuda.synchronize()
    print(time, mem)

    exp_stats["time"] = time/niters
    exp_stats["peak_mem"] = mem
    print(exp_stats)
    # rkmod.compiler.storage.ld[rkmod.output.name.split(" ")[0]].data = torch.empty(0)
    # rkmod.compiler.storage.ld["_"+rkmod.output.name.split(" ")[0]].data = torch.empty(0)
    if exp_id:
        os.makedirs(os.path.dirname(f"exp_results/{exp_id}/"), exist_ok=True)
        print("dump result")
        with open(f"exp_results/{exp_id}/res_{id}_pt.pkl", "wb") as f:
            pickle.dump(exp_stats, f)

def exp_rkmod(nlayers=1, batch_size=3, exp_id=None, num_adapters=None, id="7B"):
    # model, sample = get3Bllm_embed(3,512, nlayers=nlayers)
    model, sample = get7Bllama(batch_size, 512, nlayers=nlayers)
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

    # budget = 8 * 1024**3
    budget = torch.cuda.get_device_properties(0).total_memory - 2*1024**3#to avoid fragmentation
    niters = 5
    partitioners = [rkgb.partitioned.PartitionerRecognizeRepetitivePattern(
        strict_max_number_of_top_level_nodes=nlayers+4,
        max_number_of_patterns=nlayers+2,
        min_percentage_covered_required=0.75)]

    rkmod = ORockmate(model, sample, 1e8, solve_sched=0, 
                    ilp_solver="PULP_CBC_CMD", 
                    #   ilp_solver="HiGHS_CMD", 
                    # cpu_optim = DeepSpeedCPUAdam,
                    partitioners=partitioners,
                    optim_kwargs = {"lr":1e-4},
                    ilp_time_limit=10*60,
                    minor_param_size=4*1024**2,
                    )
    
    print(f"number of HCN: {len(rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs)}")
    print(f"number of HAN: {len(rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HANs)}")
    max_size = max(sum(kdn.mem for kdn in hcn.sub_cluster.parameter_nodes) 
        for hcn in rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs if hcn.sub_cluster)
    print(f"maximum size of subgraph: {max_size}")
    # rkmod.save_to_local(".", "GPT2-12")

    exp_stats = {}
    exp_stats["nlayer"] = nlayers
    exp_stats["input_size"] = [s.shape for s in sample]
    exp_stats["max_subgraph_size"] = max(sum(kdn.mem for kdn in hcn.sub_cluster.parameter_nodes) 
    for hcn in rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs if hcn.sub_cluster)

    exp_stats["model_size"] = sum(p.numel()*p.element_size() for p in model.parameters())
    exp_stats["model_gradient_size"] = sum(p.numel()*p.element_size() for p in model.parameters()
                                           if p.requires_grad)
    exp_stats["act_size"] = sum(kdn.mem for kdn in rkmod.rkgb_res.hierarchical_cluster.list_anodes
                                if "grad" not in kdn.name)
    exp_stats["optimize_stats"] = rkmod.gd["optimize_stats"]
    exp_stats["cpu_optim"] = str(rkmod.gd["cpu_optim"])
    exp_stats["gpu_optim"] = str(rkmod.gd["gpu_optim"])
    exp_stats["gpu_type"] = torch.cuda.get_device_name()

    ### Solve schedule
    rkmod.preprocess()
    rkmod.solve_sched(budget, rec=False)
    md = rkmod.list_solvers[0].md
    if md.feasible:
        # analyze_mem(rkmod)
        pass
    else:
        raise ValueError
    print(md.solving_time)
    exp_stats["solving_time"] = str(md.solving_time)
    # mem_ilp, mem_sched = analyze_mem(rkmod)
    # exp_stats["Mem_ILP"] = mem_ilp
    # exp_stats["Mem_sched"] = mem_sched

    # exp_stats["Before_refine"] = {}
    # add_sched_stats(exp_stats["Before_refine"], rkmod)

    # time, mem = exec_rk(rkmod, sample, niters=niters)
    # exp_stats["Before_refine_time"] = time/niters
    # exp_stats["Before_refine_peak_mem"] = mem

    ### Refine schedule
    # rkmod.op_sched.refine_optimize()
    rkmod.op_sched.simulate_update()
    # exp_stats["After_refine"] = {}
    # add_sched_stats(exp_stats["After_refine"], rkmod)
    # print(exp_stats["After_refine"])
    # rkmod.op_sched.alive_list = rkmod.op_sched.create_alive_list()
    # rkmod.op_sched.refine()
    rkmod.get_compiled_fct()

    # sleep(5)
    time, mem = exec_rk(rkmod, sample, niters=niters)
    torch.cuda.synchronize()
    print(time, mem)

    exp_stats["time"] = time/niters
    exp_stats["peak_mem"] = mem
    opts = list(rkmod.compiler.storage.ld["optimizers"].keys())
    # for s in sample:
    #     s.data = torch.empty(0)
    for opt in opts:
        del rkmod.compiler.storage.ld["optimizers"][opt]
    # rkmod.restore_exec()
    del rkmod
    gc.collect()
    print(exp_stats)
    # rkmod.compiler.storage.ld[rkmod.output.name.split(" ")[0]].data = torch.empty(0)
    # rkmod.compiler.storage.ld["_"+rkmod.output.name.split(" ")[0]].data = torch.empty(0)
    if exp_id:
        os.makedirs(os.path.dirname(f"exp_results/{exp_id}/"), exist_ok=True)
        print("dump result")
        with open(f"exp_results/{exp_id}/res_{id}.pkl", "wb") as f:
            pickle.dump(exp_stats, f)
        # rkmod.save_to_local(f"exp_results/{exp_id}", id=id)
    # return rkmod

if __name__=="__main__":
    # exp_id = datetime.now().strftime('llama-7b')
    # exp_id = "llama-7b-ds"
    # a = torch.empty(2560,1024,1024,device="cuda")
    # a.data = torch.empty(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", metavar=datetime.now().strftime('%d_%m_%H_%M'), type=str)
    parser.add_argument("--nlayers", metavar=32, type=int)
    parser.add_argument("--num_adapters", metavar=256, type=int)
    parser.add_argument("--id", metavar=None, type=str)
    parser.add_argument("--rk", metavar=1, type=int)
    parser.add_argument("--batch_size", metavar=3, type=int)

    args = parser.parse_args()
    exp_id = args.exp_id
    nlayers = args.nlayers
    num_adapters = args.num_adapters
    id = args.id if args.id is not None else f"{nlayers}_{num_adapters}"
    rk = args.rk
    batch_size = args.batch_size
    if rk:
        # print("exp on rockmate")
        exp_rkmod(nlayers=nlayers, batch_size=batch_size, exp_id=exp_id, num_adapters=num_adapters, id=id)
    else:
        exp_pt(nlayers=nlayers, batch_size=batch_size, exp_id=exp_id, num_adapters=num_adapters, id=id)
    