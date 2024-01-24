import os
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from models import *
from models.LLM import *
from rockmate import HRockmate
from rockmate.solvers.op_schedule import *
# from rkgb.Ptools import Partitioner_bottom_to_top
from rockmate.solvers.HILP_pulp_ofl_para import *
from datetime import datetime
# from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
device = torch.device("cuda")
import pickle
from time import sleep
from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
import psutil
from tmp_utils import *
import rkgb
from rkgb.lowlevel.measure import TimerCUDA

import sys
sys.setrecursionlimit(30000)
timer = TimerCUDA(torch.device("cuda"))
import tracemalloc
tracemalloc.start()
torch.random.manual_seed(0)

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

    model_r = HRockmate(model, sample, budget, 
                        solve_sched=False,
                        cpu_optim=optim,
                        gpu_optim=optim,
                        ilp_solver = "PULP_CBC_CMD")
    prepare_for_offload(model_r)
    model_r.solve_sched(budget)
    model_r.zero_grad()
    sample_r = [s.to("cuda") for s in sample]
    print("rockmate model")    
    exec(model_r, sample_r, print_mem=False, print_loss=True)


def add_sched_stats(stats, rkmod):
    stats[f"Original module iter time"] = sum(kcn.time for kcn in rkmod.rkgb_res.H_cluster.list_kcn if kcn.time)
    stats[f"Schedule_total_time"] = sum(step.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_compute_time"] = sum(step.comp_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_offload_time"] = sum(step.ofl_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_prefetch_time"] = sum(step.prf_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_total_cpu_optimize_time"] = sum(step.opt_ops.time for step in rkmod.op_sched.steps)
    stats[f"Schedule_time_from_waiting_cpu optimize"] = sum((step.time - step.comp_ops.time) for step in rkmod.op_sched.steps if step.time == step.opt_ops.time)
    stats[f"Schedule_time_from_waiting_offload"] = sum((step.time - step.max2nd()) for step in rkmod.op_sched.steps if step.time == step.ofl_ops.time)
    stats[f"Schedule_time_from_waiting_prefetch"] = sum((step.time - step.max2nd()) for step in rkmod.op_sched.steps if step.time == step.prf_ops.time)

def Loss(y, labels):
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
    labels = torch.randn(y.shape).softmax(dim=1).to(sample[0].device)
    if print_mem:print(torch.cuda.memory_allocated())
    loss = Loss(y, labels)
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
        loss = Loss(y, labels)
        if print_loss:print(f"loss: {loss}")
        loss.backward()
        torch.cuda.synchronize()
        # print(f"grad: {model.wte.weight.grad[0,0]}")
        if optimize_fct:optimize_fct()
        # for s in sample:
        #     s.grad = None
        y.data = torch.empty(0)
        y.grad = None
        del y
        
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


def exp_rkmod(nlayers=1, exp_id=None):
    model, sample = get3Bllm_embed(3,512, nlayers=nlayers)
    for n, p in model.named_parameters():
        if "mlps" in n:
            p.requires_grad = False

    budget = 8 * 1024**3
    niters = 10
    partitioners = [
                    Partitioner_bottom_to_top(max_estimate_for_main_graph=model.nlayers*2+3,
                                            can_use_rotor=False)
                ]
    rkmod = HRockmate(model, sample, 1e8, solve_sched=0, 
                    ilp_solver="PULP_CBC_CMD", 
                    #   ilp_solver="HiGHS_CMD", 
                    # cpu_optim = DeepSpeedCPUAdam,
                    partitioners=partitioners,
                    optim_kwargs = {"lr":1e-3},
                    ilp_time_limit=10*60
                    )
    prepare_for_offload(rkmod)


    exp_stats = {}
    exp_stats["nlayer"] = nlayers
    exp_stats["input_size"] = [s.shape for s in sample]
    exp_stats["model_size"] = sum(kdn.mem for kdn in 
                                rkmod.rkgb_res.H_cluster.list_kdn_parameters)
    exp_stats["act_size"] = sum(kdn.mem for kdn in rkmod.rkgb_res.H_cluster.list_kdn
                                if "grad" not in kdn.name)
    exp_stats["optimize_stats"] = rkmod.gd["optimize_stats"]
    exp_stats["cpu_optim"] = str(rkmod.gd["cpu_optim"])
    exp_stats["gpu_optim"] = str(rkmod.gd["gpu_optim"])
    exp_stats["gpu_type"] = torch.cuda.get_device_name()

    ### Solve schedule
    rkmod.solve_sched(budget, rec=True)
    md = rkmod.list_solvers[0].md
    if md.feasible:
        analyze_mem(rkmod)
    else:
        raise ValueError

    exp_stats["Before_refine"] = {}
    add_sched_stats(exp_stats["Before_refine"], rkmod)

    # time, mem = exec_rk(rkmod, sample, niters=niters)
    # exp_stats["Before_refine_time"] = time/niters
    # exp_stats["Before_refine_peak_mem"] = mem

    ### Refine schedule
    rkmod.op_sched.refine_optimize()
    exp_stats["After_refine"] = {}
    add_sched_stats(exp_stats["After_refine"], rkmod)

    rkmod.op_sched.alive_list = rkmod.op_sched.create_alive_list()
    rkmod.op_sched.refine()
    rkmod.get_compiled_fct(new_compiler=False)

    sleep(5)
    time, mem = exec_rk(rkmod, sample, niters=niters)
    torch.cuda.synchronize()
    print(time, mem)

    exp_stats["time"] = time/niters
    exp_stats["peak_mem"] = mem
    id=f"3b_{nlayers}"
    # opts = list(rkmod.compiler.storage.ld["optimizers"].keys())
    # for s in sample:
    #     s.data = torch.empty(0)
    # for opt in opts:
    #     del rkmod.compiler.storage.ld["optimizers"][opt]
    # rkmod.restore_exec()
    # rkmod.compiler.storage.ld[rkmod.output.name.split(" ")[0]].data = torch.empty(0)
    # rkmod.compiler.storage.ld["_"+rkmod.output.name.split(" ")[0]].data = torch.empty(0)
    if exp_id:
        os.makedirs(os.path.dirname(f"exp_results/{exp_id}/"), exist_ok=True)
        with open(f"exp_results/{exp_id}/res_{id}.pkl", "wb") as f:
            pickle.dump(exp_stats, f)
        rkmod.save_to_local(f"exp_results/{exp_id}", id=id)
    return rkmod

if __name__=="__main__":
    exp_id = datetime.now().strftime('%d_%m_%H_%M')
    for nlayer in [1,2,6,12,18,24,30][::-1]:
        exp_rkmod(nlayers=nlayer, exp_id=exp_id)
