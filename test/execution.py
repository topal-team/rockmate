import argparse
import os
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
# from models import *
# from models.LLM import *
from rockmate import Rockmate
from rockmate.op_schedule import *
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

import sys
sys.setrecursionlimit(30000)
timer = TimerCUDA(torch.device("cuda"))
import tracemalloc
tracemalloc.start()
torch.random.manual_seed(0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"


def check_correctness(model, sample, budget=1e9, optim=torch.optim.Adam, dtype=None):
    model_g = deepcopy(model).to("cuda")
    sample_g = [s.to("cuda") for s in sample]
    model_c = deepcopy(model).to("cpu")
    sample_c = [s.to("cpu") for s in sample]
    if dtype:
        model_g.to(dtype)
        model_c.to(dtype)
        sample_g = [s.to("cuda").to(dtype) for s in sample]
        sample_c = [s.to("cuda").to(dtype) for s in sample]

    optimizer = optim(model_g.parameters())
    def optimize():
        optimizer.step()
    print("gpu model")
    # print(model_g(*sample_g).mean())
    execution(model_g, sample_g, print_mem=False, print_loss=True, optimize_fct=optimize)
    out_g = model_g(*sample_g)

    optimizer = optim(model_c.parameters())
    def optimize():
        optimizer.step()
    print("cpu model")
    # print(model_c(*sample_c).mean())
    execution(model_c, sample_c, print_mem=False, print_loss=True, optimize_fct=optimize)
    out_c = model_c(*sample_c)

    optimizer = optim(model.parameters())
    def optimize():
        optimizer.step()
    model_r = Rockmate(model, sample, budget, 
                        solve_sched=False,
                        cpu_optim=optim,
                        gpu_optim=optim
                        )
    # model_r.solve_sched(budget)
    model_r.fast_solve()
    model_r.get_compiled_fct()
    model_r.zero_grad()
    sample_r = [s.to("cuda") for s in sample]
    print("rockmate model")    
    execution(model_r, sample_r, print_mem=False, print_loss=True, optimize_fct=optimize)
    out_r = model_r(*sample_r)
    return out_g, out_c, out_r

def Loss(y, labels=None):
    return y.mean()
    return torch.nn.CrossEntropyLoss()(y, labels)

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
    y = model(*sample, **kwargs)[0]
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
        if print_mem:print(f"Mem after bwd {torch.cuda.memory_allocated()}")
        if optimize_fct:optimize_fct()
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

def exec_pt(model, sample, optim=torch.optim.Adam, niters=10, device="cuda", **kwargs):
    torch.random.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    sample = [s.to(device) for s in sample]
    optimizer = optim(model.parameters(), lr=1e-3)
    def optimize():
        optimizer.step()
    mem = torch.cuda.memory_allocated()
    print(f"Memory allocated before exec {mem}")
    time = execution(model, sample, niters, optimize_fct=optimize, zero_grad=True, **kwargs)
    peak_mem = torch.cuda.max_memory_allocated() - mem
    return time, peak_mem
