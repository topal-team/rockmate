import torch
from rkgb.utils.imports_from_rotor import make_timer
import time
import platform
import tqdm
import os

nodename = platform.node().split('.')[0]

device = torch.device("cuda")
giga = 1024*1024*1024

def exec_mod(module, inputs, repeat=15):
    
    for n, p in module.named_parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    times = []
    timer = make_timer(device)
    try:
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        print(f"*********************CUDA memory usage before is     {max_before:8e}B ({max_before/giga:.2f} GB)")

        for _ in tqdm.trange(repeat+1):
            timer.start()
            y = module(*inputs)
            loss = y.mean()
            loss.backward()
            if hasattr(module, "backward"):
                module.backward()

            timer.end()
            times.append(timer.elapsed())

            del y
            del loss
        
        peak_mem = torch.cuda.max_memory_allocated() - max_before
        avg_time = sum(times[1:]) / (len(times) - 1)
        print(f"*********************CUDA max memory usage during is { torch.cuda.max_memory_allocated():8e}B ({ torch.cuda.max_memory_allocated()/giga:.2f} GB)")
        print(f"*********************CUDA measured peak_mem is       {peak_mem:8e}B ({peak_mem/giga:.2f} GB)")
        print(f"              *******Average time is {avg_time:.2f} ms")

    except Exception as e:
        print("********Error")
        print(e)
        peak_mem = None
        if type(e) != torch.cuda.OutOfMemoryError:
            raise e
    
    del inputs
    torch.cuda.empty_cache()
    
    return peak_mem, times


def ensure_dir(directory):
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
