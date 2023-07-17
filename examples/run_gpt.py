from rockmate import Rockmate
import rkgb
from rockmate.models import get_GPT
from utils import exec_mod, device, giga, nodename, ensure_dir
import torch
import time
import argparse
import sys
sys.setrecursionlimit(10000)

'''This is an example for using Rockmate on the well-known GPT model.
Rockmate provides an implementation of the GPT network on which the rkgb 
tracing tool works perfectly. 

This example instantiates a GPT model, and a sample input. It then compares
the behavior (peak memory usage and iteration time) of two execution modes:
* `std` is the standard Pytorch execution, without any rematerialization
* `rm` uses rockmate, with a budget of 3 GB.

Note: the memory usage *before* starting the iterations (which
represents the weight of the model) is substracted from the peak
memory usage during training, so that only the memory usage of
activations is counted in the 3GB budget.
'''

save_dir = "./solutions/"
ensure_dir(save_dir)

parser = argparse.ArgumentParser("Test script")
parser.add_argument("--load", action='store_true')
parser.add_argument("--no-std", action='store_true')
parser.add_argument("--no-rm", action='store_true')
parser.add_argument("--hostname", default=nodename)
parser.add_argument("--print", action='store_true')
parser.add_argument("--try-hard", action='store_true')
args=parser.parse_args()

nodename = args.hostname

print("--- Using Model: GPT-medium\n")
batch_size = 8
model = get_GPT(model="GPT2-medium").to(device)
sample = [ torch.randint(0,600, [batch_size, 500]).to(device) ]
budget = 3
try_hard_indicator = 'TH' if args.try_hard else 'N'
name = f"GPT2-medium-500-{batch_size}_{budget}_{try_hard_indicator}"
budget *= giga

if (not args.no_std):
    print("--- Standard model, no rematerialization   ----")
    exec_mod(model, sample)
    print("\n")

if(not args.no_rm):
    print("---  Doing rematerialization with Rockmate ----")    
    print(f"     Budget: {budget:8e}B ({budget/giga:.2f} GB)")
    if not args.load:
        print("*** Solving the optimization problem...")
        start = time.time()
        if not args.try_hard:
            rkMod = Rockmate(model, sample, budget)
        else:
            rkMod = Rockmate(model, sample, budget, nb_budget_save=20, nb_budget_peak=20)
        solve_time = time.time() - start
        print("*** Done: ", solve_time, " seconds.")
        rkMod.save_to_file(save_dir, id=f"{name}-RM_{nodename}")
    else:
        rkMod = Rockmate(model, sample, solve=False)
        print("*** Loading solution...")
        rkMod.load_from_file(save_dir, id=f"{name}-RM_{nodename}")

    if args.print:
        rkgb.print(rkMod.rkgb_res)
    print("*** Running optimized model")
    exec_mod(rkMod, sample)
    del rkMod
    print("\n")

