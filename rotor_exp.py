import torch
from rotor_exp.rotor import Checkpointable
from rotor import timing
import pickle
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import random
from rotor.inspection import tensorMsize
from copy import deepcopy
import time
from models.GPT import *
from exp_utils import *

random.seed(0)

torch.random.manual_seed(0)
tmp = torch.ones(14, 1024, 1024, 256, device=device)
del tmp


def copy_run_rt(model, x, mbudget, repeat=10):
    results = []
    budgets = mbudget if hasattr(mbudget, "__iter__") else [mbudget]
    # _model = deepcopy(model).to(device)
    _x = deepcopy(x).to(device)

    nlayers = model.nlayers
    GPT2_0 = GPT2_input(d_model=model.d_model).to(device)
    GPT2_1 = GPT2_output(d_model=model.d_model)
    h = [
        TransformerBlock(d_model=model.d_model, n_head=model.n_head)
        for _ in range(nlayers)
    ]
    model_rotor = nn.Sequential(*h + [GPT2_1]).to(device)
    allo_before = torch.cuda.memory_allocated()
    x_rotor = GPT2_0(x.clone().to(device))
    abar = torch.cuda.memory_allocated() - allo_before - tensorMsize(x_rotor)
    start = time.time()
    chk = Checkpointable(model_rotor, verbosity=0)
    for n, p in model_rotor.named_parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    chk.measure(x_rotor)
    end = time.time()
    measure_time = end - start
    for budget in budgets:
        res = {}
        res["measure time"] = measure_time
        res["input_size"] = x.size()
        res["budget"] = budget

        try:
            start = time.time()
            chk.compute_sequence(budget - abar)
            end = time.time()
            res["expect_peak"] = chk.get_expected_memory()
            res["expect_time"] = chk.get_expected_makespan()
            res["abar"] = abar
            res["feasible"] = True
            res["DP solve time"] = end - start
            # res["simulation overhead"] = newmod.simulation_overhead

        except:
            res["feasible"] = False
            results.append(res)
            continue
        timer = timing.make_timer(device)
        times = []
        try:
            _x = deepcopy(x).to(device)
            torch.cuda.reset_peak_memory_stats()
            max_before = torch.cuda.max_memory_allocated()
            for _ in range(repeat):
                timer.start()
                torch.random.manual_seed(0)
                _x_rotor = GPT2_0(_x)
                output = chk(_x_rotor)
                # output_grad = torch.ones_like(output.data).to(device)
                # output.backward(output_grad)
                loss = output.mean()
                loss.backward()
                timer.end()
                times.append(timer.elapsed())
                # del output_grad
                del loss
                output.grad = None
                _x_rotor.grad = None
                _x_rotor.data = torch.empty(0)
                output.data = torch.empty(0)
                # del output
                # del _x_rotor

            peak_mem = torch.cuda.max_memory_allocated() - max_before
            res["peak_mem"] = peak_mem
        except Exception as e:
            res["peak_mem"] = 0
            res["Error"] = f"caught {type(e)}: {e}"
            if type(e) != torch.cuda.OutOfMemoryError:
                print("OOM")
        res["times"] = times
        results.append(res)
    model_rotor.to("cpu")
    _x.to("cpu")
    return results


input_sizes = {}
input_sizes["GPT2-small"] = [8, 512]
input_sizes["GPT2-medium"] = [4, 512]
input_sizes["GPT2-large"] = [2, 512]
input_sizes["GPT2-xl"] = [1, 256]

budgets = {}
budgets["GPT2-small"] = np.arange(0.6, 7, 0.2) * 1024 ** 3
budgets["GPT2-medium"] = np.arange(0.6, 7.6, 0.2) * 1024 ** 3
budgets["GPT2-large"] = np.arange(0.4, 7, 0.2) * 1024 ** 3
budgets["GPT2-xl"] = np.arange(0.4, 2.5, 0.2) * 1024 ** 3

results = {}

for name in ["GPT2-small", "GPT2-medium", "GPT2-large", "GPT2-xl"]:
    model = get_GPT(name)
    x = torch.randint(0, 600, input_sizes[name])
    mbudget = budgets[name]

    results[f"rotor {name}"] = copy_run_rt(model, x, mbudget, repeat=10)
    with open("rotor_exp.pkl", "wb") as f:
        pickle.dump(results, f)

    results[f"rockmate {name}"] = copy_run_rk(model, x, mbudget, repeat=10)

    with open("rotor_exp.pkl", "wb") as f:
        pickle.dump(results, f)
