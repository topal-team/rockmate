import torch
import rkgb
import rockmate as rk
from copy import deepcopy
from rotor import timing
import pickle
import time
import sys
from models.GPT import *

sys.setrecursionlimit(30000)

device = torch.device("cuda")


def copy_run(model, inputs, dict_kwargs=None, repeat=10, return_mod=False):
    #  copy model, inputs and dict_kwargs
    dict_inputs = rkgb.make_inputs(model, inputs.to(device), dict_kwargs)
    _dict_inputs = dict()
    for k, v in dict_inputs.items():
        if isinstance(v, torch.Tensor):
            _dict_inputs[k] = v.clone()
        else:
            _dict_inputs[k] = deepcopy(v)
    try:
        _model = deepcopy(model).to(device)
        for n, p in _model.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        # _x = deepcopy(x).to(device)
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        timer = timing.make_timer(device)
        times = []
        for _ in range(repeat):
            timer.start()
            torch.random.manual_seed(0)
            y = _model(**_dict_inputs)
            loss = y.mean()
            loss.backward()
            timer.end()
            times.append(timer.elapsed())
        peak_mem = torch.cuda.max_memory_allocated() - max_before
        # _model.to("cpu")
        # _x.to("cpu")
        res = {"peak_mem": peak_mem, "times": times}
    except Exception as e:
        res = {}
        res["peak_mem"] = 0
        res["Error"] = f"caught {type(e)}: {e}"
    if return_mod:
        res["module"] = _model.to("cpu")
    return res


def copy_run_rk(
    model,
    inputs,
    mbudget,
    dict_kwargs=None,
    repeat=10,
    nbar=10,
    nall=2,
    return_mod=False,
):
    #  copy model, inputs and dict_kwargs
    dict_inputs = rkgb.make_inputs(model, inputs.to(device), dict_kwargs)
    _dict_inputs = dict()
    for k, v in dict_inputs.items():
        if isinstance(v, torch.Tensor):
            _dict_inputs[k] = v.clone()
        else:
            _dict_inputs[k] = deepcopy(v)
    results = []
    budgets = mbudget if hasattr(mbudget, "__iter__") else [mbudget]
    _model = deepcopy(model).to(device)
    start = time.time()
    rkmod = rk.CheckpointedModule(
        _model,
        _dict_inputs,
        mem_limit=max(budgets),
        nb_budget_abar=nbar,
        nb_budget_all=nall,
        get_sequence=False,
    )
    end = time.time()
    ilp_time = end - start
    for budget in budgets:
        res = {}
        res["ILP solve time"] = ilp_time
        res["nb_budget_abar"] = nbar
        res["nb_budget_all"] = nall
        res["budget"] = budget
        try:
            for n, p in _model.named_parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            start = time.time()
            rkmod.get_sequence(budget)
            end = time.time()
            res["feasible"] = True
            res["DP solve time"] = end - start
            res["simulation overhead"] = rkmod.simulation_overhead
            rkmod.get_compiled_fct()
        except Exception as e:
            res["feasible"] = False
            res["Error"] = f"caught {type(e)}: {e}"
            results.append(res)
            continue
            # return results
        _x = _dict_inputs.copy()
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        timer = timing.make_timer(device)
        times = []
        rkmod.reinit()

        try:
            for _ in range(repeat):
                timer.start()
                torch.random.manual_seed(0)
                y = rkmod(**_x)
                loss = y.mean()
                loss.backward()
                rkmod.backward()
                timer.end()
                times.append(timer.elapsed())
            peak_mem = torch.cuda.max_memory_allocated() - max_before

            res["peak_mem"] = peak_mem

        except Exception as e:
            res["peak_mem"] = 0
            res["Error"] = f"caught {type(e)}: {e}"
            if type(e) != torch.cuda.OutOfMemoryError:
                with open("rk_mod.pkl", "wb") as f:
                    torch.save(rkmod, f)
                raise e
        res["times"] = times
        if return_mod:
            res["module"] = rkmod.to("cpu")
        results.append(res)

    _model.to("cpu")
    # _x.to("cpu")

    return results


def sanity_check(model1, model2, inputs, dict_kwargs=None, device="cuda"):
    module = model1.original_mod
    dict_inputs = rkgb.make_inputs(model2, inputs.to(device), dict_kwargs)
    _dict_inputs = dict()
    for k, v in dict_inputs.items():
        if isinstance(v, torch.Tensor):
            _dict_inputs[k] = v.clone()
        else:
            _dict_inputs[k] = deepcopy(v)

    model1.train()
    model2.train()
    torch.random.manual_seed(0)
    y1 = model1(**_dict_inputs)
    torch.random.manual_seed(0)
    y2 = model2(**dict_inputs)
    same_train = torch.allclose(y1, y2)

    model1.eval()
    model2.eval()
    torch.random.manual_seed(0)
    y1 = model1(**_dict_inputs)
    torch.random.manual_seed(0)
    y2 = model2(**dict_inputs)
    same_eval = torch.allclose(y1, y2)

    same_grad = True
    for n, _ in model2.named_parameters():
        if not torch.allclose(model2.get_parameter(n), module.get_parameter(n)):
            print("Unequal weight found in:", n)
            same_grad = False

        if (
            model2.get_parameter(n).grad != None
            and module.get_parameter(n).grad != None
        ):
            grad1 = module.get_parameter(n).grad
            grad2 = model2.get_parameter(n).grad
            if not torch.allclose(grad1, grad2):
                print("Unequal grad found in:", n)
                print(torch.mean((grad1 - grad2) / grad1))
                same_grad = False

    return same_train, same_eval, same_grad


def compare_model(model, inputs, budgets, dict_kwargs=None, repeat=10):
    res_rk = copy_run_rk(
        model,
        inputs,
        budgets,
        dict_kwargs=dict_kwargs,
        return_mod=True,
        repeat=repeat,
    )
    res_og = copy_run(
        model, inputs, dict_kwargs=dict_kwargs, return_mod=True, repeat=repeat
    )
    mod = res_og["module"].to(device)
    for res in res_rk:
        if not res["feasible"]:
            print(res["Error"])
        rkmod = res["module"].to(device)
        same_train, same_eval, same_grad = sanity_check(
            rkmod, mod, inputs, dict_kwargs=dict_kwargs, device=device
        )
        assert same_train, "different output with model.train()"
        assert same_eval, "different output with model.eval()"
        assert same_grad, "different gradients of parameters"
        assert (
            res["peak_mem"] <= res["budget"]
        ), f"given budget {res['budget']}, peak memory {res['peak_mem']}"

