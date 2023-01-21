import torch
import pgb
import rockmate as rk
from copy import deepcopy
from rotor import timing
import numpy as np
import pickle
import time

device = torch.device("cuda")


def copy_run(model, x, repeat=10):
    try:
        _model = deepcopy(model).to(device)
        for n, p in _model.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        _x = deepcopy(x).to(device)
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        timer = timing.make_timer(device)
        times = []
        for _ in range(repeat):
            timer.start()
            y = _model(_x)
            loss = y.mean()
            loss.backward()
            timer.end()
            times.append(timer.elapsed())
        peak_mem = torch.cuda.max_memory_allocated() - max_before
        _model.to("cpu")
        _x.to("cpu")
        res = {"input_size": x.size(), "peak_mem": peak_mem, "times": times}
    except:
        res = {"input_size": 0, "peak_mem": 0, "times": 0}
    return res


def copy_run_rk(model, x, mbudget, repeat=10, nbar=10, nall=10):
    results = []
    budgets = mbudget if hasattr(budget, "__iter__") else [mbudget]
    for budget in budgets:
        res = {}
        res["nb_budget_abar"] = nbar
        res["nb_budget_all"] = nall
        res["input_size"] = x.size()
        try:
            _model = deepcopy(model).to(device)
            for n, p in _model.named_parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            _x = deepcopy(x).to(device)
            start = time.time()
            newmod = rk.CheckpointedModule(
                _model,
                _x,
                mem_limit=budget,
                nb_budget_abar=nbar,
                nb_budget_all=nall,
            )
            end = time.time()
            res["feasible"] = True
            res["solve time"] = end - start
        except:
            res["feasible"] = False
            return None
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        timer = timing.make_timer(device)
        times = []
        try:
            for _ in range(repeat):
                timer.start()
                torch.random.manual_seed(0)
                y = newmod(_x)
                loss = y.mean()
                loss.backward()
                newmod.backward()
                timer.end()
                times.append(timer.elapsed())
            peak_mem = torch.cuda.max_memory_allocated() - max_before
            _model.to("cpu")
            _x.to("cpu")
            res["peak_mem"] = peak_mem

        except:
            res["peak_mem"] = 0
        res["times"] = times
        results.append(res)
    return results


def exp(model, x, budget, run_original=False, repeat=10):
    if run_original:
        orig_res = copy_run(model, x, repeat=repeat)
    rk_res = copy_run_rk(model, x, budget)
    return rk_res


def sanity_check(module, input, mem_limit=None):
    for n, p in module.named_parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    _input = input.clone()
    _module = deepcopy(module)
    # To warm up
    y = module(input)
    loss = y.mean()
    loss.backward()

    module.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()
    timer = timing.make_timer(device)
    timer.start()
    torch.random.manual_seed(0)
    y = module(input)
    loss = y.mean()
    loss.backward()
    timer.end()
    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f"original module peak memory {peak_mem}")
    print("original module time: %.4f" % timer.elapsed())

    newmod = rk.CheckpointedModule(_module, _input, mem_limit=mem_limit)
    # for n, m in newmod.original_mod.named_modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         m.running_mean[:] = 0.0
    #         m.running_var[:] = 1.0

    torch.random.manual_seed(0)
    _y = newmod(_input)
    _loss = _y.mean()
    _loss.backward()
    newmod.backward()

    newmod.reinit()
    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()
    timer = timing.make_timer(device)
    timer.start()
    torch.random.manual_seed(0)
    _y = newmod(_input)
    _loss = _y.mean()
    _loss.backward()
    newmod.backward()
    timer.end()

    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f"rockmate module peak memory {peak_mem}")
    print("rockmate module time: %.4f" % timer.elapsed())

    if torch.allclose(loss, _loss):
        print("Same loss obtained!")

    same_grad = True
    for n, p in _module.named_parameters():
        # print(n)
        if not torch.allclose(
            _module.get_parameter(n), module.get_parameter(n)
        ):
            print("Unequal weight found in:", n)
            same_grad = False

        if (
            _module.get_parameter(n).grad != None
            and module.get_parameter(n).grad != None
        ):
            grad1 = module.get_parameter(n).grad
            grad2 = _module.get_parameter(n).grad
            if not torch.allclose(grad1, grad2):
                print("Unequal grad found in:", n)
                print(torch.mean((grad1 - grad2) / grad1))
                same_grad = False
    if same_grad:
        print("Same grad obtained!")

    module.eval()
    newmod.reinit()
    newmod.eval()
    # _module.eval()
    if torch.allclose(module(input), newmod(_input)):
        print("Same evaluation obtained!")
    else:
        print("Unequal evaluation")


def test_pgb(module, input):
    pgb_res = pgb.make_all_graphs(module, input)
    list_kg = pgb_res.K_graph_list
    kg = pgb_res.K_graph
    print("Generated all the graphs !\n")
    print(f"Equiv classes are : {pgb_res.equivalent_classes}")
    print(
        f"So we have only {len(pgb_res.equivalent_classes)} "
        f"blocks to solve ILP on, instead of {len(list_kg)}\n"
    )
    print("CONCERNING K_graph_list :")
    list_nb_kcn = [len(kg.list_kcn) for kg in list_kg]
    list_nb_kdn = [len(kg.list_kdn) for kg in list_kg]
    tot_nb_kcn = sum(list_nb_kcn)
    tot_nb_kdn = sum(list_nb_kdn)
    str_list_nb_kcn = "+".join(str(i) for i in list_nb_kcn)
    str_list_nb_kdn = "+".join(str(i) for i in list_nb_kdn)
    print(
        f"{len(list_kg)} K_graphs in seq, with :\n"
        f"{str_list_nb_kcn} = {tot_nb_kcn} Comp nodes\n"
        f"{str_list_nb_kdn} = {tot_nb_kdn} Data nodes\n"
        f"=> total of {tot_nb_kcn + tot_nb_kdn} nodes\n"
    )
    print("CONCERNING phantoms impossible to restore :")
    nb_ips = 0
    for kcn in kg.list_kcn:
        deps_ips = kcn.deps_impossible_to_restore
        if len(deps_ips) != 0:
            nb_ips += 1
            print(
                f"{kcn.main_target}'s phantoms must be "
                f"protected, because deps_impossible_to_restore :"
            )
            for kdn, ph_name in deps_ips:
                print(f"deps on {kdn} through {ph_name}")
    print(f"Total nb of special phantoms :  {nb_ips}")
    return pgb_res


def throughput_exp(module, input, batch_sizes, mem_limit=None):
    throughput = {}
    original_batch = input.shape[0]
    original_input = input[0:1]
    # print(torch.cuda.memory_allocated())
    seq_length = input.shape[1]

    def original():

        y = module(input)
        loss = y.mean()
        loss.backward()
        y.grad = None
        y.data = torch.empty(0)
        module.zero_grad()

        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        timer = timing.make_timer(device)
        times = []
        for _ in range(10):
            timer.start()
            y = module(input)
            loss = y.mean()
            loss.backward()
            timer.end()
            times.append(timer.elapsed())
        peak_mem = torch.cuda.max_memory_allocated() - max_before
        print(f"original module peak memory {peak_mem}")
        print("original module time: %.4f" % (np.mean(times)))
        print(f"batch size {original_batch}")
        print(f"throughput: {original_batch / np.mean(times)}")
        throughput[0] = times

        y.grad = None
        y.data = torch.empty(0)

    original()
    # print(torch.cuda.memory_allocated())

    def rockmate(batch_size):
        # input = torch.randint(0, 600, [batch_size, seq_length]).to(device)
        input = original_input.expand([batch_size, *original_input.shape[1:]])
        # batch_size = input.shape[0]
        try:

            newmod = rk.CheckpointedModule(module, input, mem_limit=mem_limit)
            # newmod.get_sequence(mem_limit)
        except:
            throughput[batch_size] = "infeasible"
            return None
        newmod.get_code()
        for n, p in newmod.original_mod.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        y = newmod(input)
        loss = y.mean()
        loss.backward()
        newmod.backward()
        y.grad = None
        y.data = torch.empty(0)
        timer = timing.make_timer(device)
        times = []
        try:
            for _ in range(repeat):
                # print(torch.cuda.memory_allocated())
                # newmod.reinit()

                torch.cuda.reset_peak_memory_stats()
                max_before = torch.cuda.max_memory_allocated()
                timer.start()
                torch.random.manual_seed(0)

                y = newmod.forward(input)
                loss = y.mean()
                loss.backward()
                newmod.backward()
                timer.end()

                peak_mem = torch.cuda.max_memory_allocated() - max_before
                del y
                del loss
                input.grad = None
                times.append(timer.elapsed())
        except:
            # print(newmod.full_code)
            with open("bug.pkl", "wb") as f:
                pickle.dump(newmod.full_code, f)
            raise Exception("failed execution")
        print(f"rockmate module peak memory {peak_mem}")
        print(f"rockmate module budget {newmod.mem_limit}")
        print("rockmate module time: %.4f" % np.mean(times))
        print(f"batch size {batch_size}")
        print(f"throughput: {batch_size / np.mean(times)}\n")
        throughput[batch_size] = times  # batch_size / np.mean(times)

    for batch_size in batch_sizes:
        repeat = 10

        rockmate(batch_size)
        # input.data = torch.empty(0)
        input.grad = None
        module.zero_grad()
    return throughput
