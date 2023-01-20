import torch
import math
import copy
import torch.nn as nn
import torch.nn as nn
import pgb
from pgb.utils import *
import rockmate as rk
import numpy as np
import random
from rotor import timing
import gc

# from rockmate.defs import RK_block_solution
# from rotor.inspection import tensorMsize
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# from example_modules import GPT2
# from transformers import GPT2Tokenizer
from models.GPT import GPT2

def analyze_mem(i, newmod, code=True, diff=False):
    print(f"allo memory: {newmod.allo_mem[i]}")
    print(f"expt memory: {newmod.expect_mem()[i] - newmod.expect_mem()[i - 1]}")
    if diff:
        print(
            f"{newmod.allo_mem[i]-(newmod.expect_mem()[i] - newmod.expect_mem()[i - 1])}"
        )
    if code:
        print(newmod.full_code[i])


def mod(mem=None):
    random.seed(0)
    torch.random.manual_seed(0)
    model2 = GPT2(nlayers=4, dropout=1e-10, vcb_sz=600).to(device)
    # for p in model2.parameters():
    #    p.grad = torch.zeros_like(p)
    model2.zero_grad()
    context1 = torch.randint(0, 600, [200, 20]).to(device)
    d = {"src": context1}
    src = context1
    import warnings

    warnings.filterwarnings("ignore")
    if not mem:
        mem = 5.0e8
    newmod = rk.CheckpointedModule(model2, d, mem_limit=mem)
    # for p in model2.parameters():
    #    p.grad = torch.zeros_like(p)
    model2.zero_grad()
    return newmod


def get_n(i, newmod):
    return newmod.executor.op_list[i].n


def run(newmod, context, bwd=False, stop=-1, device=torch.device("cuda")):
    context1 = context.to(device)
    torch.random.manual_seed(0)
    # for p in newmod.original_mod.parameters():
    #    p.grad = torch.zeros_like(p)
    newmod.storage.add_val("src", context1)
    exec(newmod.init_code, newmod.storage.gd, newmod.storage.ld)
    mem_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()
    doc = []
    for i, c in enumerate(newmod.fwd_code[:stop]):
        op = newmod.executor.op_list[i]  # op.n.get_code()
        # max_before = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        try:
            exec(c, newmod.storage.gd, newmod.storage.ld)
        except:
            print(f"failed to execte {c}")
        # doc.append((i,c, torch.cuda.memory_allocated()-mem_before,
        #            torch.cuda.max_memory_allocated()-max_before))
        # gc.collect()
        d = {}
        d["id"] = i
        d["code"] = c
        d["mem_a"] = torch.cuda.memory_allocated() - mem_before
        d["mem_o"] = torch.cuda.max_memory_allocated() - max_before
        doc.append(d)

        mem_before = torch.cuda.memory_allocated()
    torch.random.manual_seed(0)
    doc.append(doc[-1])
    if bwd:
        newmod.storage.ld[newmod.output].grad = torch.ones_like(
            newmod.storage.ld[newmod.output]
        )
        for i, c in enumerate(newmod.bwd_code):
            op = newmod.executor.op_list[
                i + len(newmod.fwd_code)
            ]  # op.n.get_code()
            torch.cuda.reset_peak_memory_stats()
            # max_before = torch.cuda.max_memory_allocated()
            try:
                exec(c, newmod.storage.gd, newmod.storage.ld)
            except:
                print(f"failed to execte {c}")
            # gc.collect()
            d = {}
            d["id"] = i + len(newmod.fwd_code)
            d["code"] = c
            d["mem_a"] = torch.cuda.memory_allocated() - mem_before
            d["mem_o"] = (
                torch.cuda.max_memory_allocated()
                - torch.cuda.memory_allocated()
            )
            doc.append(d)
            # doc.append((i,c, torch.cuda.memory_allocated()-mem_before,
            #            torch.cuda.max_memory_allocated()-max_before))
            mem_before = torch.cuda.memory_allocated()
    gc.collect()

    return doc


def mem_op(op, print_code=False):
    print("run mem", op.n.run_mem)
    print("del mem", op.n.del_mem)
    if print_code:
        print(op.n.main_code)


def measure(n, g, newmod):
    print(inspection(n, g=g, our_global=newmod.storage.gd))


def find_code(var, code_list):
    for i, c in enumerate(code_list):
        if var in c:
            print(i, c, "\n")


def compare(doc, newmod, print_=True):
    res = []
    for d in doc:
        i = d["id"]
        c = d["code"]
        m = d["mem_a"]
        m_s = d["mem_o"]
        op = newmod.executor.op_list[i]
        if "loss" in op.name:
            res.append((i, c, m, m_s, 0))
        else:
            res.append((i, c, m, m_s, op.mem))
        # if op.is_fgt:
        # if op.n.is_fwd:
        #    if abs(m) != abs(op.n.del_mem.v):
        #        print(i,'real:',m,'theory:',-op.n.del_mem.v)
        #        if print_code: print(c)
        # else:
        #    if abs(m) != abs(op.n.fgt_mem.v):
        #        print(i,'real:',m,'theory:',-op.n.fgt_mem.v)
        #        if print_code: print(c)
        # else:
        #    if abs(m) !=abs(op.n.run_mem.v):
        #        print(i,'real:',m,'theory:',op.n.run_mem.v)
        #        if print_code: print(c)
        if print_:
            print(res[-1])
            print("\n")
    return res


def plot_mem(mem_real, mem_theory):
    import matplotlib.pyplot as plt

    """
    for i,c,m in doc:
        mem_real.append(m)
        op = newmod.executor.op_list[i]
        if op.is_fgt:
            mem_theory.append(-op.n.del_mem.v)
        else:
            mem_theory.append(op.n.del_mem.v)
    """
    plt.plot(np.cumsum(np.array(mem_real)), label="real")
    plt.plot(np.cumsum(np.array(mem_theory)), label="theory")
    plt.legend()
    plt.show()


def experiment(mem, origin=False, check_valid=False, print_res=False):
    result = {}
    newmod = mod(mem)
    context1 = torch.randint(0, 600, [200, 20])

    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()
    allo_before = torch.cuda.memory_allocated()
    timer = timing.make_timer(device)

    timer.start()
    context1 = context1.to(device)
    torch.random.manual_seed(0)
    y1 = newmod.forward(context1)

    rk.utils.ref_print_atoms[0] = False

    # Run loss node by hand
    newmod.storage.ld["loss"] = newmod.storage.ld["_loss"] = torch.mean(y1)
    newmod.storage.ld["loss"].backward()
    torch.random.manual_seed(0)
    newmod.backward()

    timer.end()
    # print('')

    result["runtime"] = timer.elapsed()
    result["runtime_theory"] = (
        newmod.fwd_seq.compute_time() + newmod.bwd_seq.compute_time()
    )
    result["mem_limit"] = newmod.mem_limit
    result["mem_peak"] = torch.cuda.max_memory_allocated() - max_before
    if print_res:
        print("Great! You have executed the code!")
        print("=======ROCKMATE MODULE=======")
        print("Given memory limit:", result["mem_limit"])
        print("Real peak memory:", result["mem_peak"])
        print("Runtime: %.4f" % result["runtime"])

    if origin:
        # device = torch.device('cpu')
        torch.random.manual_seed(0)
        model1 = GPT2(nlayers=8, dropout=1e-8, vcb_sz=600).to(device)
        context1 = torch.clone(context1).to(device)
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        allo_before = torch.cuda.memory_allocated()
        timer = timing.make_timer(device)
        timer.start()

        torch.random.manual_seed(0)
        y = model1(context1)
        loss = torch.mean(y)

        torch.random.manual_seed(0)
        loss.backward()
        timer.end()

        print("=======ORIGINAL MODULE=======")
        print(
            "Real peak memory:", torch.cuda.max_memory_allocated() - max_before
        )
        print("Runtime: %.4f" % timer.elapsed())

    if check_valid:
        if torch.allclose(loss, newmod.storage.ld["loss"]):
            print("Same loss obtained!")

        same_grad = True
        model2 = newmod.original_mod
        for n, p in model2.named_parameters():
            if not torch.allclose(
                model2.get_parameter(n).to(device), model1.get_parameter(n)
            ):
                print("Unequal weight found in:", n)
                same_grad = False

            if model1.get_parameter(n).grad != None:
                if not torch.allclose(
                    model2.get_parameter(n).grad.to(device),
                    model1.get_parameter(n).grad,
                ):
                    print("Unequal grad found in:", n)
                    same_grad = False
        if same_grad:
            print("Same grad obtained!\n")

    return result
