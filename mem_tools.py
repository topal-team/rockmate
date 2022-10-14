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
#from rockmate.defs import RK_block_solution
#from rotor.inspection import tensorMsize
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
from example_modules import GPT2
from transformers import GPT2Tokenizer

def mod():
    random.seed(0)
    torch.random.manual_seed(0)
    model2 = GPT2(nlayers=4,dropout=1e-10, vcb_sz=600)
    context1 = torch.randint(0,600, [10,10])
    d = {"src":context1}
    src = context1
    import warnings ; warnings.filterwarnings("ignore")
    newmod = rk.CheckpointedModule(model2,d, mem_limit = 3.2e7)
    for p in model2.parameters():
        p.grad = torch.zeros_like(p)
    return newmod

def get_n(i,newmod):
    return newmod.executor.op_list[i].n

def run(newmod, context, bwd=False, 
        stop=-1, device=torch.device('cuda')):
    context1 = context.to(device)
    torch.random.manual_seed(0)
    #for p in newmod.original_mod.parameters():
    #    p.grad = torch.zeros_like(p)
    newmod.storage.add_val("src",context1)
    exec(newmod.init_code,newmod.storage.gd,newmod.storage.ld)
    mem_before  = torch.cuda.memory_allocated()
    doc = []
    for i,c in enumerate(newmod.fwd_code[:stop]):
        op = newmod.executor.op_list[i]#op.n.get_code()
        try:
            exec(c, newmod.storage.gd,newmod.storage.ld)
        except:print(f'failed to execte {c}')
        doc.append((i,c, torch.cuda.memory_allocated()-mem_before))
        mem_before = torch.cuda.memory_allocated()
    if bwd:
        newmod.storage.ld[newmod.output].grad = torch.ones_like(newmod.storage.ld[newmod.output])
        for i,c in enumerate(newmod.bwd_code):
            op = newmod.executor.op_list[i+len(newmod.fwd_code)]#op.n.get_code()
            try:
                exec(c, newmod.storage.gd,newmod.storage.ld)
            except:print(f'failed to execte {c}')
            doc.append((i+len(newmod.fwd_code),c, torch.cuda.memory_allocated()-mem_before))
            mem_before = torch.cuda.memory_allocated()

    return doc

def mem_op(op, print_code=False):
    print("run mem", op.n.run_mem)
    print("del mem", op.n.del_mem)
    if print_code:
        print(op.n.main_code)
        
def measure(n, g, newmod):
    print(inspection(n, g=g, our_global=newmod.storage.gd))

def find_code(var, code_list):
    for c in code_list:
        if var in c:
            print(c)

def compare(doc, newmod,print_code=True):
    for i,c,m in doc:
        op = newmod.executor.op_list[i]
        if op.is_fgt:
            if op.n.is_fwd:
                if abs(m) != abs(op.n.del_mem.v):
                    print(i,'real:',m,'theory:',-op.n.del_mem.v)
                    if print_code: print(c)
                    print('\n')
            else:
                if abs(m) != abs(op.n.fgt_mem.v):
                    print(i,'real:',m,'theory:',-op.n.fgt_mem.v)
                    if print_code: print(c)
                    print('\n')

        else:
            if abs(m) !=abs(op.n.run_mem.v):
                print(i,'real:',m,'theory:',op.n.run_mem.v)
                if print_code: print(c)
                print('\n')

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
    plt.plot(np.cumsum(np.array(mem_real)), label='real')
    plt.plot(np.cumsum(np.array(mem_theory)), label='theory')
    plt.legend()
    plt.show()
