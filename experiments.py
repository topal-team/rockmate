#from mem_tools import *
from rotor import timing
import pickle
import numpy as np
import random
import torch
import torch.nn as nn
import rockmate as rk
from example_modules import GPT2
from transformers import GPT2Tokenizer
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# def mod(mem=None, input_shape=[500,20], nlayers=8):
#     random.seed(0)
#     torch.random.manual_seed(0)
#     model2 = GPT2(nlayers=nlayers,dropout=1e-10, vcb_sz=600).to(device)
#     #for p in model2.parameters():
#     #    p.grad = torch.zeros_like(p)
#     model2.zero_grad()
#     context1 = torch.randint(0,600, input_shape).to(device)
#     d = {"src":context1}
#     src = context1
#     import warnings ; warnings.filterwarnings("ignore")
#     if not mem:mem = 5.e8
#     newmod = rk.CheckpointedModule(model2,d, mem_limit = mem)
#     #for p in model2.parameters():
#     #    p.grad = torch.zeros_like(p)
#     model2.zero_grad()
#     return newmod

def experiments(input_shape=[500,20], nlayers=8,iterate=False,
               origin=False, check_valid=False, print_res=False,
               nb = (10,5), budgets=[1e10], file="results.pkl"):
    context1 = torch.randint(0,600, input_shape)
    if origin:
        # device = torch.device('cpu')
        torch.random.manual_seed(0)
        model1 = GPT2(nlayers=nlayers,dropout=1e-8, vcb_sz=600).to(device)
        context1 = torch.clone(context1).to(device)
        torch.cuda.reset_peak_memory_stats()
        torch.random.manual_seed(0)
        y = model1(context1)
        loss = torch.mean(y)

        torch.random.manual_seed(0)
        loss.backward()
        del y
        model1.zero_grad()
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
        print("Real peak memory:", torch.cuda.max_memory_allocated()-max_before)
        print("Runtime: %.4f"%timer.elapsed())

    torch.random.manual_seed(0)
    model2 = GPT2(nlayers=nlayers,dropout=1e-8, vcb_sz=600).to(device)
    d = {"src":context1}
    newmod = rk.CheckpointedModule(model2,d)
    results = []
    newmod.get_chain(*nb)
    for budget in budgets:
        try:
            newmod.get_sequence(budget)
        except:
            # print(f"Not enough budget {budget}")
            continue
        newmod.get_code()
        for _ in range(1+iterate):
            newmod.reinit()
            context1 = context1.to(device)
            result = {}
            torch.cuda.reset_peak_memory_stats()
            max_before = torch.cuda.max_memory_allocated()
            allo_before = torch.cuda.memory_allocated()
            timer = timing.make_timer(device)

            timer.start()
            
            torch.random.manual_seed(0)
            y1 = newmod.forward(context1)

            rk.utils.ref_print_atoms[0] = False

            # Run loss node by hand
            newmod.storage.ld["loss"] = newmod.storage.ld["_loss"] = torch.mean(y1)
            newmod.storage.ld["loss"].backward()
            torch.random.manual_seed(0)
            newmod.backward()
            timer.end()
        
            result['nlayers'] = nlayers
            result['input_shape'] = input_shape
            result['nb_solution'] = nb
            result['runtime'] = timer.elapsed()
            result['runtime_theory'] = newmod.fwd_seq.compute_time()+newmod.bwd_seq.compute_time()
            result['mem_limit'] = newmod.mem_limit
            result['mem_peak'] = torch.cuda.max_memory_allocated()-max_before
            results.append(result)
            
        if print_res:
            print("Great! You have executed the code!")
            print("=======ROCKMATE MODULE=======")
            print("Given memory limit:", result['mem_limit'])
            print("Real peak memory:", result['mem_peak'])
            print("Runtime: %.4f"%result['runtime'])
        
        if check_valid:
            # if torch.allclose(loss, newmod.storage.ld["loss"]):
                # pass
                # print("Same loss obtained!")

            same_grad = True
            model2 = newmod.original_mod
            for n,p in model2.named_parameters():
                if not torch.allclose(model2.get_parameter(n).to(device), model1.get_parameter(n)):
                    print("Unequal weight found in:", n)
                    same_grad = False

                if model1.get_parameter(n).grad!=None:
                    if not torch.allclose(model2.get_parameter(n).grad.to(device), model1.get_parameter(n).grad):
                        print("Unequal grad found in:", n)
                        same_grad = False
            if same_grad:
                pass
                # print("Same grad obtained!\n")
        with open(file,'wb') as f:pickle.dump(results, f)
    return results

budgets = np.arange(0,50)*1e8 + 6e8

experiments(input_shape=[500,20], nlayers=8,iterate=5,
               origin=True, check_valid=True, print_res=True,
               nb = (20,10), budgets=budgets, file="results.pkl")


