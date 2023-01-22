import torch
from example_modules import *
from rotor import Checkpointable
from rotor import timing
import pickle
import numpy as np
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
import random
from rotor.inspection import tensorMsize
random.seed(0)

torch.random.manual_seed(0)

def rotor_exp(input_shape=[500,20], nlayers=8,iterate=2,
                origin=True, check_valid=True, print_res=True,
                budgets=[10e9], file="results.pkl"):
    
    GPT2_0 = GPT2_input(nlayers=nlayers,dropout=1e-9, vcb_sz=600).to(device)
    GPT2_1 = GPT2_output(nlayers=nlayers,dropout=1e-9, vcb_sz=600).to(device)
    h = [TransformerBlock().to(device) for _ in range(nlayers)]
    GPT2_rotor = nn.Sequential(*h +[GPT2_1])
    context0 = torch.randint(0,600, [500,20]).to(device)
    x1 = GPT2_0(torch.randint(0,600, [500,20]).to(device))
    chk = Checkpointable(GPT2_rotor, verbosity=0)
    chk.measure(x1)
    
    results = []
    for budget in budgets:
        try:
            chk.compute_sequence(budget - tensorMsize(x1))
        except:
            continue
        for _ in range(1+iterate):
            result = {}
            torch.cuda.reset_peak_memory_stats()
            max_before = torch.cuda.max_memory_allocated()
            allo_before = torch.cuda.memory_allocated()

            timer = timing.make_timer(device)
            timer.start()

            torch.random.manual_seed(0)
            context1 = GPT2_0(context0)
            output = chk(context1)
            torch.random.manual_seed(0)
            output_grad = torch.ones_like(output.data).to(device)
            output.backward(output_grad)
            # grad = input.grad

            timer.end()
            result['runtime'] = timer.elapsed()
            result['mem_limit'] = budget
            result['mem_peak'] = torch.cuda.max_memory_allocated()-max_before
            results.append(result)
            del output_grad
            del output
            
        if print_res:
            print("=======ROTOR MODULE=======")
            print("budget:", budget)
            print("peak memory:", torch.cuda.max_memory_allocated()-max_before)
            print("runtime: %.4f"%timer.elapsed())
            
    with open(file,"wb") as f: pickle.dump(results, f)

budgets = np.arange(0,50)*1e8 + 6e8

rotor_exp(input_shape=[500,20], nlayers=8,iterate=5,
               origin=True, check_valid=True, print_res=True,
               budgets=budgets, file="results_rotor.pkl")


