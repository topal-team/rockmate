import torch
import math
import copy
import torch.nn as nn
import torch.nn as nn
import pgb
from pgb.utils import *
import rockmate as rk
import numpy as np
#from rockmate.defs import RK_block_solution
#from rotor.inspection import tensorMsize
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


from example_modules import GPT2
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
context1  = torch.tensor([tokenizer.encode("The planet earth")])
context2  = torch.tensor([tokenizer.encode("I'm upset with those tools")], device=device)
print(context1)

torch.random.manual_seed(0)
model2 = GPT2(nlayers=1,dropout=0, d_model=768, vcb_sz=5500)
#model2.to(device)

for p in model2.parameters():
    p.grad = torch.zeros_like(p)


GPT2_graphs = pgb.make_all_graphs(model2,{"src":context1},impose_device=True)
pgb.print_all_graphs(GPT2_graphs,"GPT2",False)
GPT2_Kg = GPT2_graphs.K_graph
for k,n in GPT2_Kg.dict_nodes.items():
    if n.run_mem.v!=n.fgt_mem.v:
        pass
        #print(n.name, n.run_mem.v - n.fgt_mem.v)
#print(GPT2_Kg.dict_nodes["__15_fv"].get_main_code()) 
budget = 800000
# reload(pytorch_checkmate)
sched_result, g = rk.use_chk.make_sched(GPT2_Kg, budget, use_gurobi=True, verbose=False)


print('max_mem in the solution:', np.max(sched_result.ilp_aux_data.U))

#tmp = RK_block_solution(GPT2_Kg,budget,budget)

translator = rk.use_chk.Sched_to_ops(g, GPT2_Kg)

fwd_ops,bwd_ops = translator.generate_sched_ops(sched_result)
fwd_code = [op.code for op in fwd_ops.body]
bwd_code = [op.code for op in bwd_ops.body]
print("Great! You generated the execution code!")
model2.to(device)
context1 = context1.to("cuda")
model2.wte.weight.grad = torch.ones_like(model2.wte.weight)
print("Memory allocation of model and inputs:", torch.cuda.memory_allocated())

tmp_local = {'self':model2, 'src':context1}

torch.cuda.reset_peak_memory_stats()
max_before = torch.cuda.max_memory_allocated()
allo_before = torch.cuda.memory_allocated()
mem_timeline_real = []

exec(ast_to_str(GPT2_Kg.init_code), globals(), tmp_local)
for i,code in enumerate(fwd_code):
    #print(i,)
    if "loss" in code:
        mem_timeline_real.append(mem_timeline_real[-1])

        continue
    try:
        exec(code, globals(), tmp_local)
    except:
        print(f"fail to execute code{i}")
        print(code)
        break
    mem_timeline_real.append(torch.cuda.memory_allocated()-allo_before)

print("peak memory:", torch.cuda.max_memory_allocated()-max_before)

# Add and delete grad of output by hand. Only for test.
tmp_local["__152_fv"].grad = torch.ones_like(tmp_local["__152_fv"])
#bwd_code[2] += "__152_fv.grad = None"

for i,code in enumerate(bwd_code[:]):
    #print(i,)
    if "loss" in code:
        mem_timeline_real.append(mem_timeline_real[-1])
        continue
    try:
        exec(code, globals(), tmp_local)
    except:
        print(f"fail to execute code{i}")
        print(code)
        exec(code, globals(), tmp_local)
        break
    mem_timeline_real.append(torch.cuda.memory_allocated()-allo_before)
    #print(f"{i}-th op peak memory:", torch.cuda.max_memory_allocated()-max_before)
print("Great! You have executed the code!")
#print(bwd_code[2])
print("=======EXEC SCHED CODE=======")
#print("weight grad:", torch.mean(tmp_local['self'].ln_f.weight.grad))
print("peak memory:", torch.cuda.max_memory_allocated()-max_before)

from checkmate.core.schedule import OperatorEvaluation, DeallocateRegister, AllocateRegister
mem_timeline = []
for i,op in enumerate(sched_result.schedule):
    if isinstance(op, OperatorEvaluation) or isinstance(op, DeallocateRegister):
        mem_timeline.append(sched_result.schedule_aux_data.mem_timeline[i])
        # mem_timeline.append(sb.ram_timeline[i])

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
fig = figure(figsize=(8, 6), dpi=120)

plt.plot(mem_timeline, label='theory')
plt.plot(mem_timeline_real, label='practice')

plt.xlabel('ops')
plt.ylabel('mem')
plt.title('Memory allocation')
plt.legend()

# plt.show()
plt.savefig("mem_alloc.png")
