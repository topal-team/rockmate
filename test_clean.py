import torch
import pgb
from example_modules import GPT2
import rockmate as rk
from rotor import timing
import pickle
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.random.manual_seed(0)
#
model2 = GPT2(nlayers=12,dropout=1e-9, vcb_sz=600)
context1 = torch.randint(0,600, [1000,20])
d = {"src":context1}
src = context1
import warnings ; warnings.filterwarnings("ignore")
newmod = rk.CheckpointedModule(model2,d, mem_limit = 4e9)
#with open("/beegfs/xzhao/newmod.pk","wb") as f:
#    pickle.dump(newmod, f)
for p in model2.parameters():
    p.grad = None
    
rk.utils.ref_print_atoms[0] = False#True 
print(newmod.fwd_seq) 
print(newmod.bwd_seq) 
print("")

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

print("Great! You have executed the code!")
print("=======ROCKMATE MODULE=======")
print("peak memory:", torch.cuda.max_memory_allocated()-max_before)
print("runtime: %.4f"%timer.elapsed())

torch.random.manual_seed(0)
model1 = GPT2(nlayers=12,dropout=1e-9, vcb_sz=600).to(device)
context1 = torch.clone(context1)
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
print("peak memory:", torch.cuda.max_memory_allocated()-max_before)
print("runtime: %.4f"%timer.elapsed())

if torch.allclose(loss, newmod.storage.ld["loss"]):
    print("Same loss obtained!")

same_grad = True
for n,p in model2.named_parameters():
    if not torch.allclose(model2.get_parameter(n), model1.get_parameter(n)):
        print("Unequal weight found in:", n)
        same_grad = False
        
    if model2.get_parameter(n).grad!=None:
        grad1 = model1.get_parameter(n).grad
        grad2 = model2.get_parameter(n).grad
        if not torch.allclose(grad1,grad2):
            print("Unequal grad found in:", n)
            print(torch.mean((grad1-grad2)/grad1))
            same_grad = False
if same_grad:
    print("Same grad obtained!")
if False:
    for c in newmod.executor.done:
        print(c)
    for c in newmod.fwd_code+newmod.bwd_code:
        print(c)
