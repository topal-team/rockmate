import torch
import pgb
from example_modules import GPT2
import rockmate as rk

torch.random.manual_seed(0)
model2 = GPT2(nlayers=2,dropout=0, vcb_sz=6000)
context1 = torch.tensor([[ 464, 5440, 4534]])
d = {"src":context1}

import warnings ; warnings.filterwarnings("ignore")
import sys; sys.tracebacklimit = 0
newmod = rk.CheckpointedModule(model2,d)

for p in model2.parameters():
    p.grad = None
    
print("")
context1 = context1.to('cuda')
torch.random.manual_seed(0)
y1 = newmod.forward(context1)

rk.utils.ref_print_atoms[0] = False

# Run loss node by hand
newmod.storage.ld["loss"] = newmod.storage.ld["_loss"] = torch.mean(y1)
newmod.storage.ld["loss"].backward()
torch.random.manual_seed(0)
newmod.backward()

print("Finish execution")

torch.random.manual_seed(0)
model1 = GPT2(nlayers=2,dropout=0, vcb_sz=6000).to('cuda')
context1 = torch.tensor([[ 464, 5440, 4534]]).to('cuda')

torch.random.manual_seed(0)

y = model1(context1)
loss = torch.mean(y)
print("Same loss obtained:", torch.allclose(loss,newmod.storage.ld["loss"]))
# print("Same loss from our module and original module:", torch.eq(newmod.storage.ld["__534_fv"], y))
torch.random.manual_seed(0)
loss.backward()
same_grad = True
for n,p in model2.named_parameters():
    if not torch.allclose(model2.get_parameter(n), model1.get_parameter(n)):
        print("Unequal weight found in:", n)
        same_grad = False
        
    if model2.get_parameter(n).grad!=None:
        if not torch.allclose(model2.get_parameter(n).grad, model1.get_parameter(n).grad):
            print("Unequal grad found in:", n)
            same_grad = False
if same_grad:
    print("Same grad obtained!")