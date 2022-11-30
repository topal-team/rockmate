from mem_tools import *
newmod = mod()

from rotor import timing
context1 = torch.randint(0,600, [100,10])
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


# device = torch.device('cpu')
torch.random.manual_seed(0)
model1 = GPT2(nlayers=8,dropout=1e-8, vcb_sz=600).to(device)
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
print("peak memory:", torch.cuda.max_memory_allocated()-max_before)
print("runtime: %.4f"%timer.elapsed())

if torch.allclose(loss, newmod.storage.ld["loss"]):
    print("Same loss obtained!")

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
    print("Same grad obtained!")
