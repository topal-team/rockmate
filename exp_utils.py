import torch
import rockmate as rk
from copy import deepcopy
from rotor import timing

device = torch.device("cuda")


def test_on_module(module, input, mem_limit=None):
    for n, p in module.named_parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    _input = input.clone()
    _module = deepcopy(module)

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
