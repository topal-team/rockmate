import torch
import rockmate as rk
from copy import deepcopy
from rotor import timing

device = torch.device("cuda")


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


def throughput_exp(module, input, batch_sizes, mem_limit=None):
    newmod = rk.CheckpointedModule(module, input, mem_limit=mem_limit)
    original_batch = input.shape[0]
    throughput = {}
    y = newmod(input)
    newmod.reinit()
    for batch_size in batch_sizes:
        try:
            newmod.get_sequence(mem_limit * original_batch / batch_size)
        except:
            throughput[batch_size] = "infeasible"
            continue
        newmod.get_code()

        newmod.reinit()
        torch.cuda.reset_peak_memory_stats()
        max_before = torch.cuda.max_memory_allocated()
        timer = timing.make_timer(device)
        timer.start()
        torch.random.manual_seed(0)
        input = torch.randint(0, 600, [batch_size, input.shape[1]]).to(device)
        y = newmod(input)
        loss = y.mean()
        loss.backward()
        newmod.backward()
        timer.end()

        peak_mem = torch.cuda.max_memory_allocated() - max_before
        print(f"rockmate module peak memory {peak_mem}")
        print("rockmate module time: %.4f" % timer.elapsed())
        print(f"batch size {batch_size}")
        print(f"throughput: {batch_size / timer.elapsed()}")
        throughput[batch_size] = batch_size / timer.elapsed()
        del y
        del loss
        input.grad = None
    return throughput
