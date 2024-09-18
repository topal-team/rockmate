import torch
from rkgb.lowlevel.measure import TimerCPU
from copy import deepcopy

def get_optimize_metrics(
    _p, optim, cpu_optim=None, optim_kwargs={}, niter=20, minor_offload_size=1024**2
):
    # x = torch.ones(30, 1024, 1024, 1024)
    if optim is None:
        return None
    if cpu_optim is None:
        cpu_optim = optim
    # timer = irotor.make_timer(torch.device("cpu"))
    timer = TimerCPU()
    # p = deepcopy(_p).to("cuda")
    # if not p.is_leaf:
    p = torch.ones([10, 1024, 1024], dtype=_p.dtype).to("cuda")
    size = p.numel()
    
    p_c = torch.nn.Parameter(torch.zeros_like(p, device="cpu"))
    p_c.grad = torch.ones_like(p_c)
    optimizer = cpu_optim([p_c], **optim_kwargs)
    for i in range(niter):
        optimizer.step()
    
    torch.cuda.synchronize()

    a_c = torch.ones_like(p, device="cpu", pin_memory=True)
    a_g = torch.ones_like(p, device="cuda")
    b_c = torch.ones_like(p, device="cpu", pin_memory=True)
    b_g = torch.ones_like(p, device="cuda")
    p_stream = torch.cuda.Stream()
    o_stream = torch.cuda.Stream()
    timer.start()
    for i in range(niter):
        with torch.cuda.stream(p_stream):
            a_c.copy_(a_g, non_blocking=True)
            a_c.copy_(a_g, non_blocking=True)
        with torch.cuda.stream(o_stream):
            b_g.copy_(b_c, non_blocking=True)
            b_g.copy_(b_c, non_blocking=True)
        optimizer.step()
        p_c.grad.zero_()
    torch.cuda.synchronize()
    timer.end()
    cpu_optimize_speed = size * p.element_size() * niter / timer.elapsed()
    
    
    p.grad = torch.ones_like(p)
    optimizer = optim([p], **optim_kwargs)
    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.memory_allocated()
    optimizer.step()
    timer.start()
    for i in range(niter):
        optimizer.step()
    torch.cuda.synchronize()
    timer.end()
    
    mem_after = torch.cuda.memory_allocated()
    gpu_optimize_speed = size * p.element_size() * niter / timer.elapsed()
    # gpu_optimize_speed = 0
    opt_size = mem_after - mem
    opt_overhead = torch.cuda.max_memory_allocated() - mem_after

    
    optimize_metrics = {
        "cpu_optimize_speed": cpu_optimize_speed * 0.8,
        "gpu_optimize_speed": gpu_optimize_speed,
        "optimizer_states_factor": round(opt_size // size / p.element_size()),
        "optimizer_overhead": round(opt_overhead // size / p.element_size()),
        "cpu_optim": cpu_optim,
        "gpu_optim": optim,
        "optim_kwargs": optim_kwargs,
    }
    return optimize_metrics


def measure_bandwidth(niter=10):
    timer = TimerCPU()
    a_c = torch.ones([10, 1024, 1024], device="cpu", pin_memory=True)
    a_g = torch.ones([10, 1024, 1024], device="cuda")
    b_c = torch.ones([10, 1024, 1024], device="cpu", pin_memory=True)
    b_g = torch.ones([10, 1024, 1024], device="cuda")
    p_stream = torch.cuda.Stream()
    o_stream = torch.cuda.Stream()
    timer.start()
    for i in range(niter):
        with torch.cuda.stream(p_stream):
            a_c.copy_(a_g, non_blocking=True)
        with torch.cuda.stream(o_stream):
            b_g.copy_(b_c, non_blocking=True)
    torch.cuda.synchronize()
    timer.end()
    bandwidth = niter * a_c.numel() * a_c.element_size() / timer.elapsed()
    return bandwidth