from pulp import lpSum
from .ilp_utils import RkLpVariable

def add_offload_variables(md):
    add_offload_param_variables(md)
    add_offload_activation_variables(md)

def add_offload_constraints(md):
    add_offload_param_constraints(md)
    add_offload_activation_constraints(md)

def add_offload_activation_variables(md):

    pass

def add_offload_param_variables(md):
    optimize_metrics = md.optimize_metrics

    md.cpu_optimize = True
    md.optimizer_states_factor = optimize_metrics["optimizer_states_size"]#*weight size
    md.cpu_optimize_speed = optimize_metrics["cpu_optimize_speed"]/md.gcd#B/ms
    md.gpu_optimize_speed = optimize_metrics["gpu_optimize_speed"]/md.gcd#B/ms
    md.optimizer_overhead_factor = optimize_metrics["optimizer_overhead"]#*weight size
    md.minor_param_size = optimize_metrics["minor_param_size"]# minor weight size
    md.bandwidth = optimize_metrics["bandwidth"]# bandwidth
    
    md.req_w = RkLpVariable("Required_w", lowBound=0, upBound=1, cat="Continuous")
    md.req_w = 1.

    md.param2hcn = dict()
    md.parameters = []
    for k,v in md.hgraph.parameter_groups.items():
        md.param2hcn[len(md.parameters)] = k
        md.parameters.append(v)

    md.hcn2param = {t:[] for t in range(md.T)}

    for p,hcn_s in md.param2hcn.items():
        for hcn in hcn_s:
            md.hcn2param[hcn].append(p)

    md.parameter_size = [sum(pnode.mem for pnode in p)/md.gcd for p in md.parameters]
    md.parameter_gradient_size = [sum(pnode.mem for pnode in p if pnode.info.requires_grad
                                        )/md.gcd for p in md.parameters]
    md.W = len(md.parameters)

    param_idx = [(t, k, w) 
               for t in range(md.T) 
               for k in md.krange(t)
               for w in range(md.W)]

    md.AliveW = RkLpVariable.dicts("AliveW", param_idx)  # parameter w is alive at the start of step j.
    md.AliveG = RkLpVariable.dicts("AliveG", param_idx)  # w.grad is alive at the start of step j.
    md.AliveO = RkLpVariable.dicts("AliveO", param_idx)  # w.grad is alive at the start of step j.
    md.OflW = RkLpVariable.dicts("OflW", param_idx)# Offload weight
    md.OflG = RkLpVariable.dicts("OflG", param_idx)# Offload gradient
    md.PrfW = RkLpVariable.dicts("PrfW", param_idx)# Prefetch gradient
    md.PrfG = RkLpVariable.dicts("PrfG", param_idx)# Prefetch gradient
    md.OflO = RkLpVariable.dicts("OflO", param_idx)# Offload optimizer states
    md.PrfO = RkLpVariable.dicts("PrfO", param_idx)# Prefetch optimizer states

    md.OptC = RkLpVariable.dicts("OptC", param_idx)# Optimize on cpu
    md.sumOptC = dict()
    for w in md.param2hcn:
        md.sumOptC[w] = lpSum(md.OptC[t,k,w] for t in range(md.T) for k in md.krange(t))    

    md.param_grad_mem = {(t,k):0 for t in range(md.T) for k in md.krange(t)}

    md.bandwidthOfl = optimize_metrics["bandwidth"]/md.gcd  # byte/ms
    md.bandwidthPrf = optimize_metrics["bandwidth"]/md.gcd  # byte/ms
    prefill_offload(md)

def accumC_grad(md, w):
    #if grad_accumulation, gradient stored on CPU from previous iterations
    return md.sumOptC[w]

def accumC_optimizer_states(md, w):
    #if grad_accumulation, optimizer states stored on CPU from previous iterations
    return accumC_grad(md, w)

def instant_opt(md, w):
    # return the fraction of parameter instantly optimized after bwd
    if md.gradient_accumulation:
        return 0
    if md.grad_mode =="free":
        return md.req_w
    return md.req_w-md.sumOptC[w]

def max_OflGProg(md, t, k, w):
    return md.OflGProg[t, k, w]+(md.OflWProg[t, k, w]*(md.grad_mode=="free")
                                    *w_by_wg(md, w))

def w_by_wg(md, w):
    if md.parameter_gradient_size[w]==0:return 0
    return md.parameter_size[w]/md.parameter_gradient_size[w]

def all_param_mem(md, t, k, with_multiplier=True):
    return (parameter_mem(md, t,k)
            + md.param_grad_mem[t,k]
            + optimizer_states_mem(md, t,k)
            + (1-md.req_w)*md.peak_budget*with_multiplier)

def parameter_mem(md, t, k):
    parameter_mem = lpSum(
        (md.AliveW[t, k, w] + md.PrfW[t, k, w])
        * md.parameter_size[w]
        for w in range(md.W)
    )
    return parameter_mem

# def param_grad_mem(md, t, k):
#     grad_mem = lpSum(
#         md.AliveG[t, k, w]
#         * md.parameter_gradient_size[w]
#         for w in range(md.W)
#     )
#     return grad_mem

def optimizer_states_mem(md, t, k, with_overhead=True):    
    optimizer_states_mem = lpSum(((md.AliveO[t, k, w]+md.PrfO[t, k, w])*
                md.parameter_gradient_size[w] *
                md.optimizer_states_factor)
                for w in range(md.W))
    optimizer_overhead = 0
    if k > md.loss_idx:# and k in md.hcn2param:
        l_w = md.hcn2param[k]
        optimizer_overhead += sum((md.req_w-md.sumOptC[w])
                                    * md.parameter_gradient_size[w]
                                    * md.optimizer_overhead_factor
                                    for w in l_w)
    return optimizer_states_mem + optimizer_overhead*with_overhead


def prefill_offload(md):
    for t in range(md.T):
        for k in range(md.T):                
            for w in range(md.W):
                if k not in md.krange(t):
                    md.PrfW[t, k, w] = 0
                    md.OflW[t, k, w] = 0
                
                # fwd_i, bwd_i = md.param2hcn[w]
                fwd_i = min(md.param2hcn[w])
                bwd_i = max(md.param2hcn[w])
                if k not in md.krange(t) or (t>fwd_i and t<bwd_i):
                    md.OptC[t, k, w] = 0

    if not md.gradient_accumulation:
        for k in md.PrfG:
            md.PrfG[k] = 0

    if not md.with_optimizer_states:
        for (t,k,w) in md.AliveO:
            md.AliveO[(t,k,w)] = (md.req_w-accumC_optimizer_states(md, w))
            md.PrfO[(t,k,w)] = 0
            md.OflO[(t,k,w)] = 0
    if md.grad_mode in ["free"]:
        # If gradient is freed ASAP, OflG will be merged into OflW
        for (t,k,w) in md.OflG:
            md.OflG[(t,k,w)] = 0
        for (t,k,w) in md.AliveG:
            grad_size = md.parameter_gradient_size[w]
            if len(md.param2hcn[w]) <= 2:
                md.AliveG[(t,k,w)] = 0
                if k == max(md.param2hcn[w]):
                    # md.overhead[k] = [v+grad_size for v in md.overhead[k]]
                    md.param_grad_mem[t,k] += grad_size * md.req_w
            else:#shared weight
                bwd_first = min(x for x in md.param2hcn[w] if x>md.loss_idx)
                bwd_last = max(md.param2hcn[w])
                if t<=bwd_first or t>bwd_last:#assume single bwd
                    md.AliveG[(t,k,w)] = 0
                else:
                    md.AliveG[(t,k,w)] = 1
                    if k in md.param2hcn[w] and k>bwd_first:
                        # md.overhead[k] = [v+grad_size for v in md.overhead[k]]
                        md.param_grad_mem[t,k] += grad_size * md.req_w
    elif md.grad_mode in ["accumulate"]:
        for (t,k,w) in md.AliveG:
            md.AliveG[(t,k,w)] = 1
            # if k == max(md.param2hcn[w]):
            #     md.overhead[k] = [v+grad_size for v in md.overhead[k]]
            # TODO: add offload gradient variables for gradient accumulation
    
    for t in range(md.T):
        for k in md.krange(t):
            md.param_grad_mem[t,k] += lpSum(
                            md.AliveG[t, k, w]
                            * md.parameter_gradient_size[w]
                            for w in range(md.W)
                        )

def add_offload_param_constraints(md):
    # if with_grad, AliveG is a variable
    # if with_optimizer_states, AliveO is a variable
    md.OflWProg = dict()
    md.OflGProg = dict()
    md.OptCProg = dict()
    md.PrfWProg = dict()
    md.PrfGProg = dict()
    md.OflOProg = dict()
    md.PrfOProg = dict()
    
    def get_progress(op, t, k, w):
        bwd_i = max(md.param2hcn[w])
        if bwd_i < t:  # after bwd of w
            progress = lpSum(
                op[t, kk, w] for kk in md.krange(t) if kk < k
            ) + lpSum(
                op[tt, kk, w]
                for tt in range(bwd_i, t)  # offload right after bwd_i
                for kk in md.krange(tt)
            )
        else:
            progress = (
                lpSum(op[t, kk, w] for kk in md.krange(t) if kk < k)
                + lpSum(op[tt, kk, w] for tt in range(t) for kk in md.krange(tt))
                + lpSum(
                    op[tt, kk, w]
                    for tt in range(bwd_i, md.T)
                    for kk in md.krange(tt)
                )
            )
        return progress

    for t in range(md.T):
        for k in md.krange(t):
            t_, k_ = md.next_idx(t, k)
            md.md += md.Time[t, k] >= time_step_prefetch(md, t, k)
            md.md += md.Time[t, k] >= (time_step_offload(md, t, k)
                                       + time_step_optimize_self(md, t, k))
            md.md += md.Time[t, k] >= time_step_optimize(md, t, k)
            md.md += md.Time[t, k] >= (time_step_compute(md, t, k) 
                                       + time_step_optimize_self(md, t, k, cpu=False)
                                       + time_step_offload_self(md, t, k)
                                       + time_step_optimize_self(md, t, k, cpu=True))

            for w in range(md.W):
                md.PrfWProg[t,k,w] = get_progress(md.PrfW, t, k, w)
                md.PrfGProg[t,k,w] = get_progress(md.PrfG, t, k, w)
                md.OflWProg[t,k,w] = get_progress(md.OflW, t, k, w)
                md.OflGProg[t,k,w] = get_progress(md.OflG, t, k, w)
                md.OflOProg[t,k,w] = get_progress(md.OflO, t, k, w)
                md.PrfOProg[t,k,w] = get_progress(md.PrfO, t, k, w)
                md.OptCProg[t,k,w] = get_progress(md.OptC, t, k, w)
                
                md.md += md.AliveW[t, k, w] <= md.req_w
                md.md += md.OflWProg[t, k, w] <= md.req_w
                md.md += md.OflGProg[t, k, w] <= accumC_grad(md, w)
                md.md += md.OptCProg[t, k, w] <= max_OflGProg(md,t,k,w)
                md.md += md.OptCProg[t, k, w] <= md.PrfWProg[t, k, w]*w_by_wg(md, w)
                md.md += (md.AliveW[t, k, w] + md.OflWProg[t, k, w]
                            >= fraction_constant_param(md, w)
                            + fraction_instant_updated_param(md, w))
                md.md += (md.AliveG[t, k, w] + md.OflGProg[t, k, w] 
                            >= fraction_remaining_gradients(md, w))
                md.md += (md.AliveW[t_, k_, w] + md.PrfW[t_, k_, w] <= 
                            md.req_w
                            + (md.OptCProg[t, k, w] - md.sumOptC[w])
                            *md.parameter_gradient_size[w]/md.parameter_size[w]
                            # size that not yet finished updating cannot be prefetched
                                )
                diffW = md.AliveW[t_, k_, w] - md.AliveW[t, k, w]
                md.md += diffW <= md.PrfW[t, k, w]

                diffG = md.AliveG[t_, k_, w] - md.AliveG[t, k, w]
                md.md += (diffG <= 1*(k in md.param2hcn[w] 
                                        and k>md.loss_idx)#backward operations
                                        +md.PrfG[t, k, w])

                md.md += md.AliveO[t_, k_, w] - md.AliveO[t, k, w] <= md.PrfO[t,k,w]
                md.md += md.OflOProg[t, k, w] >= md.PrfOProg[t, k, w]
                md.md += (md.AliveO[t, k, w] + md.OflOProg[t, k, w]
                            >= md.req_w - md.sumOptC[w])
                md.md += (md.OflOProg[t, k, w]
                            <= md.req_w - md.sumOptC[w])
        
    for w in md.param2hcn:
        fwd_i = min(md.param2hcn[w])
        bwd_i = max(md.param2hcn[w])
        md.md += md.PrfWProg[fwd_i,fwd_i,w] >= md.sumOptC[w]
        md.md += md.req_w - md.sumOptC[w] <= md.AliveO[bwd_i, bwd_i, w]
        if md.gradient_accumulation:
            md.md += md.OflGProg[bwd_i, bwd_i, w] == accumC_grad(md, w)
            md.md += md.PrfGProg[bwd_i, bwd_i, w] == accumC_grad(md, w) - md.sumOptC[w]
        t_, k_ = md.next_idx(bwd_i, bwd_i)
        # md.md += md.AliveO[t_, k_, w] - md.AliveO[bwd_i, bwd_i, w] <= 1 - md.gradient_accumulation
        # md.md += md.AliveG[t_, k_, w] - md.AliveG[bwd_i, bwd_i, w] <= 1# - md.gradient_accumulation
        for t in range(md.T):
            for k in md.param2hcn[w]:
                if k not in md.krange(t):
                    continue
                md.md += md.sumComp[t, k] -1 <= md.AliveW[t, k, w] - md.req_w

def add_offload_activation_constraints(md):
    pass


def add_offload_objective(md, bandwidth_cost=0.01):
    prf_cost = (
        bandwidth_cost
        * lpSum(
            (md.PrfW[t, k, w] + md.PrfG[t, k, w] + md.PrfO[t, k, w])
            * md.parameter_size[w] / md.bandwidthPrf
            for t in range(md.T)
            for k in md.krange(t)
            for w in range(md.W)
        )
    )
    ofl_cost = (
        bandwidth_cost
        * lpSum(
            (md.OflW[t, k, w] + md.OflG[t, k, w] + md.OflO[t, k, w])
            * md.parameter_size[w] / md.bandwidthOfl
            for t in range(md.T)
            for k in md.krange(t)
            for w in range(md.W)
        )
    )
    md.md += (
            lpSum(md.Time[t, k] for t in range(md.T) for k in md.krange(t))
            + prf_cost
            + ofl_cost
        )
    
"""
The following functions are meant to be used for building constraints with
meaningful names. They are not supposed to be called outside this file.
By default, index t is for stage, k for step and w for the parameter group.
"""

def fraction_constant_param(md, w):
    return 1 - md.parameter_gradient_size[w]/md.parameter_size[w]

def fraction_instant_updated_param(md, w):
    if md.gradient_accumulation:
        # TODO: not fully supported now
        return 0
    if md.grad_mode =="free":
        return md.req_w
    return md.req_w-md.sumOptC[w]

def fraction_remaining_gradients(md, w):
    if md.gradient_accumulation:
        # TODO: not fully supported now
        return md.req_w
    if md.grad_mode =="free":
        return 0

# def fraction_after_bwd(md, w):
#     return fraction_constant_param(md, w) + fraction_instant_updated_param(md, w)

def time_step_offload(md, t, k):
    mem = 0
    for w in range(md.W):
        mem += md.parameter_size[w] * md.OflW[t, k, w]
        optim_state = md.optimizer_states_factor * md.OflO[t,k,w]
        mem += md.parameter_gradient_size[w] * (md.OflG[t, k, w] + optim_state)
        
    return mem/md.bandwidthOfl

def time_step_offload_self(md, t, k):
    mem = 0
    for w in md.hcn2param[k]:
        mem += md.parameter_size[w] * md.OflW[t, k, w]
        optim_state = md.optimizer_states_factor * md.OflO[t,k,w]
        mem += md.parameter_gradient_size[w] * (md.OflG[t, k, w] + optim_state)
        
    return mem/md.bandwidthOfl

def time_step_prefetch(md, t, k):
    mem = 0
    for w in range(md.W):
        mem += md.parameter_size[w] * md.PrfW[t, k, w]
        optim_state = md.optimizer_states_factor * md.PrfO[t,k,w]
        mem += md.parameter_gradient_size[w] * (md.PrfG[t, k, w] + optim_state)
        
    return mem/md.bandwidthPrf


def time_step_prefetch_self(md, t, k):
    mem = 0
    for w in md.hcn2param[k]:
        mem += md.parameter_size[w] * md.PrfW[t, k, w]
        optim_state = md.optimizer_states_factor * md.PrfO[t,k,w]
        mem += md.parameter_gradient_size[w] * (md.PrfG[t, k, w] + optim_state)
        
    return mem/md.bandwidthPrf

def time_step_optimize(md, t, k, cpu=True):
    """
    We assume that GPU optimization happens only right after 
    the backward, which should be accessed from time_step_optimize_self.
    """
    mem = 0
    for w in range(md.W):
        mem += md.parameter_gradient_size[w] * md.OptC[t, k, w]
    return mem/md.cpu_optimize_speed


def time_step_optimize_self(md, t, k, cpu=True):
    """
    We assume that GPU optimization happens only right after 
    the backward, but CPU optimization could happen anytime and
    represented by md.OptC.
    """
    mem = 0
    if cpu:
        for w in md.hcn2param[k]:
            opt_fraction = md.OptC[t, k, w]
            mem += md.parameter_gradient_size[w] * opt_fraction
        return mem/md.cpu_optimize_speed
    
    elif k <= md.loss_idx:#forward
        return 0
    else:
        for w in md.hcn2param[k]:
            opt_fraction = fraction_instant_updated_param(md, w)
            mem += md.parameter_gradient_size[w] * opt_fraction
        return mem/md.cpu_optimize_speed

def time_step_compute(md, t, k):
    return lpSum(md.Comp[t, k, o] * md.time[k][o] for o in range(md.nComp[k]))