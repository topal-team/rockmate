from pulp import lpSum
from .ilp_utils import RkLpVariable

def offload(md):
    add_offload_variables(md)
    prefill_offload(md)
    add_offload_constraints(md)
    # add_offload_objective()

def add_offload_variables(md):
    optimize_metrics = md.optimize_metrics

    md.cpu_optimize = True
    md.optimizer_states_factor = optimize_metrics["optimizer_states_size"]#*weight size
    md.cpu_optimize_speed = optimize_metrics["cpu_optimize_speed"]#B/ms
    md.gpu_optimize_speed = optimize_metrics["gpu_optimize_speed"]#B/ms
    md.optimizer_overhead_factor = optimize_metrics["optimizer_overhead"]#*weight size
    md.minor_param_size = optimize_metrics["minor_param_size"]# minor weight size
    md.bandwidth = optimize_metrics["bandwidth"]# bandwidth
    batch_multiplier = 4
    # md.BatMpl = RkLpVariable("BMpl", lowBound=0, upBound=md.batch_multiplier, cat="Integer")
    # md.param_multiplier = 1-md.BatMpl*1/md.batch_multiplier
    md.param_multiplier = RkLpVariable("BMpl", lowBound=0, upBound=1-1/batch_multiplier, cat="Continuous")
    md.param_multiplier = 0.

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
    md.W = W = len(md.parameters)

    ofl_idx = [(t, k, w) 
               for t in range(md.T) 
               for k in md.krange(t)
               for w in range(W)]
    md.AliveW = RkLpVariable.dicts("AliveW", ofl_idx)  # parameter w is alive at the start of step j.
    md.AliveG = RkLpVariable.dicts("AliveG", ofl_idx)  # w.grad is alive at the start of step j.
    md.AliveO = RkLpVariable.dicts("AliveO", ofl_idx)  # w.grad is alive at the start of step j.
    md.OflW = RkLpVariable.dicts("OflW", ofl_idx)# Offload weight
    md.OflG = RkLpVariable.dicts("OflG", ofl_idx)# Offload gradient
    md.PrfW = RkLpVariable.dicts("PrfW", ofl_idx)# Prefetch gradient
    md.PrfG = RkLpVariable.dicts("PrfG", ofl_idx)# Prefetch gradient
    md.OptC = RkLpVariable.dicts("OptC", ofl_idx)# Optimize on cpu
    md.OflO = RkLpVariable.dicts("OflO", ofl_idx)# Offload optimizer states
    md.PrfO = RkLpVariable.dicts("PrfO", ofl_idx)# Prefetch optimizer states
    
    md.param_grad_mem = {(t,k):0 for t in range(md.T) for k in md.krange(t)}

    md.bandwidthOfl = optimize_metrics["bandwidth"]/md.gcd  # byte/ms
    md.bandwidthPrf = optimize_metrics["bandwidth"]/md.gcd  # byte/ms
    prefill_offload(md)

def req_w(md):
        return 1 - md.param_multiplier

def accumC_grad(md, w):
    #if grad_accumulation, gradient stored on CPU from previous iterations
    return md.sumOptC[w]

def accumC_optimizer_states(md, w):
    #if grad_accumulation, optimizer states stored on CPU from previous iterations
    return md.accumC_grad(w)

def instant_opt(md, w):
    # return the fraction of parameter instantly optimized after bwd
    if md.gradient_accumulation:
        return 0
    if md.grad_mode =="free":
        return md.req_w()
    return 1-md.sumOptC[w]- md.param_multiplier

def max_OflGProg(md, t, k, w):
    return md.OflGProg[t, k, w]+(md.OflWProg[t, k, w]*(md.grad_mode=="free")
                                    *md.w_by_wg(w))

def w_by_wg(md, w):
    if md.parameter_gradient_size[w]==0:return 0
    return md.parameter_size[w]/md.parameter_gradient_size[w]

def all_param_mem(md, t, k, with_multiplier=True):
    return (md.parameter_mem(t,k) 
            + md.param_grad_mem[t,k]
            + md.optimizer_states_mem(t,k)  
            + md.param_multiplier*md.peak_budget*with_multiplier)

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
        optimizer_overhead += sum((md.req_w()-md.sumOptC[w])
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

    md.sumOptC = dict()
    for w in md.param2hcn:
        md.sumOptC[w] = lpSum(md.OptC[t,k,w] for t in range(md.T) for k in md.krange(t))

    if not md.gradient_accumulation:
        for k in md.PrfG:
            md.PrfG[k] = 0

    if not md.with_optimizer_states:
        for (t,k,w) in md.AliveO:
            md.AliveO[(t,k,w)] = (1-md.accumC_optimizer_states(w)- md.param_multiplier)
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
                    md.param_grad_mem[t,k] += grad_size * md.req_w()
            else:#shared weight
                bwd_first = min(x for x in md.param2hcn[w] if x>md.loss_idx)
                bwd_last = max(md.param2hcn[w])
                if t<=bwd_first or t>bwd_last:#assume single bwd
                    md.AliveG[(t,k,w)] = 0
                else:
                    md.AliveG[(t,k,w)] = 1
                    if k in md.param2hcn[w] and k>bwd_first:
                        # md.overhead[k] = [v+grad_size for v in md.overhead[k]]
                        md.param_grad_mem[t,k] += grad_size * md.req_w()
    elif md.grad_mode in ["accumulate"]:
        for (t,k,w) in md.AliveG:
            md.AliveG[(t,k,w)] = 1
            # if k == max(md.param2hcn[w]):
            #     md.overhead[k] = [v+grad_size for v in md.overhead[k]]
            # TODO: add offload gradient variables for gradient accumulation


def add_offload_constraints(md):
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

            md.md += md.Time[t, k] >= 1/ md.bandwidthOfl *lpSum(
                md.parameter_size[w] 
                * md.PrfW[t, k, w]
                    +md.parameter_gradient_size[w] *
                    md.optimizer_states_factor*md.PrfO[t,k,w]
                for w in range(md.W))
            md.md += md.Time[t, k] >= (1/ md.bandwidthOfl *lpSum(
                md.parameter_size[w] 
                * md.OflW[t, k, w]
                    +md.parameter_gradient_size[w] *
                    (md.OflG[t, k, w]+
                    md.optimizer_states_factor*md.OflO[t,k,w])
                for w in range(md.W))
                + lpSum(md.parameter_gradient_size[w]
                / md.cpu_optimize_speed*md.gcd
                * md.OptC[t, k, w]
                for w in md.hcn2param[k]))
            md.md += md.Time[t, k] >= (lpSum(md.Comp[t, k, o] * md.time[k][o] 
                                                    for o in range(md.nComp[k]))
            + 1/ md.bandwidthOfl *lpSum(
                md.parameter_size[w] 
                * md.OflW[t, k, w]
                    +md.parameter_gradient_size[w]
                    * (md.OflG[t, k, w]
                + md.optimizer_states_factor*md.OflO[t,k,w])
                for w in md.hcn2param[k])# current layer offload
                + 1/md.gpu_optimize_speed*md.gcd
                * lpSum(md.parameter_gradient_size[w]
                * (md.req_w() - md.sumOptC[w])
                for w in md.hcn2param[k]
                ))
            md.md += md.Time[t, k] >= lpSum(
                md.parameter_size[w] 
                / md.cpu_optimize_speed*md.gcd
                * md.OptC[t, k, w]
                for w in range(md.W)
            )
            md.param_grad_mem[t,k] += lpSum(
                            md.AliveG[t, k, w]
                            * md.parameter_gradient_size[w]
                            for w in range(md.W)
                        )
            for w in range(md.W):
                md.PrfWProg[t,k,w] = get_progress(md.PrfW, t, k, w)
                md.PrfGProg[t,k,w] = get_progress(md.PrfG, t, k, w)
                md.OflWProg[t,k,w] = get_progress(md.OflW, t, k, w)
                md.OflGProg[t,k,w] = get_progress(md.OflG, t, k, w)
                md.OflOProg[t,k,w] = get_progress(md.OflO, t, k, w)
                md.PrfOProg[t,k,w] = get_progress(md.PrfO, t, k, w)
                md.OptCProg[t,k,w] = get_progress(md.OptC, t, k, w)
                
                md.md += (md.AliveW[t, k, w] <= md.req_w())
                md.md += md.OflWProg[t, k, w] <= md.req_w()
                md.md += md.OflGProg[t, k, w] <= md.accumC_grad(w)
                md.md += md.OptCProg[t, k, w] <= md.max_OflGProg(t,k,w)
                md.md += md.OptCProg[t, k, w] <= md.PrfWProg[t, k, w]*md.w_by_wg(w)
                md.md += (md.AliveW[t, k, w] + md.OflWProg[t, k, w]
                            >= md.instant_opt(w))
                md.md += (md.AliveG[t, k, w] + md.OflGProg[t, k, w] 
                            >= md.req_w() - md.instant_opt(w))
                md.md += (md.AliveW[t_, k_, w] + md.PrfW[t_, k_, w] <= 
                            md.req_w()
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
                            >= md.req_w() - md.sumOptC[w])
                md.md += (md.OflOProg[t, k, w]
                            <= md.req_w() - md.sumOptC[w])
        
    for w in md.param2hcn:
        fwd_i = min(md.param2hcn[w])
        bwd_i = max(md.param2hcn[w])
        md.md += md.PrfWProg[fwd_i,fwd_i,w] >= md.sumOptC[w]
        md.md += md.req_w() - md.sumOptC[w] <= md.AliveO[bwd_i, bwd_i, w]
        if md.gradient_accumulation:
            md.md += md.OflGProg[bwd_i, bwd_i, w] == md.accumC_grad(w)
            md.md += md.PrfGProg[bwd_i, bwd_i, w] == md.accumC_grad(w) - md.sumOptC[w]
        t_, k_ = md.next_idx(bwd_i, bwd_i)
        # md.md += md.AliveO[t_, k_, w] - md.AliveO[bwd_i, bwd_i, w] <= 1 - md.gradient_accumulation
        # md.md += md.AliveG[t_, k_, w] - md.AliveG[bwd_i, bwd_i, w] <= 1# - md.gradient_accumulation
        for t in range(md.T):
            for k in md.param2hcn[w]:
                if k not in md.krange(t):
                    continue
                md.md += md.sumComp[t, k] -1 <= md.AliveW[t, k, w] - md.req_w()

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
        if md.with_parameters
        else 0
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
        if md.with_parameters
        else 0
    )
    md.md += (
            lpSum(md.Time[t, k] for t in range(md.T) for k in md.krange(t))
            + prf_cost
            + ofl_cost
        )