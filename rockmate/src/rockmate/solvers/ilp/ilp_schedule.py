from typing import Dict, Any
import time
import numpy as np
from copy import deepcopy
from ...op_schedule import (
    Activation,
    Parameter,
    Buffer,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    OpSchedule,
)
from ..main import get_sched, add_sched, translate
from .ilp_model import ModelPULP
from .ilp_offload import ModelPULPOffload

class knapsack:
    def __init__(self, parameter_sizes: list, pre_solve_size=10):
        size = [s[1] for s in parameter_sizes]
        self.parameter_sizes = parameter_sizes
        self.sizes = [s / sum(size) for s in size]
        self.pre_solve_size = pre_solve_size

    def get_size(self, indices, sizes):
        return sum(sizes[i] for i in indices)

    # @lru_cache(maxsize=4096 * 4096)
    def solve(self, frac: float, i: int = 0, sizes=[]):
        sizes = sizes or self.sizes
        if frac < 0:
            return []
        if i == len(sizes):
            return list(range(i))
        res1 = self.solve(frac, i + 1, sizes)
        res2 = self.solve(frac - sizes[i], i + 1, sizes)
        res2 = [i] + res2
        if self.get_size(res1, sizes) <= self.get_size(res2, sizes) + sizes[i]:
            return res1
        else:
            return res2

    def select(self, frac: float):
        sizes = self.sizes.copy()
        parameter_sizes = self.parameter_sizes.copy()
        selections = []
        while len(sizes) > self.pre_solve_size and frac>0:
            sel_i = self.presolve(frac, sizes)
            if sel_i is None:break
            selections.append(parameter_sizes.pop(sel_i)[0])
            frac -= sizes.pop(sel_i)
        indices = self.solve(frac, sizes=sizes)
        selections += [parameter_sizes[i][0] for i in indices]
        return selections

    def select_size(self, size: int):
        if not self.parameter_sizes:return []
        return self.select(size / sum(s[1] for s in self.parameter_sizes))
    
    def presolve(self, frac, sizes):
        array = np.array(sizes)
        sel_i = np.argmax(array*(array<frac))
        if array[sel_i]>frac:return np.argmin(array)
        return sel_i


def schedule(md: ModelPULP, hgraph=None, check_valid=False):
    """
    Given the solution from HILP, we want to translate the result
    to a OpSchedule that can be used in a higher level.
    """
    hgraph = hgraph if hgraph else md.hgraph

    init_op_list = []
    restore_op_list = []
    init_alive_status = {}
    loss_op = ComputeOp(md.hgraph.cluster.loss_cnode, disabled=True)
    if md.with_offload:
        W = len(md.parameter_size)
        (
            op_list,
            init_alive_status,
            init_op_list,
            restore_op_list,
        ) = schedule_offload(md, hgraph)
    else:
        op_list = []
        
        for t in range(md.T):
            for k in md.krange(t):
                if t == md.loss_idx and k == md.loss_idx:
                    op_list.append(loss_op)
                op_list += schedule_compute(md,t,k,hgraph)
    
    # print("finish scheduling")
    for anode in md.hgraph.cluster.interfaces["input_data_anodes"]:
        init_alive_status[anode.name] = True  # anode share the name as alloc
    op_sched = OpSchedule(
        op_list,
        loss_idx=op_list.index(loss_op),
        cluster=md.hgraph.cluster,
        init_alive_status=init_alive_status,
        init_op_list=init_op_list,
        restore_op_list=restore_op_list,
        with_parameters=md.with_offload,
        optim_states_multiplier = md.optimize_metrics["optimizer_states_size"]
    )
    # check_valid = True
    if check_valid:
        for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
            if op.is_del:
                continue
            for kdn in op.kn.deps_real:
                if not alive_status[kdn.name]:
                    print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                    raise ValueError
    return op_sched

def schedule_compute(md: ModelPULP, t,k,hgraph):
    op_list = []
    sol = md.sol
    
    j = md.hcn2sub_c[k]
    # if md.sumComp[t, k].value() == 1:
    if sol(md.sumComp[t, k]):
        hcn = hgraph.list_HCNs[k]
        opt = -1
        for o in range(md.nSched[k]):
            if sol(md.Comp[t, k, o]):
                opt = o
                break
        if opt > -1:
            h_obj = md.list_list_sched[j][opt]
            if hcn.is_fwd:
                sub_op_list = h_obj.op_list[: h_obj.loss_idx]
            else:
                sub_op_list = h_obj.op_list[h_obj.loss_idx + 1 :]

                # if md.sumAliveP[(j, t + 1)].value() == 0:
                # sub_op_list.append()
            sub_op_list = deepcopy(sub_op_list)

            if (
                not hcn.is_fwd
                # and md.sumAliveP[(j, t + 1)].value() > 0
                and sol(md.sumAliveP[t + 1, j])
            ):  # phantoms should be kept
                phantoms_to_keep = h_obj.phantoms
                # for op in sub_op_list[::-1]:
                #     if (
                #         op.is_del
                #         and not op.disabled
                #         and op.kn in phantoms_to_keep
                #     ):
                #         # Only the last del should be disabled
                #         op.disabled = True
                #         phantoms_to_keep.remove(op.kn)
            # translating sub_op_list
            if (
                hcn.sub_cluster
                is not hcn.sub_cluster.representee_cluster
            ):
                sub_op_list = translate(hcn.sub_cluster, sub_op_list)
        else:
            h_obj = hcn
            sub_op_list = deepcopy(h_obj.ff_op_list)

        op_list += sub_op_list

    for eidx, (k_, i) in enumerate(md.delete_list):
        # print(k_, i)
        # if k == k_ and md.delete[t, eidx].value()==1:
        if k == k_ and sol(md.delete[t, eidx]):
            han = hgraph.list_HANs[i]
            op_list.append(DeleteOp(Activation(han.anode)))
    return op_list


def schedule_offload(md: ModelPULPOffload, hgraph=None):
    """
    V1: md.grouping = False:
    merge every cluster, ofl/prf/del partially, high memory overhead
    """
    hgraph = hgraph if hgraph else md.hgraph

    ### Handle multiplier
    md.params_vars = [md.AliveW, md.OflWProg, md.OflW, 
                        md.PrfW, md.PrfWProg, md.OptC,
                        md.AliveG, md.OflGProg, md.OflG,
                        md.PrfG, md.PrfGProg, md.AliveO,
                        md.OflO, md.OflOProg, md.PrfO, 
                        md.PrfOProg
                        ]
    if isinstance(md.req_w, float):
        multiplier = 1- md.req_w
    else:
        multiplier = 1- md.req_w.value()
    for p in md.params_vars:
        for k,v in p.items():
            p[k] = v*1/ (1-multiplier)
            pass

    md.ofl_ops = []
    md.prf_ops = []
    md.del_ops = []
    md.opt_ops = []
    md.cpu_optimized_params = {}
    md.cpu_optimized_steps = {step:[] for step in md.active_steps}
    init_op_list = []
    restore_op_list = []
    init_alive_status = dict()
    if md.grouping:
        for w in range(md.W)[::-1]:
            o_l, p_l, d_l, t_l, i_l, r_l, init_alive = group(md, w)
            md.ofl_ops.extend(o_l)
            md.prf_ops.extend(p_l)
            md.del_ops.extend(d_l)
            md.opt_ops.extend(t_l)
            init_op_list.extend([ops[2] for ops in i_l])
            restore_op_list.extend([ops[2] for ops in r_l])
            for alloc in init_alive:
                init_alive_status[alloc.name] = True
    # else:
    #     init_op_list = md.schedule_init_op_list()

    sol = md.sol
    # offload_buffers = {w:[] for w in range(W)}
    op_list = []
    
    # for op in init_op_list:
    #     if isinstance(op, AllocateOp):
    #         init_alive_status[op.target] = True
    
    for t in range(md.T):
        for k in md.krange(t):
            op_list.append(SynchronizeOp(f"{(t,k)}"))
            if t == md.loss_idx and k == md.loss_idx:
                # loss_idx = len(op_list)
                # loss_op = Op(K_C_node("loss"))

                op_list.append(ComputeOp(md.hgraph.cluster.loss_cnode, disabled=True))
            if not sol(md.sumComp[t, k]):
                continue
            j = md.hcn2sub_c[k]
            # if md.sumComp[t, k].value() == 1:
            prefetch_list = []
            for w in range(md.W):
                prefetch_ops = create_prefetch_ops(md,t, k, w)
                op_list.extend(prefetch_ops[0])
                prefetch_list.extend(prefetch_ops[1])
            op_list += schedule_compute(md,t,k,hgraph)
            wait_op_1 = []
            wait_op_2 = []
            wait_op_3 = []
            for w in range(md.W):
                if k in md.param2hcn[w]:
                    wait_op_1.extend(create_optimize_ops(md,t, k, w))
                    wait_op_2.extend(create_offload_ops(md,t, k, w))
                    wait_op_3.extend(create_delete_ops(md,t, k, w))
                    # op_list.extend(create_prefetch_ops(md,t,k,w))
                else:
                    op_list.extend(create_offload_ops(md,t, k, w))
                    op_list.extend(create_delete_ops(md,t, k, w))
                    op_list.extend(create_optimize_ops(md,t, k, w))
            if wait_op_1:# for the current layer, need to synchronize first
                op_list.extend([SynchronizeOp(str(k))]+wait_op_1)
            if wait_op_2:# for the current layer, need to synchronize first
                op_list.extend([SynchronizeOp(str(k))]+wait_op_2)
            if wait_op_3:# for the current layer, need to synchronize first
                op_list.extend([SynchronizeOp(str(k))]+wait_op_3)

            op_list.extend(prefetch_list)
    return op_list, init_alive_status, init_op_list, restore_op_list

def create_optimize_ops(md, t, k, w, itemsize=4):
    op_list = []
    # sub_cluster = md.hgraph.list_HCNs[min(md.param2hcn[w])].sub_cluster
    if md.grouping:
        for t_, k_, op in md.opt_ops:
            if (
                t_ == t
                and k_ == k
                and op.target.pnode.param_name in [k.param_name for k in md.parameters[w]]
            ):
                op_list.append(op)
        return op_list

def create_delete_ops(md, t, k, w, itemsize=4):
    op_list = []
    sub_cluster = md.hgraph.list_HCNs[min(md.param2hcn[w])].sub_cluster
    if md.grouping:
        for t_, k_, op in md.del_ops:
            if (
                t_ == t
                and k_ == k
                and op.target.pnode.param_name in [k.param_name for k in md.parameters[w]]
            ):
                op_list.append(op)
        return op_list

def create_prefetch_ops(md, t, k, w, itemsize=4):
    pre_op_list = []
    post_op_list = []
    sub_cluster = md.hgraph.list_HCNs[min(md.param2hcn[w])].sub_cluster

    if md.grouping:
        for t_, k_, op in md.prf_ops:
            if (
                t_ == t
                and k_ == k
                and op.target.pnode.param_name in [k.param_name for k in md.parameters[w]]
            ):
                pre_op_list.append(op)

        return pre_op_list, post_op_list

def create_offload_ops(md, t, k, w, itemsize=4):
    op_list = []
    sub_cluster = md.hgraph.list_HCNs[min(md.param2hcn[w])].sub_cluster
    if md.grouping:
        for t_, k_, op in md.ofl_ops:
            if (
                t_ == t
                and k_ == k
                and op.target.pnode.param_name in [k.param_name for k in md.parameters[w]]
            ):
                op_list.append(op)
        return op_list

def group(md, w, tol=1):
    # Group the parameters of each block for the task
    fwd_i = min(md.param2hcn[w])
    bwd_i = max(md.param2hcn[w])
    early_fwd = []
    for t in range(bwd_i, md.T):
        if not md.single_fwd and md.sol(md.sumComp[t,fwd_i]):
            early_fwd.append(t)#if recompute fwd after bwd
    hcn = md.hgraph.list_HCNs[fwd_i]
    parameters = {pnode.param_name: pnode for pnode in md.parameters[w]}
    parameter_size = sum(pnode.mem for pnode in parameters.values())

    Alive = {p: 1 for p in parameters.keys()}
    Offloaded = {p: False for p in parameters.keys()}

    ofl_ops = []
    prf_ops = []
    del_ops = []
    opt_ops = []
    init_ops = []
    restore_ops = []
    init_alive = []
    cpu_optimize_candidates = {p: 0 for p in parameters.keys()}

    def apply_cpu_optimize(p):
        for (t,k,op) in ofl_ops:
            if op.target.target_name == p:
                op.target.is_grad = True
                # op.grad = True
                break
        # for (t,k,op) in del_ops:
        #     if op.target.name == p:
        #         op.grad = True
        del_ops.append((t,k,DeleteOp(Parameter(parameters[p], is_grad=True))))
        i = md.active_steps.index((t,k))+1# TODO: distribute cpu optimization based on time
        p_alloc = Parameter(parameters[p])
        op = OptimizeOp(list_params=[p], cpu=True, alloc=p_alloc,
                        time=parameters[p].mem/md.cpu_optimize_speed,
                        )
        opt_ops.append((*md.active_steps[i], op))
        md.cpu_optimized_steps[md.active_steps[i]].append(p)
        # del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p]))))

        #if cpu optimize, do not keep w after bwd
    def apply_gpu_optimize(p):
        p_alloc = Parameter(parameters[p])
        op = OptimizeOp(list_params=[p], alloc=p_alloc,
                        time = parameters[p].mem/md.gpu_optimize_speed,
                        overhead=parameters[p].mem*md.optimizer_overhead_factor)
        opt_ops.append((bwd_i, bwd_i, op))# optimize after bwd
        del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p], is_grad=True))))
        

    assert (bwd_i, bwd_i) in md.active_steps
    idx = md.active_steps.index((bwd_i, bwd_i))
    for t, k in md.active_steps[idx:] + md.active_steps[:idx]:
        t_, k_ = md.next_idx(t, k)
        current_alive_size = sum(parameters[p].mem * a for p, a in Alive.items())
        current_offloaded_size = sum(parameters[p].mem * a for p, a in Offloaded.items())
        next_alive_size = round((md.AliveG[(t_, k_, w)]+md.AliveW[(t_, k_, w)]).value() * parameter_size)
        next_offloaded_size = round((md.OflGProg[(t_, k_, w)]+md.OflWProg[(t_, k_, w)]).value() * parameter_size)
        
        # assert current_alive_size <= round(md.AliveW[(t, k, w)].value() * parameter_size)

        if (t, k) == (0, 0):  # init
            for p, a in Alive.items():
                if a:
                    p_alloc = Parameter(parameters[p])
                    init_ops.append((t, k, AllocateOp(p_alloc)))
                    init_alive.append(p_alloc)
                    op = PrefetchOp(
                        alloc=p_alloc, indices=(0, None), 
                        time=parameters[p].mem/md.bandwidthPrf/md.gcd
                    )
                    init_ops.append((t, k, op))
                    op = OffloadOp(
                        alloc=p_alloc, indices=(0, None),
                        time=parameters[p].mem/md.bandwidthOfl/md.gcd
                    )
                    restore_ops.append((t, k, op))
                    restore_ops.append((t, k, DeleteOp(p_alloc)))

        if next_offloaded_size > current_offloaded_size:
            # print(t,k, next_offloaded_size, current_offloaded_size)
            ofl_size = next_offloaded_size - current_offloaded_size
            candidates = {
                p: parameters[p].mem * (1 - o)
                for p, o in Offloaded.items()
                if o < 1
            }
            if not candidates:
                if ofl_size<1024:
                    ofl_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(ofl_size)
            # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
            #     pass
            for p in select_paras:
                op = OffloadOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                time=parameters[p].mem/md.bandwidthOfl/md.gcd)
                ofl_ops.append((t, k, op))
                Offloaded[p] = 1

        if current_alive_size > next_alive_size:
            del_size = current_alive_size - next_alive_size
            candidates = {}
            for p, o in Offloaded.items():
                if Alive[p]>0 and o>0:
                    candidates[p] = min(o, Alive[p])*parameters[p].mem
            if not candidates:
                if del_size<1024:
                    del_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(del_size)
            # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
            #     pass
            for p in select_paras:
                del_ops.append((t, k, DeleteOp(Parameter(parameters[p]))))
                Alive[p] = 0
        if current_alive_size < next_alive_size:
            # prefetch should be smaller than solution
            prf_size = next_alive_size - current_alive_size
            candidates = {
                p: parameters[p].mem * (1 - a) for p, a in Alive.items() if a < 1
            }
            if not candidates:
                if prf_size<1024:
                    prf_size=0
                else:
                    raise ValueError
            if md.sol(md.AliveW[(t_, k_, w)]):
                select_paras = list(candidates.keys())
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
            else:
                selector = knapsack(list(candidates.items()))
                unselect_paras = selector.select_size(sum(candidates.values()) - prf_size)
                
                select_paras = [
                    p for p in candidates.keys() if p not in unselect_paras
                ]
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
            #     pass
            for p in select_paras:
                prf_ops.append((t, k, AllocateOp(Parameter(parameters[p]))))
                op = PrefetchOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                time=parameters[p].mem/md.bandwidthPrf/md.gcd)
                prf_ops.append((t, k, op))
                Alive[p] = 1
                if (t > bwd_i and t < min(early_fwd + [md.T + 1])) or t < fwd_i:
                    # cpu optimize only if prefetch before fwd
                    if parameters[p].info.requires_grad:
                        # only trainable parameters will be optimize candidate
                        cpu_optimize_candidates[p] = 1
    
    candidates = {
                p: parameters[p].mem * a for p, a in cpu_optimize_candidates.items() if a >0
            }
    select_paras = []
    # assert sum(candidates.values())/parameter_size >= md.sumOptC[w].value()-0.01
    
        # cpu_optimize_size = md.sumOptC[w].value()*parameter_size# size by subgraph
    if isinstance(md.req_w, float):
        multiplier = 1-md.req_w
    else:
        multiplier = 1-md.req_w.value()
    # cpu_optimize_size = (sum(md.sumOptC[w_].value() * 
    #                         md.parameter_gradient_size[w_] *md.gcd
    #                         for w_ in range(w, md.W)) / (1-multiplier)
    #                     - sum(md.cpu_optimized_params.values()))# size by all graphs
    cpu_optimize_size = min(sum(candidates.values()),
                            (md.sumOptC[w].value() * 
                            md.parameter_gradient_size[w] *md.gcd
                            ) / (1-multiplier))
    
    if candidates and cpu_optimize_size>0:
        # print(candidates, cpu_optimize_size)
        selector = knapsack(list(candidates.items()))
        select_paras = selector.select_size(cpu_optimize_size)
        if cpu_optimize_size>sum(candidates[p] for p in select_paras):
            raise ValueError
        # print(select_paras)
        
    # Optimize parameters which requires grad
    gpu_optimze_param = []
    for p, pnode in parameters.items():
        if not pnode.info.requires_grad:continue
        if p in select_paras:
            md.cpu_optimized_params[p] = parameters[p].mem
            apply_cpu_optimize(p)
        else:
            apply_gpu_optimize(p)
            gpu_optimze_param.append(pnode)
    if md.with_optimizer_states and gpu_optimze_param:
        ofl_ops_os, prf_ops_os, del_ops_os, init_alive_os = group_optimizer_states(md, w, gpu_optimze_param)
        ofl_ops += ofl_ops_os
        prf_ops += prf_ops_os
        del_ops += del_ops_os
        init_alive += init_alive_os
    return ofl_ops, prf_ops, del_ops, opt_ops, init_ops, restore_ops, init_alive

def group_optimizer_states(md, w, gpu_optimize_param):
    # To offload and prefetch optimizer states witin the gpu_optimize_param
    ofl_ops = []
    prf_ops = []
    del_ops = []
    init_alive = []
    fwd_i = min(md.param2hcn[w])
    bwd_i = max(md.param2hcn[w])
    hcn = md.hgraph.list_HCNs[fwd_i]
    parameters = {pnode.param_name: pnode for pnode in md.parameters[w] if pnode.requires_grad}
    parameter_size = sum(pnode.mem for pnode in parameters.values())
    gpu_optimize_size = sum(pnode.mem for pnode in gpu_optimize_param)

    Alive = {pnode.param_name: 1 for pnode in gpu_optimize_param}
    Offloaded = {pnode.param_name: False for pnode in gpu_optimize_param}
    assert (bwd_i, bwd_i) in md.active_steps
    idx = md.active_steps.index((bwd_i, bwd_i))
    for t, k in md.active_steps[idx:] + md.active_steps[:idx]:
        if (t, k) == (0, 0):  # init
            for p, a in Alive.items():
                if a:
                    init_alive.append(Parameter(parameters[p],is_optim_states=True))

        t_, k_ = md.next_idx(t, k)
        current_alive_size = sum(parameters[p].mem * a for p, a in Alive.items())
        current_offloaded_size = sum(parameters[p].mem * a for p, a in Offloaded.items())
        next_alive_size = min(gpu_optimize_size,
                                round((md.AliveO[(t_, k_, w)]).value() * parameter_size))
        next_offloaded_size = min(gpu_optimize_size,
            round((md.OflOProg[(t_, k_, w)]).value() * parameter_size))
        if parameter_size * (1-md.sumOptC[w]).value()<gpu_optimize_size:
            next_offloaded_size += gpu_optimize_size - parameter_size * (1-md.sumOptC[w]).value()

        # assert current_alive_size <= round(md.AliveW[(t, k, w)].value() * parameter_size)
        if next_offloaded_size > current_offloaded_size:
            # print(t,k, next_offloaded_size, current_offloaded_size)
            ofl_size = next_offloaded_size - current_offloaded_size
            candidates = {
                p: parameters[p].mem * (1 - o)
                for p, o in Offloaded.items()
                if o < 1
            }
            if not candidates:
                if ofl_size<1024:
                    ofl_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(ofl_size)
            # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
            #     pass
            for p in select_paras:
                op = OffloadOp(alloc=Parameter(parameters[p],
                                                is_optim_states=True), indices=(0, None),
                                time=parameters[p].mem/md.bandwidthOfl/md.gcd*md.optimizer_states_factor)
                ofl_ops.append((t, k, op))
                Offloaded[p] = 1

        if current_alive_size > next_alive_size:
            if k_ ==bwd_i:continue
            del_size = current_alive_size - next_alive_size
            candidates = {}
            for p, o in Offloaded.items():
                if Alive[p]>0 and o>0:
                    candidates[p] = min(o, Alive[p])*parameters[p].mem
            if not candidates:
                if del_size<1024:
                    del_size = 0
                else:
                    raise ValueError
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(del_size)
            # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
            #     pass
            for p in select_paras:
                del_ops.append((t, k, DeleteOp(Parameter(parameters[p],
                                                            is_optim_states=True)
                                                )))
                Alive[p] = 0
        if current_alive_size < next_alive_size or k_==bwd_i:
            # if w == 15:print(md.active_steps[k_]==bwd_i)
            # prefetch should be smaller than solution
            prf_size = next_alive_size - current_alive_size
            candidates = {
                p: parameters[p].mem * (1 - a) for p, a in Alive.items() if a < 1
            }
            if not candidates:
                if prf_size<1024:
                    prf_size=0
                else:
                    raise ValueError
            if md.sol(md.AliveO[(t_, k_, w)]+md.sumOptC[w]-md.req_w+1) or k_==bwd_i:
                
                select_paras = list(candidates.keys())
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
            else:
                selector = knapsack(list(candidates.items()))
                unselect_paras = selector.select_size(sum(candidates.values()) - prf_size)
                
                select_paras = [
                    p for p in candidates.keys() if p not in unselect_paras
                ]
                # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
            # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
            #     pass
            for p in select_paras:
                prf_ops.append((t, k, AllocateOp(Parameter(parameters[p],
                                                            is_optim_states=True))))
                op = PrefetchOp(alloc=Parameter(parameters[p],is_optim_states=True), indices=(0, None),
                                time=parameters[p].mem/md.bandwidthPrf/md.gcd*md.optimizer_states_factor)
                prf_ops.append((t, k, op))
                Alive[p] = 1
        if k_==bwd_i:assert 0 not in Alive.values()

    return ofl_ops, prf_ops, del_ops, init_alive

