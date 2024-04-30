import rkgb
import torch
import numpy as np
from copy import deepcopy

from rkgb.lowlevel.ast_add_on import ast_to_str
from rkgb.lowlevel.measure import TimerCPU
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from rkgb.core.backward import ComputationNode
from rkgb.lowlevel.constants import init_target_string

from .main import Solver
from ..op_schedule import OpSchedule, ComputeOp, DeleteOp, Activation, OffloadOp, PrefetchOp, AllocateOp
import time
import psutil

class CheapSolver(Solver):
    """
    Cheap solver solves a cluster with heuristic: 
    all the cheap operations are recomputed and 
    all the phantoms are offloaded during forward and prefetched during backward.
    """
    class Config:
        def __init__(self):
            pass

    def __init__(self, config=None):
        super().__init__(config)

    def solve(self, cluster: HierarchicalCluster, budget=None):
        avg_time = np.mean([cnode.time
                for cnode in cluster.list_cnodes 
                if cnode.time is not None])
        
        def is_cheap(cnode):
            if not cnode.is_fwd:return False
            if cnode.time is None:return False
            return cnode.time < avg_time/2
        cheap_cnodes = [cnode for cnode in cluster.list_cnodes if is_cheap(cnode)]
        cnode_idx = {cnode.name: i for i,cnode in enumerate(cluster.list_cnodes)}
        
        anodes_del_idx = {i:[] for i, _ in enumerate(cluster.list_cnodes)}
        output_anodes = []
        for anode in cluster.list_anodes:
            if "source" in anode.name:continue
            user_indices = [cnode_idx[cnode.name]
                                for cnode in anode.users_real
                                if cnode.name in cnode_idx]
            if user_indices:
                last_user_idx = max(user_indices)
                anodes_del_idx[last_user_idx].append(anode)
            else:
                output_anodes.append(anode)
            

            if (all(is_cheap(cnode) for cnode in anode.deps) 
                and anode.allocation_type == "data"):
                user_indices_fwd = [cnode_idx[cnode.name] 
                                    for cnode in anode.users_real
                                    if cnode.is_fwd and 
                                    cnode.name in cnode_idx]
                if user_indices_fwd:
                    last_user_idx_fwd = max(user_indices_fwd)

                anodes_del_idx[last_user_idx_fwd].append(anode)
        
        fwd_op_list = []
        bwd_op_list = []
        for i, cnode in enumerate(cluster.list_cnodes):
            fwd_op_list.append(ComputeOp(cnode, disabled="loss" in cnode.name))
            if "loss" in cnode.name:
                loss_idx = i
                break
            for anode in anodes_del_idx[i]:
                fwd_op_list.append(DeleteOp(Activation(anode)))

        for anode in output_anodes:
            bwd_op_list.append(DeleteOp(Activation(anode)))
        for cnode in cheap_cnodes:
            bwd_op_list.append(ComputeOp(cnode))
        for i, cnode in enumerate(cluster.list_cnodes[loss_idx+1:]):
            bwd_op_list.append(ComputeOp(cnode))
            for anode in anodes_del_idx[i+loss_idx+1]:
                bwd_op_list.append(DeleteOp(Activation(anode)))

        op_sched = OpSchedule(fwd_op_list+bwd_op_list, 
                              loss_idx=len(fwd_op_list)-1,
                              cluster=cluster)
        return [add_activation_offload(op_sched)]
    
    
def add_activation_offload(op_sched:OpSchedule) -> OpSchedule:
    """
    With all the torch.cuda.Event and stream.wait_event() to synchornize
    between offload/prefetch, in the order of Main>Ofl>Del during fwd and Prf>Main during bwd

    """
    cluster = op_sched.cluster
    op_list = op_sched.op_list
    fwd_op_list = op_list[:op_sched.loss_idx]
    bwd_op_list = op_list[op_sched.loss_idx:]
    prefetch_op_list = []
    phantoms = op_sched.phantoms
    for anode in phantoms:
        if not anode.allocation_type == "data":continue
        i = max(min(op_sched.occurrences[ComputeOp(cnode).name]) for cnode in anode.deps)
        op_sched.op_list[i].record_event = True
        comp_op = op_sched.op_list[i]
        offload_op = OffloadOp(Activation(anode), time=anode.mem/1e7)
        offload_op.wait_events.append(comp_op)
        offload_op.record_event = True
        fwd_op_list.append(offload_op)
        del_op = DeleteOp(Activation(anode))
        del_op.wait_events.append(offload_op)
        fwd_op_list.append(del_op)

        prefetch_op_list.append(AllocateOp(Activation(anode)))
        prefetch_op = PrefetchOp(Activation(anode), time=anode.mem/1e7)
        prefetch_op.record_event = True
        prefetch_op.wait_events.append(offload_op)
        prefetch_op_list.append(prefetch_op)
        for op in bwd_op_list:
            if isinstance(op, ComputeOp) and op.target in anode.users_real:
                op.wait_events.append(prefetch_op)

    bwd_op_list = bwd_op_list[:1] + prefetch_op_list+bwd_op_list[1:]


    new_op_sched = OpSchedule(fwd_op_list+bwd_op_list, 
                              loss_idx=len(fwd_op_list), 
                              cluster=cluster,
                              init_op_list=op_sched.init_op_list)
    return new_op_sched