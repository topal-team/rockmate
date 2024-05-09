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
from ..op_schedule import (
    OpSchedule,
    ComputeOp,
    DeleteOp,
    Activation,
    OffloadOp,
    PrefetchOp,
    AllocateOp,
)
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
        avg_time = np.mean(
            [cnode.time for cnode in cluster.list_cnodes if cnode.time is not None]
        )

        def is_cheap(cnode):
            if not cnode.is_fwd:
                return False
            if cnode.time is None:
                return False
            return cnode.time < avg_time / 2

        cnode_idx = {cnode.name: i for i, cnode in enumerate(cluster.list_cnodes)}
        loss_idx = len([cnode for cnode in cluster.list_cnodes if cnode.is_fwd]) - 1
        cheap_cnodes = {}
        for cnode in cluster.list_cnodes:
            user_cnodes = [
                user_cnode
                for anode in cnode.users
                for user_cnode in anode.users_real
                if user_cnode.name in cnode_idx
            ]
            # has_bwd_user = any(not user_cnode.is_fwd for user_cnode in user_cnodes)
            if is_cheap(cnode):  # and has_bwd_user:
                cheap_cnodes[cnode.name] = cnode

        anodes_del_idx = {i: [] for i, _ in enumerate(cluster.list_cnodes)}
        output_anodes = []
        for anode in cluster.list_anodes:
            if "source" in anode.name:
                continue
            user_indices = []
            for cnode in anode.users_real:
                if cnode.name in cnode_idx:
                    user_indices.append(cnode_idx[cnode.name])
                if cnode.name in cheap_cnodes:
                    user_indices.append(loss_idx)

            if user_indices:
                last_user_idx = max(user_indices)
                anodes_del_idx[last_user_idx].append(anode)
            else:
                output_anodes.append(anode)

            regenerated = any(cnode.name in cheap_cnodes for cnode in anode.deps)
            if regenerated and all(cnode.is_fwd for cnode in anode.deps):
                anodes_del_idx[loss_idx].append(anode)

            if (
                all(is_cheap(cnode) for cnode in anode.deps)
                and anode.allocation_type == "data"
            ):
                user_indices_fwd = [
                    cnode_idx[cnode.name]
                    for cnode in anode.users_real
                    if cnode.is_fwd and cnode.name in cnode_idx
                ]
                if user_indices_fwd:
                    last_user_idx_fwd = max(user_indices_fwd)

                anodes_del_idx[last_user_idx_fwd].append(anode)

        fwd_op_list = []
        bwd_op_list = []
        for i, cnode in enumerate(cluster.list_cnodes[:loss_idx]):
            fwd_op_list.append(ComputeOp(cnode, disabled="loss" in cnode.name))
            # if "loss" in cnode.name:
            #     loss_idx = i
            #     break
            for anode in anodes_del_idx[i]:
                fwd_op_list.append(DeleteOp(Activation(anode)))

        fwd_op_list.append(ComputeOp(cluster.list_cnodes[loss_idx], disabled=True))

        for anode in output_anodes:
            bwd_op_list.append(DeleteOp(Activation(anode)))
        for cnode in cheap_cnodes.values():
            bwd_op_list.append(ComputeOp(cnode))
        for anode in anodes_del_idx[loss_idx]:
            bwd_op_list.append(DeleteOp(Activation(anode)))

        for i, cnode in enumerate(cluster.list_cnodes[loss_idx + 1 :]):
            bwd_op_list.append(ComputeOp(cnode))
            for anode in anodes_del_idx[i + loss_idx + 1]:
                bwd_op_list.append(DeleteOp(Activation(anode)))

        op_sched = OpSchedule(
            fwd_op_list + bwd_op_list, loss_idx=len(fwd_op_list) - 1, cluster=cluster
        )
        return [add_activation_offload(op_sched)]


def add_activation_offload(op_sched: OpSchedule) -> OpSchedule:
    """
    With all the torch.cuda.Event and stream.wait_event() to synchornize
    between offload/prefetch, in the order of Main>Ofl>Del during fwd and Prf>Main during bwd

    """
    cluster = op_sched.cluster
    op_list = op_sched.op_list
    fwd_op_list = op_list[: op_sched.loss_idx]
    bwd_op_list = op_list[op_sched.loss_idx :]
    prefetch_op_list = []
    phantoms = op_sched.phantoms
    for anode in phantoms:
        if not anode.allocation_type == "data":
            continue
        i = max(
            min(op_sched.occurrences[ComputeOp(cnode).name]) for cnode in anode.deps
        )
        op_sched.op_list[i].record_event = True
        comp_op = op_sched.op_list[i]
        offload_op = OffloadOp(Activation(anode), time=anode.mem / 1e7)
        offload_op.wait_events.append([comp_op.op_type, comp_op.target.name])
        offload_op.record_event = True
        fwd_op_list.append(offload_op)
        del_op = DeleteOp(Activation(anode))
        del_op.wait_events.append([offload_op.op_type, offload_op.target.name])
        fwd_op_list.append(del_op)

        prefetch_op_list.append(AllocateOp(Activation(anode)))
        prefetch_op = PrefetchOp(Activation(anode), time=anode.mem / 1e7)
        prefetch_op.record_event = True
        prefetch_op.wait_events.append([offload_op.op_type, offload_op.target.name])
        prefetch_op_list.append(prefetch_op)
        for op in bwd_op_list:
            if isinstance(op, ComputeOp) and op.target in anode.users_real:
                op.wait_events.append([prefetch_op.op_type, prefetch_op.target.name])

    bwd_op_list = bwd_op_list[:1] + prefetch_op_list + bwd_op_list[1:]

    new_op_sched = OpSchedule(
        fwd_op_list + bwd_op_list,
        loss_idx=len(fwd_op_list),
        cluster=cluster,
        init_op_list=op_sched.init_op_list,
    )
    return new_op_sched
