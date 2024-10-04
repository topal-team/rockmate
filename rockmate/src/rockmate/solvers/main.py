import rkgb
import torch
import numpy as np
from copy import deepcopy, copy

from rkgb.lowlevel.ast_add_on import ast_to_str
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from rkgb.core.backward import ComputationNode
from rkgb.lowlevel.constants import init_target_string

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


class Solver:
    class Config:
        def __init__(self):
            pass

    def __init__(self, config=None, **kwargs):
        self.config = config if config is not None else type(self).Config(**kwargs)

    def __call__(self, cluster: HierarchicalCluster, budgets=None, *args, **kargs):
        return self.solve(cluster, budgets, *args, **kargs)

    def solve(self, cluster: HierarchicalCluster, budgets=None):
        # -> RETURN list of Op_sched
        pass


class FastSolver(Solver):
    def __init__(self, config=None, recompute_sched=True):
        super().__init__(config)
        self.recompute_sched=recompute_sched

    def solve_hcn(self, hcn, cluster, no_del_names):
        if hcn.sub_cluster is None:  # fwd with no grad
            if hcn.name in cluster.dict_nodes:
                cnode = cluster.dict_nodes[hcn.name]
                hcn.ff_time = cnode.time
                hcn.ff_overhead = cnode.mem_overhead
                hcn.ff_op_list = [ComputeOp(cnode, fast_forward=True)]
            else:  # loss node
                hcn.ff_time = 0
                hcn.ff_overhead = 0
                hcn.ff_op_list = []
            return

        re_cluster = hcn.sub_cluster.representee_cluster
        list_sched = self.solve(re_cluster)

        if re_cluster.list_schedules == []:
            # re_cluster.list_schedules += self.solve(re_cluster)
            re_cluster.list_schedules.append(list_sched[0])
            if self.recompute_sched and not re_cluster.is_bottom:
                re_cluster.list_schedules.append(list_sched[1])
        autograd_sched = list_sched[1]
        hcn.ff_time = autograd_sched.fwd_time
        hcn.ff_overhead = autograd_sched.fwd_overhead
        hcn.ff_op_list = translate(
            hcn.sub_cluster, autograd_sched.op_list[: autograd_sched.loss_idx]
        )

    def solve(
        self,
        cluster: HierarchicalCluster,
        no_del_names=[f"{init_target_string} data", f"{init_target_string} grad"],
        recompute_sched=False,
    ):
        """
        Return basic schedules for the cluster:
        1. PyTorch autograd schedule.
        2. Save_none: forward saves no intermediate activations and delete ASAP;
                      backward runs autograd schedule (fwd+bwd).
        """
        assert cluster is cluster.representee_cluster

        list_sched = []
        # if not cluster is cluster.representee_cluster:
        #     return list_sched
        ff_op_list = self.single_compute_op_list(
            cluster,
            with_backward=False,
            no_del_names=no_del_names,
            fast_forward=True,
        )
        # list_sched.append(
        #     OpSchedule(
        #     ff_op_list + [ComputeOp(ComputationNode("loss"))],
        #     cluster=cluster,
        #     loss_idx=len(ff_op_list)
        #     )
        # )  # not real sched, only for info

        autograd_op_list = self.single_compute_op_list(
            cluster,
            no_del_names=no_del_names,
        )
        autograd_loss_idx = autograd_op_list.index(
            ComputeOp(ComputationNode("loss"), disabled=True)
        )
        autograd_sched = OpSchedule(
            autograd_op_list, cluster=cluster, loss_idx=autograd_loss_idx
        )
        re_autograd_op_list = copy(autograd_op_list) ##LED was deepcopy()
        loss_op = re_autograd_op_list.pop(autograd_loss_idx)
        recompute_op_list = ff_op_list + [loss_op] + re_autograd_op_list
        list_sched.append(autograd_sched)

        list_sched.append(
            OpSchedule(recompute_op_list, cluster=cluster, loss_idx=len(ff_op_list))
        )
        return list_sched

    def single_compute_op_list(
        self,
        cluster: HierarchicalCluster,
        with_backward=True,
        no_del_names=[],
        fast_forward=False,
    ):
        list_cnode = cluster.list_cnodes.copy()
        if not with_backward:
            list_cnode = list_cnode[: cluster.loss_idx]
        loss_i = (
            list_cnode.index(cluster.loss_cnode)
            if cluster.loss_cnode in list_cnode
            else -1
        )

        def _can_del(i, anode):
            if anode.name in no_del_names:
                return False
            for cnode in anode.users_real:  # .union(anode.users_fake):
                if cnode in list_cnode[i + 1 :]:
                    return False

            fwd = not with_backward or i <= loss_i
            if anode in cluster.loss_cnode.deps_real and fwd:
                return False
            # if anode in cluster.interfaces["input_data_anodes"]:
            #     return False
            if anode in cluster.interfaces["input_grad_anodes"]:
                return False
            return True

        op_list = []
        alive_status = {}

        for anode in cluster.list_anodes:
            # if hdn not in cluster.interfaces:
            alive_status[anode.name] = (
                1 if (anode in cluster.interfaces["input_data_anodes"]) else 0
            )

        loss_idx = None
        for i, cnode in enumerate(list_cnode):
            for anode in cnode.users:
                if "phantom" in anode.name and fast_forward:
                    continue
                alive_status[anode.name] = 1
            if cnode == cluster.loss_cnode:
                loss_idx = len(op_list)
            op_list.append(
                ComputeOp(
                    cnode,
                    detach=True,  # not with_bwd,
                    fast_forward=fast_forward,
                    disabled=("loss" in cnode.name),
                )
            )

            for anode_name, alive in alive_status.items():
                if alive:
                    anode = cluster.dict_nodes[anode_name]
                    if _can_del(i, anode):
                        op_list.append(DeleteOp(Activation(anode)))
                        alive_status[anode_name] = 0

        return op_list

    # def recursive_solve_hcn(self,
    #                         hcn,
    #                         no_del_names=[f"{init_target_string} data",
    #                         f"{init_target_string} grad"]):
    #     cluster = hcn.sub_cluster
    #     if not cluster is cluster.representee_cluster or cluster.is_bottom:
    #         return
    #     for hg in cluster.partitionings:
    #         for hcn in hg.list_HCNs:
    #             self.recursive_solve_hcn(hcn, hcn, no_del_names)
    #             self.solve_hcn(hcn, no_del_names)

    def preprocess(
        self,
        cluster: HierarchicalCluster,
        no_del_names=[f"{init_target_string} data", f"{init_target_string} grad"],
    ):
        if not cluster is cluster.representee_cluster or cluster.is_bottom:
            return
        for hg in cluster.partitionings:
            for hcn in hg.list_HCNs:
                self.solve_hcn(hcn, cluster, no_del_names)


def add_sched(cluster: HierarchicalCluster, sched):
    cluster.list_schedules.append(sched)


def get_sched(cluster: HierarchicalCluster, pareto=False):
    representee = cluster.representee_cluster
    if not pareto:
        return representee.list_schedules
    else:
        list_schedules = representee.list_schedules
        time_mem = np.array(
            [(sum(op_sched.time), op_sched.mem) for op_sched in list_schedules]
        )
        is_pareto = np.ones(len(list_schedules), dtype=bool)
        for i, c in enumerate(time_mem):
            is_pareto[i] = np.all(np.any(time_mem >= c, axis=1))

    return [list_schedules[i] for i, p in enumerate(is_pareto) if p]


def translate(cluster: HierarchicalCluster, op_list):
    if cluster is cluster.representee_cluster:
        return op_list
    translator_re = cluster.representee_cluster.translator
    translator = cluster.translator
    translated_op_list = [ copy(op) for op in op_list]

    def translate_op(op):
        if isinstance(op, ComputeOp):
            ana_kn = translator_re.to_ano(op.target)
            op.target = translator.from_ano(ana_kn)
        elif isinstance(op.target, Activation):
            ana_kn = translator_re.to_ano(op.target.anode)
            op.target = Activation(translator.from_ano(ana_kn))
        else:
            raise ValueError

    for op in translated_op_list:
        translate_op(op)
        for e in op.wait_events:
            re_node = translator_re.to_ano(e[1])
            e[1] = translator.from_ano(re_node)
    return translated_op_list


def get_hgraph_budget_lb(hgraph: HierarchicalGraph):
    # Lower bound for minimum feasible budget given schedules
    hcn_memory_budget = []
    for hcn in hgraph.list_HCNs:
        if hcn.sub_cluster is not None:
            # list_schedules = hcn.sub_cluster.get_sched()
            list_schedules = get_sched(hcn.sub_cluster)
            hcn_memory_budget.append(
                min(op_sched.peak_mem for op_sched in list_schedules)
            )
        else:
            hcn_memory_budget.append(hcn.ff_overhead)
    return max(hcn_memory_budget, default=0)


def get_hgraph_budget_ub(hgraph: HierarchicalGraph):
    # Upper bound for minimum feasible budget given schedules
    cluster = hgraph.cluster
    if cluster.representee_cluster.list_schedules == []:
        autograd_op_list = get_single_compute_op_list(
            cluster,
            with_bwd=True,
        )
        autograd_sched = OpSchedule(
            autograd_op_list,
            cluster=cluster,
        )
    else:
        autograd_sched = cluster.representee_cluster.list_schedules[0]
    max_bdg = autograd_sched.mem + autograd_sched.bwd_overhead
    return max_bdg


def get_cluster_budget(
    cluster: HierarchicalCluster,
    nb_bdg_peak=3,
    nb_bdg_save=6,
    overall_bdg=None,
    with_save_budget=False,
):
    # assuming solving budget does not based on lower level solution
    budgets = []

    sizes = [anode.mem for anode in cluster.list_anodes]
    overheads = [
        cnode.mem_overhead for cnode in cluster.list_cnodes if cnode.mem_overhead is not None
    ]

    # overheads = [hcn.sub_cluster.ff_overhead for hcn in hg.list_HCNs] + [
    #     op_sched.bwd_overhead for op_sched in hg.list_schedules
    # ]
    # max_bdg = sum(sizes) + max(overheads)
    if cluster.representee_cluster.list_schedules == []:
        autograd_op_list = get_single_compute_op_list(
            cluster,
            with_bwd=True,
        )
        autograd_sched = OpSchedule(
            autograd_op_list,
            cluster=cluster,
        )
    else:
        autograd_sched = cluster.representee_cluster.list_schedules[0]
    interfaces_mem = sum(anode.mem for anode in cluster.all_interfaces)
    max_bdg = autograd_sched.mem + autograd_sched.bwd_overhead
    if overall_bdg is not None:
        max_bdg = min(max_bdg, overall_bdg)
    min_bdg = max(overheads)
    # max_bdg = hg.list_schedules[0].mem + max(overheads)

    # TODO: find the minimum feasible budget
    # min_bdg = hg.fast_fwd_overhead()[0]
    # min_bdg = min(op_sched.mem for op_sched in hg.list_schedules) + max(overheads)

    l_bd_peak = np.linspace(min_bdg, max_bdg, nb_bdg_peak) + interfaces_mem
    if not with_save_budget:
        return l_bd_peak
    for bd_peak in l_bd_peak:
        l_bd_save = (
            np.linspace(
                0,
                min(bd_peak, autograd_sched.mem),
                nb_bdg_save,
            )
            + interfaces_mem
        )
        # for bd_save in l_bd_save:
        #     budgets.append((bd_peak, bd_save))
        budgets.append((bd_peak, l_bd_save))
    return budgets


def solve_recursive(h_cluster: HierarchicalCluster, list_solvers=[], skip_self=False):
    # assume it's representee
    # print(h_cluster.name)
    for hg in h_cluster.partitionings:
        # print(hg.name)
        for hcn in hg.list_HCNs:
            if (
                hcn.is_fwd
                and hcn.sub_cluster is not None
                and not hcn.sub_cluster.is_bottom
                and hcn.sub_cluster is hcn.sub_cluster.representee_cluster
            ):
                sub_cluster = hcn.sub_cluster
                solve_recursive(sub_cluster, list_solvers)
    if not skip_self:
        for solver in list_solvers:
            # h_cluster.solve(solver)
            if h_cluster is h_cluster.representee_cluster:
                last_time = time.time()
                h_cluster.list_schedules.extend(solver(h_cluster))


# Preprocessing Cluster: add fast_forward and autograd option
def preprocess_rec(cluster: HierarchicalCluster):
    if cluster is cluster.representee_cluster:
        if not cluster.is_bottom:
            for hg in cluster.partitionings:
                for hcn in hg.list_HCNs:
                    if hcn.is_fwd and hcn.sub_cluster is not None:
                        # if not hcn.sub_cluster.list_schedules:
                        preprocess_rec(hcn.sub_cluster)
            preprocess(cluster)


def preprocess(
    cluster: HierarchicalCluster,
    no_del_names=[f"{init_target_string} data", f"{init_target_string} grad"],
    add_no_save_sched=True,
):
    if cluster is cluster.representee_cluster:
        for hg in cluster.partitionings:
            for hcn in hg.list_HCNs:
                if hcn.sub_cluster is None:  # fwd with no grad
                    if hcn.name in cluster.dict_nodes:
                        cnode = cluster.dict_nodes[hcn.name]
                        ff_op_list = [ComputeOp(cnode, fast_forward=True)]
                        hcn.ff_time = cnode.time
                        hcn.ff_overhead = cnode.mem_overhead
                    else:  # loss node
                        ff_op_list = []
                        hcn.ff_time = 0
                        hcn.ff_overhead = 0
                else:
                    ff_op_list = get_single_compute_op_list(
                        hcn.sub_cluster,
                        with_bwd=False,
                        no_del_names=no_del_names,
                        ff=True,
                    )
                    ff_op_sched = OpSchedule(
                        ff_op_list + [ComputeOp(ComputationNode("loss"))],
                        cluster=cluster,
                        loss_idx=len(ff_op_list),
                        # correct_overhead=False,
                    )  # not real sched, only for info
                    hcn.ff_time = ff_op_sched.fwd_time
                    hcn.ff_overhead = ff_op_sched.fwd_overhead
                    if (
                        hcn.sub_cluster.representee_cluster is hcn.sub_cluster
                        and hcn.sub_cluster.list_schedules == []
                    ):
                        autograd_op_list = get_single_compute_op_list(
                            hcn.sub_cluster,
                            with_bwd=True,
                            no_del_names=no_del_names,
                        )
                        hcn.sub_cluster.list_schedules.append(
                            OpSchedule(
                                autograd_op_list,
                                cluster=hcn.sub_cluster,
                                loss_idx=autograd_op_list.index(
                                    ComputeOp(ComputationNode("loss"), disabled=True)
                                ),
                            )
                        )
                        if add_no_save_sched:
                            hcn.sub_cluster.list_schedules.append(
                                OpSchedule(
                                    ff_op_list
                                    + [ComputeOp(ComputationNode("loss"))]
                                    + autograd_op_list,
                                    cluster=hcn.sub_cluster,
                                    loss_idx=len(ff_op_list),
                                )
                            )
                hcn.ff_op_list = ff_op_list


def get_single_compute_op_list(
    cluster: HierarchicalCluster, with_bwd=True, no_del_names=[], ff=False
):
    list_cnode = cluster.list_cnodes.copy()
    if not with_bwd:
        list_cnode = list_cnode[: cluster.loss_idx]
    loss_i = (
        list_cnode.index(cluster.loss_cnode) if cluster.loss_cnode in list_cnode else -1
    )

    def _can_del(i, anode):
        if anode.name in no_del_names:
            return False
        # for cnode in list_cnode[i + 1 :]:
        #     if anode in cnode.deps_real:
        #         return False
        for cnode in anode.users_real:
            if cnode in list_cnode[i + 1 :]:
                return False

        if anode in cluster.loss_cnode.deps_real and i <= loss_i:
            return False
        # if anode in cluster.interfaces["input_data_anodes"]:
        #     return False
        if anode in cluster.interfaces["input_grad_anodes"]:
            return False
        return True

    op_list = []
    # alive_list = []
    alive_status = {}
    for anode in cluster.list_anodes:
        # if hdn not in cluster.interfaces:
        alive_status[anode.name] = (
            1 if (anode in cluster.interfaces["input_data_anodes"]) else 0
        )

    loss_idx = None
    for i, cnode in enumerate(list_cnode):
        for anode in cnode.users:
            if "phantom" in anode.name and ff:
                continue
            alive_status[anode.name] = 1
        if cnode == cluster.loss_cnode:
            loss_idx = len(op_list)
        op_list.append(
            ComputeOp(
                cnode,
                detach=True,  # not with_bwd,
                fast_forward=ff,
                disabled=("loss" in cnode.name),
            )
        )

        for anode_name, alive in alive_status.items():
            if alive:
                anode = cluster.dict_nodes[anode_name]
                if _can_del(i, anode):
                    op_list.append(DeleteOp(Activation(anode)))
                    alive_status[anode_name] = 0

    return op_list  # , loss_idx


