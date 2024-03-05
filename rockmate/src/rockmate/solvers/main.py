import rkgb
import torch
import numpy as np
from copy import deepcopy
# from rkgb.Htools import H_C_node, H_D_node, H_graph, H_cluster
# from rkgb.Ktools import K_C_node, K_D_node
# from rkgb.utils.ast_add_on import ast_to_str
# from rkgb.utils.def_info import Var_info
# from rkgb.utils import irotor

from rkgb.lowlevel.ast_add_on import ast_to_str
from rkgb.lowlevel.measure import TimerCPU
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from rkgb.core.backward import ComputationNode
from rkgb.lowlevel.constants import init_target_string

from ..op_schedule import OpSchedule, ComputeOp, DeleteOp, Activation
import time
import psutil


class Solver:
    class Config:
        def __init__(self):
            pass

    def __init__(self, config=None):
        self.config = config if config is not None else type(self).Config()

    def __call__(self, cluster: HierarchicalCluster, budgets=None, *args, **kargs):
        return self.solve(cluster, budgets, *args, **kargs)

    def solve(self, cluster: HierarchicalCluster, budgets=None):
        # -> RETURN list of Op_sched
        pass


def H_cluster_method_get_sched(self, pareto=False):
    representee = self.representee_cluster
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
    # if self is not representee:
    #     repr_sols = representee.get_sched()
    #     ano_sols = [representee.translator.translate(sol) for sol in repr_sols]
    #     return [self.translator.reverse_translate(sol) for sol in ano_sols]
    # else:
    #     return self.list_schedules


def H_cluster_method_translate_op_list(self, op_list):
    if self is self.representee_cluster:
        return op_list
    translator_re = self.representee_cluster.translator
    translator = self.translator
    translated_op_list = deepcopy(op_list)
    for op in translated_op_list:
        if isinstance(op, DeleteOp):
            # ana_kn = translator_re.dict_name_to_ano_triplet[op.target.name]
            # op.target = Activation(translator.dict_ano_triplet_to_anode[ana_kn])
            ana_kn = translator_re.to_ano(op.target.anode)
            op.target = Activation(translator.from_ano(ana_kn))
        else:
            # ana_kn = translator_re.dict_name_to_ano_triplet[op.cnode.name]
            # op.cnode = translator.dict_ano_triplet_to_cnode[ana_kn]
            ana_kn = translator_re.to_ano(op.target)
            op.target = translator.from_ano(ana_kn)
    return translated_op_list


def H_cluster_method_solve(self, solver: Solver):
    if self is not self.representee_cluster:
        self.representee_cluster.solve(solver)
    elif solver.stop_condition(self):
        pass
    else:
        self.sched.extend(solver(self))


setattr(HierarchicalCluster, "get_sched", H_cluster_method_get_sched)
setattr(HierarchicalCluster, "solve", H_cluster_method_solve)
setattr(HierarchicalCluster, "translate_op_list", H_cluster_method_translate_op_list)


def get_hgraph_budget_lb(hgraph: HierarchicalGraph):
    # Lower bound for minimum feasible budget given schedules
    hcn_memory_budget = []
    for hcn in hgraph.list_HCNs:
        if hcn.sub_cluster is not None:
            list_schedules = hcn.sub_cluster.get_sched()
            hcn_memory_budget.append(min(op_sched.peak_mem for op_sched in list_schedules))
        else:
            hcn_memory_budget.append(hcn.ff_overhead)
    return max(hcn_memory_budget)


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

    # hg = cluster
    # return [1e12]
    # def get_hg_budgets(hg, nb_bdg_peak=3, nb_bdg_save=6):
    # return reasonable budget list
    budgets = []
    sizes = []
    # fwd_hdns = set()
    # for hcn in hg.list_HCNs:
    #     # if hcn.is_fwd:
    #     for hdn in hcn.users:
    #         # if hdn not in hg.interfaces:
    #         #     fwd_hdns.add(hdn)
    #         if not hcn.sub_cluster is None:
    #             sizes.append(hcn.sub_cluster.list_schedules[0].mem)
    # sizes += [hdn.mem for hdn in hg.list_hdn]
    sizes = [anode.mem for anode in cluster.list_anodes]
    overheads = [cnode.mem_overhead for cnode in cluster.list_cnodes if cnode.mem_overhead]

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
                # print(
                #     f"Time to solve {h_cluster.name} of size {len(h_cluster.list_cnodes)}: {time.time() - last_time}"
                # )
                # mem = psutil.virtual_memory()
                # print(
                #     f"The CPU mem usage when solving {h_cluster.name} is: ",
                #     # psutil.cpu_percent(4)
                #     mem.used,
                # )


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


def preprocess(cluster: HierarchicalCluster, 
               protect_names=[f"{init_target_string} data", 
                              f"{init_target_string} grad"], 
               add_no_save_sched=True):
    if cluster is cluster.representee_cluster:
        # assert cluster.list_schedules == []  # only visit representee once
        # autograd_op_list, autograd_loss_idx = get_single_compute_op_list(
        #     cluster, with_bwd=True, protect_names=protect_names
        # )
        # cluster.list_schedules.append(
        #     OpSchedule(autograd_op_list, autograd_loss_idx, cluster=cluster)
        # )
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
                        protect_names=protect_names,
                        ff=True,
                    )
                    ff_op_sched = OpSchedule(
                        ff_op_list + [ComputeOp(ComputationNode("loss"))],
                        cluster=cluster,
                        loss_idx=len(ff_op_list)
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
                            protect_names=protect_names,
                        )
                        hcn.sub_cluster.list_schedules.append(
                            OpSchedule(
                                autograd_op_list,
                                cluster=hcn.sub_cluster,
                                loss_idx=autograd_op_list.index(ComputeOp(ComputationNode("loss"), disabled=True))
                            )
                        )
                        if add_no_save_sched:
                            hcn.sub_cluster.list_schedules.append(
                                OpSchedule(
                                    ff_op_list + [ComputeOp(ComputationNode("loss"))]+autograd_op_list,
                                    cluster=hcn.sub_cluster,
                                    loss_idx=len(ff_op_list)
                                )
                            )
                hcn.ff_op_list = ff_op_list


def get_single_compute_op_list(
    cluster: HierarchicalCluster, with_bwd=True, protect_names=[], ff=False
):
    list_cnode = cluster.list_cnodes.copy()
    if not with_bwd:
        list_cnode = list_cnode[: cluster.loss_idx]
    loss_idx = list_cnode.index(cluster.loss_cnode) if cluster.loss_cnode in list_cnode else -1

    def _can_del(i, anode):
        if anode.name in protect_names:
            return False
        # for cnode in list_cnode[i + 1 :]:
        #     if anode in cnode.deps_real:
        #         return False
        for cnode in anode.users_real:
            if cnode in list_cnode[i + 1 :]:
                return False
        
        if anode in cluster.loss_cnode.deps_real and i<= loss_idx:
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

    for i, cnode in enumerate(list_cnode):
        # if i == cluster.loss_idx:
        #     loss_idx = len(op_list)
        for anode in cnode.users:
            if "phantom" in anode.name and ff:
                continue
            alive_status[anode.name] = 1
        op_list.append(
            ComputeOp(
                cnode,
                detach=True,  # not with_bwd,
                fast_forward=ff,
                disabled=("loss" in cnode.name),
            )
        )
        # alive_list.append(alive_status.copy())

        for anode_name, alive in alive_status.items():
            if alive:
                anode = cluster.dict_nodes[anode_name]
                if _can_del(i, anode):
                    op_list.append(DeleteOp(Activation(anode)))

                    alive_status[anode_name] = 0
                    # alive_list.append(alive_status.copy())
    # print(alive_status)
    return op_list  # , loss_idx


def get_optimize_stats(_p, cpu_optim, gpu_optim, optim_kwargs={}, niter=10):
    # timer = irotor.make_timer(torch.device("cpu"))
    timer = TimerCPU()
    a_c = torch.ones([10, 1024,1024], device="cpu", pin_memory=True)
    a_g = torch.ones([10, 1024,1024], device="cuda")
    b_c = torch.ones([10, 1024,1024], device="cpu", pin_memory=True)
    b_g = torch.ones([10, 1024,1024], device="cuda")

    p = deepcopy(_p).to("cuda")
    # if not p.is_leaf:
    p = torch.ones([10,1024,1024], dtype=_p.dtype).to("cuda")
    size = p.numel()
    p.grad = torch.ones_like(p)
    optimizer = gpu_optim([p], **optim_kwargs)
    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.memory_allocated()
    timer.start()
    for i in range(niter):
        optimizer.step()
    timer.end()
    mem_after = torch.cuda.memory_allocated()
    gpu_optimize_speed = size*p.element_size()*niter/timer.elapsed()
    opt_size = mem_after - mem
    opt_overhead = torch.cuda.max_memory_allocated() - mem_after

    p_c = torch.zeros_like(p, device="cpu")
    p_c.grad = torch.ones_like(p_c)
    optimizer = cpu_optim([p_c], **optim_kwargs)
    optimizer.step()
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
    bandwidth = niter*a_c.numel()*a_c.element_size()/timer.elapsed()
    timer.start()
    for i in range(niter):
        with torch.cuda.stream(torch.cuda.Stream()):
            a_c.copy_(a_g, non_blocking=True)
            a_c.copy_(a_g, non_blocking=True)
            a_c.copy_(a_g, non_blocking=True)
            a_c.copy_(a_g, non_blocking=True)
            a_c.copy_(a_g, non_blocking=True)
        with torch.cuda.stream(torch.cuda.Stream()):
            b_g.copy_(b_c, non_blocking=True)
            b_g.copy_(b_c, non_blocking=True)
            b_g.copy_(b_c, non_blocking=True)
            b_g.copy_(b_c, non_blocking=True)
            b_g.copy_(b_c, non_blocking=True)
        optimizer.step()
    timer.end()
    optimize_stats = {"optimizer_states_size": round(opt_size//size/p.element_size()),
                          "optimizer_overhead":round(opt_overhead//size/p.element_size()),
                          "cpu_optimize_speed": size*p.element_size()*niter/timer.elapsed(),
                          "gpu_optimize_speed":gpu_optimize_speed,
                          "bandwidth": bandwidth}
    return optimize_stats

