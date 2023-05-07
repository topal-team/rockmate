import rkgb
import numpy as np
from rkgb.Htools import H_C_node, H_D_node, H_graph, H_cluster
from rkgb.Ktools import K_C_node, K_D_node
from solvers.op_schedule import Op, OpSchedule


class Solver:
    class Config:
        def __init__(self):
            pass

    def __init__(self, config=None):
        self.config = config if config is not None else type(self).Config()

    def __call__(self, cluster: H_cluster, budgets=None, accurate_mem=False):
        return self.solve(cluster, budgets, accurate_mem=accurate_mem)

    def solve(self, cluster: H_cluster, budgets=None):
        # -> RETURN list of Op_sched
        pass


def H_cluster_method_get_sched(self):
    representee = self.representee_cluster
    return representee.list_sched
    # if self is not representee:
    #     repr_sols = representee.get_sched()
    #     ano_sols = [representee.translator.translate(sol) for sol in repr_sols]
    #     return [self.translator.reverse_translate(sol) for sol in ano_sols]
    # else:
    #     return self.list_sched


def H_cluster_method_solve(self, solver: Solver):
    if self is not self.representee_cluster:
        self.representee_cluster.solve(solver)
    elif solver.stop_condition(self):
        pass
    else:
        self.sched.extend(solver(self))


setattr(H_cluster, "get_sched", H_cluster_method_get_sched)
setattr(H_cluster, "solve", H_cluster_method_solve)


def get_cluster_budget(
    cluster: H_cluster, nb_bdg_peak=3, nb_bdg_save=6, overall_bdg=None
):
    # assuming solving budget does not based on lower level solution

    # hg = cluster
    # return [1e12]
    # def get_hg_budgets(hg, nb_bdg_peak=3, nb_bdg_save=6):
    # return reasonable budget list
    budgets = []
    sizes = []
    # fwd_hdns = set()
    # for hcn in hg.list_hcn:
    #     # if hcn.is_fwd:
    #     for hdn in hcn.users:
    #         # if hdn not in hg.interfaces:
    #         #     fwd_hdns.add(hdn)
    #         if not hcn.sub_cluster is None:
    #             sizes.append(hcn.sub_cluster.list_sched[0].mem)
    # sizes += [hdn.mem for hdn in hg.list_hdn]
    sizes = [kdn.mem for kdn in cluster.list_kdn]
    overheads = [kcn.overhead for kcn in cluster.list_kcn if kcn.overhead]

    # overheads = [hcn.sub_cluster.ff_overhead for hcn in hg.list_hcn] + [
    #     op_sched.bwd_overhead for op_sched in hg.list_sched
    # ]
    # max_bdg = sum(sizes) + max(overheads)
    if cluster.representee_cluster.list_sched == []:
        autograd_op_list= get_single_compute_op_list(
            cluster,
            with_bwd=True,
        )
        autograd_sched = OpSchedule(
            autograd_op_list,
            cluster=cluster,
        )
    else:
        autograd_sched = cluster.representee_cluster.list_sched[0]
    interfaces_mem = sum(kdn.mem for kdn in cluster.all_interfaces)
    max_bdg = autograd_sched.mem + autograd_sched.bwd_overhead
    if overall_bdg is not None:
        max_bdg = min(max_bdg, overall_bdg)
    min_bdg = max(overheads)
    # max_bdg = hg.list_sched[0].mem + max(overheads)

    # TODO: find the minimum feasible budget
    # min_bdg = hg.fast_fwd_overhead()[0]
    # min_bdg = min(op_sched.mem for op_sched in hg.list_sched) + max(overheads)

    l_bd_peak = np.linspace(min_bdg, max_bdg, nb_bdg_peak) + interfaces_mem
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


def solve_recursive(h_cluster: H_cluster, list_solvers=[]):
    # assume it's representee
    for hg in h_cluster.possible_hg:
        for hcn in hg.list_hcn:
            if (
                hcn.is_fwd
                and hcn.sub_cluster is not None
                and not hcn.sub_cluster.is_bottom
            ):
                sub_cluster = hcn.sub_cluster
                # if not stop_condition(
                #     sub_cluster.representee, h_cluster
                # ):  # e.g. already solved/bottom level
                solve_recursive(sub_cluster, list_solvers)
    for solver in list_solvers:
        # h_cluster.solve(solver)
        if h_cluster is h_cluster.representee_cluster:
            h_cluster.list_sched.extend(solver(h_cluster))


# Preprocessing Cluster: add fast_forward and autograd option
def preprocess_rec(cluster: H_cluster):
    if cluster is cluster.representee_cluster:
        if not cluster.is_bottom:
            for hg in cluster.possible_hg:
                for hcn in hg.list_hcn:
                    if hcn.is_fwd and hcn.sub_cluster is not None:
                        # if not hcn.sub_cluster.list_sched:
                        preprocess_rec(hcn.sub_cluster)
            preprocess(cluster)


def preprocess(cluster: H_cluster, protect_names=[]):
    if cluster is cluster.representee_cluster:
        # assert cluster.list_sched == []  # only visit representee once
        # autograd_op_list, autograd_loss_idx = get_single_compute_op_list(
        #     cluster, with_bwd=True, protect_names=protect_names
        # )
        # cluster.list_sched.append(
        #     OpSchedule(autograd_op_list, autograd_loss_idx, cluster=cluster)
        # )
        for hg in cluster.possible_hg:
            for hcn in hg.list_hcn:
                if hcn.sub_cluster is None:  # fwd with no grad
                    if hcn.name in cluster.dict_kn:
                        kcn = cluster.dict_kn[hcn.name]
                        ff_op_list = [Op(kcn, fast_forward=True)]
                        hcn.ff_time = kcn.time
                        hcn.ff_overhead = kcn.overhead
                    else:  # loss node
                        ff_op_list = []
                        hcn.ff_time = 0
                        hcn.ff_overhead = 0
                else:
                    if (
                        hcn.sub_cluster.representee_cluster is hcn.sub_cluster
                        and hcn.sub_cluster.list_sched == []
                    ):
                        autograd_op_list = get_single_compute_op_list(
                            hcn.sub_cluster,
                            with_bwd=True,
                            protect_names=protect_names,
                        )
                        hcn.sub_cluster.list_sched.append(
                            OpSchedule(
                                autograd_op_list,
                                cluster=hcn.sub_cluster,
                            )
                        )
                    ff_op_list = get_single_compute_op_list(
                        hcn.sub_cluster,
                        with_bwd=False,
                        protect_names=protect_names,
                        ff=True,
                    )
                    ff_op_sched = OpSchedule(
                        ff_op_list+[Op(K_C_node("loss"))], cluster=cluster, correct_overhead=False
                    )  # not real sched, only for info
                    hcn.ff_time = ff_op_sched.fwd_time
                    hcn.ff_overhead = ff_op_sched.fwd_overhead
                hcn.ff_op_list = ff_op_list


def get_single_compute_op_list(
    cluster: H_cluster, with_bwd=True, protect_names=[], ff=False
):
    list_kcn = cluster.list_kcn.copy()
    if not with_bwd:
        list_kcn = list_kcn[: cluster.loss_idx]

    def _can_del(i, kdn):
        if kdn.name in protect_names:
            return False
        for kcn in list_kcn[i + 1 :]:
            if kdn in kcn.deps_real:
                return False
        return True

    op_list = []
    # alive_list = []
    alive_status = {}
    for kdn in cluster.list_kdn:
        # if hdn not in cluster.interfaces:
        alive_status[kdn.name] = (
            1 if (kdn in cluster.interfaces["inputs_kdn_data"]) else 0
        )

    for i, kcn in enumerate(list_kcn):
        # if i == cluster.loss_idx:
        #     loss_idx = len(op_list)
        for kdn in kcn.users:
            alive_status[kdn.name] = 1
        op_list.append(
            Op(
                kcn,
                detach=True,  # not with_bwd,
                fast_forward=ff,
                disabled=("loss" in kcn.name),
            )
        )
        # alive_list.append(alive_status.copy())

        for kdn_name, alive in alive_status.items():
            kdn = cluster.dict_kn[kdn_name]
            if alive and _can_del(i, kdn):
                op_list.append(Op(kdn))
                alive_status[kdn_name] = 0
                # alive_list.append(alive_status.copy())

    return op_list  # , loss_idx


# def add_autograd_sched(cluster: H_cluster, protect_names=[]):
#     if cluster is cluster.representee_cluster:
#         op_list, loss_idx = get_single_compute_op_list(
#             cluster, with_bwd=True, protect_names=protect_names
#         )
#         cluster.list_sched.append(
#             OpSchedule(op_list, loss_idx=loss_idx, cluster=cluster)
#         )

#     list_kcn = cluster.list_kcn

#     def _can_del(i, kdn):
#         if kdn.name in protect_names:
#             return False
#         for kcn in cluster.list_kcn[i + 1 :]:
#             if kdn in kcn.deps_real:
#                 return False
#         # for kcn in kdn.users_real:
#         #     if cluster.list_kcn.index(kcn) > i:
#         #         return False
#         return True

#     if len(list_kcn) == 3:  # leaf nodes: fwd+bwd pair & loss
#         op_list = [
#             Op(list_kcn[0], detach=True),
#             Op(list_kcn[1], disabled=True),
#             Op(list_kcn[2], detach=True),
#         ]
#         for kdn in list_kcn[2].deps_real:
#             if "phantoms" in kdn.name:
#                 op_list.append(Op(kdn))
#         loss_idx = 1
#     else:
#         op_list = []
#         alive_list = []
#         alive_status = {}
#         for kdn in cluster.list_kdn:
#             # if hdn not in cluster.interfaces:
#             alive_status[kdn.name] = (
#                 1 if (kdn in cluster.interfaces["inputs_kdn_data"]) else 0
#             )

#         for i, kcn in enumerate(cluster.list_kcn):
#             if i == cluster.loss_idx:
#                 loss_idx = len(op_list)
#             for kdn in kcn.users:
#                 alive_status[kdn.name] = 1
#             op_list.append(Op(kcn, detach=True, disabled=("loss" in kcn.name)))
#             alive_list.append(alive_status.copy())

#             for kdn_name, alive in alive_status.items():
#                 kdn = cluster.dict_kn[kdn_name]
#                 if alive and _can_del(i, kdn):
#                     op_list.append(Op(kdn))
#                     alive_status[kdn_name] = 0
#                     alive_list.append(alive_status.copy())

#     return OpSchedule(op_list, loss_idx, cluster.interfaces)


# save_all & fast_forward
# they should be based on the toposorted order of KCN
# adding them to H_cluster recursively


# def fast_fwd_overhead(self):
#     def _can_del(i, hdn):
#         for hcn in hdn.users:
#             if not hcn.is_fwd:
#                 continue
#             if self.list_hcn.index(hcn) > i:
#                 return False
#         return True

#     loss_idx = self.list_hcn.index(self.loss_hcn)
#     op_list = []
#     alive_mem = []
#     alive_list = []
#     alive_status = np.zeros(len(self.list_hdn), dtype=bool)
#     for hdn in self.inputs_hdn_data:
#         alive_status[self.list_hdn.index(hdn)] = 1
#     loss_idx = self.list_hcn.index(self.loss_hcn)
#     for i, hcn in enumerate(self.list_hcn[:loss_idx]):
#         op_list += hcn.ff_op_list
#         for hdn in hcn.users:
#             alive_status[self.list_hdn.index(hdn)] = 1
#         alive_list.append(alive_status.copy())
#         alive_mem.append(
#             sum(
#                 hdn.mem
#                 for j, hdn in enumerate(self.list_hdn)  # No phantom in FF
#                 if alive_status[j]
#             )
#         )

#         for j, hdn in enumerate(self.list_hdn):
#             # if False:  # TODO: if hdn is interface
#             # if hdn in self.inputs_hdn_data:
#             #     continue
#             if alive_status[j] and _can_del(i, hdn):
#                 alive_status[j] = 0
#                 # op_list.append(H_op(hdn.name, hdn, is_del=True))
#                 op_list.append(Op(hdn.kdn))
#                 alive_list.append(alive_status.copy())
#                 alive_mem.append(
#                     sum(
#                         hdn.mem
#                         for j, hdn in enumerate(self.list_hdn)
#                         if alive_status[j]
#                     )
#                 )
#     return max(alive_mem) - alive_mem[-1], op_list


# def add_sched(self, sched):
#     pareto = True
#     for opt in self.list_sched:
#         if (
#             opt.fwd_time + opt.bwd_time < sched.fwd_time + sched.bwd_time
#         ) and (opt.mem < sched.mem):
#             # should consider other factors like req_inputs
#             pareto = False
#         if (
#             opt.fwd_time + opt.bwd_time > sched.fwd_time + sched.bwd_time
#         ) and (opt.mem > sched.mem):
#             self.list_sched.remove(opt)
#     if pareto:
#         self.list_sched.append(sched)
#     # self.refine_scheds()

# def refine_scheds(self, expect_num=10):
#     def exclude_one():
#         def worse_sched(sched1, sched2):
#             if (
#                 sched1.fwd_time + sched1.bwd_time
#                 > sched2.fwd_time + sched2.bwd_time
#             ):
#                 return sched1
#             else:
#                 return sched2

#         # find the pair to remove one
#         diff = 100
#         for i, _sched1 in enumerate(self.list_sched[1:]):
#             for _sched2 in self.list_sched[i + 1 :]:

#                 if (
#                     abs((_sched2.mem - _sched1.mem + 1) / (_sched2.mem + 1))
#                     < diff
#                 ):
#                     sched1 = _sched1
#                     sched2 = _sched2
#                     diff = abs(
#                         (_sched2.mem - _sched1.mem + 1) / (_sched2.mem + 1)
#                     )
#         # assert sched1 in self.list_sched
#         # assert sched2 in self.list_sched
#         self.list_sched.remove(worse_sched(sched1, sched2))

#     while len(self.list_sched) > expect_num:
#         # print(len(self.list_sched))
#         exclude_one()


# class Solver_HILP(Solver):
#     # take_h_cluster = False # otherwise take h_graph
#     class Config():
#         sub_solvers : list[Solver] = None
#         nb_diff_budget = 10
#         def __init__(self, list_solvers, nb_diff_budget = 10):
#             self.sub_solvers = list_solvers
#         def get_budgets_to_solve_on(self,cluster,max_budget = None):
#             # return the list of budgets
#             pass

#     def __init__(list_solvers):


#     def __call__(self, H_obj, recursive=True):
#         if self.stop_solving:
#             return []
#         if isinstance()

#         for sub_clusters: sub_c.solve()
