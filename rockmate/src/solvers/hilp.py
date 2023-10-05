import time
import torch
import rkgb
import numpy as np
from rkgb.Htools import H_cluster, H_graph, H_C_node

from .main import (
    Solver,
    get_cluster_budget,
    get_hgraph_budget_lb,
    get_hgraph_budget_ub,
)
from .HILP_gurobi import ModelGurobi
from .rotor_solver import seq_builder, solve_dp_functional
from .op_schedule import OpSchedule


default_time_limit = [60 * 60]


class HILP(Solver):
    class Config:
        def __init__(
            self,
            mem_unit=1024**2,
            gurobi_params={
                "LogToConsole": 0,
                "IntegralityFocus": 1,
            },
            protected_names=["sources data", "sources grad"],
            nb_total_sched=100,
            nb_total_nodes_top_level=30,
            nb_total_nodes=20,
            nb_bdg_save=6,
            nb_bdg_peak=4,
            time_limit=None,
        ):
            self.mem_unit = mem_unit
            self.gurobi_params = gurobi_params
            self.protected_names = protected_names
            self.nb_total_sched = nb_total_sched
            self.nb_total_nodes_top_level = nb_total_nodes_top_level
            self.nb_total_nodes = nb_total_nodes
            self.nb_bdg_save = nb_bdg_save
            self.nb_bdg_peak = nb_bdg_peak
            self.solve_top_level = False
            self.time_limit_ = time_limit

        @property
        def time_limit(self):
            if self.time_limit_ is None:
                return default_time_limit[0]
            else:
                return self.time_limit_

    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    # def __repr__(self):
    #     return f"HILP solver"

    def can_solve(self, hg: H_graph):
        if self.config.solve_top_level:
            limit = self.config.nb_total_nodes_top_level
        else:
            limit = self.config.nb_total_nodes
        return len(hg.list_hcn) // 2 <= limit

    def get_budget_list(self, hgraph: H_graph):
        min_bdg = get_hgraph_budget_lb(hgraph)
        max_bdg = get_hgraph_budget_ub(hgraph)
        interfaces_mem = sum(kdn.mem for kdn in hgraph.cluster.all_interfaces)

        budgets = []
        l_bd_peak = (
            np.linspace(min_bdg, max_bdg, self.config.nb_bdg_peak) + interfaces_mem
        )
        for bd_peak in l_bd_peak:
            l_bd_save = (
                np.linspace(
                    0,
                    # min(bd_peak, autograd_sched.mem),
                    bd_peak,
                    self.config.nb_bdg_save,
                )
                + interfaces_mem
            )

            budgets.append((bd_peak, l_bd_save))
        return budgets

    def _select_sched(self, hg, overall_budget=None):
        # for fwd hcn, select sched from hcn.sub_cluster and put in hcn.list_sched
        weights = []
        overall_budget = overall_budget or np.inf
        for hcn in hg.list_hcn:
            if hcn.is_fwd:
                if hcn.sub_cluster is None:
                    weights.append(0)
                else:
                    weights.append(len(hcn.sub_cluster.list_kcn))

        for hcn, w in zip(hg.list_hcn, weights):
            nb_sched = max(
                self.config.nb_total_sched * w // sum(weights), 1
            )  # at least 1 sched
            if hcn.sub_cluster is not None:
                list_sched = hcn.sub_cluster.get_sched(pareto=True)
                list_sched = [
                    op_sched
                    for op_sched in list_sched
                    if op_sched.mem <= overall_budget
                ]
                if nb_sched >= len(list_sched):
                    hcn.list_sched = list_sched
                    continue
                indices = np.array(
                    [(i, op_sched.mem) for i, op_sched in enumerate(list_sched)]
                )
                sel_sched = [list_sched[0]]
                sel_mem = [list_sched[0].mem]

                while len(sel_sched) < nb_sched:
                    # add the one with most different .mem with all selected sched
                    argmax_diff = np.argmax(
                        [min(abs(x - y) for y in sel_mem) for x in indices[:, 1]]
                    )
                    sel_mem.append(indices[argmax_diff][1])
                    sel_sched.append(list_sched[argmax_diff])
                    indices[argmax_diff][1] = 0
                hcn.list_sched = sel_sched
                # hcn.list_sched = list_sched[:nb_sched]
            else:
                hcn.list_sched = []

    def solve(
        self, cluster: H_cluster, budgets=None, accurate_mem=False, gc_collect=True
    ):
        list_op_sched = []

        for hg in cluster.representee_cluster.possible_hg:
            if budgets is None:
                # self.budgets = get_cluster_budget(
                #     cluster.representee_cluster, with_save_budget=True
                # )
                self.budgets = self.get_budget_list(hg)
            else:
                self.budgets = budgets

            for budget in self.budgets:
                if not hasattr(budget, "__iter__"):
                    budget = [budget]
                list_op_sched.extend(
                    self.solve_hg(
                        hg,
                        *budget,
                        accurate_mem=accurate_mem,
                    )
                )
        if gc_collect:
            gc.collect()
        return list_op_sched

    def solve_hg(
        self,
        hg: H_graph,
        peak_budget,
        save_budget=None,
        accurate_mem=False,
        print_result=False,
    ):
        gc.collect()
        if not self.can_solve(hg):
            return []
        if save_budget is not None:
            save_budget = save_budget
        else:
            save_budget = peak_budget

        list_op_sched = []
        self._select_sched(hg, overall_budget=peak_budget)
        if not hasattr(save_budget, "__iter__"):
            save_budget = [save_budget]
        # start = time.time()
        gurobi_params = self.config.gurobi_params
        gurobi_params["TimeLimit"] = self.config.time_limit
        # md = ModelGurobi(
        #     hg,
        #     peak_budget=peak_budget,
        #     save_budget=max(save_budget),
        #     gurobi_params=gurobi_params,
        #     accurate_mem=accurate_mem,
        #     protected_names=self.config.protected_names,
        # )
        # # print(f"model building: {time.time()-start}")
        # sols = set()
        # for sv_budget in np.sort(save_budget)[::-1]:
        #     md.add_abar_constraint(sv_budget)
        #     md.solve()
        #     # if not md.feasible:
        #     # if print_result:
        #     # print("Not feasible solution")
        #     # return []
        #     if md.feasible:
        #         if print_result:
        #             # if True:
        #             # print(
        #             #     f"Solution with obj: {md.md.getObjective().getValue()}"
        #             # )
        #             print(
        #                 f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {md.solve_time:03f}s"
        #             )
        #         loss_idx = md.loss_idx
        #         time_mem = (
        #             md.md.getObjective().getValue(),  # time
        #             md.U[(loss_idx, loss_idx)].getValue(),  # save_mem
        #         )
        #         if not time_mem in sols:
        #             # start = time.time()

        #             sols.add(time_mem)
        #             self.op_sched = md.schedule()
        #             if md.md.status == 2:
        #                 status = "opt"
        #             elif md.md.status == 9:
        #                 status = "early_stp"
        #             else:
        #                 status = md.md.status

        #             self.op_sched.solver = f"HILP_{status}"
        #             list_op_sched.append(self.op_sched)
        #             # print(f"scheduling: {time.time()-start}")
        #     else:  # if infeasible, no need to try smaller budget
        #         return list_op_sched
        # del md
        list_op_sched = solve_ilp(
            hg,
            peak_budget,
            save_budget,
            gurobi_params,
            accurate_mem,
            self.config.protected_names,
        )

        return list_op_sched


def solve_ilp(
    hg, peak_budget, save_budget, gurobi_params, accurate_mem, protected_names
):
    list_op_sched = []
    md = ModelGurobi(
        hg,
        peak_budget=peak_budget,
        save_budget=max(save_budget),
        gurobi_params=gurobi_params,
        accurate_mem=accurate_mem,
        protected_names=protected_names,
    )
    # print(f"model building: {time.time()-start}")
    sols = set()
    for sv_budget in np.sort(save_budget)[::-1]:
        md.add_abar_constraint(sv_budget)
        md.solve()
        # if not md.feasible:
        # if print_result:
        # print("Not feasible solution")
        # return []
        if md.feasible:
            if True:  # print_result:
                # if True:
                # print(
                #     f"Solution with obj: {md.md.getObjective().getValue()}"
                # )
                print(
                    f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {md.solve_time:03f}s"
                )
            loss_idx = md.loss_idx
            time_mem = (
                md.md.getObjective().getValue(),  # time
                md.U[(loss_idx, loss_idx)].getValue(),  # save_mem
            )
            if not time_mem in sols:
                # start = time.time()

                sols.add(time_mem)
                op_sched = md.schedule()
                if md.md.status == 2:
                    status = "opt"
                elif md.md.status == 9:
                    status = "early_stp"
                else:
                    status = md.md.status

                op_sched.solver = f"HILP_{status}"
                list_op_sched.append(op_sched)
                # print(f"scheduling: {time.time()-start}")
        else:  # if infeasible, no need to try smaller budget
            return list_op_sched
    return list_op_sched


# def solve(
#     self,
#     rkgb_res,
#     mem_limit,
#     recursive=True,
#     print_info=False,
#     protect_names=["sources data", "sources grad"],
#     return_hg=False,
# ):
#     if isinstance(rkgb_res, rkgb.Htools.H_graph):
#         return self.solve_hg(
#             rkgb_res,
#             mem_limit,
#             mem_limit,
#             print_info=print_info,
#             protect_names=protect_names,
#         )
#     self.mem_limit = mem_limit
#     #  -- build Hgraph --

#     kg = rkgb_res.K_graph
#     sg = rkgb_res.S_graph
#     if recursive:
#         ps = rkgb.Ptools.S_to_P(sg, None)  # TO TODO None=model
#         self.hg = rkgb.Htools.P_and_K_to_H(ps, kg)
#         print(f"Size of Hgraph {len(self.hg.list_hcn)}")
#         solve_hg_recursive(self.hg, solve_self=False, print_info=print_info)
#         print("Low level finished")
#     if return_hg:
#         return self.hg
#     md = ModelGurobi(
#         self.hg,
#         mem_limit,
#         mem_limit,
#         gurobi_params=self.config.gurobi_params,
#         accurate_mem=True,
#         protected_names=[
#             kg.output_kdn_data.name
#         ],  # output data is protected
#     )
#     md.solve()
#     if not md.feasible:
#         print("Not feasible solution")
#         return OpSchedule([])
#     else:
#         print(f"Solution with obj: {md.md.getObjective().getValue()}")
#     self.op_sched = md.schedule_()
#     for op in self.op_sched.op_list:
#         if op.name in protect_names:
#             op.disabled = True
#     return self.op_sched

# def solve_hg(
#     self,
#     hg: rkgb.Htools.H_graph,
#     save_budget,
#     peak_budget,
#     print_info=False,
#     protect_names=["sources data", "sources grad"],
#     gurobi_params=None,
#     accurate_mem=False,
# ):
#     gurobi_params = gurobi_params or self.config.gurobi_params
#     md = ModelGurobi(
#         hg,
#         save_budget,
#         peak_budget,
#         gurobi_params=gurobi_params,
#         accurate_mem=accurate_mem,
#     )
#     md.solve()
#     if md.feasible:
#         op_sched = md.schedule_()
#         for op in op_sched.op_list:
#             if op.name in protect_names:
#                 op.disabled = True
#         if print_info:
#             print(
#                 f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {md.solve_time:03f}s"
#             )
#         return op_sched
