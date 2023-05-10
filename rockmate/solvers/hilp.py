import time
import torch
import rockmate.rkgb as rkgb
import numpy as np
from rockmate.rkgb.Htools import H_cluster, H_graph, H_C_node

from .main import Solver, get_cluster_budget
from .HILP_gurobi import ModelGurobi
from .rotor_solver import seq_builder, solve_dp_functional
from .op_schedule import OpSchedule


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
            nb_total_nodes=20,
        ):
            self.mem_unit = mem_unit
            self.gurobi_params = gurobi_params
            self.protected_names = protected_names
            self.nb_total_sched = nb_total_sched
            self.nb_total_nodes = nb_total_nodes

    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    def can_solve(self, hg: H_graph):
        return len(hg.list_hcn) // 2 < self.config.nb_total_nodes

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
                        [
                            min(abs(x - y) for y in sel_mem)
                            for x in indices[:, 1]
                        ]
                    )
                    sel_mem.append(indices[argmax_diff][1])
                    sel_sched.append(list_sched[argmax_diff])
                    indices[argmax_diff][1] = 0
                hcn.list_sched = sel_sched
                # hcn.list_sched = list_sched[:nb_sched]
            else:
                hcn.list_sched = []

    def solve(self, cluster: H_cluster, budgets=None, accurate_mem=False):
        self.config.protected_names.extend(
            [kdn.name for kdn in cluster.interfaces["outputs_kdn_data"]]
        )
        list_op_sched = []
        if budgets is None:
            self.budgets = get_cluster_budget(
                cluster.representee_cluster, with_save_budget=True
            )
        else:
            self.budgets = budgets

        for budget in self.budgets:
            if not hasattr(budget, "__iter__"):
                budget = [budget]
            for hg in cluster.representee_cluster.possible_hg:
                # if isinstance(budget, )
                list_op_sched.extend(
                    self.solve_hg(
                        hg,
                        *budget,
                        accurate_mem=accurate_mem,
                    )
                )
        return list_op_sched

    def solve_hg(
        self,
        hg: H_graph,
        peak_budget,
        save_budget=None,
        accurate_mem=False,
        print_result=False,
    ):
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
        self.md = ModelGurobi(
            hg,
            peak_budget=peak_budget,
            save_budget=max(save_budget),
            gurobi_params=self.config.gurobi_params,
            accurate_mem=accurate_mem,
            protected_names=self.config.protected_names,
        )
        # print(f"model building: {time.time()-start}")
        sols = set()
        for sv_budget in np.sort(save_budget)[::-1]:
            self.md.add_abar_constraint(sv_budget)
            self.md.solve()
            # if not self.md.feasible:
            # if print_result:
            # print("Not feasible solution")
            # return []
            if self.md.feasible:
                if print_result:
                    # if True:
                    # print(
                    #     f"Solution with obj: {self.md.md.getObjective().getValue()}"
                    # )
                    print(
                        f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {self.md.solve_time:03f}s"
                    )
                loss_idx = self.md.loss_idx
                time_mem = (
                    self.md.md.getObjective().getValue(),  # time
                    self.md.U[(loss_idx, loss_idx)].getValue(),  # save_mem
                )
                if not time_mem in sols:
                    # start = time.time()

                    sols.add(time_mem)
                    self.op_sched = self.md.schedule()
                    list_op_sched.append(self.op_sched)
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
    #     self.md = ModelGurobi(
    #         self.hg,
    #         mem_limit,
    #         mem_limit,
    #         gurobi_params=self.config.gurobi_params,
    #         accurate_mem=True,
    #         protected_names=[
    #             kg.output_kdn_data.name
    #         ],  # output data is protected
    #     )
    #     self.md.solve()
    #     if not self.md.feasible:
    #         print("Not feasible solution")
    #         return OpSchedule([])
    #     else:
    #         print(f"Solution with obj: {self.md.md.getObjective().getValue()}")
    #     self.op_sched = self.md.schedule_()
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
