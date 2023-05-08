import time
import torch
import rkgb
from rkgb.utils import print_debug, np, irotor
from rkgb.utils.global_vars import ref_verbose, solver_name
from rkgb.utils.small_fcts import get_device
from rkgb.utils.ast_add_on import ast_to_str
from rkgb.Htools import H_cluster, H_graph, H_C_node

from solvers.main import Solver, get_cluster_budget
from solvers.HILP_gurobi import ModelGurobi
from solvers.rotor_solver import seq_builder, solve_dp_functional
from solvers.op_schedule import OpSchedule


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
        ):
            self.mem_unit = mem_unit
            self.gurobi_params = gurobi_params
            self.protected_names = protected_names
            self.nb_total_sched = nb_total_sched

    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    def _select_sched(self, hg, overall_budget=None):
        # for fwd hcn, select sched from hcn.sub_cluster and put in hcn.list_sched
        weights = []
        for hcn in hg.list_hcn:
            if hcn.is_fwd:
                if hcn.sub_cluster is None:
                    weights.append(0)
                else:
                    weights.append(len(hcn.sub_cluster.list_kcn))

        for hcn, w in zip(hg.list_hcn, weights):
            nb_sched = self.config.nb_total_sched * w // sum(weights)
            if hcn.sub_cluster is not None:
                list_sched = hcn.sub_cluster.get_sched(pareto=True)
                list_sched = [
                    op_sched
                    for op_sched in list_sched
                    if op_sched.mem <= overall_budget
                ]
                hcn.list_sched = list_sched[:nb_sched]
            else:
                hcn.list_sched = []

    def solve(self, cluster: H_cluster, budgets=None, accurate_mem=False):
        self.config.protected_names.extend(
            [kdn.name for kdn in cluster.interfaces["outputs_kdn_data"]]
        )
        list_op_sched = []
        if budgets is None:
            self.budgets = get_cluster_budget(cluster.representee_cluster)
        else:
            self.budgets = budgets

        for budget in self.budgets:
            if not hasattr(budget, "__iter__"):
                budget = [budget]
            # for hg in
            # if isinstance(budget, )
            list_op_sched.extend(
                self.solve_hg(
                    cluster.representee_cluster.possible_hg[0],
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
