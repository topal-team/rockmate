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
        ):
            self.mem_unit = mem_unit
            self.gurobi_params = gurobi_params
            self.protected_names = protected_names

    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    def _select_sched(self, hg):
        nb_sched = 200 // len(hg.list_hcn)
        for hcn in hg.list_hcn:
            if hcn.sub_cluster is not None:
                hcn.list_sched = hcn.sub_cluster.get_sched().copy()[:nb_sched]
            else:
                hcn.list_sched = []

    def solve(self, cluster: H_cluster, budgets=None):
        if budgets is None:
            self.budgets = get_cluster_budget(cluster.representee_cluster)
        else:
            self.budgets = budgets
        self.config.protected_names.extend(
            [kdn.name for kdn in cluster.interfaces["outputs_kdn_data"]]
        )
        return self.solve_hg(
            cluster.representee_cluster.possible_hg[0],
            self.budgets[0],
            self.budgets[0],
        )

    def solve_hg(
        self,
        hg: H_graph,
        peak_budget,
        save_budget,
        accurate_mem=False,
        print_result=False,
    ):
        self._select_sched(hg)
        self.md = ModelGurobi(
            hg,
            peak_budget=peak_budget,
            save_budget=save_budget,
            gurobi_params=self.config.gurobi_params,
            accurate_mem=accurate_mem,
            protected_names=self.config.protected_names,
        )
        self.md.solve()
        if not self.md.feasible:
            # if print_result:
            print("Not feasible solution")
            return None
        else:
            if print_result:
                print(
                    f"Solution with obj: {self.md.md.getObjective().getValue()}"
                )
            self.op_sched = self.md.schedule()
            return self.op_sched

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
