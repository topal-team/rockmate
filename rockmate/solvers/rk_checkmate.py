import time
import torch
from rockmate import rkgb
import os
from rockmate.rkgb.utils import print_debug, np, irotor
from rockmate.rkgb.utils.global_vars import ref_verbose, solver_name
from rockmate.rkgb.utils.small_fcts import get_device
from rockmate.rkgb.utils.ast_add_on import ast_to_str
from rockmate.rkgb.Htools import H_cluster, H_graph, H_C_node
from rockmate.rkgb.Ktools import K_graph

from .main import Solver, get_cluster_budget
from .ILP_gurobi import ModelGurobi
from .op_schedule import OpSchedule


class RK_checkmate(Solver):
    class Config:
        def __init__(
            self,
            mem_unit=1024**2,
            gurobi_params={
                "LogToConsole": 1,
                "IntegralityFocus": 1,
                "TimeLimit": 4 * 60,
                "Threads": os.cpu_count(),
                "OptimalityTol": 1e-4,
                "IntFeasTol": 1e-5,
            },
            protected_names=["sources data", "sources grad"],
            nb_total_nodes=100,
        ):
            self.mem_unit = mem_unit
            self.gurobi_params = gurobi_params
            self.protected_names = protected_names
            self.nb_total_nodes = nb_total_nodes

    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    def can_solve(self, kg: K_graph):
        return len(kg.list_kcn) // 2 < self.config.nb_total_nodes

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
            list_op_sched.extend(
                self.solve_kg(
                    cluster,
                    *budget,
                    accurate_mem=accurate_mem,
                )
            )
        return list_op_sched

    def solve_kg(
        self,
        kg,
        peak_budget,
        save_budget=None,
        print_result=False,
    ):
        if not self.can_solve(kg):
            return []
        if save_budget is not None:
            save_budget = save_budget
        else:
            save_budget = peak_budget

        list_op_sched = []
        if not hasattr(save_budget, "__iter__"):
            save_budget = [save_budget]
        # start = time.time()
        self.md = ModelGurobi(
            kg,
            peak_budget=peak_budget,
            save_budget=max(save_budget),
            gurobi_params=self.config.gurobi_params,
            # protected_names=self.config.protected_names,
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
                        f"Solve Kgraph {kg.name} with {len(kg.list_hcn)} nodes takes {self.md.solve_time:03f}s"
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
