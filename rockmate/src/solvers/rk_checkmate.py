import time
import torch
import rkgb
import os
from rkgb.utils import print_debug, np, irotor
from rkgb.utils.global_vars import ref_verbose, solver_name
from rkgb.utils.small_fcts import get_device
from rkgb.utils.ast_add_on import ast_to_str
from rkgb.Htools import H_cluster, H_graph, H_C_node
from rkgb.Ktools import K_graph

from .main import Solver, get_cluster_budget
from .HILP_gurobi import ModelGurobi
from .op_schedule import OpSchedule


class RK_checkmate(Solver):
    class Config:
        def __init__(
            self,
            mem_unit=1024**2,
            gurobi_params={
                "LogToConsole": 0,
                "IntegralityFocus": 1,
                "Threads": os.cpu_count(),
                "OptimalityTol": 1e-4,
                "IntFeasTol": 1e-5,
            },
            protected_names=["sources data", "sources grad"],
            nb_total_nodes=100,
        ):
            self.mem_unit = mem_unit
            self.protected_names = protected_names
            self.nb_total_nodes = nb_total_nodes
            self.time_limit = 20 * 60
            self.gurobi_params = gurobi_params

    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)

    def can_solve(self, hg: H_graph, cluster: H_cluster):
        return len(hg.list_hcn) == len(cluster.list_kcn)

    def _select_sched(self, hg):
        # for fwd hcn, select sched from hcn.sub_cluster and put in hcn.list_sched
        for hcn in hg.list_hcn:
            if hcn.sub_cluster is not None:
                hcn.list_sched = hcn.sub_cluster.get_sched()

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
            for hg in cluster.possible_hg:
                if self.can_solve(hg, cluster):
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
        if save_budget is not None:
            save_budget = save_budget
        else:
            save_budget = peak_budget

        list_op_sched = []
        self._select_sched(hg)
        if not hasattr(save_budget, "__iter__"):
            save_budget = [save_budget]
        # start = time.time()
        gurobi_params = self.config.gurobi_params
        gurobi_params["TimeLimit"] = self.config.time_limit
        self.md = ModelGurobi(
            hg,
            peak_budget=peak_budget,
            save_budget=max(save_budget),
            gurobi_params=gurobi_params,
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
                    self.op_sched.solver = "HILP"
                    list_op_sched.append(self.op_sched)
                    # print(f"scheduling: {time.time()-start}")
            else:  # if infeasible, no need to try smaller budget
                return list_op_sched
        return list_op_sched
