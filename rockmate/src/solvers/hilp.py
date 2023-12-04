import time
import torch
import rkgb
import numpy as np
from rkgb.Htools import H_cluster, H_graph, H_C_node
import gc
from .main import (
    Solver,
    get_cluster_budget,
    get_hgraph_budget_lb,
    get_hgraph_budget_ub,
)
import pulp
from .HILP_gurobi import ModelGurobi
# from .HILP_pulp import ModelPULP
from .HILP_pulp_ofl_para import ModelPULP
from .rotor_solver import seq_builder, solve_dp_functional
from .op_schedule import OpSchedule
from rkgb.utils.global_vars import solver_name
import psutil


default_time_limit = [60 * 60]


class HILP(Solver):
    class Config:
        def __init__(
            self,
            mem_unit=1024**2,
            ilp_solver_params={
                "LogToConsole": 0,
                "IntegralityFocus": 1,
                "NodeFileStart":0.5,
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
            self.ilp_solver_params = ilp_solver_params
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

    def __init__(self, config=None, ilp_solver=None):
        super().__init__(config)
        self.ilp_solver = ilp_solver or solver_name[0]
        if self.ilp_solver == "gurobi":
            self.model_ilp = ModelGurobi
            print("Using GUROBI to solve ILP")
        else:
            self.model_ilp = ModelPULP
            try:
                solver = pulp.get_solver(self.ilp_solver, msg=0)
            except:
                avail_solver = pulp.listSolvers(onlyAvailable=True)[0]
                print(f"Cannot get {ilp_solver}, will use {avail_solver}")
                self.ilp_solver = avail_solver
            print(f"Using {self.ilp_solver} to solve ILP")

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
            l_bd_save = np.linspace(
                interfaces_mem,
                # min(bd_peak, autograd_sched.mem),
                bd_peak,
                self.config.nb_bdg_save,
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
            # mem = psutil.virtual_memory()
            # print(
            #         f"The CPU mem usage before solving {cluster.name} is: ",
            #         # psutil.cpu_percent(4)
            #         mem.used,
            #     )
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
        ilp_solver_params = self.config.ilp_solver_params
        ilp_solver_params["TimeLimit"] = self.config.time_limit

        def solve_md(ilp_solver_params=ilp_solver_params, 
                     model_ilp = self.model_ilp,
                     protected_names=self.config.protected_names,
                     ilp_solver = self.ilp_solver):
            
            md = model_ilp(
                hg,
                peak_budget=peak_budget,
                save_budget=max(save_budget),
                ilp_solver_params=ilp_solver_params,
                accurate_mem=accurate_mem,
                protected_names=protected_names,
            )
            # print(f"model building: {time.time()-start}")
            sols = set()
            for sv_budget in np.sort(save_budget)[::-1]:
                md.add_abar_constraint(sv_budget)
                md.solve(ilp_solver)
                # if not md.feasible:
                # if print_result:
                # print("Not feasible solution")
                # return []
                if md.feasible == 1:
                    if print_result:
                        # if True:
                        # print(
                        #     f"Solution with obj: {md.md.getObjective().getValue()}"
                        # )
                        print(
                            f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {md.solve_time:03f}s, used {psutil.virtual_memory().used}B memory"
                        )
                    loss_idx = md.loss_idx
                    if solver_name[0] == "gurobi":
                        time_mem = (
                            md.md.getObjective().getValue(),  # time
                            md.U[(loss_idx, loss_idx)].getValue(),  # save_mem
                        )
                    else:
                        time_mem = (
                            md.md.objective.value(),
                            md.U[(loss_idx, loss_idx)].value(),  # save_mem
                        )
                    if not time_mem in sols:
                        # start = time.time()

                        sols.add(time_mem)
                        op_sched = md.schedule()
                        # if md.md.status==2:
                        #     status = "opt"
                        # elif md.md.status==9:
                        #     status = "early_stp"
                        # else:
                        #     status = md.md.status

                        op_sched.solver = f"HILP_{md.status}"
                        list_op_sched.append(op_sched)
                        # print(f"scheduling: {time.time()-start}")
                else:  # if infeasible, no need to try smaller budget
                    return list_op_sched, md
            return list_op_sched, md if accurate_mem else None
        
        list_op_sched, md = solve_md(ilp_solver_params=ilp_solver_params, 
                     model_ilp = self.model_ilp,
                     protected_names=self.config.protected_names,
                     ilp_solver = self.ilp_solver)
        self.md = md
        return list_op_sched
