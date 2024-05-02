import time
import torch
import rkgb
import numpy as np
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
import gc
from ..main import (
    Solver,
    get_cluster_budget,
    get_hgraph_budget_lb,
    get_hgraph_budget_ub,
)
import pulp
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from rkgb.lowlevel.constants import init_target_string
from .ilp_model import ModelPULP
from .ilp_offload import ModelPULPOffload
from .ilp_schedule import schedule
from .ilp_utils import (
    set_hcn_list_sched, 
    set_hg_parameter_groups,
    clean_hcn_list_sched, 
    clean_hg_parameter_groups
)
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
            protected_names=[f"{init_target_string} data", f"{init_target_string} grad"],
            nb_total_sched=100,
            nb_total_nodes_top_level=100,
            nb_total_nodes=20,
            nb_bdg_save=6,
            nb_bdg_peak=4,
            time_limit=None,
            offload=False,
            activation_offload=False,
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
            self.time_limit_top = time_limit
            self.optimize_metrics = {}
            self.offload = offload
            self.activation_offload = activation_offload

        @property
        def time_limit(self):
            if self.time_limit_ is None:
                return default_time_limit[0]
            else:
                return self.time_limit_

    def __init__(self, config=None, ilp_solver=None):
        super().__init__(config)
        self.ilp_solver = ilp_solver# or solver_name[0]
        
        try:
            solver = pulp.get_solver(self.ilp_solver, msg=0)
        except:
            avail_solver = pulp.listSolvers(onlyAvailable=True)[0]
            print(f"Cannot get {ilp_solver}, will use {avail_solver}")
            self.ilp_solver = avail_solver
        print(f"Using {self.ilp_solver} to solve ILP")

    # def __repr__(self):
    #     return f"HILP solver"

    def can_solve(self, hg: HierarchicalGraph):
        if self.config.solve_top_level:
            limit = self.config.nb_total_nodes_top_level
        else:
            limit = self.config.nb_total_nodes
        return len(hg.list_HCNs) // 2 <= limit

    def get_budget_list(self, hgraph: HierarchicalGraph):
        min_bdg = get_hgraph_budget_lb(hgraph)
        max_bdg = get_hgraph_budget_ub(hgraph)
        # interfaces_mem = sum(kdn.mem for kdn in hgraph.cluster.all_interfaces)
        interfaces_mem = sum(kdn.mem for kdn in hgraph.cluster.interfaces["input_data_anodes"])
        interfaces_mem += sum(kdn.mem for kdn in hgraph.cluster.interfaces["output_data_anodes"])

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

    def _group_parameters(self, hg: HierarchicalGraph):
        all_params = {}
        param2hcn = {}

        for i,hcn in enumerate(hg.list_HCNs):
            if not hasattr(hcn, "required_parameter_nodes_real"):continue
            if hcn.sub_cluster is not None and hasattr(hcn.sub_cluster, "parameter_nodes"):
                # if FWD/BWD hcns have different req_pnodes, parameters may be needed for recomputation
                req_pnodes = hcn.sub_cluster.parameter_nodes
            else:
                h_pnodes = hcn.required_parameter_nodes_real|hcn.required_parameter_nodes_fake
                req_pnodes = [pnode.original_param_node for pnode in h_pnodes]
            for pnode in req_pnodes:
                if pnode.is_buffer:continue
                if pnode.mem < self.config.optimize_metrics["minor_offload_size"]:continue
                if pnode not in param2hcn:
                    param2hcn[pnode] = {i}
                else:
                    param2hcn[pnode].add(i)

                # # sub_c2params[sub_cluster.name].add(pnode.param_name)
                # if hasattr(pnode, "original_param_node"):
                #     all_params[pnode.param_name] = pnode.original_param_node
                # else:
                #     all_params[pnode.param_name] = pnode
                # if pnode.param_name not in param2hcn:
                #     param2hcn[pnode.param_name] = {i}
                # else:
                #     param2hcn[pnode.param_name].add(i)
                    
        parameter_groups = {}
        for p, c in param2hcn.items():
            c_ = tuple(sorted(c))
            if c_ not in parameter_groups:
                parameter_groups[c_] = {p}
            else:
                parameter_groups[c_].add(p)

        set_hg_parameter_groups(hg, parameter_groups)

        # parameters = []
        # param_group2hcn = {}
        # for hcn, v in result.items():
        #     param_group2hcn[len(parameters)] = 
        #     parameters.append([all_params[p] for p in v])
        # return param_group2hcn, parameters

    def _select_sched(self, hg, overall_budget=None):
        # for fwd hcn, select sched from hcn.sub_cluster and put in hcn.list_sched
        weights = []
        overall_budget = overall_budget or np.inf
        for hcn in hg.list_HCNs:
            if hcn.is_fwd:
                if hcn.sub_cluster is None:
                    weights.append(0)
                else:
                    weights.append(len(hcn.sub_cluster.list_cnodes))

        for i, (hcn, w) in enumerate(zip(hg.list_HCNs, weights)):
            # if i<len(hg.list_HCNs)//2-2 and hcn.sub_cluster is not None:
            #     hcn.list_sched = [min(hcn.sub_cluster.representee_cluster.list_schedules,
            #                key=lambda x: x.mem)]
            #     continue
            nb_sched = max(
                self.config.nb_total_sched * w // sum(weights), 1
            )  # at least 1 sched
            nb_sched = 3 if i<30 else nb_sched
            if hcn.sub_cluster is not None:
                # list_sched = hcn.sub_cluster.get_sched(pareto=True)
                list_sched = hcn.sub_cluster.representee_cluster.list_schedules
                list_sched = [
                    op_sched
                    for op_sched in list_sched
                    if op_sched.mem <= overall_budget
                ]
                if nb_sched >= len(list_sched):
                    # hcn.list_sched = list_sched
                    set_hcn_list_sched(hcn, list_sched)
                    continue
                indices = np.array(
                    [(i, op_sched.mem) for i, op_sched in enumerate(list_sched)]
                )
                sel_sched = [list_sched[0]]
                sel_mem = [list_sched[0].mem]

                while len(sel_sched) < nb_sched:
                    # add the one with most different .mem with all selected sched
                    if np.max(
                        [min(abs(x - y) for y in sel_mem) for x in indices[:, 1]]
                    ) == 0:#no different schedules:
                        break
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
        self, cluster: HierarchicalCluster, budgets=None, accurate_mem=False, gc_collect=True
    ):
        print(f"solving {cluster.name}")
        list_op_sched = []

        if self.config.solve_top_level and self.config.offload:
            self.model_ilp = ModelPULPOffload
        else:
            self.model_ilp = ModelPULP

        # for hg in cluster.representee_cluster.possible_hg:
        for hg in cluster.representee_cluster.partitionings:
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
        hg: HierarchicalGraph,
        peak_budget,
        save_budget=None,
        accurate_mem=False,
        print_result=False,
    ):
        # print(accurate_mem)
        gc.collect()
        if not self.can_solve(hg):
            return []
        if save_budget is not None:
            save_budget = save_budget
        else:
            save_budget = peak_budget

        list_op_sched = []
        self._select_sched(hg, overall_budget=peak_budget)
        self._group_parameters(hg)
        if not hasattr(save_budget, "__iter__"):
            save_budget = [save_budget]
        # start = time.time()
        ilp_solver_params = self.config.ilp_solver_params
        if accurate_mem: 
            ilp_solver_params["TimeLimit"] = self.config.time_limit_top
        else:
            ilp_solver_params["TimeLimit"] = self.config.time_limit

        def solve_md(ilp_solver_params=ilp_solver_params, 
                     model_ilp = self.model_ilp,
                     protected_names=self.config.protected_names,
                     accurate_mem=False,
                     ilp_solver = self.ilp_solver):
            
            md = model_ilp(
                hg,
                peak_budget=peak_budget,
                save_budget=max(save_budget),
                ilp_solver_params=ilp_solver_params,
                accurate_mem=accurate_mem,
                protected_names=protected_names,
                optimize_metrics = self.config.optimize_metrics,
                activation_offload=self.config.activation_offload
            )
            md.build()
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
                            f"Solve Hgraph {hg.name} with {len(hg.list_HCNs)} nodes takes {md.solve_time:03f}s, used {psutil.virtual_memory().used}B memory"
                        )
                    loss_idx = md.loss_idx
                    # if solver_name[0] == "gurobi":
                    #     time_mem = (
                    #         md.md.getObjective().getValue(),  # time
                    #         md.U[(loss_idx, loss_idx)].getValue(),  # save_mem
                    #     )
                    # else:
                    time_mem = (
                        md.md.objective.value(),
                        md.U[(loss_idx, loss_idx)].value(),  # save_mem
                    )
                    if not time_mem in sols:
                        # start = time.time()

                        sols.add(time_mem)
                        op_sched = schedule(md)
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
            # if accurate_mem:print(md.feasible)
            return list_op_sched, md if accurate_mem else None
        
        list_op_sched, md = solve_md(ilp_solver_params=ilp_solver_params, 
                     model_ilp = self.model_ilp,
                     accurate_mem=accurate_mem,
                     protected_names=self.config.protected_names,
                     ilp_solver = self.ilp_solver)
        self.md = md
        return list_op_sched
