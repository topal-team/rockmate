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
from ...op_schedule import *
from .ilp_model import ModelPULP
from .ilp_offload import ModelPULPOffload
from .ilp_schedule import schedule
from .ilp_utils import (
    set_hcn_list_sched,
    set_hg_parameter_groups,
    clean_hcn_list_sched,
    clean_hg_parameter_groups,
)
import psutil
from dataclasses import dataclass, field


class HILP(Solver):
    '''ILP-based rockmate solver. Generic implementation, can be used in rematerialization or offload mode

    The original idea for the ILP formulation is from Checkmate: https://github.com/parasj/checkmate
    described in

    Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Kurt Keutzer, Ion Stoica,
    Joseph E. Gonzalez. Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization. MLSys
    2020, https://arxiv.org/abs/1910.02653

    This version is heavily modified from this original implementation, adding a hierarchical approach
    and many features to support activation and parameter offloading.
    '''
    @dataclass
    class Config:
        '''Configuration options for HILP solver'''
        mem_unit: int = 1024**2
        '''How memory values (in bytes) are quantized to obtain reasonable integers'''
        ilp_solver_params: dict = field(default_factory=lambda:
                                        {
                                            "LogToConsole": 0,
                                            "IntegralityFocus": 1,
                                            "NodeFileStart": 0.5,
                                        })
        '''Arguments passed to the ILP solver directly'''
        ilp_solver: str = "PULP_CBC_CMD"
        '''Which solver to use in the PuLP library.'''
        protected_names: list = field(default_factory=list)
        nb_total_sched: int = 100
        '''Total number of options to consider when solving one level of ILP.'''
        nb_total_nodes: int = 20
        nb_bdg_save: int = 6
        nb_bdg_peak: int = 4
        time_limit: int = 1*60
        offload: bool = False ## Only to be used in top solver
        add_offload_sched: bool = False ## Only use in top solver, to automatically add offload versions of computed schedules
        activation_offload: bool = False
        model_kwargs: dict = field(default_factory=dict) ## Passed to the ModelILP class
        accurate_mem: bool = True ## If True, include correction terms
        minor_offload_size:int = 10*1024**2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ilp_solver = self.config.ilp_solver
        try:
            solver = pulp.get_solver(self.ilp_solver, msg=0)
        except:
            avail_solver = pulp.listSolvers(onlyAvailable=True)[0]
            print(f"Cannot get {self.ilp_solver}, will use {avail_solver}")
            self.ilp_solver = avail_solver
        print(f"Using {self.ilp_solver} to solve ILP")

    # def __repr__(self):
    #     return f"HILP solver"

    def can_solve(self, hg: HierarchicalGraph):
        return len(hg.list_HCNs) // 2 <= self.config.nb_total_nodes

    def get_budget_list(self, hgraph: HierarchicalGraph):
        min_bdg = get_hgraph_budget_lb(hgraph)
        max_bdg = get_hgraph_budget_ub(hgraph)
        # interfaces_mem = sum(kdn.mem for kdn in hgraph.cluster.all_interfaces)
        interfaces_mem = sum(
            kdn.mem for kdn in hgraph.cluster.interfaces["input_data_anodes"]
        )
        interfaces_mem += sum(
            kdn.mem for kdn in hgraph.cluster.interfaces["output_data_anodes"]
        )

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

    def _group_parameters(self, hg: HierarchicalGraph, minor_size=10*1024**2):
        all_params = {}
        param2hcn = {}

        for i, hcn in enumerate(hg.list_HCNs):
            if not hasattr(hcn, "required_parameter_nodes_real"):
                continue
            if hcn.sub_cluster is not None and hasattr(
                hcn.sub_cluster, "parameter_nodes"
            ):
                # if FWD/BWD hcns have different req_pnodes, parameters may be needed for recomputation
                req_pnodes = hcn.sub_cluster.parameter_nodes
            else:
                h_pnodes = (
                    hcn.required_parameter_nodes_real
                    | hcn.required_parameter_nodes_fake
                )
                req_pnodes = [pnode.original_param_node for pnode in h_pnodes]
            for pnode in req_pnodes:
                if pnode.is_buffer:
                    continue
                if pnode.mem<minor_size:
                    continue
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
            # nb_sched = 3 if i < 30 else nb_sched
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
                    if self.config.add_offload_sched:
                        self.add_offload_sched_hcn(hcn)
                    continue
                indices = np.array(
                    [(i, op_sched.mem) for i, op_sched in enumerate(list_sched)]
                )
                sel_sched = [list_sched[0]]
                sel_mem = [list_sched[0].mem]

                while len(sel_sched) < nb_sched:
                    # add the one with most different .mem with all selected sched
                    if (
                        np.max(
                            [min(abs(x - y) for y in sel_mem) for x in indices[:, 1]]
                        )
                        == 0
                    ):  # no different schedules:
                        break
                    argmax_diff = np.argmax(
                        [min(abs(x - y) for y in sel_mem) for x in indices[:, 1]]
                    )
                    sel_mem.append(indices[argmax_diff][1])
                    sel_sched.append(list_sched[argmax_diff])
                    indices[argmax_diff][1] = 0
                hcn.list_sched = sel_sched
                if self.config.add_offload_sched:
                    self.add_offload_sched_hcn(hcn)
                # hcn.list_sched = list_sched[:nb_sched]
            else:
                hcn.list_sched = []
            


    def solve(
        self,
        cluster: HierarchicalCluster,
        budgets=None,
        gc_collect=True,
    ):
        print(f"solving {cluster.name} with budgets {budgets}")
        list_op_sched = []

        if self.config.offload:
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
        
        if not hasattr(save_budget, "__iter__"):
            save_budget = [save_budget]
        # start = time.time()
        if self.config.offload:
            self._group_parameters(hg, minor_size=self.config.minor_offload_size)
        ilp_solver_params = dict(self.config.ilp_solver_params)
        ilp_solver_params["TimeLimit"] = self.config.time_limit

        def solve_md(
            ilp_solver_params=ilp_solver_params,
            model_ilp=self.model_ilp,
            protected_names=self.config.protected_names,
            accurate_mem=False,
            ilp_solver=self.ilp_solver,
        ):

            md = model_ilp(
                hg,
                peak_budget=peak_budget,
                save_budget=max(save_budget),
                ilp_solver_params=ilp_solver_params,
                accurate_mem=accurate_mem,
                protected_names=protected_names,
                activation_offload=self.config.activation_offload,
                minor_offload_size = self.config.minor_offload_size,
                **self.config.model_kwargs
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

        list_op_sched, md = solve_md(
            ilp_solver_params=ilp_solver_params,
            model_ilp=self.model_ilp,
            accurate_mem=self.config.accurate_mem,
            protected_names=self.config.protected_names,
            ilp_solver=self.ilp_solver,
        )
        if md is not None:
            self.md = md
            self.solving_time = None if not md.feasible else md.solving_time
            self.status = md.status
        return list_op_sched

    def get_activation_offload(self, op_sched):
        self.bandwidth = 1e7
        """
        With all the torch.cuda.Event and stream.wait_event() to synchornize
        between offload/prefetch, in the order of Main>Ofl>Del during fwd and Prf>Main during bwd

        """
        cluster = op_sched.cluster
        op_list = op_sched.op_list
        fwd_op_list = op_list[: op_sched.loss_idx]
        bwd_op_list = op_list[op_sched.loss_idx :]
        prefetch_op_list = []
        phantoms = op_sched.phantoms

        # for op in op_sched.op_list:
        #     if isinstance(op, ComputeOp):

        offload_ops = {}
        prefetch_ops = {}

        for anode in phantoms:
            if not anode.allocation_type == "data":
                continue
            i = max(
                min(op_sched.occurrences[ComputeOp(cnode).name]) for cnode in anode.deps
            )
            op_sched.op_list[i].record_event = True
            comp_op = op_sched.op_list[i]
            offload_op = OffloadOp(Activation(anode), time=anode.mem / self.bandwidth)
            offload_op.wait_events.append([comp_op.op_type, comp_op.target.name])
            offload_op.record_event = True
            # fwd_op_list.append(offload_op)
            offload_ops[offload_op] = i


            del_op = DeleteOp(Activation(anode))
            del_op.wait_events.append([offload_op.op_type, offload_op.target.name])
            users = [min(op_sched.occurrences[ComputeOp(cnode).name]) 
                    for cnode in anode.users_real
                    if cnode.is_fwd]
            if users:
                user_op = op_sched.op_list[max(users)]
                user_op.record_event = True
                del_op.wait_events.append([user_op.op_type, user_op.target.name])
            # fwd_op_list.append(del_op)
            offload_ops[del_op] = i

            # prefetch_op_list.append()
            alloc_op = AllocateOp(Activation(anode))
            prefetch_op = PrefetchOp(Activation(anode), time=anode.mem / self.bandwidth)
            prefetch_op.record_event = True
            prefetch_op.wait_events.append([offload_op.op_type, offload_op.target.name])
            # prefetch_op_list.append(prefetch_op)
            i = 0
            for j, op in enumerate(bwd_op_list):
                if isinstance(op, ComputeOp) and op.target in anode.users_real.union(anode.users_fake):
                    op.wait_events.append([prefetch_op.op_type, prefetch_op.target.name])
                    i = j

            prefetch_ops[alloc_op] = j
            prefetch_ops[prefetch_op] = j

        def simulate_schedule(comp_ops, prf_ops, ofl_ops):
            op_time = {}
            time = 0
            for op in prf_ops:
                op_time[op.name] = time
                time += op.time

            time = 0
            for op in comp_ops:
                wait_time = [op_time[f"{e[0]}({e[1]})"] 
                            for e in op.wait_events 
                            if f"{e[0]}({e[1]})" in op_time]
                if wait_time:
                    time = max(wait_time+[time])
                time += op.time
                op_time[op.name] = time
            
            time = 0
            for op in ofl_ops:
                wait_time = [op_time[f"{e[0]}({e[1]})"] 
                            for e in op.wait_events 
                            if f"{e[0]}({e[1]})" in op_time]
                if wait_time:
                    time = max(wait_time+[time])
                time += op.time
                op_time[op.name] = time
            return op_time

        fwd_op_time = simulate_schedule(fwd_op_list, [], offload_ops)
        bwd_op_time = simulate_schedule(bwd_op_list, prefetch_ops, [])

        fwd_op_list += list(sorted(offload_ops.keys(), key=lambda x:offload_ops[x]))
        bwd_op_list = bwd_op_list[:1] + list(sorted(prefetch_ops.keys(), key=lambda x:prefetch_ops[x])) + bwd_op_list[1:]

        new_op_sched = OpSchedule(
            fwd_op_list + bwd_op_list,
            loss_idx=len(fwd_op_list),
            cluster=cluster,
            init_op_list=op_sched.init_op_list,
        )

        fwd_wait_time = max(fwd_op_time.values()) - new_op_sched.fwd_time
        bwd_wait_time = max(bwd_op_time.values()) - new_op_sched.bwd_time
        
        new_op_sched.fwd_wait_time = fwd_wait_time
        new_op_sched.bwd_wait_time = bwd_wait_time

        return new_op_sched

    def add_offload_sched_hcn(self, hcn):
        scheds = [sched for sched in hcn.list_sched 
                    if sched.offload_mem == 0 
                    and sched.prefetch_mem == 0 ]
        for sched in scheds:
            hcn.list_sched.append(self.get_activation_offload(sched))
