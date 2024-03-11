from typing import Dict, Any
import time
import numpy as np
from copy import deepcopy
from pulp import (
    LpVariable,
    LpProblem,
    LpMinimize,
    lpSum,
    get_solver,
    listSolvers,
    LpStatus,
)

from ..op_schedule import (
    Activation,
    Parameter,
    Buffer,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    OpSchedule,
)
from .main import get_sched, add_sched, translate
from rkgb.core.hierarchical import HierarchicalGraph

class knapsack:
    def __init__(self, parameter_sizes: list, pre_solve_size=10):
        size = [s[1] for s in parameter_sizes]
        self.parameter_sizes = parameter_sizes
        self.sizes = [s / sum(size) for s in size]
        self.pre_solve_size = pre_solve_size

    def get_size(self, indices, sizes):
        return sum(sizes[i] for i in indices)

    # @lru_cache(maxsize=4096 * 4096)
    def solve(self, frac: float, i: int = 0, sizes=[]):
        sizes = sizes or self.sizes
        if frac < 0:
            return []
        if i == len(sizes):
            return list(range(i))
        res1 = self.solve(frac, i + 1, sizes)
        res2 = self.solve(frac - sizes[i], i + 1, sizes)
        res2 = [i] + res2
        if self.get_size(res1, sizes) <= self.get_size(res2, sizes) + sizes[i]:
            return res1
        else:
            return res2

    def select(self, frac: float):
        sizes = self.sizes.copy()
        parameter_sizes = self.parameter_sizes.copy()
        selections = []
        while len(sizes) > self.pre_solve_size and frac>0:
            sel_i = self.presolve(frac, sizes)
            if sel_i is None:break
            selections.append(parameter_sizes.pop(sel_i)[0])
            frac -= sizes.pop(sel_i)
        indices = self.solve(frac, sizes=sizes)
        selections += [parameter_sizes[i][0] for i in indices]
        return selections

    def select_size(self, size: int):
        if not self.parameter_sizes:return []
        return self.select(size / sum(s[1] for s in self.parameter_sizes))
    
    def presolve(self, frac, sizes):
        array = np.array(sizes)
        sel_i = np.argmax(array*(array<frac))
        if array[sel_i]>frac:return np.argmin(array)
        return sel_i


class RkLpVariable(LpVariable):
    def __init__(self, name, lowBound=None, upBound=None, cat="Continuous", e=None):
        super().__init__(name=name, lowBound=lowBound, upBound=upBound, cat=cat, e=e)
        self.solution = None

    def value(self):
        return self.solution or self.varValue

    @classmethod
    def dicts(
        cls,
        name,
        indices=None,
        lowBound=None,
        upBound=None,
        cat="Continuous",
        indexStart=[],
    ):
        d = {}
        for index in indices:
            var_name = name + "_" + "_".join([str(i) for i in index])
            d[index] = RkLpVariable(
                var_name, lowBound=lowBound, upBound=upBound, cat=cat
            )
        return d

    def __repr__(self):
        if self.varValue:return str(self.varValue)
        return super().__repr__()

    def prefill(self, value):
        self.setInitialValue(value)
        self.fixValue()

class ModelPULP:
    """
    Build the ILP model by given Hgraph and budget.
    RN this model will take a rk_chain to solve the solution.
    """

    def __init__(
        self,
        hgraph: HierarchicalGraph,
        peak_budget: int,
        save_budget=None,
        ilp_solver_params: Dict[str, Any] = {
            "LogToConsole": 0,
            "IntegralityFocus": 1,
            "TimeLimit": 4 * 60,
        },
        gcd=None,
        accurate_mem=False,
        protected_names=[],
        grouping=True,
        grad_mode="free", #["free", "accumulate"]
        cpu_optimize_kwargs = None,
        batch_multiplier = 1
    ):
        self.gcd = gcd if gcd else 1024**2
        self.peak_budget = peak_budget / self.gcd
        if save_budget:
            self.save_budget = save_budget / self.gcd
        else:
            self.save_budget = peak_budget / self.gcd

        self.ilp_solver_params = ilp_solver_params
        self.feasible = None
        self.solve_time = None
        self.with_parameters = accurate_mem
        self.with_grad = accurate_mem
        self.with_optimizer_states = accurate_mem#optimizer states will be offloaded
        self.gradient_accumulation = 0# if 0, no gradient/optimizer states alive from previous iters
        self.single_fwd = accurate_mem
        self.single_bwd = accurate_mem
        self.grouping = grouping
        self.grad_mode = grad_mode
        self.cpu_optimize_kwargs = cpu_optimize_kwargs

        #############################
        self.hgraph = hgraph
        self.hcn2sub_c = []
        self.list_list_sched = []
        self.sub_clusters = []
        self.nOpts = []  # number of opts
        self.nR = []  # number to run R, =nOpts if bwd, =nOpts+1 if fwd
        self.time = []
        self.overhead = []
        self.bin_type = "Binary"

        for i, hcn in enumerate(self.hgraph.list_HCNs):
            if "Loss" in hcn.name:
                self.loss_idx = i
            if hcn.sub_cluster is None:
                # only when hcn is fwd with requires_grad=False
                self.hcn2sub_c.append(None)
                self.nR.append(1)
                self.nOpts.append(0)
                self.time.append([hcn.ff_time])
                self.overhead.append([hcn.ff_overhead / self.gcd])
            else:
                if hcn.is_fwd:
                    self.list_list_sched.append(hcn.list_sched)
                    self.sub_clusters.append(hcn.sub_cluster)
                j = self.sub_clusters.index(hcn.sub_cluster)
                list_sched = self.list_list_sched[j]  # hcn bwd does not have list_sched
                self.hcn2sub_c.append(j)
                # self.hcn2sub_c.append(len(self.list_list_sched) - 1)
                self.nR.append(len(list_sched) + (1 if hcn.is_fwd else 0))
                self.nOpts.append(len(list_sched))

                if hcn.is_fwd:
                    # add fast forward to the options (final one)
                    self.time.append(
                        [op_sched.fwd_time for op_sched in list_sched] + [hcn.ff_time]
                        if hcn.is_fwd
                        else []
                    )
                    self.overhead.append(
                        [op_sched.fwd_overhead / self.gcd for op_sched in list_sched]
                        + [hcn.ff_overhead / self.gcd]
                    )
                else:
                    self.time.append([op_sched.bwd_time for op_sched in list_sched])
                    self.overhead.append(
                        [op_sched.bwd_overhead / self.gcd for op_sched in list_sched]
                    )
        self.sub_c2hcn = [
            [] for _ in self.list_list_sched
        ]  # index of sub_cluster to index of hcn
        for i, j in enumerate(self.hcn2sub_c):
            if j is None:
                continue
            self.sub_c2hcn[j].append(i)
        self.mem = [han.mem / self.gcd for han in self.hgraph.list_HANs]
        self.saved_mem = [
            [op_sched.mem / self.gcd for op_sched in list_sched]
            for list_sched in self.list_list_sched
        ]

        self.T = T = len(self.hgraph.list_HCNs)
        self.I = I = len(self.hgraph.list_HANs)
        self.J = J = len(self.list_list_sched)

        self.hcn2idx = {hcn:i for i, hcn in enumerate(self.hgraph.list_HCNs)}
        self.han2idx = {hcn:i for i, hcn in enumerate(self.hgraph.list_HANs)}
        
        self.protected_indices = []
        self.input_grad_indices = []
        self.input_data_indices = []
        self.han_deps = []
        self.han_users = []

        for i, han in enumerate(self.hgraph.list_HANs):
            if han.anode.name in protected_names:
                self.protected_indices.append(i)
            if han in self.hgraph.input_grad_HANs:
                self.input_grad_indices.append(i)
            if han in self.hgraph.input_data_HANs:
                self.input_data_indices.append(i)
            self.han_deps.append([self.hcn2idx[hcn] for hcn in han.deps])
            self.han_users.append([self.hcn2idx[hcn] for hcn in han.users
                                   if hcn in self.hcn2idx])# current sub-graph users

        self.hcn_users = [
            [self.han2idx[han] for han in self.hgraph.list_HCNs[k].users]
            for k in range(self.T)
        ]  # outputs of hcn

        #### Update edges based on .dep_interfaces_data
        #### In certain schedules, BWD depends on input/output data
        for k, hcn in enumerate(self.hgraph.list_HCNs):
            if hcn.sub_cluster is None:
                continue
            list_sched = self.list_list_sched[self.hcn2sub_c[k]]
            for op_sched in list_sched:
                # for i_ in op_sched.dep_interfaces_data:
                for anode in op_sched.dep_interfaces_data:
                    # Without specifying schedule, we assume it's possible to use han here
                    for i in range(I):
                        if (anode.name
                            == self.hgraph.list_HANs[i].anode.name
                            and k not in self.han_users[i]
                        ):
                            self.han_users[i].append(k)

        ##############################

        self.create_list = [(k, i) for k in range(self.T) for i in self.hcn_users[k]]
        self.delete_list = [(k, i) for i in range(I) for k in self.han_deps[i] + self.han_users[i]]

        self.Cr = len(self.create_list)
        self.De = len(self.delete_list)
        self.W = len(self.sub_clusters)

        self.add_variables()
        if accurate_mem and self.with_parameters:
            self.add_offload_variables()


        # define objective function
        prf_cost = (
            0.01
            * lpSum(
                (self.PrfW[t, k, w] + self.PrfG[t, k, w] + self.PrfO[t, k, w])
                * self.parameter_size[w] / self.bandwidthPrf
                for t in range(self.T)
                for k in self.krange(t)
                for w in range(self.W)
            )
            if self.with_parameters
            else 0
        )
        ofl_cost = (
            0.01
            * lpSum(
                (self.OflW[t, k, w] + self.OflG[t, k, w] + self.OflO[t, k, w])
                * self.parameter_size[w] / self.bandwidthOfl
                for t in range(self.T)
                for k in self.krange(t)
                for w in range(self.W)
            )
            if self.with_parameters
            else 0
        )
        self.md = LpProblem(f"rockmateMILP", LpMinimize)
        self.md += (
            lpSum(self.Time[t, k] for t in range(self.T) for k in self.krange(t))
            + prf_cost
            + ofl_cost
        )

        print("adding constraints")
        ##### Time constraints
        for t in range(self.T):
            for k in self.krange(t):
                # if k==self.loss_idx:continue
                if self.with_parameters:# and k in self.hcn2param:
                    # w = self.hcn2param[k] if k != self.loss_idx else None
                    ofl_time = lpSum(
                        self.parameter_size[w] / self.bandwidthOfl * self.OflW[t, k, w]
                        for w in self.hcn2param[k]
                    )
                else:
                    ofl_time = 0

                self.md += (
                    self.Time[t, k]
                    >= lpSum(
                        self.Comp[t, k, o] * self.time[k][o] for o in range(self.nR[k])
                    )
                    + ofl_time
                )

        
        self.add_valid_constraints()
        
        if self.with_parameters:
            self.add_offload_constraints()

        self.add_memory_constrains()

    def add_variables(self):
        self.Comp = RkLpVariable.dicts(
            f"Comp",
            [
                (t, k, o)
                for t in range(self.T)
                for k in self.krange(t)
                for o in range(self.nR[k])
            ],
            cat=self.bin_type,
        )

        self.sumComp = {}
        for t in range(self.T):
            for k in self.krange(t):
                self.sumComp[t, k] = lpSum(
                    self.Comp[t, k, o] for o in range(self.nR[k])
                )

        for t in range(self.T):
            for k in range(self.T):
                if k not in self.krange(t):
                    for o in range(self.nR[k]):
                        self.Comp[t, k, o] = 0
                    self.sumComp[t, k] = 0

        self.AliveP = RkLpVariable.dicts(
            f"AliveP",
            [
                (t, j, o)
                for t in range(self.T + 1)
                for j, list_sched in enumerate(self.list_list_sched)
                for o in range(len(list_sched))
                if t-1 in self.sub_c2hcn[j]
            ],
            cat=self.bin_type,
        )
        for j, list_sched in enumerate(self.list_list_sched):
            for o in range(len(list_sched)):
                if (0,j,o) not in self.AliveP:
                    self.AliveP[0,j,o] = 0
                for t in range(1,self.T + 1):
                    if (t,j,o) not in self.AliveP:
                        self.AliveP[t,j,o] = self.AliveP[t-1,j,o]

        self.sumAliveP = {}
        for j in range(self.J):
            for t in range(self.T + 1):
                self.sumAliveP[t,j] = lpSum(
                    self.AliveP[t, j, o] for o in range(len(self.list_list_sched[j]))
                )
        self.active_stages = dict()
        for i in range(self.I):
            self.active_stages[i] = []
            for t in range(self.T):
                for k in self.krange(t):
                    if k in self.han_deps[i]+self.han_users[i]:
                        self.active_stages[i].append(t)
            if not self.hgraph.list_HANs[i].deps:# src node
                self.active_stages[i].append(-1)
        # to present whether one saved tensor can be inherited from the last stage
        self.AliveA = RkLpVariable.dicts(
            "AliveA", [(t, c)
                       for t in range(1,self.T) 
                       for c, (k, i) in enumerate(self.create_list)
                       if t-1 in self.active_stages[i]], cat=self.bin_type
        )  # activation
        for c, (k, i) in enumerate(self.create_list):
            self.AliveA[0,c] = 0
            for t in range(1, self.T):
                if (t,c) not in self.AliveA:
                    self.AliveA[t,c] = self.AliveA[t-1,c]

        self.AliveT = RkLpVariable.dicts(
            "AliveT", [(t, i)
                       for t in range(self.T) 
                       for i in range(self.I)
                       if t-1 in self.active_stages[i]], cat=self.bin_type
        )  # tensor that can be shared by acts
        for i in range(self.I):
            if (0,i) not in self.AliveT:
                self.AliveT[0,i] = 0
            for t in range(1, self.T):
                if (t,i) not in self.AliveT:
                    self.AliveT[t,i] = self.AliveT[t-1,i]

        self.create = RkLpVariable.dicts(
            "create",
            [
                (t, i)
                for t in range(self.T)
                for i in range(self.Cr)
                if self.create_list[i][0] in self.krange(t)
            ],
            cat=self.bin_type,
        )
        self.delete = RkLpVariable.dicts(
            "delete",
            [
                (t, i)
                for t in range(self.T)
                for i in range(self.De)
                if self.delete_list[i][0] in self.krange(t)
            ],
            cat=self.bin_type,
        )
        for t in range(self.T):
            for i in range(self.Cr):
                if self.create_list[i][0] not in self.krange(t):
                    self.create[t, i] = 0
            for i in range(self.De):
                if self.delete_list[i][0] not in self.krange(t):
                    self.delete[t, i] = 0
            if self.single_fwd:
                for d in range(self.De):
                    (k, i) = self.delete_list[d]
                    if i in self.protected_indices:continue
                    if k == max(self.han_deps[i] + self.han_users[i]) and k == t:
                        self.delete[t, d] = 1
                        pass
 
        self.Time = RkLpVariable.dicts(
            "Time", [(t, k) for t in range(self.T) for k in self.krange(t)], cat="Continuous"
        )

    def add_offload_variables(self):
        cpu_optimize_kwargs = self.cpu_optimize_kwargs

        self.cpu_optimize = True
        self.optimizer_states_factor = cpu_optimize_kwargs["optimizer_states_size"]#*weight size
        self.cpu_optimize_speed = cpu_optimize_kwargs["cpu_optimize_speed"]#B/ms
        self.gpu_optimize_speed = cpu_optimize_kwargs["gpu_optimize_speed"]#B/ms
        self.optimizer_overhead_factor = cpu_optimize_kwargs["optimizer_overhead"]#*weight size
        self.minor_param_size = cpu_optimize_kwargs["minor_param_size"]# minor weight size
        self.bandwidth = cpu_optimize_kwargs["bandwidth"]# bandwidth
        batch_multiplier = 4
        # self.BatMpl = RkLpVariable("BMpl", lowBound=0, upBound=self.batch_multiplier, cat="Integer")
        # self.param_multiplier = 1-self.BatMpl*1/self.batch_multiplier
        self.param_multiplier = RkLpVariable("BMpl", lowBound=0, upBound=1-1/batch_multiplier, cat="Continuous")
        self.param_multiplier = 0.


        def get_parameters(hierarchical_nodes):
            all_params = {}
            # all_clusters = {sub_cluster.name:sub_cluster for sub_cluster in sub_clusters if sub_cluster}
            # sub_c2params = {sub_cluster.name:set() for sub_cluster in sub_clusters if sub_cluster}
            param2sub_c = {}

            for i,hcn in enumerate(hierarchical_nodes):
                if not hasattr(hcn, "required_parameter_nodes_real"):continue
                if hcn.sub_cluster is not None and hasattr(hcn.sub_cluster, "parameter_nodes"):
                    # if FWD/BWD hcns have different req_pnodes, parameters may be needed for recomputation
                    req_pnodes = hcn.sub_cluster.parameter_nodes
                else:
                    req_pnodes = hcn.required_parameter_nodes_real|hcn.required_parameter_nodes_fake
                for pnode in req_pnodes:
                    if pnode.is_buffer:continue
                    if pnode.mem < self.minor_param_size:continue
                    # sub_c2params[sub_cluster.name].add(pnode.param_name)
                    if hasattr(pnode, "original_param_node"):
                        all_params[pnode.param_name] = pnode.original_param_node
                    else:
                        all_params[pnode.param_name] = pnode
                    if pnode.param_name not in param2sub_c:
                        param2sub_c[pnode.param_name] = {i}
                    else:
                        param2sub_c[pnode.param_name].add(i)
            result = {}
            for p, c in param2sub_c.items():
                c_ = tuple(sorted(c))
                if c_ not in result:
                    result[c_] = {p}
                else:
                    result[c_].add(p)

            parameters = []
            params2sub_c = {}
            for k,v in result.items():
                params2sub_c[len(parameters)] = k
                parameters.append([all_params[p] for p in v])
            return params2sub_c, parameters
        
        if self.with_parameters:
            self.param2hcn, self.parameters = get_parameters(self.hgraph.list_HCNs)
            self.hcn2param = {t:[] for t in range(self.T)}

            for p,hcn_s in self.param2hcn.items():
                for hcn in hcn_s:
                    self.hcn2param[hcn].append(p)

            self.parameter_size = [sum(pnode.mem for pnode in p)/self.gcd for p in self.parameters]
            self.parameter_gradient_size = [sum(pnode.mem for pnode in p if pnode.info.requires_grad
                                                )/self.gcd for p in self.parameters]
            self.W = W = len(self.parameters)

            self.AliveW = RkLpVariable.dicts(
                "AliveW",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # parameter w is alive at the start of step j.
            self.AliveG = RkLpVariable.dicts(
                "AliveG",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # w.grad is alive at the start of step j.
            self.AliveO = RkLpVariable.dicts(
                "AliveO",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # w.grad is alive at the start of step j.

            self.OflW = RkLpVariable.dicts(
                "OflW",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.OflG = RkLpVariable.dicts(
                "OflG",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.PrfW = RkLpVariable.dicts(
                "PrfW",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.PrfG = RkLpVariable.dicts(
                "PrfG",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.OptC = RkLpVariable.dicts(
                "OptC",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.OflO = RkLpVariable.dicts(
            "OflO",
            [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
            cat="Continuous",
            lowBound=0,
            upBound=1,
            )
            self.PrfO = RkLpVariable.dicts(
                "PrfO",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.OflP = RkLpVariable.dicts(
            "OflP",
            [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
            cat="Continuous",
            lowBound=0,
            upBound=1,
            )
            self.PrfP = RkLpVariable.dicts(
                "PrfP",
                [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.param_grad_mem = {(t,k):0 for t in range(self.T) for k in self.krange(t)}
            self.prefill()

            self.bandwidthOfl = cpu_optimize_kwargs["bandwidth"]/self.gcd  # byte/ms
            self.bandwidthPrf = cpu_optimize_kwargs["bandwidth"]/self.gcd  # byte/ms

    def add_valid_constraints(self):
        # In the last stage, every source edge of input_grad should be alive or executed
        for i in self.input_grad_indices:
            for j_, (k_, i_) in enumerate(self.create_list):
                if i_ == i:
                    self.md += self.AliveA[self.T - 1, j_] + self.sumComp[self.T - 1, k_] == 1

        # # In the first stage, assume input data is alive
        # for i in self.input_data_indices:
        #     for j_, (k_, i_) in enumerate(self.create_list):
        #         if i_ == i:
        #             self.md += (self.AliveA[0, j_] ==  1)

        for j in range(self.J):
            bwd_i = max(self.sub_c2hcn[j])
            # Forward start with no phantoms
            # self.md += (
            #     lpSum(
            #         (self.AliveP[0, j, o])  # - self.Comp[bwd_i][T - 1, o])
            #         for o in range(self.nOpts[bwd_i])
            #     )
            #     == 0
            # )
            # in the end of bwd, del every phantoms
            self.md += (
                lpSum(
                    (self.AliveP[self.T, j, o])  # - self.Comp[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                )
                == 0
            )

        # options don't conflict
        for t in range(self.T):
            for k in self.krange(t):
                self.md += self.sumComp[t, k] <= 1
        for t in range(self.T + 1):
            for j in range(self.J):
                self.md += (
                    self.sumAliveP[t,j] <= 1
                )  # assuming two copies of saved tensors won't be kept at the same time

        #### Option-free constraints: from rk-checkmate
        self.md += (
            lpSum(self.sumComp[t, t] for t in range(self.T)) == self.T
        )  # diagonal should be executed
        self.md += (
            lpSum(
                self.sumComp[t, self.loss_idx]
                for t in range(self.T)
                if self.loss_idx in self.krange(t)
            )
            == 1
        )  # loss should be executed exactly once

        for t in range(self.T):
            for j in range(self.Cr):
                self.md += (
                    self.AliveA[t, j] <= self.AliveT[t, self.create_list[j][1]]
                )  # one edge created, memory is occupied
        for t in range(self.T - 1):
            for j in range(self.Cr):
                src_i = self.create_list[j][0]
                src = self.sumComp[t, src_i] if src_i in self.krange(t) else 0
                self.md += self.AliveA[t + 1, j] <= self.AliveA[t, j] + src
        for t in range(self.T):
            for j, (k, i) in enumerate(self.create_list):
                for k_ in self.han_users[i]:
                    self.md += (
                        self.sumComp[t, k_] <= self.sumComp[t, k] + self.AliveA[t, j]
                    )

        self.dep_interfaces = {hcn.name :[] for hcn in self.hgraph.list_HCNs}
        #### Options-related constraints
        for t in range(self.T):
            for j in range(self.J):
                fwd_i = min(self.sub_c2hcn[j])
                bwd_i = max(self.sub_c2hcn[j])
                for o in range(self.nOpts[fwd_i]):
                    self.md += (
                        self.AliveP[t + 1, j, o]
                        <= self.AliveP[t, j, o] + self.Comp[t, fwd_i, o]
                    )  # phantoms can only be generated by fwd
                    self.md += (
                        self.AliveP[t + 1, j, o]
                        >= self.AliveP[t, j, o]
                        - self.Comp[t, bwd_i, o]
                        + self.Comp[t, fwd_i, o]
                    )  # phantoms can only be deleted by bwd
                    self.md += (
                        self.Comp[t, bwd_i, o]
                        <= self.AliveP[t, j, o] + self.Comp[t, fwd_i, o]
                    )

                    list_sched = self.list_list_sched[j]
                    # for i in list_sched[o].dep_interfaces_data:
                    for anode in list_sched[o].dep_interfaces_data:
                        hcn = self.hgraph.list_HCNs[bwd_i]
                        self.dep_interfaces[hcn.name].append((o, anode.name))
                        # Tensor req_i is required by BWD
                        req_i = [han.anode.name for han in self.hgraph.list_HANs].index(
                            anode.name
                        )
                        for j_, (k_, i_) in enumerate(self.create_list):
                            if i_ == req_i:
                                self.md += (
                                    self.Comp[t, bwd_i, o]
                                    <= self.sumComp[t, k_] + self.AliveA[t, j_]
                                )

    def add_memory_constrains(self):
                # we don't keep eyes on the alive status all the time
        # only the steps when changes can happen
        self.alive = {}
        for t in range(self.T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.alive[(t, k, i)] = self.AliveT[t, i]
                self.alive[(t, k, i)] += lpSum(
                    self.create[t, eidx_c]
                    for eidx_c, (k_, i_) in enumerate(self.create_list)
                    if i_ == i and k_ <= k
                )
                self.alive[(t, k, i)] -= lpSum(
                    self.delete[t, eidx_d]
                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                    if i_ == i and k_ <= k
                )
                self.md += self.alive[(t, k, i)] >= 0
                self.md += self.alive[(t, k, i)] <= 1
                if (k, i) in self.create_list:
                    didx = self.delete_list.index((k, i))
                    self.md += (
                        self.alive[(t, k, i)] + self.delete[t, didx]
                        >= self.sumComp[t, k]
                    )

            for eidx, (k, i) in enumerate(self.create_list):
                # if k not in self.krange(t):
                #     continue
                self.md += self.create[t, eidx] <= self.sumComp[t, k]
                # self.md += self.create[t, eidx] >= self.sumComp[t, k] - self.alive[(t,k,i)]
            for i in range(self.I):
                if t + 1 < self.T:
                    pass
                    self.md += (
                        self.AliveT[t + 1, i]
                        == self.alive[(t, max(self.han_deps[i] + self.han_users[i]), i)]
                    )
                elif i not in self.protected_indices:
                    # in the end of bwd, del every HDN
                    step = None
                    for k in self.krange(t):
                        if k in self.han_deps[i] + self.han_users[i]:
                            step = k
                    if step is not None:
                        self.md += self.alive[(t, step, i)] == 0
                    # elif i>9:
                    #     print(i)
                    else:
                        self.md += self.AliveT[self.T-1, i] == 0

        def _num_hazards(t, i, k):
            if i in self.protected_indices:
                return _max_num_hazards(t, i, k)
            if t + 1 < self.T:
                return (
                    1
                    - self.sumComp[t, k]
                    + self.AliveT[t + 1, i]
                    + lpSum(self.sumComp[t, j] for j in self.han_users[i] if j > k)
                )
            return (
                1
                - self.sumComp[t, k]
                + lpSum(self.sumComp[t, j] for j in self.han_users[i] if j > k)
            )

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in self.han_users[i] if j > k)
            if t + 1 < self.T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # delete when not needed
        for t in range(self.T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += 1 - self.delete[t, eidx] <= _num_hazards(t, i, k)

        # don't delete if still needed
        for t in range(self.T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += _max_num_hazards(t, i, k) * (
                    1 - self.delete[t, eidx]
                ) >= _num_hazards(t, i, k)
                if i in self.protected_indices:
                    self.md += self.delete[t, eidx] == 0

        self.U = {}
        for t in range(self.T):
            self.U[t, 0] = (
                lpSum(self.AliveT[t, i] * self.mem[i] for i in range(self.I))
                + lpSum(
                    self.create[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.create_list)
                    if k_ == 0 and k_ in self.krange(t)
                )
                - lpSum(
                    self.delete[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.delete_list)
                    if k_ == 0 and k_ in self.krange(t)
                )
                + lpSum(
                    self.AliveP[t, j, o] * save_mem
                    for j in range(self.J)
                    for o, save_mem in enumerate(self.saved_mem[j])
                )
                + lpSum(  # if the first fwd operation creates phantoms
                    self.Comp[t, 0, o] * self.saved_mem[self.hcn2sub_c[0]][o]
                    for o in range(self.nOpts[0])
                )
                # - lpSum(
                #     self.Bwd[f_to_b[i]][t_, o] * self.saved_mem[i][o]
                #     for i in range(self.chain.ln)
                #     for o in range(nb_opt[i])
                #     for t_ in range(t)
                # )
            )

        for t in range(self.T):
            for k in range(1, self.T):
                j = self.hcn2sub_c[k]
                self.U[t, k] = (
                    self.U[(t, k - 1)]
                    + lpSum(
                        self.create[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.create_list)
                        if k_ == k and k_ in self.krange(t)
                    )
                    - lpSum(
                        self.delete[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.delete_list)
                        if k_ == k and k_ in self.krange(t)
                    )
                )
                # if k < self.loss_idx:
                if self.hgraph.list_HCNs[k].is_fwd:
                    self.U[t, k] += lpSum(
                        self.Comp[t, k, o] * self.saved_mem[j][o]
                        for o in range(self.nOpts[k])
                    )
                else:
                    if j is None:
                        continue
                    fwd_i = min(self.sub_c2hcn[j])
                    self.U[t, k] += lpSum(
                        (
                            self.AliveP[t + 1, j, o]
                            - self.Comp[t, fwd_i, o]
                            - self.AliveP[t, j, o]
                        )
                        * self.saved_mem[j][o]
                        for o in range(self.nOpts[k])
                    )
        for t in range(self.T):
            for k in self.krange(t):
                parameter_mem = self.all_param_mem(t, k) if self.with_parameters else 0
                j = self.hcn2sub_c[k]
                self.md += self.U[t, k] >= 0
                self.md += self.U[t, k] <= (self.peak_budget - parameter_mem)
                # if j is None or not accurate_mem:
                if True:
                    # don't consider correction_term
                    self.md += (
                        self.U[t, k]
                        + lpSum(
                            self.Comp[t, k, o] * self.overhead[k][o]
                            for o in range(self.nR[k])
                        )
                        + lpSum(
                            self.mem[i_] * self.delete[t, eidx_d]
                            for eidx_d, (k_, i_) in enumerate(self.delete_list)
                            if k == k_
                        )
                        <= self.peak_budget - parameter_mem
                    )
                else:
                    hcn = self.hgraph.list_HCNs[k]
                    for o, op_sched in enumerate(self.list_list_sched[j]):
                        for correction in (
                            op_sched.fwd_overhead_correction
                            if hcn.is_fwd
                            else op_sched.bwd_overhead_correction
                        ):
                            correction_term = 0
                            overhead = (correction["save"] + correction["overhead"]) - (
                                op_sched.mem if hcn.is_fwd else 0
                            )
                            for inter_position, inter_mem in correction.items():
                                if (
                                    inter_position == "save"
                                    or inter_position == "overhead"
                                ):
                                    continue

                                i_ = [
                                    han.anode.name for han in self.hgraph.list_HANs
                                ].index(
                                    self.sub_clusters[j]
                                    .list_anodes[inter_position[0]]
                                    .name
                                )
                                if inter_position[1] == "always":
                                    not_kept_alive = 1
                                elif not inter_position[1]:  # ending status
                                    if (k, i_) in self.delete_list:
                                        eidx = self.delete_list.index((k, i_))
                                        not_kept_alive = self.delete[t, eidx]
                                    else:  # when output_data is not deps, but we care about it
                                        # eidx = self.delete_list.index((k, i_))
                                        k_ = max([kk for kk in self.han_deps[i_] if kk < k])
                                        not_kept_alive = self.alive[(t, k_, i_)]
                                else:  # start status
                                    eidx = self.create_list.index((k, i_))
                                    not_kept_alive = self.create[t, eidx]
                                correction_term += not_kept_alive * inter_mem
                            self.md += (
                                self.U[t, k]
                                + self.Comp[t, k, o] * overhead / self.gcd
                                + correction_term
                                + lpSum(
                                    self.mem[i_] * self.delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                                    if k == k_
                                )
                                <= self.peak_budget - parameter_mem
                            )
                        if not (
                            op_sched.fwd_overhead_correction
                            if hcn.is_fwd
                            else op_sched.bwd_overhead_correction
                        ):
                            self.md += (
                                self.U[t, k]
                                + self.Comp[t, k, o] * self.overhead[k][o]
                                + lpSum(
                                    self.mem[i_] * self.delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                                    if k == k_
                                )
                                <= self.peak_budget - parameter_mem
                            )
                if t == self.loss_idx and self.save_budget:
                    self.md += self.U[t, k] <= self.save_budget

    
    def krange(self, t):
        if self.single_fwd:
            return [t]
        elif self.single_bwd and t > self.loss_idx:
            return list(range(self.loss_idx)) + [t]
        return list(range(t + 1))
        # return range(self.T)

    def next_index(self, t, i, upper_triangle=False):
        # if upper_triangle, consider the case when i>t
        if t == self.T - 1:
            if i < max(self.krange(t)):
                t_ = t
                i_ = self.krange(t)[self.krange(t).index(i) + 1]
            else:
                t_ = 0
                i_ = 0
        else:
            # end = self.T - 1 if upper_triangle else t
            end = max(self.krange(t))
            if i < end:
                t_ = t
                i_ = self.krange(t)[self.krange(t).index(i) + 1]
            else:
                i_ = min(self.krange(t + 1))
                t_ = t + 1
        return (t_, i_)

    def add_abar_constraint(self, save_budget):
        T = len(self.hgraph.list_HCNs)
        self.save_budget = save_budget / self.gcd
        for k in range(self.T):
            self.md += self.U[(self.loss_idx, k)] <= self.save_budget

    def req_w(self):
        return 1 - self.param_multiplier

    def accumC_grad(self, w):
        #if grad_accumulation, gradient stored on CPU from previous iterations
        return self.sumOptC[w]
    
    def accumC_optimizer_states(self, w):
        #if grad_accumulation, optimizer states stored on CPU from previous iterations
        return self.accumC_grad(w)
    
    def instant_opt(self, w):
        # return the fraction of parameter instantly optimized after bwd
        if self.gradient_accumulation:
            return 0
        if self.grad_mode =="free":
            return self.req_w()
        return 1-self.sumOptC[w]- self.param_multiplier
    
    def max_OflGProg(self, t, k, w):
        return self.OflGProg[t, k, w]+(self.OflWProg[t, k, w]*(self.grad_mode=="free")
                                      *self.w_by_wg(w))

    def w_by_wg(self, w):
        if self.parameter_gradient_size[w]==0:return 0
        return self.parameter_size[w]/self.parameter_gradient_size[w]

    def all_param_mem(self, t, k, with_multiplier=True):
        return (self.parameter_mem(t,k) 
                + self.param_grad_mem[t,k]
                + self.optimizer_states_mem(t,k)  
                + self.param_multiplier*self.peak_budget*with_multiplier)

    def parameter_mem(self, t, k):
        parameter_mem = lpSum(
            (self.AliveW[t, k, w] + self.PrfW[t, k, w])
            * self.parameter_size[w]
            for w in range(self.W)
        )
        return parameter_mem
    
    # def param_grad_mem(self, t, k):
    #     grad_mem = lpSum(
    #         self.AliveG[t, k, w]
    #         * self.parameter_gradient_size[w]
    #         for w in range(self.W)
    #     )
    #     return grad_mem
    
    def optimizer_states_mem(self, t, k, with_overhead=True):    
        optimizer_states_mem = lpSum(((self.AliveO[t, k, w]+self.PrfO[t, k, w])*
                    self.parameter_gradient_size[w] *
                    self.optimizer_states_factor)
                    for w in range(self.W))
        optimizer_overhead = 0
        if k > self.loss_idx:# and k in self.hcn2param:
            l_w = self.hcn2param[k]
            optimizer_overhead += sum((self.req_w()-self.sumOptC[w])
                                      * self.parameter_gradient_size[w]
                                      * self.optimizer_overhead_factor
                                      for w in l_w)
        return optimizer_states_mem + optimizer_overhead*with_overhead

    def prefill(self):
        for t in range(self.T):
            for k in range(self.T):                
                for w in range(self.W):
                    if k not in self.krange(t):
                        self.PrfW[t, k, w] = 0
                        self.OflW[t, k, w] = 0
                    
                    # fwd_i, bwd_i = self.param2hcn[w]
                    fwd_i = min(self.param2hcn[w])
                    bwd_i = max(self.param2hcn[w])
                    if k not in self.krange(t) or (t>fwd_i and t<bwd_i):
                        self.OptC[t, k, w] = 0

        self.sumOptC = dict()
        for w in self.param2hcn:
            self.sumOptC[w] = lpSum(self.OptC[t,k,w] for t in range(self.T) for k in self.krange(t))

        if not self.gradient_accumulation:
            for k in self.PrfG:
                self.PrfG[k] = 0

        if not self.with_optimizer_states:
            for (t,k,w) in self.AliveO:
                self.AliveO[(t,k,w)] = (1-self.accumC_optimizer_states(w)- self.param_multiplier)
                self.PrfO[(t,k,w)] = 0
                self.OflO[(t,k,w)] = 0
        if self.grad_mode in ["free"]:
            # If gradient is freed ASAP, OflG will be merged into OflW
            for (t,k,w) in self.OflG:
                self.OflG[(t,k,w)] = 0
            for (t,k,w) in self.AliveG:
                grad_size = self.parameter_gradient_size[w]
                if len(self.param2hcn[w]) <= 2:
                    self.AliveG[(t,k,w)] = 0
                    if k == max(self.param2hcn[w]):
                        # self.overhead[k] = [v+grad_size for v in self.overhead[k]]
                        self.param_grad_mem[t,k] += grad_size * self.req_w()
                else:#shared weight
                    bwd_first = min(x for x in self.param2hcn[w] if x>self.loss_idx)
                    bwd_last = max(self.param2hcn[w])
                    if t<=bwd_first or t>bwd_last:#assume single bwd
                        self.AliveG[(t,k,w)] = 0
                    else:
                        self.AliveG[(t,k,w)] = 1
                        if k in self.param2hcn[w] and k>bwd_first:
                            # self.overhead[k] = [v+grad_size for v in self.overhead[k]]
                            self.param_grad_mem[t,k] += grad_size * self.req_w()
        elif self.grad_mode in ["accumulate"]:
            for (t,k,w) in self.AliveG:
                self.AliveG[(t,k,w)] = 1
                # if k == max(self.param2hcn[w]):
                #     self.overhead[k] = [v+grad_size for v in self.overhead[k]]
                # TODO: add offload gradient variables for gradient accumulation


    def add_offload_constraints(self):
        # if with_grad, AliveG is a variable
        # if with_optimizer_states, AliveO is a variable
        self.OflWProg = dict()
        self.OflGProg = dict()
        self.OptCProg = dict()
        self.PrfWProg = dict()
        self.PrfGProg = dict()
        self.OflOProg = dict()
        self.PrfOProg = dict()
        
        def get_progress(op, t, k, w):
            bwd_i = max(self.param2hcn[w])
            if bwd_i < t:  # after bwd of w
                progress = lpSum(
                    op[t, kk, w] for kk in self.krange(t) if kk < k
                ) + lpSum(
                    op[tt, kk, w]
                    for tt in range(bwd_i, t)  # offload right after bwd_i
                    for kk in self.krange(tt)
                )
            else:
                progress = (
                    lpSum(op[t, kk, w] for kk in self.krange(t) if kk < k)
                    + lpSum(op[tt, kk, w] for tt in range(t) for kk in self.krange(tt))
                    + lpSum(
                        op[tt, kk, w]
                        for tt in range(bwd_i, self.T)
                        for kk in self.krange(tt)
                    )
                )
            return progress

        for t in range(self.T):
            for k in self.krange(t):
                t_, k_ = self.next_index(t, k)

                self.md += self.Time[t, k] >= 1/ self.bandwidthOfl *lpSum(
                    self.parameter_size[w] 
                    * self.PrfW[t, k, w]
                       +self.parameter_gradient_size[w] *
                       self.optimizer_states_factor*self.PrfO[t,k,w]
                    for w in range(self.W))
                self.md += self.Time[t, k] >= (1/ self.bandwidthOfl *lpSum(
                    self.parameter_size[w] 
                    * self.OflW[t, k, w]
                       +self.parameter_gradient_size[w] *
                       (self.OflG[t, k, w]+
                        self.optimizer_states_factor*self.OflO[t,k,w])
                    for w in range(self.W))
                    + lpSum(self.parameter_gradient_size[w]
                    / self.cpu_optimize_speed*self.gcd
                    * self.OptC[t, k, w]
                    for w in self.hcn2param[k]))
                self.md += self.Time[t, k] >= (lpSum(self.Comp[t, k, o] * self.time[k][o] 
                                                     for o in range(self.nR[k]))
                + 1/ self.bandwidthOfl *lpSum(
                    self.parameter_size[w] 
                    * self.OflW[t, k, w]
                       +self.parameter_gradient_size[w]
                       * (self.OflG[t, k, w]
                    + self.optimizer_states_factor*self.OflO[t,k,w])
                    for w in self.hcn2param[k])# current layer offload
                    + 1/self.gpu_optimize_speed*self.gcd
                    * lpSum(self.parameter_gradient_size[w]
                    * (self.req_w() - self.sumOptC[w])
                    for w in self.hcn2param[k]
                    ))
                self.md += self.Time[t, k] >= lpSum(
                    self.parameter_size[w] 
                    / self.cpu_optimize_speed*self.gcd
                    * self.OptC[t, k, w]
                    for w in range(self.W)
                )
                self.param_grad_mem[t,k] += lpSum(
                                self.AliveG[t, k, w]
                                * self.parameter_gradient_size[w]
                                for w in range(self.W)
                            )
                for w in range(self.W):
                    self.PrfWProg[t,k,w] = get_progress(self.PrfW, t, k, w)
                    self.PrfGProg[t,k,w] = get_progress(self.PrfG, t, k, w)
                    self.OflWProg[t,k,w] = get_progress(self.OflW, t, k, w)
                    self.OflGProg[t,k,w] = get_progress(self.OflG, t, k, w)
                    self.OflOProg[t,k,w] = get_progress(self.OflO, t, k, w)
                    self.PrfOProg[t,k,w] = get_progress(self.PrfO, t, k, w)
                    self.OptCProg[t,k,w] = get_progress(self.OptC, t, k, w)
                    
                    self.md += (self.AliveW[t, k, w] <= self.req_w())
                    self.md += self.OflWProg[t, k, w] <= self.req_w()
                    self.md += self.OflGProg[t, k, w] <= self.accumC_grad(w)
                    self.md += self.OptCProg[t, k, w] <= self.max_OflGProg(t,k,w)
                    self.md += self.OptCProg[t, k, w] <= self.PrfWProg[t, k, w]*self.w_by_wg(w)
                    self.md += (self.AliveW[t, k, w] + self.OflWProg[t, k, w]
                                >= self.instant_opt(w))
                    self.md += (self.AliveG[t, k, w] + self.OflGProg[t, k, w] 
                                >= self.req_w() - self.instant_opt(w))
                    self.md += (self.AliveW[t_, k_, w] + self.PrfW[t_, k_, w] <= 
                                self.req_w()
                                + (self.OptCProg[t, k, w] - self.sumOptC[w])
                                *self.parameter_gradient_size[w]/self.parameter_size[w]
                                # size that not yet finished updating cannot be prefetched
                                 )
                    diffW = self.AliveW[t_, k_, w] - self.AliveW[t, k, w]
                    self.md += diffW <= self.PrfW[t, k, w]

                    diffG = self.AliveG[t_, k_, w] - self.AliveG[t, k, w]
                    self.md += (diffG <= 1*(k in self.param2hcn[w] 
                                           and k>self.loss_idx)#backward operations
                                           +self.PrfG[t, k, w])

                    self.md += self.AliveO[t_, k_, w] - self.AliveO[t, k, w] <= self.PrfO[t,k,w]
                    self.md += self.OflOProg[t, k, w] >= self.PrfOProg[t, k, w]
                    self.md += (self.AliveO[t, k, w] + self.OflOProg[t, k, w]
                                >= self.req_w() - self.sumOptC[w])
                    self.md += (self.OflOProg[t, k, w]
                                <= self.req_w() - self.sumOptC[w])
            
        for w in self.param2hcn:
            fwd_i = min(self.param2hcn[w])
            bwd_i = max(self.param2hcn[w])
            self.md += self.PrfWProg[fwd_i,fwd_i,w] >= self.sumOptC[w]
            self.md += self.req_w() - self.sumOptC[w] <= self.AliveO[bwd_i, bwd_i, w]
            if self.gradient_accumulation:
                self.md += self.OflGProg[bwd_i, bwd_i, w] == self.accumC_grad(w)
                self.md += self.PrfGProg[bwd_i, bwd_i, w] == self.accumC_grad(w) - self.sumOptC[w]
            t_, k_ = self.next_index(bwd_i, bwd_i)
            # self.md += self.AliveO[t_, k_, w] - self.AliveO[bwd_i, bwd_i, w] <= 1 - self.gradient_accumulation
            # self.md += self.AliveG[t_, k_, w] - self.AliveG[bwd_i, bwd_i, w] <= 1# - self.gradient_accumulation
            for t in range(self.T):
                for k in self.param2hcn[w]:
                    if k not in self.krange(t):
                        continue
                    self.md += self.sumComp[t, k] -1 <= self.AliveW[t, k, w] - self.req_w()

    def solve(self, solver=""):
        # some solvers have no support of 'Time limit reached' status
        # if self.with_parameters:
        #     self.add_single_fwd_constraints()
        #     self.add_single_bwd_constraints()
        print(f"time limit {self.ilp_solver_params['TimeLimit']}")
        try:
            solver = get_solver(
                solver, msg=0, timeLimit=self.ilp_solver_params["TimeLimit"]
            )
        except:
            avail_solver = listSolvers(onlyAvailable=True)[0]
            #     print(f"Cannot get {solver}, will use {avail_solver}")
            solver = get_solver(
                avail_solver, msg=0, timeLimit=self.ilp_solver_params["TimeLimit"]
            )
        # print("start solving")
        last_time = time.time()
        status = self.md.solve(solver)
        time_taken = time.time() - last_time
        clean_time_taken = time.strftime("%H:%M:%S", time.gmtime(time_taken))
        self.solving_time = clean_time_taken
        self.status = LpStatus[status]  # readable status
        self.feasible = status == 1

        sol = self.sol
        if self.feasible:
            print(f"finished solving in {self.md.solutionTime}")
            print(f"objective {self.md.objective.value()}")
            self.solve_time = self.md.solutionTime
            self.active_steps = []
            for t in list(range(self.loss_idx + 1, self.T)) + list(
                range(self.loss_idx + 1)
            ):
                for k in self.krange(t):
                    if not sol(self.sumComp[t, k]):
                        continue
                    self.active_steps.append((t, k))

    def sol(self, value):
        if hasattr(value, "value"):
            return value.value() > 0.9999
        return value > 0.9999

    def group(self, w, tol=1):
        # Group the parameters of each block for the task
        fwd_i = min(self.param2hcn[w])
        bwd_i = max(self.param2hcn[w])
        early_fwd = []
        for t in range(bwd_i, self.T):
            if not self.single_fwd and self.sol(self.sumComp[t,fwd_i]):
                early_fwd.append(t)#if recompute fwd after bwd
        hcn = self.hgraph.list_HCNs[fwd_i]
        parameters = {pnode.param_name: pnode for pnode in self.parameters[w]}
        parameter_size = sum(pnode.mem for pnode in parameters.values())

        Alive = {p: 1 for p in parameters.keys()}
        Offloaded = {p: False for p in parameters.keys()}

        ofl_ops = []
        prf_ops = []
        del_ops = []
        opt_ops = []
        init_ops = []
        restore_ops = []
        init_alive = []
        cpu_optimize_candidates = {p: 0 for p in parameters.keys()}

        def apply_cpu_optimize(p):
            for (t,k,op) in ofl_ops:
                if op.target.name == p:
                    op.grad = True
                    break
            # for (t,k,op) in del_ops:
            #     if op.target.name == p:
            #         op.grad = True
            del_ops.append((t,k,DeleteOp(Parameter(parameters[p], is_grad=True))))
            i = self.active_steps.index((t,k))+1# TODO: distribute cpu optimization based on time
            p_alloc = Parameter(parameters[p])
            op = OptimizeOp(list_params=[p], cpu=True, alloc=p_alloc,
                            time=parameters[p].mem/self.cpu_optimize_speed,
                            )
            opt_ops.append((*self.active_steps[i], op))
            self.cpu_optimized_steps[self.active_steps[i]].append(p)
            # del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p]))))

            #if cpu optimize, do not keep w after bwd
        def apply_gpu_optimize(p):
            p_alloc = Parameter(parameters[p])
            op = OptimizeOp(list_params=[p], alloc=p_alloc,
                            time = parameters[p].mem/self.gpu_optimize_speed,
                            overhead=parameters[p].mem*self.optimizer_overhead_factor)
            opt_ops.append((bwd_i, bwd_i, op))# optimize after bwd
            del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p], is_grad=True))))
            

        assert (bwd_i, bwd_i) in self.active_steps
        idx = self.active_steps.index((bwd_i, bwd_i))
        for t, k in self.active_steps[idx:] + self.active_steps[:idx]:
            t_, k_ = self.next_index(t, k)
            current_alive_size = sum(parameters[p].mem * a for p, a in Alive.items())
            current_offloaded_size = sum(parameters[p].mem * a for p, a in Offloaded.items())
            next_alive_size = round((self.AliveG[(t_, k_, w)]+self.AliveW[(t_, k_, w)]).value() * parameter_size)
            next_offloaded_size = round((self.OflGProg[(t_, k_, w)]+self.OflWProg[(t_, k_, w)]).value() * parameter_size)
            
            # assert current_alive_size <= round(self.AliveW[(t, k, w)].value() * parameter_size)

            if (t, k) == (0, 0):  # init
                for p, a in Alive.items():
                    if a:
                        p_alloc = Parameter(parameters[p])
                        init_ops.append((t, k, AllocateOp(p_alloc)))
                        init_alive.append(p_alloc)
                        op = PrefetchOp(
                            alloc=p_alloc, indices=(0, None), 
                            time=parameters[p].mem/self.bandwidthPrf/self.gcd
                        )
                        init_ops.append((t, k, op))
                        op = OffloadOp(
                            alloc=p_alloc, indices=(0, None),
                            time=parameters[p].mem/self.bandwidthOfl/self.gcd
                        )
                        restore_ops.append((t, k, op))
                        restore_ops.append((t, k, DeleteOp(p_alloc)))

            if next_offloaded_size > current_offloaded_size:
                # print(t,k, next_offloaded_size, current_offloaded_size)
                ofl_size = next_offloaded_size - current_offloaded_size
                candidates = {
                    p: parameters[p].mem * (1 - o)
                    for p, o in Offloaded.items()
                    if o < 1
                }
                if not candidates:
                    if ofl_size<1024:
                        ofl_size = 0
                    else:
                        raise ValueError
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select_size(ofl_size)
                # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
                #     pass
                for p in select_paras:
                    op = OffloadOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                   time=parameters[p].mem/self.bandwidthOfl/self.gcd)
                    ofl_ops.append((t, k, op))
                    Offloaded[p] = 1

            if current_alive_size > next_alive_size:
                del_size = current_alive_size - next_alive_size
                candidates = {}
                for p, o in Offloaded.items():
                    if Alive[p]>0 and o>0:
                        candidates[p] = min(o, Alive[p])*parameters[p].mem
                if not candidates:
                    if del_size<1024:
                        del_size = 0
                    else:
                        raise ValueError
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select_size(del_size)
                # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
                #     pass
                for p in select_paras:
                    del_ops.append((t, k, DeleteOp(Parameter(parameters[p]))))
                    Alive[p] = 0
            if current_alive_size < next_alive_size:
                # prefetch should be smaller than solution
                prf_size = next_alive_size - current_alive_size
                candidates = {
                    p: parameters[p].mem * (1 - a) for p, a in Alive.items() if a < 1
                }
                if not candidates:
                    if prf_size<1024:
                        prf_size=0
                    else:
                        raise ValueError
                if self.sol(self.AliveW[(t_, k_, w)]):
                    select_paras = list(candidates.keys())
                    # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
                else:
                    selector = knapsack(list(candidates.items()))
                    unselect_paras = selector.select_size(sum(candidates.values()) - prf_size)
                    
                    select_paras = [
                        p for p in candidates.keys() if p not in unselect_paras
                    ]
                    # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
                #     pass
                for p in select_paras:
                    prf_ops.append((t, k, AllocateOp(Parameter(parameters[p]))))
                    op = PrefetchOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                    time=parameters[p].mem/self.bandwidthPrf/self.gcd)
                    prf_ops.append((t, k, op))
                    Alive[p] = 1
                    if (t > bwd_i and t < min(early_fwd + [self.T + 1])) or t < fwd_i:
                        # cpu optimize only if prefetch before fwd
                        if parameters[p].info.requires_grad:
                            # only trainable parameters will be optimize candidate
                            cpu_optimize_candidates[p] = 1
        
        candidates = {
                    p: parameters[p].mem * a for p, a in cpu_optimize_candidates.items() if a >0
                }
        select_paras = []
        # assert sum(candidates.values())/parameter_size >= self.sumOptC[w].value()-0.01
        
            # cpu_optimize_size = self.sumOptC[w].value()*parameter_size# size by subgraph
        if isinstance(self.param_multiplier, float):
            multiplier = self.param_multiplier
        else:
            multiplier = self.param_multiplier.value()
        # cpu_optimize_size = (sum(self.sumOptC[w_].value() * 
        #                         self.parameter_gradient_size[w_] *self.gcd
        #                         for w_ in range(w, self.W)) / (1-multiplier)
        #                     - sum(self.cpu_optimized_params.values()))# size by all graphs
        cpu_optimize_size = min(sum(candidates.values()),
                                (self.sumOptC[w].value() * 
                                self.parameter_gradient_size[w] *self.gcd
                             ) / (1-multiplier))
        
        if candidates and cpu_optimize_size>0:
            # print(candidates, cpu_optimize_size)
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(cpu_optimize_size)
            if cpu_optimize_size>sum(candidates[p] for p in select_paras):
                raise ValueError
            # print(select_paras)
            
        # Optimize parameters which requires grad
        gpu_optimze_param = []
        for p, pnode in parameters.items():
            if not pnode.info.requires_grad:continue
            if p in select_paras:
                self.cpu_optimized_params[p] = parameters[p].mem
                apply_cpu_optimize(p)
            else:
                apply_gpu_optimize(p)
                gpu_optimze_param.append(pnode)
        if self.with_optimizer_states and gpu_optimze_param:
            ofl_ops_os, prf_ops_os, del_ops_os, init_alive_os = self.group_optimizer_states(w, gpu_optimze_param)
            ofl_ops += ofl_ops_os
            prf_ops += prf_ops_os
            del_ops += del_ops_os
            init_alive += init_alive_os
        return ofl_ops, prf_ops, del_ops, opt_ops, init_ops, restore_ops, init_alive

    def group_optimizer_states(self, w, gpu_optimize_param):
        # To offload and prefetch optimizer states witin the gpu_optimize_param
        ofl_ops = []
        prf_ops = []
        del_ops = []
        init_alive = []
        fwd_i = min(self.param2hcn[w])
        bwd_i = max(self.param2hcn[w])
        hcn = self.hgraph.list_HCNs[fwd_i]
        parameters = {pnode.param_name: pnode for pnode in self.parameters[w] if pnode.requires_grad}
        parameter_size = sum(pnode.mem for pnode in parameters.values())
        gpu_optimize_size = sum(pnode.mem for pnode in gpu_optimize_param)

        Alive = {pnode.param_name: 1 for pnode in gpu_optimize_param}
        Offloaded = {pnode.param_name: False for pnode in gpu_optimize_param}
        assert (bwd_i, bwd_i) in self.active_steps
        idx = self.active_steps.index((bwd_i, bwd_i))
        for t, k in self.active_steps[idx:] + self.active_steps[:idx]:
            if (t, k) == (0, 0):  # init
                for p, a in Alive.items():
                    if a:
                        init_alive.append(Parameter(parameters[p],is_optim_states=True))

            t_, k_ = self.next_index(t, k)
            current_alive_size = sum(parameters[p].mem * a for p, a in Alive.items())
            current_offloaded_size = sum(parameters[p].mem * a for p, a in Offloaded.items())
            next_alive_size = min(gpu_optimize_size,
                                  round((self.AliveO[(t_, k_, w)]).value() * parameter_size))
            next_offloaded_size = min(gpu_optimize_size,
                round((self.OflOProg[(t_, k_, w)]).value() * parameter_size))
            if parameter_size * (1-self.sumOptC[w]).value()<gpu_optimize_size:
                next_offloaded_size += gpu_optimize_size - parameter_size * (1-self.sumOptC[w]).value()

            # assert current_alive_size <= round(self.AliveW[(t, k, w)].value() * parameter_size)
            if next_offloaded_size > current_offloaded_size:
                # print(t,k, next_offloaded_size, current_offloaded_size)
                ofl_size = next_offloaded_size - current_offloaded_size
                candidates = {
                    p: parameters[p].mem * (1 - o)
                    for p, o in Offloaded.items()
                    if o < 1
                }
                if not candidates:
                    if ofl_size<1024:
                        ofl_size = 0
                    else:
                        raise ValueError
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select_size(ofl_size)
                # assert ofl_size==0 or sum(candidates[p] for p in select_paras)/ofl_size>0.99
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
                #     pass
                for p in select_paras:
                    op = OffloadOp(alloc=Parameter(parameters[p],
                                                   is_optim_states=True), indices=(0, None),
                                   time=parameters[p].mem/self.bandwidthOfl/self.gcd*self.optimizer_states_factor,
                                   is_optim_states=True)
                    ofl_ops.append((t, k, op))
                    Offloaded[p] = 1

            if current_alive_size > next_alive_size:
                if k_ ==bwd_i:continue
                del_size = current_alive_size - next_alive_size
                candidates = {}
                for p, o in Offloaded.items():
                    if Alive[p]>0 and o>0:
                        candidates[p] = min(o, Alive[p])*parameters[p].mem
                if not candidates:
                    if del_size<1024:
                        del_size = 0
                    else:
                        raise ValueError
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select_size(del_size)
                # assert del_size==0 or sum(candidates[p] for p in select_paras)/del_size>0.99
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
                #     pass
                for p in select_paras:
                    del_ops.append((t, k, DeleteOp(Parameter(parameters[p],
                                                             is_optim_states=True),
                                                   is_optim_states=True)))
                    Alive[p] = 0
            if current_alive_size < next_alive_size or k_==bwd_i:
                # if w == 15:print(self.active_steps[k_]==bwd_i)
                # prefetch should be smaller than solution
                prf_size = next_alive_size - current_alive_size
                candidates = {
                    p: parameters[p].mem * (1 - a) for p, a in Alive.items() if a < 1
                }
                if not candidates:
                    if prf_size<1024:
                        prf_size=0
                    else:
                        raise ValueError
                if self.sol(self.AliveO[(t_, k_, w)]+self.sumOptC[w]-self.req_w()+1) or k_==bwd_i:
                    
                    select_paras = list(candidates.keys())
                    # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size>0.99
                else:
                    selector = knapsack(list(candidates.items()))
                    unselect_paras = selector.select_size(sum(candidates.values()) - prf_size)
                    
                    select_paras = [
                        p for p in candidates.keys() if p not in unselect_paras
                    ]
                    # assert prf_size==0 or sum(candidates[p] for p in select_paras)/prf_size<1.01
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
                #     pass
                for p in select_paras:
                    prf_ops.append((t, k, AllocateOp(Parameter(parameters[p],
                                                               is_optim_states=True),
                                                     is_optim_states=True)))
                    op = PrefetchOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                    time=parameters[p].mem/self.bandwidthPrf/self.gcd*self.optimizer_states_factor,
                                    is_optim_states=True)
                    prf_ops.append((t, k, op))
                    Alive[p] = 1
            if k_==bwd_i:assert 0 not in Alive.values()

        return ofl_ops, prf_ops, del_ops, init_alive


    def schedule(self, hgraph=None, check_valid=False):
        """
        Given the solution from HILP, we want to translate the result
        to a OpSchedule that can be used in a higher level.
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"

        sol = self.sol

        T = len(hgraph.list_HCNs)
        I = len(hgraph.list_HANs)
        J = len(self.list_list_sched)
        init_op_list = []
        restore_op_list = []
        init_alive_status = {}
        loss_op = ComputeOp(self.hgraph.cluster.loss_cnode, disabled=True)
        if self.with_parameters:
            W = len(self.parameter_size)
            (
                op_list,
                init_alive_status,
                init_op_list,
                restore_op_list,
            ) = self.schedule_offload(hgraph)
        else:
            op_list = []
            
            for t in range(self.T):
                for k in self.krange(t):
                    if t == self.loss_idx and k == self.loss_idx:
                        op_list.append(loss_op)
                    op_list += self.schedule_compute(t,k,hgraph)
        
        # print("finish scheduling")
        for anode in self.hgraph.cluster.interfaces["input_data_anodes"]:
            init_alive_status[anode.name] = True  # anode share the name as alloc
        op_sched = OpSchedule(
            op_list,
            # ofl_list=ofl_list,
            # prf_list=prf_list,
            loss_idx=op_list.index(loss_op),
            cluster=self.hgraph.cluster,
            init_alive_status=init_alive_status,
            init_op_list=init_op_list,
            restore_op_list=restore_op_list,
            with_parameters=self.with_parameters,
        )
        # check_valid = True
        if check_valid:
            for op, alive_status in zip(op_sched.op_list, op_sched.alive_list):
                if op.is_del:
                    continue
                for kdn in op.kn.deps_real:
                    if not alive_status[kdn.name]:
                        print(f"Invalid sched found: try to run {op.kn} without {kdn}")
                        raise ValueError
        return op_sched
    
    def schedule_compute(self, t,k,hgraph):
        op_list = []
        sol = self.sol
        
        j = self.hcn2sub_c[k]
        # if self.sumComp[t, k].value() == 1:
        if sol(self.sumComp[t, k]):
            hcn = hgraph.list_HCNs[k]
            opt = -1
            for o in range(self.nOpts[k]):
                if sol(self.Comp[t, k, o]):
                    opt = o
                    break
            if opt > -1:
                h_obj = self.list_list_sched[j][opt]
                if hcn.is_fwd:
                    sub_op_list = h_obj.op_list[: h_obj.loss_idx]
                else:
                    sub_op_list = h_obj.op_list[h_obj.loss_idx + 1 :]

                    # if self.sumAliveP[(j, t + 1)].value() == 0:
                    # sub_op_list.append()
                sub_op_list = deepcopy(sub_op_list)

                if (
                    not hcn.is_fwd
                    # and self.sumAliveP[(j, t + 1)].value() > 0
                    and sol(self.sumAliveP[t + 1, j])
                ):  # phantoms should be kept
                    phantoms_to_keep = h_obj.phantoms
                    # for op in sub_op_list[::-1]:
                    #     if (
                    #         op.is_del
                    #         and not op.disabled
                    #         and op.kn in phantoms_to_keep
                    #     ):
                    #         # Only the last del should be disabled
                    #         op.disabled = True
                    #         phantoms_to_keep.remove(op.kn)
                # translating sub_op_list
                if (
                    hcn.sub_cluster
                    is not hcn.sub_cluster.representee_cluster
                ):
                    sub_op_list = translate(hcn.sub_cluster, sub_op_list)
            else:
                h_obj = hcn
                sub_op_list = deepcopy(h_obj.ff_op_list)

            op_list += sub_op_list

        for eidx, (k_, i) in enumerate(self.delete_list):
            # print(k_, i)
            # if k == k_ and self.delete[t, eidx].value()==1:
            if k == k_ and sol(self.delete[t, eidx]):
                han = hgraph.list_HANs[i]
                op_list.append(DeleteOp(Activation(han.anode)))
        return op_list

    def schedule_offload(self, hgraph=None):
        """
        V1: self.grouping = False:
        merge every cluster, ofl/prf/del partially, high memory overhead
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(hgraph.list_HCNs)
        I = len(hgraph.list_HANs)
        J = len(self.list_list_sched)
        W = len(self.parameter_size)

        ### Handle multiplier
        self.params_vars = [self.AliveW, self.OflWProg, self.OflW, 
                            self.PrfW, self.PrfWProg, self.OptC,
                            self.AliveG, self.OflGProg, self.OflG,
                            self.PrfG, self.PrfGProg, self.AliveO,
                            self.OflO, self.OflOProg, self.PrfO, 
                            self.PrfOProg
                            ]
        if isinstance(self.param_multiplier, float):
            multiplier = self.param_multiplier
        else:
            multiplier = self.param_multiplier.value()
        for p in self.params_vars:
            for k,v in p.items():
                p[k] = v*1/ (1-multiplier)
                pass

        self.ofl_ops = []
        self.prf_ops = []
        self.del_ops = []
        self.opt_ops = []
        self.cpu_optimized_params = {}
        self.cpu_optimized_steps = {step:[] for step in self.active_steps}
        init_op_list = []
        restore_op_list = []
        init_alive_status = dict()
        if self.grouping:
            for w in range(self.W)[::-1]:
                o_l, p_l, d_l, t_l, i_l, r_l, init_alive = self.group(w)
                self.ofl_ops.extend(o_l)
                self.prf_ops.extend(p_l)
                self.del_ops.extend(d_l)
                self.opt_ops.extend(t_l)
                init_op_list.extend([ops[2] for ops in i_l])
                restore_op_list.extend([ops[2] for ops in r_l])
                for alloc in init_alive:
                    init_alive_status[alloc.name] = True
        # else:
        #     init_op_list = self.schedule_init_op_list()

        sol = self.sol
        # offload_buffers = {w:[] for w in range(W)}
        op_list = []
        
        # for op in init_op_list:
        #     if isinstance(op, AllocateOp):
        #         init_alive_status[op.target] = True
        
        for t in range(self.T):
            for k in self.krange(t):
                op_list.append(SynchronizeOp(f"{(t,k)}"))
                if t == self.loss_idx and k == self.loss_idx:
                    # loss_idx = len(op_list)
                    # loss_op = Op(K_C_node("loss"))

                    op_list.append(ComputeOp(self.hgraph.cluster.loss_cnode, disabled=True))
                if not sol(self.sumComp[t, k]):
                    continue
                j = self.hcn2sub_c[k]
                # if self.sumComp[t, k].value() == 1:
                prefetch_list = []
                for w in range(W):
                    prefetch_ops = self.create_prefetch_ops(t, k, w)
                    op_list.extend(prefetch_ops[0])
                    prefetch_list.extend(prefetch_ops[1])
                op_list += self.schedule_compute(t,k,hgraph)
                wait_op_1 = []
                wait_op_2 = []
                wait_op_3 = []
                for w in range(W):
                    if k in self.param2hcn[w]:
                        wait_op_1.extend(self.create_optimize_ops(t, k, w))
                        wait_op_2.extend(self.create_offload_ops(t, k, w))
                        wait_op_3.extend(self.create_delete_ops(t, k, w))
                        # op_list.extend(self.create_prefetch_ops(t,k,w))
                    else:
                        op_list.extend(self.create_offload_ops(t, k, w))
                        op_list.extend(self.create_delete_ops(t, k, w))
                        op_list.extend(self.create_optimize_ops(t, k, w))
                if wait_op_1:# for the current layer, need to synchronize first
                    op_list.extend([SynchronizeOp(str(k))]+wait_op_1)
                if wait_op_2:# for the current layer, need to synchronize first
                    op_list.extend([SynchronizeOp(str(k))]+wait_op_2)
                if wait_op_3:# for the current layer, need to synchronize first
                    op_list.extend([SynchronizeOp(str(k))]+wait_op_3)

                op_list.extend(prefetch_list)
        return op_list, init_alive_status, init_op_list, restore_op_list

    def create_optimize_ops(self, t, k, w, itemsize=4):
        op_list = []
        # sub_cluster = self.hgraph.list_HCNs[min(self.param2hcn[w])].sub_cluster
        if self.grouping:
            for t_, k_, op in self.opt_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.pnode.param_name in [k.param_name for k in self.parameters[w]]
                ):
                    op_list.append(op)
            return op_list

    def create_delete_ops(self, t, k, w, itemsize=4):
        op_list = []
        sub_cluster = self.hgraph.list_HCNs[min(self.param2hcn[w])].sub_cluster
        if self.grouping:
            for t_, k_, op in self.del_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.pnode.param_name in [k.param_name for k in self.parameters[w]]
                ):
                    op_list.append(op)
            return op_list

    def create_prefetch_ops(self, t, k, w, itemsize=4):
        pre_op_list = []
        post_op_list = []
        sub_cluster = self.hgraph.list_HCNs[min(self.param2hcn[w])].sub_cluster

        if self.grouping:
            for t_, k_, op in self.prf_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.pnode.param_name in [k.param_name for k in self.parameters[w]]
                ):
                    pre_op_list.append(op)

            return pre_op_list, post_op_list

    def create_offload_ops(self, t, k, w, itemsize=4):
        op_list = []
        sub_cluster = self.hgraph.list_HCNs[min(self.param2hcn[w])].sub_cluster
        if self.grouping:
            for t_, k_, op in self.ofl_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.pnode.param_name in [k.param_name for k in self.parameters[w]]
                ):
                    op_list.append(op)
            return op_list

    def add_boundary_constraint(self):
        self.md += (
            lpSum(self.sumComp[t, k] for t in range(self.T) for k in range(t + 1, self.T)) == 0
        )
        self.md += (
            lpSum(
                self.sumAliveP[(self.hcn2sub_c[i], t)]
                for t in range(self.loss_idx)
                for i in range(t + 1, self.loss_idx)
                if self.hcn2sub_c[i]
            )
            == 0
        )
        self.md += (
            lpSum(
                self.AliveA[t, j]
                for j in range(self.Cr)
                for t in range(self.create_list[j][0] + 1)
            )
            == 0
        )
        for i in range(self.I):
            if self.han_deps[i]:
                self.md += (
                    lpSum(self.AliveT[t, i] for t in range(min(self.han_deps[i]) + 1)) == 0
                )