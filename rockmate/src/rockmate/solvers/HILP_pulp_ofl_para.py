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

from .HILP_scheduler import schedule
from rkgb.core.hierarchical import HierarchicalGraph

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
        lowBound=0,
        upBound=1,
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
        optimize_metrics = None,
        activation_offload = False,
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
        self.optimize_metrics = optimize_metrics
        self.activation_offload = activation_offload
        self.protected_names = protected_names
        self.hgraph = hgraph
        #############################
        self.config()

        self.add_variables()
        if self.with_parameters:
            self.add_offload_variables()

        self.add_objective()

        print("adding constraints")
        self.add_constraints()
        
    def config(self):
        self.hcn2sub_c = []
        self.list_list_sched = []
        self.sub_clusters = []
        self.nSched = []  # number of fwd-bwd schedule
        self.nComp = []  # number to run compute, =nSched if bwd, =nSched + 1 if fwd
        self.time = []
        self.overhead = []
        self.bin_type = "Binary"

        for i, hcn in enumerate(self.hgraph.list_HCNs):
            if "Loss" in hcn.name:
                self.loss_idx = i
            if hcn.sub_cluster is None:
                # only when hcn is fwd with requires_grad=False
                self.hcn2sub_c.append(None)
                self.nComp.append(1)
                self.nSched.append(0)
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
                self.nComp.append(len(list_sched) + (1 if hcn.is_fwd else 0))
                self.nSched.append(len(list_sched))

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

        self.T = len(self.hgraph.list_HCNs)
        self.I = len(self.hgraph.list_HANs)
        self.J = len(self.list_list_sched)

        self.hcn2idx = {hcn:i for i, hcn in enumerate(self.hgraph.list_HCNs)}
        self.han2idx = {hcn:i for i, hcn in enumerate(self.hgraph.list_HANs)}
        
        self.protected_indices = []
        self.input_grad_indices = []
        self.input_data_indices = []
        self.han_deps = []
        self.han_users = []

        for i, han in enumerate(self.hgraph.list_HANs):
            if han.anode.name in self.protected_names:
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
                    for i in range(self.I):
                        if (anode.name
                            == self.hgraph.list_HANs[i].anode.name
                            and k not in self.han_users[i]
                        ):
                            self.han_users[i].append(k)

        ##############################

        self.create_list = [(k, i) for k in range(self.T) for i in self.hcn_users[k]]
        self.delete_list = [(k, i) for i in range(self.I) for k in self.han_deps[i] + self.han_users[i]]

        self.Cr = len(self.create_list)
        self.De = len(self.delete_list)
        self.W = len(self.sub_clusters)

    def krange(self, t):
        if self.single_fwd:
            return [t]
        elif self.single_bwd and t > self.loss_idx:
            return list(range(self.loss_idx)) + [t]
        return list(range(t + 1))
        # return range(self.T)

    def crange(self, t):
        """
        Concerning range for computation
        """
        return self.krange(t)
    
    def orange(self, t):
        """
        Concerning range for offloading
        """
        return self.krange(t)

    def next_idx(self, t, i, upper_triangle=False):
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


    def add_objective(self, bandwidth_cost=0.01):
        # define objective function
        prf_cost = (
            bandwidth_cost
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
            bandwidth_cost
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

    def add_constraints(self):
        self.add_valid_constraints()
        self.add_memory_constrains()
        if self.with_parameters:
            self.add_offload_constraints()
        ##### Time constraints
        for t in range(self.T):
            for k in self.krange(t):
                if self.with_parameters:
                    ofl_time = lpSum(
                        self.parameter_size[w] / self.bandwidthOfl * self.OflW[t, k, w]
                        for w in self.hcn2param[k]
                    )
                else:
                    ofl_time = 0
                self.md += (
                    self.Time[t, k]
                    >= lpSum(
                        self.Comp[t, k, o] * self.time[k][o] for o in range(self.nComp[k])
                    )
                    + ofl_time
                )


    def add_variables(self):
        self.Comp = RkLpVariable.dicts(
            f"Comp",
            [
                (t, k, o)
                for t in range(self.T)
                for k in self.krange(t)
                for o in range(self.nComp[k])
            ],
            cat=self.bin_type,
        )

        self.sumComp = {}
        for t in range(self.T):
            for k in self.krange(t):
                self.sumComp[t, k] = lpSum(
                    self.Comp[t, k, o] for o in range(self.nComp[k])
                )

        for t in range(self.T):
            for k in range(self.T):
                if k not in self.krange(t):
                    for o in range(self.nComp[k]):
                        self.Comp[t, k, o] = 0
                    self.sumComp[t, k] = 0

        self.AliveP = RkLpVariable.dicts(
            f"AliveP",
            [
                (t, j, o)
                for t in range(self.T + 1)
                for j, list_sched in enumerate(self.list_list_sched)
                for o in range(len(list_sched))
                # if t-1 in self.sub_c2hcn[j]
            ],
            cat=self.bin_type,
        )
        
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
                       ], 
                       cat=self.bin_type
        )  # activation

        self.AliveT = RkLpVariable.dicts(
            "AliveT", [(t, i)
                       for t in range(self.T) 
                       for i in range(self.I)
                       ], 
                       cat=self.bin_type
        )  # tensor that can be shared by acts

        self.Create = RkLpVariable.dicts(
            "Create",
            [
                (t, i)
                for t in range(self.T)
                for i in range(self.Cr)
                if self.create_list[i][0] in self.krange(t)
            ],
            cat=self.bin_type,
        )
        self.Delete = RkLpVariable.dicts(
            "Delete",
            [
                (t, i)
                for t in range(self.T)
                for i in range(self.De)
                if self.delete_list[i][0] in self.krange(t)
            ],
            cat=self.bin_type,
        )
        
        self.Time = RkLpVariable.dicts(
            "Time", [(t, k) for t in range(self.T) for k in self.krange(t)], cat="Continuous"
        )
        self.prefill_compute()

    def add_offload_variables(self):
        optimize_metrics = self.optimize_metrics

        self.cpu_optimize = True
        self.optimizer_states_factor = optimize_metrics["optimizer_states_size"]#*weight size
        self.cpu_optimize_speed = optimize_metrics["cpu_optimize_speed"]#B/ms
        self.gpu_optimize_speed = optimize_metrics["gpu_optimize_speed"]#B/ms
        self.optimizer_overhead_factor = optimize_metrics["optimizer_overhead"]#*weight size
        self.minor_param_size = optimize_metrics["minor_param_size"]# minor weight size
        self.bandwidth = optimize_metrics["bandwidth"]# bandwidth
        batch_multiplier = 4
        # self.BatMpl = RkLpVariable("BMpl", lowBound=0, upBound=self.batch_multiplier, cat="Integer")
        # self.param_multiplier = 1-self.BatMpl*1/self.batch_multiplier
        self.param_multiplier = RkLpVariable("BMpl", lowBound=0, upBound=1-1/batch_multiplier, cat="Continuous")
        self.param_multiplier = 0.

        self.param2hcn = dict()
        self.parameters = []
        for k,v in self.hgraph.parameter_groups.items():
            self.param2hcn[len(self.parameters)] = k
            self.parameters.append(v)

        self.hcn2param = {t:[] for t in range(self.T)}

        for p,hcn_s in self.param2hcn.items():
            for hcn in hcn_s:
                self.hcn2param[hcn].append(p)

        self.parameter_size = [sum(pnode.mem for pnode in p)/self.gcd for p in self.parameters]
        self.parameter_gradient_size = [sum(pnode.mem for pnode in p if pnode.info.requires_grad
                                            )/self.gcd for p in self.parameters]
        self.W = W = len(self.parameters)

        ofl_idx = [(t, k, w) for t in range(self.T) for k in self.krange(t) for w in range(W)] 
        self.AliveW = RkLpVariable.dicts("AliveW", ofl_idx)  # parameter w is alive at the start of step j.
        self.AliveG = RkLpVariable.dicts("AliveG", ofl_idx)  # w.grad is alive at the start of step j.
        self.AliveO = RkLpVariable.dicts("AliveO", ofl_idx)  # w.grad is alive at the start of step j.
        self.OflW = RkLpVariable.dicts("OflW", ofl_idx)# Offload weight
        self.OflG = RkLpVariable.dicts("OflG", ofl_idx)# Offload gradient
        self.PrfW = RkLpVariable.dicts("PrfW", ofl_idx)# Prefetch gradient
        self.PrfG = RkLpVariable.dicts("PrfG", ofl_idx)# Prefetch gradient
        self.OptC = RkLpVariable.dicts("OptC", ofl_idx)# Optimize on cpu
        self.OflO = RkLpVariable.dicts("OflO", ofl_idx)# Offload optimizer states
        self.PrfO = RkLpVariable.dicts("PrfO", ofl_idx)# Prefetch optimizer states
        
        self.param_grad_mem = {(t,k):0 for t in range(self.T) for k in self.krange(t)}
        self.prefill_offload()

        self.bandwidthOfl = optimize_metrics["bandwidth"]/self.gcd  # byte/ms
        self.bandwidthPrf = optimize_metrics["bandwidth"]/self.gcd  # byte/ms

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
            #         for o in range(self.nSched[bwd_i])
            #     )
            #     == 0
            # )
            # in the end of bwd, del every phantoms
            self.md += (
                lpSum(
                    (self.AliveP[self.T, j, o])  # - self.Comp[bwd_i][T - 1, o])
                    for o in range(self.nSched[bwd_i])
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

        #### Options-related constraints
        for t in range(self.T):
            for j in range(self.J):
                fwd_i = min(self.sub_c2hcn[j])
                bwd_i = max(self.sub_c2hcn[j])
                for o in range(self.nSched[fwd_i]):
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
                        # Tensor req_i is required by BWD
                        req_i = [han.anode.name for han in self.hgraph.list_HANs].index(
                            anode.name
                        )
                        # req_i = self.han2idx[anode]
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
                    self.Create[t, eidx_c]
                    for eidx_c, (k_, i_) in enumerate(self.create_list)
                    if i_ == i and k_ <= k
                )
                self.alive[(t, k, i)] -= lpSum(
                    self.Delete[t, eidx_d]
                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                    if i_ == i and k_ <= k
                )
                self.md += self.alive[(t, k, i)] >= 0
                self.md += self.alive[(t, k, i)] <= 1
                if (k, i) in self.create_list:
                    didx = self.delete_list.index((k, i))
                    self.md += (
                        self.alive[(t, k, i)] + self.Delete[t, didx]
                        >= self.sumComp[t, k]
                    )

            for eidx, (k, i) in enumerate(self.create_list):
                # if k not in self.krange(t):
                #     continue
                self.md += self.Create[t, eidx] <= self.sumComp[t, k]
                # self.md += self.Create[t, eidx] >= self.sumComp[t, k] - self.alive[(t,k,i)]
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
                self.md += 1 - self.Delete[t, eidx] <= _num_hazards(t, i, k)

        # don't delete if still needed
        for t in range(self.T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += _max_num_hazards(t, i, k) * (
                    1 - self.Delete[t, eidx]
                ) >= _num_hazards(t, i, k)
                if i in self.protected_indices:
                    self.md += self.Delete[t, eidx] == 0

        self.U = {}
        for t in range(self.T):
            self.U[t, 0] = (
                lpSum(self.AliveT[t, i] * self.mem[i] for i in range(self.I))
                + lpSum(
                    self.Create[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.create_list)
                    if k_ == 0 and k_ in self.krange(t)
                )
                - lpSum(
                    self.Delete[t, eidx] * self.mem[i]
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
                    for o in range(self.nSched[0])
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
                        self.Create[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.create_list)
                        if k_ == k and k_ in self.krange(t)
                    )
                    - lpSum(
                        self.Delete[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.delete_list)
                        if k_ == k and k_ in self.krange(t)
                    )
                )
                # if k < self.loss_idx:
                if self.hgraph.list_HCNs[k].is_fwd:
                    self.U[t, k] += lpSum(
                        self.Comp[t, k, o] * self.saved_mem[j][o]
                        for o in range(self.nSched[k])
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
                        for o in range(self.nSched[k])
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
                            for o in range(self.nComp[k])
                        )
                        + lpSum(
                            self.mem[i_] * self.Delete[t, eidx_d]
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
                                        not_kept_alive = self.Delete[t, eidx]
                                    else:  # when output_data is not deps, but we care about it
                                        # eidx = self.delete_list.index((k, i_))
                                        k_ = max([kk for kk in self.han_deps[i_] if kk < k])
                                        not_kept_alive = self.alive[(t, k_, i_)]
                                else:  # start status
                                    eidx = self.create_list.index((k, i_))
                                    not_kept_alive = self.Create[t, eidx]
                                correction_term += not_kept_alive * inter_mem
                            self.md += (
                                self.U[t, k]
                                + self.Comp[t, k, o] * overhead / self.gcd
                                + correction_term
                                + lpSum(
                                    self.mem[i_] * self.Delete[t, eidx_d]
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
                                    self.mem[i_] * self.Delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                                    if k == k_
                                )
                                <= self.peak_budget - parameter_mem
                            )
                if t == self.loss_idx and self.save_budget:
                    self.md += self.U[t, k] <= self.save_budget

    
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

    def prefill_compute(self):
        self.active_stages = dict()
        for i in range(self.I):
            self.active_stages[i] = []
            for t in range(self.T):
                for k in self.krange(t):
                    if k in self.han_deps[i]+self.han_users[i]:
                        self.active_stages[i].append(t)
            if not self.hgraph.list_HANs[i].deps:# src node
                self.active_stages[i].append(-1)

        for j, list_sched in enumerate(self.list_list_sched):
            for o in range(len(list_sched)):
                # if (0,j,o) not in self.AliveP:
                self.AliveP[0,j,o] = 0
                for t in range(1,self.T + 1):
                    # if (t,j,o) not in self.AliveP:
                    if not t-1 in self.sub_c2hcn[j]:
                        self.AliveP[t,j,o] = self.AliveP[t-1,j,o]

        for c, (k, i) in enumerate(self.create_list):
            self.AliveA[0,c] = 0
            for t in range(1, self.T):
                if not t-1 in self.active_stages[i]:
                    self.AliveA[t,c] = self.AliveA[t-1,c]

        for i in range(self.I):
            if not -1 in self.active_stages[i]:#src node
                self.AliveT[0,i] = 0
            for t in range(1, self.T):
                if not t-1 in self.active_stages[i]:
                    self.AliveT[t,i] = self.AliveT[t-1,i]
    
        for t in range(self.T):
            for i in range(self.Cr):
                if self.create_list[i][0] not in self.krange(t):
                    self.Create[t, i] = 0
            for i in range(self.De):
                if self.delete_list[i][0] not in self.krange(t):
                    self.Delete[t, i] = 0
            if self.single_fwd:
                for d in range(self.De):
                    (k, i) = self.delete_list[d]
                    if i in self.protected_indices:continue
                    if k == max(self.han_deps[i] + self.han_users[i]) and k == t:
                        self.Delete[t, d] = 1
                        pass
 

    def prefill_offload(self):
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
                t_, k_ = self.next_idx(t, k)

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
                                                     for o in range(self.nComp[k]))
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
            t_, k_ = self.next_idx(bwd_i, bwd_i)
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
    
    def schedule(self):
        # scheduler = Scheduler(self)
        return schedule(self)