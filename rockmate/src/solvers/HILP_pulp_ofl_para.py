# import logging
# import math
from typing import Dict, Any
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

from .op_schedule import (
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
# from rkgb.Htools import *
from rkgb.core.hierarchical import HierarchicalGraph
# from rkgb.utils.global_vars import solver_name
# from functools import lru_cache

class knapsack:
    def __init__(self, parameter_size: list):
        size = [s[1] for s in parameter_size]
        self.parameter_size = parameter_size
        self.size = [s / sum(size) for s in size]

    def get_size(self, indices):
        return sum(self.size[i] for i in indices)

    # @lru_cache(maxsize=4096 * 4096)
    def solve(self, frac: float, i: int = 0):
        if frac < 0:
            return []
        if i == len(self.size):
            return list(range(i))
        res1 = self.solve(frac, i + 1)
        res2 = self.solve(frac - self.size[i], i + 1)
        res2 = [i] + res2
        if self.get_size(res1) <= self.get_size(res2) + self.size[i]:
            return res1
        else:
            return res2

    def select(self, frac: float):
        indices = self.solve(frac)
        return [self.parameter_size[i][0] for i in indices]

    def select_size(self, size: int):
        if not self.parameter_size:return []
        return self.select(size / sum(s[1] for s in self.parameter_size))


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
        offload=False,
        protected_names=[],
        grouping=True,
        grad_mode="free", #["free", "accumulate"]
        cpu_optimize_kwargs = None,
        batch_multiplier = 1
    ):
        self.gcd = gcd if gcd else 1
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
        self.with_optimizer_states = 0#accurate_mem
        self.gradient_accumulation = 0# if 0, no gradient/optimizer states alive from previous iters
        self.single_fwd = accurate_mem#False
        self.single_bwd = accurate_mem
        self.grouping = grouping
        self.grad_mode = grad_mode
        if cpu_optimize_kwargs and accurate_mem:
            self.cpu_optimize = True
            self.optimizer_states_factor = cpu_optimize_kwargs["optimizer_states_size"]#*weight size
            self.cpu_optimize_speed = cpu_optimize_kwargs["cpu_optimize_speed"]#B/ms
            self.optimizer_overhead_factor = cpu_optimize_kwargs["optimizer_overhead"]#*weight size
            batch_multiplier = 4
            # self.BatMpl = RkLpVariable("BMpl", lowBound=0, upBound=self.batch_multiplier, cat="Integer")
            # self.param_multiplier = 1-self.BatMpl*1/self.batch_multiplier
            self.param_multiplier = RkLpVariable("BMpl", lowBound=0, upBound=1-1/batch_multiplier, cat="Continuous")
            self.param_multiplier = 0.

        #############################
        self.hgraph = hgraph
        self.hcn2sub_c = []
        self.list_list_sched = []
        self.sub_clusters = []
        self.nOpts = []  # number of opts
        self.nR = []  # number to run R, =nOpts if bwd, =nOpts+1 if fwd
        self.time = []
        self.overhead = []

        for i, hcn in enumerate(self.hgraph.list_hcn):
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
        self.mem = [hdn.mem / self.gcd for hdn in self.hgraph.list_hdn]
        self.saved_mem = [
            [op_sched.mem / self.gcd for op_sched in list_sched]
            for list_sched in self.list_list_sched
        ]

        self.T = T = len(self.hgraph.list_hcn)
        self.I = I = len(self.hgraph.list_hdn)
        self.J = J = len(self.list_list_sched)

        self.protected_indices = [
            i
            for i, hdn in enumerate(self.hgraph.list_hdn)
            if hdn.kdn.name in protected_names
        ]
        if accurate_mem:
            self.protected_indices += [
                i
                for i, hdn in enumerate(self.hgraph.list_hdn)
                if hdn.kdn in self.hgraph.cluster.interfaces["outputs_kdn_data"]
            ]

        self.input_grad_indices = [
            self.hgraph.list_hdn.index(hdn)
            for hdn in self.hgraph.inputs_hdn_grad
            if hdn in self.hgraph.list_hdn
        ]
        self.input_data_indices = [
            self.hgraph.list_hdn.index(hdn)
            for hdn in self.hgraph.inputs_hdn_data
            if hdn in self.hgraph.list_hdn
        ]

        _deps_d = [
            [self.hgraph.list_hcn.index(hcn) for hcn in hdn.deps]
            for hdn in self.hgraph.list_hdn
        ]  # source of hdn
        _users_d = [
            [
                self.hgraph.list_hcn.index(hcn)
                for hcn in self.hgraph.list_hdn[i].users
                if hcn in self.hgraph.list_hcn
            ]
            for i in range(I)
        ]  # outputs of hdn
        _users_c = [
            [self.hgraph.list_hdn.index(hdn) for hdn in self.hgraph.list_hcn[k].users]
            for k in range(T)
        ]  # outputs of hcn

        #### Update edges based on .dep_interfaces_data
        #### In certain schedules, BWD depends on input/output data
        # for i in range(I):
        #     for k, hcn in enumerate(self.hgraph.list_hcn):
        #         if hcn.sub_cluster is None:
        #             continue
        #         for op_sched in hcn.list_sched:
        #             # Without specifying schedule, we assume it's possible to use hdn here
        #             for i_ in op_sched.dep_interfaces_data:
        #                 if (
        #                     op_sched.list_kdn[i_]
        #                     == self.hgraph.list_hdn[i].kdn.name
        #                     and k not in _users_d[i]
        #                 ):
        #                     _users_d[i].append(k)

        ##############################

        # self.md = Model(f"rockmateMILP_{T}_{peak_budget}")
        # if ilp_solver_params is not None:
        #     for k, v in ilp_solver_params.items():
        #         setattr(self.md.Params, k, v)

        self.create_list = [(k, i) for k in range(T) for i in _users_c[k]]
        self.delete_list = [(k, i) for i in range(I) for k in _deps_d[i] + _users_d[i]]

        Cr = len(self.create_list)
        De = len(self.delete_list)
        # print(Cr, De, I)
        # ======build variables======
        # For every HCN[i], R[i] is of size T*nR[i]
        # self.Comp = [
        #     RkLpVariable.dicts(
        #         f"Comp{k}",
        #         [(t, o) for t in range(T) for o in range(self.nR[k])],
        #         cat="Binary",
        #     )
        #     for k in range(T)
        # ]
        self.Comp = RkLpVariable.dicts(
            f"Comp",
            [
                (t, k, o)
                for t in range(T)
                for k in self.krange(t)
                for o in range(self.nR[k])
            ],
            cat="Binary",
        )

        self.sumComp = {}
        for t in range(T):
            for k in self.krange(t):
                self.sumComp[t, k] = lpSum(
                    self.Comp[t, k, o] for o in range(self.nR[k])
                )

        for t in range(T):
            for k in range(T):
                if k not in self.krange(t):
                    for o in range(self.nR[k]):
                        self.Comp[t, k, o] = 0
                    self.sumComp[t, k] = 0

        # Sp for saved Phantoms, option-related
        # self.AliveP = [
        #     RkLpVariable.dicts(
        #         f"Alivep{j}",
        #         [(t, k) for t in range(T + 1) for k in range(len(list_sched))],
        #         cat="Binary",
        #     )
        #     for j, list_sched in enumerate(self.list_list_sched)
        # ]

        self.AliveP = RkLpVariable.dicts(
            f"AliveP",
            [
                (t, j, o)
                for t in range(T + 1)
                for j, list_sched in enumerate(self.list_list_sched)
                for o in range(len(list_sched))
                if t-1 in self.sub_c2hcn[j]
            ],
            cat="Binary",
        )
        for j, list_sched in enumerate(self.list_list_sched):
            for o in range(len(list_sched)):
                if (0,j,o) not in self.AliveP:
                    self.AliveP[0,j,o] = 0
                for t in range(1,T + 1):
                    if (t,j,o) not in self.AliveP:
                        self.AliveP[t,j,o] = self.AliveP[t-1,j,o]

        # self.AliveP = RkLpVariable.dicts(
        #     f"AliveP",
        #     [
        #         (t, j, o)
        #         for t in range(T + 1)
        #         for j, list_sched in enumerate(self.list_list_sched)
        #         for o in range(len(list_sched))
        #     ],
        #     cat="Binary",
        # )

        self.sumAliveP = {}
        for j in range(J):
            for t in range(T + 1):
                self.sumAliveP[j, t] = lpSum(
                    self.AliveP[t, j, o] for o in range(len(self.list_list_sched[j]))
                )
        self.active_stages = dict()
        for i in range(I):
            self.active_stages[i] = []
            for t in range(T):
                for k in self.krange(t):
                    if k in _deps_d[i]+_users_d[i]:
                        self.active_stages[i].append(t)
            if not self.hgraph.list_hdn[i].deps:# src node
                self.active_stages[i].append(-1)
        # to present whether one saved tensor can be inherited from the last stage
        self.AliveA = RkLpVariable.dicts(
            "AliveA", [(t, c)
                       for t in range(1,T) 
                       for c, (k, i) in enumerate(self.create_list)
                       if t-1 in self.active_stages[i]], cat="Binary"
        )  # activation
        for c, (k, i) in enumerate(self.create_list):
            self.AliveA[0,c] = 0
            for t in range(1, self.T):
                if (t,c) not in self.AliveA:
                    self.AliveA[t,c] = self.AliveA[t-1,c]

        # self.AliveA = RkLpVariable.dicts(
        #     "AliveA", [(t, i) for t in range(T) for i in range(Cr)], cat="Binary"
        # )  # activation

        # self.AliveT = RkLpVariable.dicts(
        #     "AliveT", [(t, i) for t in range(T) for i in range(I)], cat="Binary"
        # )  # tensor that can be shared by acts

        self.AliveT = RkLpVariable.dicts(
            "AliveT", [(t, i)
                       for t in range(T) 
                       for i in range(I)
                       if t-1 in self.active_stages[i]], cat="Binary"
        )  # tensor that can be shared by acts
        for i in range(I):
            if (0,i) not in self.AliveT:
                self.AliveT[0,i] = 0
            for t in range(1, self.T):
                if (t,i) not in self.AliveT:
                    self.AliveT[t,i] = self.AliveT[t-1,i]

        self.create = RkLpVariable.dicts(
            "create",
            [
                (t, i)
                for t in range(T)
                for i in range(Cr)
                if self.create_list[i][0] in self.krange(t)
            ],
            cat="Binary",
        )
        self.delete = RkLpVariable.dicts(
            "delete",
            [
                (t, i)
                for t in range(T)
                for i in range(De)
                if self.delete_list[i][0] in self.krange(t)
            ],
            cat="Binary",
        )
        for t in range(T):
            for i in range(Cr):
                if self.create_list[i][0] not in self.krange(t):
                    self.create[t, i] = 0
            for i in range(De):
                if self.delete_list[i][0] not in self.krange(t):
                    self.delete[t, i] = 0

        self.W = W = len(self.sub_clusters)
        def get_parameter_cluster(sub_clusters):
            all_params = {}
            # all_clusters = {sub_cluster.name:sub_cluster for sub_cluster in sub_clusters if sub_cluster}
            # sub_c2params = {sub_cluster.name:set() for sub_cluster in sub_clusters if sub_cluster}
            param2sub_c = {}

            for i,sub_cluster in enumerate(sub_clusters):
                if not hasattr(sub_cluster, "list_kdn_parameters"):continue
                for kdn in sub_cluster.list_kdn_parameters:
                    # sub_c2params[sub_cluster.name].add(kdn.name)
                    all_params[kdn.name] = kdn
                    if kdn.name not in param2sub_c:
                        param2sub_c[kdn.name] = {i}
                    else:
                        param2sub_c[kdn.name].add(i)
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
            # self.param2sub_c, self.parameters = get_parameter_cluster(self.sub_clusters)
            # self.param2hcn = {w:[] for w in range(W)}
            # self.hcn2param = {}# is no param for hcn[k], k not in hcn2param
            # for w,l_c in self.param2sub_c.items():
            #     for c in l_c:
            #         for k in self.sub_c2hcn[c]:
            #             if k in self.hcn2param:
            #                 self.hcn2param[k].append(w)
            #             else:
            #                 self.hcn2param[k] = [w]
            #             self.param2hcn[w].append(k)

            self.param2hcn, self.parameters = get_parameter_cluster(self.hgraph.list_hcn)
            self.hcn2param = {t:[] for t in range(self.T)}

            for p,hcn_s in self.param2hcn.items():
                for hcn in hcn_s:
                    self.hcn2param[hcn].append(p)

            self.parameter_size = [sum(kdn.mem for kdn in p)/self.gcd for p in self.parameters]
            self.parameter_gradient_size = [sum(kdn.mem for kdn in p if kdn.info.requires_grad
                                                )/self.gcd for p in self.parameters]
            self.W = W = len(self.parameters)

            self.AliveW = RkLpVariable.dicts(
                "AliveW",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # parameter w is alive at the start of step j.
            self.AliveG = RkLpVariable.dicts(
                "AliveG",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # w.grad is alive at the start of step j.
            self.AliveO = RkLpVariable.dicts(
                "AliveO",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # w.grad is alive at the start of step j.

            self.OflW = RkLpVariable.dicts(
                "OflW",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.OflG = RkLpVariable.dicts(
                "OflG",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.PrfW = RkLpVariable.dicts(
                "PrfW",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.PrfG = RkLpVariable.dicts(
                "PrfG",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.OptC = RkLpVariable.dicts(
                "OptC",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.UpdW = RkLpVariable.dicts(
                "UpdW",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )
            self.param_grad_mem = {(t,k):0 for t in range(T) for k in self.krange(t)}
            self.prefill()

            self.bandwidthOfl = cpu_optimize_kwargs["bandwidth"]#6 * 1024**2  # byte/ms
            self.bandwidthPrf = cpu_optimize_kwargs["bandwidth"]#6 * 1024**2  # byte/ms

        self.Time = RkLpVariable.dicts(
            "Time", [(t, k) for t in range(T) for k in self.krange(t)], cat="Continuous"
        )

        # define objective function
        prf_cost = (
            0.01
            * lpSum(
                (self.PrfW[t, k, w] + self.PrfG[t, k, w])
                * self.parameter_size[w] / self.bandwidthPrf
                for t in range(T)
                for k in self.krange(t)
                for w in range(W)
            )
            if self.with_parameters
            else 0
        )
        ofl_cost = (
            0.01
            * lpSum(
                (self.OflW[t, k, w] + self.OflG[t, k, w])
                * self.parameter_size[w] / self.bandwidthOfl
                for t in range(T)
                for k in self.krange(t)
                for w in range(W)
            )
            if self.with_parameters
            else 0
        )
        self.md = LpProblem(f"rockmateMILP", LpMinimize)
        self.md += (
            lpSum(self.Time[t, k] for t in range(T) for k in self.krange(t))
            + prf_cost
            + ofl_cost
        )

        ##### Time constraints
        for t in range(T):
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

        ##### Boundary constraints
        # self.md += (
        #     lpSum(self.sumComp[t, k] for t in range(T) for k in range(t + 1, T)) == 0
        # )
        # self.md += (
        #     lpSum(
        #         self.sumAliveP[(self.hcn2sub_c[i], t)]
        #         for t in range(self.loss_idx)
        #         for i in range(t + 1, self.loss_idx)
        #         if self.hcn2sub_c[i]
        #     )
        #     == 0
        # )
        # self.md += (
        #     lpSum(
        #         self.AliveA[t, j]
        #         for j in range(Cr)
        #         for t in range(self.create_list[j][0] + 1)
        #     )
        #     == 0
        # )
        # for i in range(I):
        #     if _deps_d[i]:
        #         self.md += (
        #             lpSum(self.AliveT[t, i] for t in range(min(_deps_d[i]) + 1)) == 0
        #         )

        ##### Validity constraints

        # In the last stage, every source edge of input_grad should be alive or executed
        for i in self.input_grad_indices:
            for j_, (k_, i_) in enumerate(self.create_list):
                if i_ == i:
                    self.md += self.AliveA[T - 1, j_] + self.sumComp[T - 1, k_] == 1

        # # In the first stage, assume input data is alive
        # for i in self.input_data_indices:
        #     for j_, (k_, i_) in enumerate(self.create_list):
        #         if i_ == i:
        #             self.md += (self.AliveA[0, j_] ==  1)

        for j in range(J):
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
                    (self.AliveP[T, j, o])  # - self.Comp[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                )
                == 0
            )

        # options don't conflict
        for t in range(T):
            for k in self.krange(t):
                self.md += self.sumComp[t, k] <= 1
        for t in range(T + 1):
            for j in range(J):
                self.md += (
                    self.sumAliveP[j, t] <= 1
                )  # assuming two copies of saved tensors won't be kept at the same time

        #### Option-free constraints: from rk-checkmate
        self.md += (
            lpSum(self.sumComp[t, t] for t in range(T)) == T
        )  # diagonal should be executed
        self.md += (
            lpSum(
                self.sumComp[t, self.loss_idx]
                for t in range(T)
                if self.loss_idx in self.krange(t)
            )
            == 1
        )  # loss should be executed exactly once

        for t in range(T):
            for j in range(Cr):
                self.md += (
                    self.AliveA[t, j] <= self.AliveT[t, self.create_list[j][1]]
                )  # one edge created, memory is occupied
        for t in range(T - 1):
            for j in range(Cr):
                src_i = self.create_list[j][0]
                src = self.sumComp[t, src_i] if src_i in self.krange(t) else 0
                self.md += self.AliveA[t + 1, j] <= self.AliveA[t, j] + src
        for t in range(T):
            for j, (k, i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md += (
                        self.sumComp[t, k_] <= self.sumComp[t, k] + self.AliveA[t, j]
                    )

        #### Options-related constraints
        for t in range(T):
            for j in range(J):
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
                    for i in list_sched[o].dep_interfaces_data:
                        hcn = self.hgraph.list_hcn[fwd_i]
                        name = self.sub_clusters[j].list_kdn[i].name
                        # Tensor req_i is required by BWD
                        req_i = [hdn.kdn.name for hdn in self.hgraph.list_hdn].index(
                            name
                        )
                        for j_, (k_, i_) in enumerate(self.create_list):
                            if i_ == req_i:
                                self.md += (
                                    self.Comp[t, bwd_i, o]
                                    <= self.sumComp[t, k_] + self.AliveA[t, j_]
                                )

        #### Offload constraints
        if self.with_parameters:
            self.add_parameter_constraints()

        ##### Memory constraints
        # we don't keep eyes on the alive status all the time
        # only the steps when changes can happen
        self.alive = {}
        for t in range(T):
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
            for i in range(I):
                if t + 1 < T:
                    self.md += (
                        self.AliveT[t + 1, i]
                        == self.alive[(t, max(_deps_d[i] + _users_d[i]), i)]
                    )
                elif i not in self.protected_indices:
                    # in the end of bwd, del every HDN
                    self.md += self.alive[(t, max(_deps_d[i] + _users_d[i]), i)] == 0

        def _num_hazards(t, i, k):
            if i in self.protected_indices:
                return _max_num_hazards(t, i, k)
            if t + 1 < T:
                return (
                    1
                    - self.sumComp[t, k]
                    + self.AliveT[t + 1, i]
                    + lpSum(self.sumComp[t, j] for j in _users_d[i] if j > k)
                )
            return (
                1
                - self.sumComp[t, k]
                + lpSum(self.sumComp[t, j] for j in _users_d[i] if j > k)
            )

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in _users_d[i] if j > k)
            if t + 1 < T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # delete when not needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += 1 - self.delete[t, eidx] <= _num_hazards(t, i, k)

        # don't delete if still needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += _max_num_hazards(t, i, k) * (
                    1 - self.delete[t, eidx]
                ) >= _num_hazards(t, i, k)
                if i in self.protected_indices:
                    self.md += self.delete[t, eidx] == 0

        self.U = {}
        for t in range(T):
            self.U[t, 0] = (
                lpSum(self.AliveT[t, i] * self.mem[i] for i in range(I))
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
                    for j in range(J)
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

        for t in range(T):
            for k in range(1, T):
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
                if self.hgraph.list_hcn[k].is_fwd:
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
        for t in range(T):
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
                    hcn = self.hgraph.list_hcn[k]
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
                                    hdn.kdn.name for hdn in self.hgraph.list_hdn
                                ].index(
                                    self.sub_clusters[j]
                                    .list_kdn[inter_position[0]]
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
                                        k_ = max([kk for kk in _deps_d[i_] if kk < k])
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
        T = len(self.hgraph.list_hcn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
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
        return (self.OflGProg[t, k, w]+self.OflWProg[t, k, w]*(self.grad_mode=="free"))

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
        optimizer_states_mem = lpSum((self.AliveO[t, k, w]*
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


    def add_parameter_constraints(self, 
                                  with_grad=False,
                                  with_optimizer_stats=False):
        # if with_grad, AliveG is a variable
        # if with_optimizer_states, AliveO is a variable
        self.OflWProg = dict()
        self.OflGProg = dict()
        self.OptCProg = dict()
        self.UpdWProg = dict()
        self.PrfWProg = dict()
        self.PrfGProg = dict()
        
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

                self.md += self.Time[t, k] >= lpSum(
                    self.parameter_size[w] / self.bandwidthPrf * self.PrfW[t, k, w]
                    for w in range(self.W)
                )
                self.md += self.Time[t, k] >= lpSum(
                    self.parameter_size[w] / self.bandwidthOfl 
                    * (self.OflW[t, k, w]+self.OflG[t, k, w])
                    + self.parameter_size[w]
                    / self.cpu_optimize_speed
                    * self.OptC[t, k, w]
                    for w in range(self.W)
                )
                # self.md += self.Time[t, k] >= lpSum(
                #     self.parameter_size[w] / self.cpu_optimize_speed * self.OptC[t, k, w]
                #     for w in range(self.W)
                # )
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
                    self.OptCProg[t,k,w] = get_progress(self.OptC, t, k, w)
                    
                    self.md += (self.AliveW[t, k, w] <= self.req_w())
                    self.md += self.OflWProg[t, k, w] <= self.req_w()
                    self.md += self.OflGProg[t, k, w] <= self.accumC_grad(w)
                    self.md += self.OptCProg[t, k, w] <= self.max_OflGProg(t,k,w)
                    self.md += (self.AliveW[t, k, w] + self.OflWProg[t, k, w]
                                >= self.instant_opt(w))
                    self.md += (self.AliveG[t, k, w] + self.OflGProg[t, k, w] 
                                >= self.req_w() - self.instant_opt(w))
                    self.md += (self.AliveW[t_, k_, w] + self.PrfW[t_, k_, w] <= 
                                self.req_w() - self.sumOptC[w]# update on GPU
                                + self.OptCProg[t, k, w])
                    diffW = self.AliveW[t_, k_, w] - self.AliveW[t, k, w]
                    self.md += diffW <= self.PrfW[t, k, w]

                    diffG = self.AliveG[t_, k_, w] - self.AliveG[t, k, w]
                    self.md += (diffG <= 1*(k in self.param2hcn[w] 
                                           and k>self.loss_idx)#backward operations
                                           +self.PrfG[t, k, w])

                    self.md += self.AliveO[t_, k_, w] - self.AliveO[t, k, w] <= 0#no prefetch now
            
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

        status = self.md.solve(solver)
        self.status = LpStatus[status]  # readable status
        self.feasible = status == 1

        sol = self.sol
        if self.feasible:
            # print("finished solving")
            self.solve_time = self.md.solutionTime
            self.active_steps = []
            for t in list(range(self.loss_idx + 1, self.T)) + list(
                range(self.loss_idx + 1)
            ):
                for k in self.krange(t):
                    if not sol(self.sumComp[t, k].value()):
                        continue
                    self.active_steps.append((t, k))

    def sol(self, value):
        return value > 0.9999

    def group(self, w, tol=1):
        # Group the parameters of each block for the task
        fwd_i = min(self.param2hcn[w])
        bwd_i = max(self.param2hcn[w])
        early_fwd = []
        for t in range(bwd_i, self.T):
            if not self.single_fwd and self.sol(self.sumComp[t,fwd_i].value()):
                early_fwd.append(t)#if recompute fwd after bwd
        hcn = self.hgraph.list_hcn[fwd_i]
        # parameters = {kdn.name: kdn for kdn in hcn.sub_cluster.list_kdn_parameters}
        parameters = {kdn.name: kdn for kdn in self.parameters[w]}
        parameter_size = sum(kdn.mem for kdn in parameters.values())

        Alive = {p: 1 for p in parameters.keys()}
        Offloaded = {p: False for p in parameters.keys()}

        ofl_ops = []
        prf_ops = []
        del_ops = []
        opt_ops = []
        init_ops = []
        restore_ops = []
        cpu_optimize_candidates = {p: 0 for p in parameters.keys()}

        def apply_cpu_optimize(p):
            for (t,k,op) in ofl_ops:
                if op.target.name == p:
                    op.grad = True
                    break
            # for (t,k,op) in del_ops:
            #     if op.target.name == p:
            #         op.grad = True
            i = self.active_steps.index((t,k))+1# TODO: distribute cpu optimization based on time
            op = OptimizeOp(name="cpu_"+p,list_params=["cpu_"+p], alloc=Parameter(parameters[p]),
                            time=parameters[p].mem/self.cpu_optimize_speed,
                            )
            opt_ops.append((*self.active_steps[i], op))
            del_ops.append((*self.active_steps[i],DeleteOp(Parameter(parameters[p]), grad=True)))
            self.cpu_optimized_steps[self.active_steps[i]].append(p)
            del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p]))))

            #if cpu optimize, do not keep w after bwd
        def apply_gpu_optimize(p):
            op = OptimizeOp(name=p,list_params=[p], alloc=Parameter(parameters[p]),
                            overhead=parameters[p].mem*self.optimizer_overhead_factor)
            opt_ops.append((bwd_i, bwd_i, op))# optimize after bwd
            del_ops.append((bwd_i, bwd_i, DeleteOp(Parameter(parameters[p]), grad=True)))

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
                        init_ops.append((t, k, AllocateOp(Parameter(parameters[p]))))
                        op = PrefetchOp(
                            alloc=Parameter(parameters[p]), indices=(0, None), 
                            time=parameters[p].mem/self.bandwidthPrf
                        )
                        init_ops.append((t, k, op))
                        op = OffloadOp(
                            alloc=Parameter(parameters[p]), indices=(0, None),
                            time=parameters[p].mem/self.bandwidthOfl
                        )
                        restore_ops.append((t, k, op))
                        restore_ops.append((t, k, DeleteOp(Parameter(parameters[p]))))

            if next_offloaded_size > current_offloaded_size:
                # print(t,k, next_offloaded_size, current_offloaded_size)
                ofl_size = next_offloaded_size - current_offloaded_size
                candidates = {
                    p: parameters[p].mem * (1 - o)
                    for p, o in Offloaded.items()
                    if o < 1
                }
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select_size(ofl_size)
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
                #     pass
                for p in select_paras:
                    op = OffloadOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                   time=parameters[p].mem/self.bandwidthOfl)
                    ofl_ops.append((t, k, op))
                    Offloaded[p] = 1

            if current_alive_size > next_alive_size:
                del_size = current_alive_size - next_alive_size
                candidates = {}
                for p, o in Offloaded.items():
                    if Alive[p]>0 and o>0:
                        candidates[p] = min(o, Alive[p])*parameters[p].mem

                selector = knapsack(list(candidates.items()))
                select_paras = selector.select_size(del_size)
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
                    continue
                if self.sol(self.AliveW[(t_, k_, w)].value()):
                    select_paras = list(candidates.keys())
                else:
                    selector = knapsack(list(candidates.items()))
                    unselect_paras = selector.select_size(sum(candidates.values()) - prf_size)
                    select_paras = [
                        p for p in candidates.keys() if p not in unselect_paras
                    ]
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
                #     pass
                for p in select_paras:
                    prf_ops.append((t, k, AllocateOp(Parameter(parameters[p]))))
                    op = PrefetchOp(alloc=Parameter(parameters[p]), indices=(0, None),
                                    time=parameters[p].mem/self.bandwidthPrf)
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
        
            # cpu_optimize_size = self.sumOptC[w].value()*parameter_size# size by subgraph
        if isinstance(self.param_multiplier, float):
            multiplier = self.param_multiplier
        else:
            multiplier = self.param_multiplier.value()
        cpu_optimize_size = (sum(self.sumOptC[w_].value() * 
                                self.parameter_gradient_size[w_] 
                                for w_ in range(w, self.W)) / (1-multiplier)
                            - sum(self.cpu_optimized_params.values()))# size by all graphs
        if candidates and cpu_optimize_size>0:
            # print(candidates, cpu_optimize_size)
            selector = knapsack(list(candidates.items()))
            select_paras = selector.select_size(cpu_optimize_size)
            # print(select_paras)
            
        # Optimize parameters which requires grad
        for p, kdn in parameters.items():
            if not kdn.info.requires_grad:continue
            if p in select_paras:
                self.cpu_optimized_params[p] = parameters[p].mem
                apply_cpu_optimize(p)
            else:
                apply_gpu_optimize(p)
        return ofl_ops, prf_ops, del_ops, opt_ops, init_ops, restore_ops

    def distribute_cpu_optimize(self):
        # after grouping the optimize ops, re-distribute them to reduce the overhead
        pass

    def schedule(self, hgraph=None, check_valid=False):
        """
        Given the solution from HILP, we want to translate the result
        to a OpSchedule that can be used in a higher level.
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"

        sol = self.sol

        T = len(hgraph.list_hcn)
        I = len(hgraph.list_hdn)
        J = len(self.list_list_sched)
        init_op_list = []
        restore_op_list = []
        init_alive_status = {}

        if self.with_parameters:
            W = len(self.parameter_size)
            (
                op_list,
                init_alive_status,
                init_op_list,
                restore_op_list,
            ) = self.greedy_post_processing(hgraph)
        else:
            op_list = []
            
            for t in range(T):
                for k in self.krange(t):
                    if t == self.loss_idx and k == self.loss_idx:
                        op_list.append(ComputeOp(self.hgraph.cluster.loss_kcn))
                    op_list += self.schedule_compute(t,k,hgraph)
        
        op_sched = OpSchedule(
            op_list,
            # ofl_list=ofl_list,
            # prf_list=prf_list,
            loss_idx=None,
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
        if sol(self.sumComp[t, k].value()):
            hcn = hgraph.list_hcn[k]
            opt = -1
            for o in range(self.nOpts[k]):
                if sol(self.Comp[t, k, o].value()):
                    opt = o
                    break
            if opt > -1:
                h_obj = self.list_list_sched[j][opt]
                if hcn.is_fwd:
                    # sub_op_list = deepcopy(
                    #     h_obj.op_list[: h_obj.loss_idx]
                    # )
                    sub_op_list = h_obj.op_list[: h_obj.loss_idx]
                else:
                    sub_op_list = h_obj.op_list[h_obj.loss_idx + 1 :]

                    # if self.sumAliveP[(j, t + 1)].value() == 0:
                    # sub_op_list.append()
                sub_op_list = deepcopy(sub_op_list)

                if (
                    not hcn.is_fwd
                    # and self.sumAliveP[(j, t + 1)].value() > 0
                    and sol(self.sumAliveP[(j, t + 1)].value())
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
                    sub_op_list = hcn.sub_cluster.translate_op_list(
                        sub_op_list
                    )
            else:
                h_obj = hcn
                sub_op_list = deepcopy(h_obj.ff_op_list)

            op_list += sub_op_list

        for eidx, (k_, i) in enumerate(self.delete_list):
            # print(k_, i)
            # if k == k_ and self.delete[t, eidx].value()==1:
            if k == k_ and sol(self.delete[t, eidx].value()):
                hdn = hgraph.list_hdn[i]
                op_list.append(DeleteOp(Activation(hdn.kdn)))
        return op_list

    def greedy_post_processing(self, hgraph=None):
        """
        V1: self.grouping = False:
        merge every cluster, ofl/prf/del partially, high memory overhead
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(hgraph.list_hcn)
        I = len(hgraph.list_hdn)
        J = len(self.list_list_sched)
        W = len(self.parameter_size)

        ### Handle multiplier
        self.params_vars = [self.AliveW, self.OflWProg, self.OflW, 
                            self.PrfW, self.PrfWProg, self.OptC,
                            self.AliveG, self.OflGProg, self.OflG,
                            self.PrfG, self.PrfGProg, self.AliveO
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
        if self.grouping:
            for w in range(self.W)[::-1]:
                o_l, p_l, d_l, t_l, i_l, r_l = self.group(w)
                self.ofl_ops.extend(o_l)
                self.prf_ops.extend(p_l)
                self.del_ops.extend(d_l)
                self.opt_ops.extend(t_l)
                init_op_list.extend([ops[2] for ops in i_l])
                restore_op_list.extend([ops[2] for ops in r_l])
        # else:
        #     init_op_list = self.schedule_init_op_list()

        sol = self.sol
        # offload_buffers = {w:[] for w in range(W)}
        op_list = []
        init_alive_status = dict()
        # for kdn in self.hgraph.cluster.list_kdn_parameters:
        #     init_alive_status[kdn.name] = True
        # for w in range(W):
        #     hcn = self.hgraph.list_hcn[self.param2hcn[w]]
        #     self.current_buffers[w] = Parameter(hcn.sub_cluster.name)
        # self.cpu_optimized_params = []
        # for (_,_,op) in self.opt_ops:
        #     if isinstance(op, OptimizeOp):
        #         self.cpu_optimized_params.append(op.target.name)

        for t in range(T):
            for k in self.krange(t):
                op_list.append(SynchronizeOp(f"{(t,k)}"))
                if t == self.loss_idx and k == self.loss_idx:
                    # loss_idx = len(op_list)
                    # loss_op = Op(K_C_node("loss"))

                    op_list.append(ComputeOp(self.hgraph.cluster.loss_kcn))
                if not sol(self.sumComp[t, k].value()):
                    continue
                j = self.hcn2sub_c[k]
                # if self.sumComp[t, k].value() == 1:
                prefetch_list = []
                for w in range(W):
                    prefetch_ops = self.create_prefetch_ops(t, k, w)
                    op_list.extend(prefetch_ops[0])
                    prefetch_list.extend(prefetch_ops[1])
                op_list += self.schedule_compute(t,k,hgraph)
                wait_op = []
                for w in range(W):
                    if k in self.param2hcn[w]:
                        wait_op.extend(self.create_offload_ops(t, k, w))
                        wait_op.extend(self.create_optimize_ops(t, k, w))
                        wait_op.extend(self.create_delete_ops(t, k, w))
                        # op_list.extend(self.create_prefetch_ops(t,k,w))
                    else:
                        op_list.extend(self.create_offload_ops(t, k, w))
                        op_list.extend(self.create_delete_ops(t, k, w))
                        op_list.extend(self.create_optimize_ops(t, k, w))
                if wait_op:# for the current layer, need to synchronize first
                    op_list.extend([SynchronizeOp(str(k))]+wait_op)
                op_list.extend(prefetch_list)
        return op_list, init_alive_status, init_op_list, restore_op_list

    def create_optimize_ops(self, t, k, w, itemsize=4):
        op_list = []
        # sub_cluster = self.hgraph.list_hcn[min(self.param2hcn[w])].sub_cluster
        if self.grouping:
            for t_, k_, op in self.opt_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.kdn.name in [k.name for k in self.parameters[w]]
                ):
                    op_list.append(op)
            return op_list

    def create_delete_ops(self, t, k, w, itemsize=4):
        op_list = []
        sub_cluster = self.hgraph.list_hcn[min(self.param2hcn[w])].sub_cluster
        if self.grouping:
            for t_, k_, op in self.del_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.kdn.name in [k.name for k in self.parameters[w]]
                ):
                    op_list.append(op)
            return op_list

    def create_prefetch_ops(self, t, k, w, itemsize=4):
        pre_op_list = []
        post_op_list = []
        sub_cluster = self.hgraph.list_hcn[min(self.param2hcn[w])].sub_cluster

        if self.grouping:
            for t_, k_, op in self.prf_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.kdn.name in [k.name for k in self.parameters[w]]
                ):
                    pre_op_list.append(op)

            return pre_op_list, post_op_list

    def create_offload_ops(self, t, k, w, itemsize=4):
        op_list = []
        sub_cluster = self.hgraph.list_hcn[min(self.param2hcn[w])].sub_cluster
        if self.grouping:
            for t_, k_, op in self.ofl_ops:
                if (
                    t_ == t
                    and k_ == k
                    and op.target.kdn.name in [k.name for k in self.parameters[w]]
                ):
                    op_list.append(op)
            return op_list
