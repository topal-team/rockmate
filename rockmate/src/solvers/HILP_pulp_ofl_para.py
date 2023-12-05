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
    OpSchedule,
)
from rkgb.Htools import *
from rkgb.utils.global_vars import solver_name


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


class ModelPULP:
    """
    Build the ILP model by given Hgraph and budget.
    RN this model will take a rk_chain to solve the solution.
    """

    def __init__(
        self,
        hgraph: H_graph,
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
        grouping = True
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
        self.enable_offload = accurate_mem
        self.grouping = grouping

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
        self.W = W = (
            len(self.hgraph.list_hcn) // 2
        )  # for now, one weight for each layer
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
            [(t, k, o) for t in range(T) for k in self.krange(t) for o in range(self.nR[k])],
            cat="Binary",
        )

        self.sumComp = {}
        for t in range(T):
            for k in self.krange(t):
                self.sumComp[t, k] = lpSum(
                    self.Comp[t, k, o] for o in range(self.nR[k])
                )

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
            ],
            cat="Binary",
        )

        self.sumAliveP = {}
        for j in range(J):
            for t in range(T + 1):
                self.sumAliveP[j, t] = lpSum(
                    self.AliveP[t, j, o] for o in range(len(self.list_list_sched[j]))
                )

        # to present whether one saved tensor can be inheritaged from the last stage
        self.AliveA = RkLpVariable.dicts(
            "AliveA", [(t, i) for t in range(T) for i in range(Cr)], cat="Binary"
        )  # activation
        self.AliveT = RkLpVariable.dicts(
            "AliveT", [(t, i) for t in range(T) for i in range(I)], cat="Binary"
        )  # tensor that can be shared by acts
        self.create = RkLpVariable.dicts(
            "create", [(t, i) for t in range(T) for i in range(Cr)], cat="Binary"
        )
        self.delete = RkLpVariable.dicts(
            "delete", [(t, i) for t in range(T) for i in range(De)], cat="Binary"
        )

        if self.enable_offload:
            self.AliveW = RkLpVariable.dicts(
                "AliveW",
                [(t, k, w) for t in range(T) for k in self.krange(t) for w in range(W)],
                cat="Continuous",
                lowBound=0,
                upBound=1,
            )  # weight w is alive at the start of step j.
            self.OflW = RkLpVariable.dicts(
                "OflW",
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
            # self.weights_size = [3e7 for _ in range(W)]
            self.weights_size = []
            for i in range(W):
                sub_cluster = self.hgraph.list_hcn[i].sub_cluster
                if hasattr(sub_cluster, "list_kdn_parameters"):
                    self.weights_size.append(
                        sum(kdn.mem for kdn in sub_cluster.list_kdn_parameters)
                        / self.gcd
                    )
                else:
                    self.weights_size.append(0)
            # print(self.weights_size)
            self.weight2hcn = {w: [w, T - w - 1] for w in range(W)}
            self.hcn2weight = {
                k: w for w in self.weight2hcn for k in self.weight2hcn[w]
            }
            self.bandwidthOfl = 6 * 1024**2  # byte/ms
            self.bandwidthPrf = 6 * 1024**2  # byte/ms

        self.Time = RkLpVariable.dicts(
            "Time", [(t, k) for t in range(T) for k in self.krange(t)], cat="Continuous"
        )

        # define objective function
        prf_cost = 0.01*lpSum(self.PrfW[t, k, w]*self.weights_size[w] /self.bandwidthPrf
                         for t in range(T) for k in self.krange(t) 
                         for w in range(W)) if self.enable_offload else 0
        ofl_cost = 0.01*lpSum(self.OflW[t, k, w]*self.weights_size[w] /self.bandwidthOfl
                         for t in range(T) for k in self.krange(t) 
                         for w in range(W)) if self.enable_offload else 0
        self.md = LpProblem(f"rockmateMILP", LpMinimize)
        self.md += lpSum(self.Time[t, k] for t in range(T) for k in self.krange(t)) + prf_cost + ofl_cost

        ##### Time constraints
        for t in range(T):
            for k in self.krange(t):
                # if k==self.loss_idx:continue
                if self.enable_offload and k != self.loss_idx:
                    w = self.hcn2weight[k] if k != self.loss_idx else None
                    ofl_time = (
                        self.weights_size[w] / self.bandwidthOfl * self.OflW[t, k, w]
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
        self.md += (
            lpSum(self.sumComp[t, k] for t in range(T) for k in range(t + 1, T)) == 0
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
                for j in range(Cr)
                for t in range(self.create_list[j][0] + 1)
            )
            == 0
        )
        for i in range(I):
            if _deps_d[i]:
                self.md += (
                    lpSum(self.AliveT[t, i] for t in range(min(_deps_d[i]) + 1)) == 0
                )

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
            self.md += (
                lpSum(
                    (self.AliveP[0, j, o])  # - self.Comp[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                )
                == 0
            )
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
            lpSum(self.sumComp[t, self.loss_idx] for t in range(T)) == 1
        )  # loss should be executed exactly once

        for t in range(T):
            for j in range(Cr):
                self.md += (
                    self.AliveA[t, j] <= self.AliveT[t, self.create_list[j][1]]
                )  # one edge created, memory is occupied
        for t in range(T - 1):
            for j in range(Cr):
                src_i = self.create_list[j][0]
                self.md += (
                    self.AliveA[t + 1, j] <= self.AliveA[t, j] + self.sumComp[t, src_i]
                )
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
        if self.enable_offload:
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
                    if k_ == 0
                )
                - lpSum(
                    self.delete[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.delete_list)
                    if k_ == 0
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
                        if k_ == k
                    )
                    - lpSum(
                        self.delete[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.delete_list)
                        if k_ == k
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
                if self.enable_offload:
                    # weight_mem = 0
                    weight_mem = lpSum(
                        self.AliveW[t, k, w] * self.weights_size[w] for w in range(W)
                    )
                else:
                    weight_mem = 0
                j = self.hcn2sub_c[k]
                self.md += self.U[t, k] >= 0
                self.md += self.U[t, k] <= self.peak_budget - weight_mem
                if j is None or not accurate_mem:
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
                        <= self.peak_budget - weight_mem
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
                                <= self.peak_budget - weight_mem
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
                                <= self.peak_budget - weight_mem
                            )
                if t == self.loss_idx and self.save_budget:
                    self.md += self.U[t, k] <= self.save_budget

    def krange(self, t):
        # return range(t)
        return range(self.T)

    def next_index(self, t, i, upper_triangle=True):
        # if upper_triangle, consider the case when i>t
        if t == self.T - 1:
            if i < t:
                t_ = t
                i_ = i + 1
            else:
                t_ = 0
                i_ = 0
        else:
            end = self.T - 1 if upper_triangle else t + 1
            if i < end:
                t_ = t
                i_ = i + 1
            else:
                i_ = 0
                t_ = t + 1
        return (t_, i_)

    def add_abar_constraint(self, save_budget):
        T = len(self.hgraph.list_hcn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
            self.md += self.U[(self.loss_idx, k)] <= self.save_budget

    def add_single_bwd_constraints(self):
        for i in range(self.loss_idx, self.T):  # only bwd after loss
            self.md += lpSum(self.sumComp[t, i] for t in range(self.T)) == 1

    def add_single_fwd_constraints(self):
        for i in range(self.loss_idx):  # only fwd before loss
            self.md += lpSum(self.sumComp[t, i] for t in range(self.T)) == 1

    def add_parameter_constraints(self):
        self.OflWProg = dict()
        for t in range(self.T):
            for w in self.weight2hcn:
                for k in self.weight2hcn[w]:
                    self.md += self.sumComp[t, k] <= self.AliveW[t, k, w]
            for i in range(self.T):
                self.md += self.Time[t, i] >= lpSum(
                    self.weights_size[w] / self.bandwidthPrf * self.PrfW[t, i, w]
                    for w in range(self.W)
                )
                self.md += self.Time[t, i] >= lpSum(
                    self.weights_size[w] / self.bandwidthOfl * self.OflW[t, i, w]
                    for w in range(self.W)
                )
                for w in range(self.W):
                    bwd_i = max(self.weight2hcn[w])
                    if bwd_i < t:  # after bwd of w
                        self.OflWProg[(t, i, w)] = lpSum(
                            self.OflW[t, ii, w] for ii in range(i)
                        ) + lpSum(
                            self.OflW[tt, ii, w]
                            for tt in range(bwd_i, t)  # offload right after bwd_i
                            for ii in range(self.T)
                        )
                    else:
                        self.OflWProg[(t, i, w)] = (
                            lpSum(self.OflW[t, ii, w] for ii in range(i))
                            + lpSum(
                                self.OflW[tt, ii, w]
                                for tt in range(t)
                                for ii in range(self.T)
                            )
                            + lpSum(
                                self.OflW[tt, ii, w]
                                for tt in range(bwd_i, self.T)
                                for ii in range(self.T)
                            )
                        )
                    self.md += self.OflWProg[(t, i, w)] <= 1
                    self.md += self.AliveW[t, i, w] + self.OflWProg[(t, i, w)] >= 1
                    self.md += self.AliveW[t, i, w] + self.PrfW[(t, i, w)] <= 1
                    self.md += self.OflW[t, i, w] <= self.sumComp[t, i]
                    self.md += self.PrfW[t, i, w] <= self.sumComp[t, i]

                    t_, i_ = self.next_index(t, i)
                    diff = self.AliveW[t_, i_, w] - self.AliveW[t, i, w]
                    self.md += diff <= self.sumComp[t, i]
                    self.md += -diff <= self.sumComp[t, i]
                    self.md += diff <= self.PrfW[t, i, w]

    def solve(self, solver=""):
        # some solvers have no support of 'Time limit reached' status

        self.add_single_fwd_constraints()
        self.add_single_bwd_constraints()
        try:
            solver = get_solver(solver, msg=0, timeLimit=self.ilp_solver_params["TimeLimit"])
        except:
            avail_solver = listSolvers(onlyAvailable=True)[0]
            #     print(f"Cannot get {solver}, will use {avail_solver}")
            solver = get_solver(avail_solver, msg=0, timeLimit=self.ilp_solver_params["TimeLimit"])

        status = self.md.solve(solver)
        self.status = LpStatus[status]  # readable status
        self.feasible = status == 1

        sol = self.sol
        if self.feasible:
            self.solve_time = self.md.solutionTime
            self.active_steps = []
            for t in list(range(self.loss_idx + 1, self.T)) + list(range(self.loss_idx + 1)):
                for i in range(t + 1):
                    if not sol(self.sumComp[t, i].value()):
                        continue
                    self.active_steps.append((t, i))
    
    def sol(self, value):
            return value > 0.9999

    def _refine_solution(self):
        # greedily group offload/prefetch values by updating .sol
        assert self.feasible, "Cannot refine an infeasible model!"

        sol = self.sol

        # preparation for the greedy algo
        active_steps = []
        offload_size = dict()
        prefetch_size = dict()
        offload_progs = {w: 0 for w in range(self.W)}
        prefetch_progs = {w: 0 for w in range(self.W)}
        offload_pieces = {w: [] for w in range(self.W)}
        prefetch_pieces = {w: [] for w in range(self.W)}

        for t in list(range(self.loss_idx + 1, self.T)) + list(range(self.loss_idx + 1)):
            for i in range(t + 1):
                if not sol(self.sumComp[t, i].value()):
                    continue
                active_steps.append((t, i))
                offload_size[(t, i)] = 0
                prefetch_size[(t, i)] = 0
                for w in range(self.W):
                    if i in self.weight2hcn[w]:
                        if offload_progs[w] > 0:
                            offload_pieces[w].append((t, i, offload_progs[w]))
                            offload_progs[w] = 0
                        if prefetch_progs[w] > 0:
                            prefetch_pieces[w].append((t, i, prefetch_progs[w]))
                            prefetch_progs[w] = 0
                    if self.OflW[t, i, w].value() > 0:
                        offload_progs[w] += self.OflW[t, i, w].value()
                        offload_size[(t, i)] += (
                            self.OflW[t, i, w].value() * self.weights_size[w]
                        )
                    if self.PrfW[t, i, w].value() > 0:
                        prefetch_progs[w] += self.PrfW[t, i, w].value()
                        prefetch_size[(t, i)] += (
                            self.PrfW[t, i, w].value() * self.weights_size[w]
                        )

        # start to re-organize the offload/prefetch operations
        for w in range(self.W - 1, -1, -1):
            avail_size = ...
            offload_pieces[w]

    def group(self, w, tol=1):
        # Group the parameters of each block for the task
        fwd_i, bwd_i = self.weight2hcn[w]
        hcn = self.hgraph.list_hcn[fwd_i]
        parameters = {
            kdn.name: kdn for kdn in hcn.sub_cluster.list_kdn_parameters
        }
        parameter_size = sum(kdn.mem for kdn in parameters.values())

        Alive = {p:1 for p in parameters.keys()}
        Offloaded = {p:False for p in parameters.keys()}

        ofl_ops = []
        prf_ops = []
        del_ops = []
        init_ops = []

        assert (bwd_i, bwd_i) in self.active_steps
        idx = self.active_steps.index((bwd_i, bwd_i))
        for (t,k) in self.active_steps[idx:]+self.active_steps[:idx]:
            t_, k_ = self.next_index(t, k)
            current_size = round(self.AliveW[(t, k, w)].value() * parameter_size)
            next_size = round(self.AliveW[(t_, k_, w)].value() * parameter_size)
            ofl_size = self.OflW[t,k,w].value()

            if (t,k) == (0,0):#init
                for p,a in Alive.items():
                    if a:
                        init_ops.append((t,k,AllocateOp(Parameter(parameters[p]))))
                        op = PrefetchOp(alloc=Parameter(parameters[p]),
                                   indices=(0,None))
                        init_ops.append((t,k,op))

            if ofl_size>0:
                candidates = {p: parameters[p].mem*(1-o) for p,o in Offloaded.items() if o<1}
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select(ofl_size)
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-ofl_size>tol:
                #     pass
                for p in select_paras:
                    # start = parameters[p].info.tsize.numel()
                    op = OffloadOp(alloc=Parameter(parameters[p]),
                                   indices=(0,None))
                    ofl_ops.append((t,k,op))
                    Offloaded[p] = 1

            if current_size > next_size:
                del_size = current_size - next_size
                candidates = {p: parameters[p].mem*o for p,o in Offloaded.items() if o>0}
                selector = knapsack(list(candidates.items()))
                select_paras = selector.select(del_size)
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-del_size>tol:
                #     pass
                for p in select_paras:
                    del_ops.append((t,k,DeleteOp(Parameter(parameters[p]))))
                    Alive[p] = 0

            if current_size < next_size:
                # prefetch should be smaller than solution
                prf_size = next_size - current_size
                candidates = {p: parameters[p].mem*(1-a) for p,a in Alive.items() if a<1}
                if self.sol(next_size):
                    select_paras = list(candidates.keys())
                else:
                    selector = knapsack(list(candidates.items()))
                    unselect_paras = selector.select(1-prf_size)
                    select_paras = [p for p in candidates.keys() if p not in unselect_paras]
                # if sum(candidates[p] for p in select_paras)/sum(candidates.values())-prf_size>tol:
                #     pass
                for p in select_paras:
                    prf_ops.append((t,k,AllocateOp(Parameter(parameters[p]))))
                    op = PrefetchOp(alloc=Parameter(parameters[p]),
                                   indices=(0,None))
                    prf_ops.append((t,k,op))
                    Alive[p] = 1

        return ofl_ops, prf_ops, del_ops, init_ops


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

        if self.enable_offload:
            W = len(self.weights_size)
            (op_list, init_alive_status, init_op_list) = self.greedy_post_processing(
                hgraph
            )
        else:
            op_list = []
            init_op_list = []
            for t in range(T):
                for k in self.krange(t):
                    if t == self.loss_idx and k == self.loss_idx:
                        op_list.append(ComputeOp(self.hgraph.cluster.loss_kcn))
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

        ### debug
        # no_bwd = True
        # for op in op_list:
        #     if "bwd" in op.name:
        #         no_bwd = False
        # if no_bwd:
        #     raise("wrong solution")
        op_sched = OpSchedule(
            op_list,
            # ofl_list=ofl_list,
            # prf_list=prf_list,
            loss_idx=None,
            cluster=self.hgraph.cluster,
            # init_alive_status=init_alive_status,
            init_op_list=init_op_list,
            with_parameters=self.enable_offload,
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
        W = len(self.weights_size)

        self.ofl_ops = []
        self.prf_ops = []
        self.del_ops = []
        init_op_list = []
        if self.grouping:
            for w in range(self.W):
                o_l, p_l, d_l, i_l = self.group(w)
                self.ofl_ops.extend(o_l)
                self.prf_ops.extend(p_l)
                self.del_ops.extend(d_l)
                init_op_list.extend([ops[2] for ops in i_l])
        else:
            init_op_list = self.schedule_init_op_list()

        sol = self.sol
        # offload_buffers = {w:[] for w in range(W)}
        op_list = []
        init_alive_status = dict()
        # for kdn in self.hgraph.cluster.list_kdn_parameters:
        #     init_alive_status[kdn.name] = True
        # for w in range(W):
        #     hcn = self.hgraph.list_hcn[self.weight2hcn[w]]
        #     self.current_buffers[w] = Parameter(hcn.sub_cluster.name)

        for t in range(T):
            for k in self.krange(t):
                if t == self.loss_idx and k == self.loss_idx:
                    # loss_idx = len(op_list)
                    # loss_op = Op(K_C_node("loss"))

                    op_list.append(ComputeOp(self.hgraph.cluster.loss_kcn))
                j = self.hcn2sub_c[k]
                # if self.sumComp[t, k].value() == 1:
                prefetch_list = []
                for w in range(W):
                    prefetch_ops = self.create_prefetch_ops(t, k, w)
                    op_list.extend(prefetch_ops[0])
                    prefetch_list.extend(prefetch_ops[1])
                for w in range(W):
                    if not k in self.weight2hcn[w]:
                        op_list.extend(self.create_offload_ops(t, k, w))
                if sol(self.sumComp[t, k].value()):
                    # print(t,k)
                    hcn = hgraph.list_hcn[k]

                    opt = -1
                    for o in range(self.nOpts[k]):
                        if sol(self.Comp[t, k, o].value()):
                            opt = o
                            break
                    if opt > -1:
                        # print(k, t, opt)
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
                        if hcn.sub_cluster is not hcn.sub_cluster.representee_cluster:
                            sub_op_list = hcn.sub_cluster.translate_op_list(sub_op_list)

                    else:
                        h_obj = hcn
                        sub_op_list = deepcopy(h_obj.ff_op_list)

                    if hcn.sub_cluster is None:
                        continue
                    parameters = [
                        kdn.name for kdn in hcn.sub_cluster.list_kdn_parameters
                    ]

                    list_alloc_para = [
                        Parameter(kdn) for kdn in hcn.sub_cluster.list_kdn_parameters
                    ]
                    w = self.hcn2weight[k]

                    if not self.grouping and self.current_buffers[w] is not None:  # first time
                        self.current_buffers[w] = Buffer(
                            hcn.sub_cluster.name,
                            mem=sum(alloc.mem for alloc in list_alloc_para),
                        )
                        # Map buffer to parameter tensors
                        # op_list.extend([AllocateOp(alloc) for alloc in list_alloc_para])
                        op_list.append(
                            MappingOp(
                                name=hcn.sub_cluster.name + "_split",
                                sources=[self.current_buffers[w]],
                                targets=list_alloc_para,
                            )
                        )
                        op_list.append(DeleteOp(self.current_buffers[w]))

                    # print(t, k, len(sub_op_list), len(op_list))
                    op_list += sub_op_list

                    # Map parameter tensors to buffer
                    if not self.grouping:
                        if self.current_buffers[w] is None:
                            self.current_buffers[w] = Buffer(
                                hcn.sub_cluster.name,
                                mem=sum(alloc.mem for alloc in list_alloc_para),
                            )
                            # op_list.append(AllocateOp(self.current_buffers[w]))
                        op_list.append(
                            MappingOp(
                                name=hcn.sub_cluster.name + "_merge",
                                sources=list_alloc_para,
                                targets=[self.current_buffers[w]],
                            )
                        )
                        op_list.extend([DeleteOp(alloc) for alloc in list_alloc_para])

                for eidx, (k_, i) in enumerate(self.delete_list):
                    # print(k_, i)
                    # if k == k_ and self.delete[t, eidx].value()==1:
                    if k == k_ and sol(self.delete[t, eidx].value()):
                        hdn = hgraph.list_hdn[i]
                        op_list.append(DeleteOp(Activation(hdn.kdn)))

                for w in range(W):
                    # op_list.extend(self.create_prefetch_ops(t,k,w))
                    if k in self.weight2hcn[w]:
                        op_list.extend(self.create_offload_ops(t, k, w))
                    op_list.extend(self.create_delete_ops(t, k, w))
                op_list.extend(prefetch_list)
        return op_list, init_alive_status, init_op_list

    def create_delete_ops(self, t, k, w, itemsize=4):
        op_list = []
        sub_cluster = self.hgraph.list_hcn[self.weight2hcn[w][0]].sub_cluster
        if self.grouping:
            for (t_,k_,op) in self.del_ops:
                if t_==t and k_==k and op.target.kdn in sub_cluster.list_kdn_parameters:
                    op_list.append(op)
            return op_list

        parameter_mem = sum(kdn.mem for kdn in sub_cluster.list_kdn_parameters)
        parameter_size = round(parameter_mem / itemsize)
        t_, k_ = self.next_index(t, k)
        current_size = round(self.AliveW[(t, k, w)].value() * parameter_size)
        next_size = round(self.AliveW[(t_, k_, w)].value() * parameter_size)

        if current_size <= next_size:  # assume no prefetch then delete
            return op_list

        next_buffer = Buffer(sub_cluster.name, size=next_size)
        delete_buffer = Buffer(
            sub_cluster.name + "_delete", size=current_size - next_size
        )
        op_list.append(AllocateOp(delete_buffer))
        op_list.append(
            MappingOp(
                name=sub_cluster.name + "_divide",
                targets=[delete_buffer, next_buffer],
                sources=[self.current_buffers[w]],
            )
        )
        op_list.append(DeleteOp(delete_buffer))
        self.current_buffers[w] = next_buffer
        return op_list

    def create_prefetch_ops(self, t, k, w, itemsize=4):
        pre_op_list = []
        post_op_list = []
        sub_cluster = self.hgraph.list_hcn[self.weight2hcn[w][0]].sub_cluster

        if self.grouping:
            for (t_,k_,op) in self.prf_ops:
                if t_==t and k_==k and op.target.kdn in sub_cluster.list_kdn_parameters:
                    pre_op_list.append(op)

            return pre_op_list, post_op_list

        parameter_mem = sum(kdn.mem for kdn in sub_cluster.list_kdn_parameters)
        parameter_size = round(parameter_mem / itemsize)
        t_, k_ = self.next_index(t, k)
        current_size = round(self.AliveW[(t, k, w)].value() * parameter_size)
        next_size = round(self.AliveW[(t_, k_, w)].value() * parameter_size)
        if current_size >= next_size:  # assume no prefetch then delete
            return pre_op_list, post_op_list

        prefetch_buffer = Buffer(
            sub_cluster.name + "_prefetch",
            size=next_size - current_size,
        )
        pre_op_list.append(AllocateOp(prefetch_buffer))
        next_buffer = Buffer(sub_cluster.name, size=next_size)

        pre_op_list.append(
            PrefetchOp(
                alloc=prefetch_buffer,
                indices=(parameter_size - next_size, parameter_size - current_size),
                # after=op_list[-1],
            )
        )

        post_op_list.append(
            MappingOp(
                name=sub_cluster.name + "_add",
                sources=[prefetch_buffer, self.current_buffers[w]],
                targets=[next_buffer],
            )
        )
        self.current_buffers[w] = next_buffer
        post_op_list.append(DeleteOp(prefetch_buffer))
        return pre_op_list, post_op_list

    def create_offload_ops(self, t, k, w, itemsize=4):
        op_list = []
        sub_cluster = self.hgraph.list_hcn[self.weight2hcn[w][0]].sub_cluster
        if self.grouping:
            for (t_,k_,op) in self.ofl_ops:
                if t_==t and k_==k and op.target.kdn in sub_cluster.list_kdn_parameters:
                    op_list.append(op)
            return op_list

        parameter_mem = sum(kdn.mem for kdn in sub_cluster.list_kdn_parameters)
        parameter_size = round(parameter_mem / itemsize)
        progress_size = round(self.OflWProg[(t, k, w)].value() * parameter_size)
        if max(self.weight2hcn[w]) == t and t == k:  # bwd step
            progress_size = 0
        offload_size = -progress_size + round(
            (self.OflWProg[(t, k, w)].value() + self.OflW[(t, k, w)].value())
            * parameter_size
        )

        if offload_size == 0:
            return op_list

        start = -(parameter_size - progress_size)  # assumming progress cannot be full
        end = start + offload_size if start < -offload_size else None

        op_list.append(
            OffloadOp(
                alloc=self.current_buffers[w],
                indices=(start, end),
                # after=op_list[-1],
            )
        )
        return op_list

    def schedule_init_op_list(
        self,
        from_cpu=True,  # if assumming weights are all in cpu
    ):
        W = len(self.weights_size)
        init_op_list = []
        init_alive_status = dict()
        self.current_buffers = {}

        for w in range(W):
            hcn = self.hgraph.list_hcn[self.weight2hcn[w][0]]
            list_alloc_para = [
                Parameter(kdn) for kdn in hcn.sub_cluster.list_kdn_parameters
            ]
            parameter_mem = sum(alloc.mem for alloc in list_alloc_para)

            self.current_buffers[w] = Buffer(hcn.sub_cluster.name, mem=parameter_mem)
            target_buffer = (
                Buffer("cpu_" + hcn.sub_cluster.name, mem=parameter_mem)
                if from_cpu
                else self.current_buffers[w]
            )
            init_op_list.append(
                MappingOp(
                    name=hcn.sub_cluster.name + "_merge",
                    sources=list_alloc_para,
                    targets=[target_buffer],
                    copy=True,
                )
            )
            init_op_list.extend([DeleteOp(alloc) for alloc in list_alloc_para])
            parameter_size = round(parameter_mem / self.current_buffers[w].itemsize)

            if from_cpu:
                if self.AliveW[(0, 0, w)].value() > 0:
                    prefetch_size = round(
                        parameter_size * self.AliveW[(0, 0, w)].value()
                    )
                    # prefetch_size = round(parameter_size)
                    prefetch_buffer = Buffer(hcn.sub_cluster.name, size=prefetch_size)
                    init_op_list.append(AllocateOp(prefetch_buffer))

                    remaining_size = parameter_size - prefetch_size

                    init_op_list.append(
                        PrefetchOp(
                            alloc=prefetch_buffer,
                            indices=(remaining_size, None),
                            after=init_op_list[-1],
                        )
                    )
                    self.current_buffers[w] = prefetch_buffer
            else:
                if self.OflWProg[(0, 0, w)].value() > 0:
                    offload_size = round(
                        parameter_mem
                        * self.OflWProg[(0, 0, w)].value()
                        / self.current_buffers[w].itemsize
                    )
                    init_op_list.append(
                        OffloadOp(
                            alloc=self.current_buffers[w],
                            indices=(0, offload_size),
                        )
                    )

                    next_buffer = Buffer(
                        hcn.sub_cluster.name,
                        mem=parameter_mem * self.AliveW[(0, 0, w)].value(),
                    )
                    offload_buffer = Buffer(
                        hcn.sub_cluster.name + "_offload",
                        mem=parameter_mem - next_buffer.mem,
                    )

                    # init_op_list.append(AllocateOp(offload_buffer))
                    init_op_list.append(
                        MappingOp(
                            name=hcn.sub_cluster.name + "_divide",
                            targets=[offload_buffer, next_buffer],
                            sources=[self.current_buffers[w]],
                        )
                    )
                    self.current_buffers[w] = next_buffer
                    init_op_list.append(DeleteOp(offload_buffer))

        for kdn in self.hgraph.cluster.list_kdn_parameters:
            init_alive_status[kdn.name] = True
        return init_op_list
