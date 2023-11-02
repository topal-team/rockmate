# import logging
# import math
from typing import Dict, Any
import numpy as np
from copy import deepcopy
from pulp import *

from .op_schedule import Op, OpSchedule, OflOp, PrfOp
from rkgb.Htools import *
from rkgb.utils.global_vars import solver_name


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

        T = len(self.hgraph.list_hcn)
        W = len(self.hgraph.list_hcn)//2  # for now, one weight for each layer
        I = len(self.hgraph.list_hdn)
        J = len(self.list_list_sched)

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
            [self.hgraph.list_hdn.index(hdn) for hdn in self.hgraph.list_hcn[i].users]
            for i in range(T)
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
        self.Comp = [
            LpVariable.dicts(
                f"Comp{i}",
                [(t, k) for t in range(T) for k in range(self.nR[i])],
                cat="Binary",
            )
            for i in range(T)
        ]

        self.sumComp = {}
        for i in range(T):
            for t in range(T):
                self.sumComp[(i, t)] = lpSum(
                    self.Comp[i][t, o] for o in range(self.nR[i])
                )

        # Sp for saved Phantoms, option-related
        self.AliveP = [
            LpVariable.dicts(
                f"Alivep{j}",
                [(t, k) for t in range(T + 1) for k in range(len(list_sched))],
                cat="Binary",
            )
            for j, list_sched in enumerate(self.list_list_sched)
        ]
        self.sumAliveP = {}
        for j in range(J):
            for t in range(T + 1):
                self.sumAliveP[(j, t)] = lpSum(
                    self.AliveP[j][t, o] for o in range(len(self.list_list_sched[j]))
                )

        # to present whether one saved tensor can be inheritaged from the last stage
        self.AliveA = LpVariable.dicts(
            "AliveA", [(t, i) for t in range(T) for i in range(Cr)], cat="Binary"
        )# activation
        self.AliveT = LpVariable.dicts(
            "AliveT", [(t, i) for t in range(T) for i in range(I)], cat="Binary"
        )# tensor that can be shared by acts
        self.create = LpVariable.dicts(
            "create", [(t, i) for t in range(T) for i in range(Cr)], cat="Binary"
        )
        self.delete = LpVariable.dicts(
            "delete", [(t, i) for t in range(T) for i in range(De)], cat="Binary"
        )

        if self.enable_offload:
            self.AliveW = LpVariable.dicts(
                "AliveW",
                [(t, i, j) for t in range(T) for i in range(T) for j in range(W)],
                cat="Continuous",
                lowBound = 0,
                upBound=1
            )  # weight w is alive at the start of step j.
            self.OflW = LpVariable.dicts(
                "OflW",
                [(t, i, j) for t in range(T) for i in range(T) for j in range(W)],
                cat="Continuous",
                lowBound = 0,
                upBound=1
            )
            self.PrfW = LpVariable.dicts(
                "PrfW",
                [(t, i, j) for t in range(T) for i in range(T) for j in range(W)],
                cat="Continuous",
                lowBound = 0,
                upBound=1
            )
            # self.weights_size = [3e7 for _ in range(W)]
            self.weights_size = []
            for i in range(W):
                sub_cluster = self.hgraph.list_hcn[i].sub_cluster
                if hasattr(sub_cluster, "list_kdn_parameters"):
                    self.weights_size.append(sum(kdn.mem for kdn in sub_cluster.list_kdn_parameters) / self.gcd)
                else:
                    self.weights_size.append(0)
            # print(self.weights_size)
            self.weight2hcn = {w: [w, T-w-1] for w in range(W)}
            self.hcn2weight = {k:w for w in self.weight2hcn for k in self.weight2hcn[w]}
            self.bandwidthOfl = 8 * 1024**2  # byte/ms
            self.bandwidthPrf = 8 * 1024**2  # byte/ms

        self.Time = LpVariable.dicts(
            "Time", [(t, i) for t in range(T) for i in range(T)], cat="Continuous"
        )

        # define objective function
        self.md = LpProblem(f"rockmateMILP", LpMinimize)
        self.md += lpSum(
            self.Time[t, i]
            for t in range(T)
            for i in range(T)
        )

        ##### Time constraints
        for t in range(T):
            for i in range(T):
                self.md += (
                    self.Time[t, i] >= 
                    lpSum(self.Comp[i][t,o] * self.time[i][o]
                    for o in range(self.nR[i])),
                    "",
                )
                if self.enable_offload:
                    self.md += (
                        self.Time[t, i] >= 
                        lpSum(self.weights_size[w] /self.bandwidthPrf
                        * self.PrfW[t,i,w] for w in range(W)),
                        "",
                    )
                    self.md += (
                        self.Time[t, i] >= 
                        lpSum(self.weights_size[w]/self.bandwidthOfl
                        * self.OflW[t,i,w] for w in range(W)),
                        "",
                    )

        ##### Boundary constraints
        self.md += (
            lpSum(self.sumComp[(i, t)] for t in range(T) for i in range(t + 1, T)) == 0,
            "",
        )
        self.md += (
            lpSum(
                self.sumAliveP[(self.hcn2sub_c[i], t)]
                for t in range(self.loss_idx)
                for i in range(t + 1, self.loss_idx)
                if self.hcn2sub_c[i]
            )
            == 0,
            "",
        )
        self.md += (
            lpSum(
                self.AliveA[t, j]
                for j in range(Cr)
                for t in range(self.create_list[j][0] + 1)
            )
            == 0,
            "",
        )
        for i in range(I):
            if _deps_d[i]:
                self.md += (
                    lpSum(self.AliveT[t, i] for t in range(min(_deps_d[i]) + 1)) == 0,
                    "",
                )

        ##### Validity constraints

        # In the last stage, every source edge of input_grad should be alive or executed
        for i in self.input_grad_indices:
            for j_, (k_, i_) in enumerate(self.create_list):
                if i_ == i:
                    self.md += (
                        self.AliveA[T - 1, j_] + self.sumComp[(k_, T - 1)] == 1,
                        "",
                    )

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
                    (self.AliveP[j][0, o])  # - self.Comp[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                )
                == 0,
                "",
            )
            # in the end of bwd, del every phantoms
            self.md += (
                lpSum(
                    (self.AliveP[j][T, o])  # - self.Comp[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                )
                == 0,
                "",
            )

        

        # options don't conflict
        for i in range(T):
            for t in range(T):
                self.md += (self.sumComp[(i, t)] <= 1, "")
        for j in range(J):
            for t in range(T + 1):
                self.md += (
                    self.sumAliveP[(j, t)] <= 1,
                    "",
                )  # assuming two copies of saved tensors won't be kept at the same time

        #### Option-free constraints: from rk-checkmate
        self.md += (
            lpSum(self.sumComp[(t, t)] for t in range(T)) == T,
            "",
        )  # diagonal should be executed
        self.md += (
            lpSum(self.sumComp[(self.loss_idx, t)] for t in range(T)) == 1,
            "",
        )  # loss should be executed exactly once

        for t in range(T):
            for j in range(Cr):
                self.md += (
                    self.AliveA[t, j] <= self.AliveT[t, self.create_list[j][1]],
                    "",
                )  # one edge created, memory is occupied
        for t in range(T - 1):
            for j in range(Cr):
                src_i = self.create_list[j][0]
                self.md += (
                    self.AliveA[t + 1, j]
                    <= self.AliveA[t, j] + self.sumComp[(src_i, t)],
                    "",
                )
        for t in range(T):
            for j, (k, i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md += (
                        self.sumComp[(k_, t)]
                        <= self.sumComp[(k, t)] + self.AliveA[t, j],
                        "",
                    )
            if self.enable_offload:
                for w in self.weight2hcn:
                    for k in self.weight2hcn[w]:
                        self.md += (self.sumComp[(k, t)] <= self.AliveW[t, k, w], "")

        #### Options-related constraints
        for j in range(J):
            fwd_i = min(self.sub_c2hcn[j])
            bwd_i = max(self.sub_c2hcn[j])
            for t in range(T):
                for o in range(self.nOpts[fwd_i]):
                    self.md += (
                        self.AliveP[j][t + 1, o]
                        <= self.AliveP[j][t, o] + self.Comp[fwd_i][t, o],
                        "",
                    )  # phantoms can only be generated by fwd
                    self.md += (
                        self.AliveP[j][t + 1, o]
                        >= self.AliveP[j][t, o]
                        - self.Comp[bwd_i][t, o]
                        + self.Comp[fwd_i][t, o],
                        "",
                    )  # phantoms can only be deleted by bwd
                    self.md += (
                        self.Comp[bwd_i][t, o]
                        <= self.AliveP[j][t, o] + self.Comp[fwd_i][t, o],
                        "",
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
                                    self.Comp[bwd_i][t, o]
                                    <= self.sumComp[(k_, t)] + self.AliveA[t, j_],
                                    "",
                                )


        #### Offload constraints
        if self.enable_offload:
            self.OflWProg = dict()
            for t in range(T):
                for i in range(T):
                    for w in range(W):
                        bwd_i = max(self.weight2hcn[w])
                        if bwd_i<t:#after bwd of w
                            self.OflWProg[(t,i,w)] = (lpSum(self.OflW[t,ii,w] for ii in range(i)) +
                                lpSum(self.OflW[tt,ii,w] for tt in range(bwd_i+1, t) for ii in range(T)))
                        else:
                            self.OflWProg[(t,i,w)] = (lpSum(self.OflW[t,ii,w] for ii in range(i)) + 
                                lpSum(self.OflW[tt,ii,w] for tt in range(t) for ii in range(T)) +
                                lpSum(self.OflW[tt,ii,w] for tt in range(bwd_i+1, T) for ii in range(T)))
                        self.md += (
                            self.AliveW[t, i, w] + self.OflWProg[(t,i,w)]
                            >= 1,
                            "",
                        )
                        if i < T - 1:
                            self.md += (
                                self.AliveW[t, i + 1, w]
                                <= self.AliveW[t, i, w] + self.sumComp[(i,t)],
                                "",
                            )
                            self.md += (
                                self.AliveW[t, i + 1, w]
                                >= self.AliveW[t, i, w] - self.sumComp[(i,t)],
                                "",
                            )
                            self.md += (
                                self.AliveW[t, i + 1, w]
                                <= self.AliveW[t, i, w] + self.PrfW[t, i, w],
                                ""
                            )
            for w in range(W):
                for t in range(T-1):
                    self.md += (
                        self.AliveW[t+1, 0, w]
                        <= self.AliveW[t, T-1, w] + self.PrfW[t, T-1, w],
                        ""
                    )
                self.md += (
                    self.AliveW[0, 0, w]
                    <= self.AliveW[T-1, T-1, w] + self.PrfW[T-1, T-1, w],
                    ""
                )

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
                self.md += (self.alive[(t, k, i)] >= 0, "")
                self.md += (self.alive[(t, k, i)] <= 1, "")
                if (k, i) in self.create_list:
                    didx = self.delete_list.index((k, i))
                    self.md += (
                        self.alive[(t, k, i)] + self.delete[t, didx]
                        >= self.sumComp[(k, t)],
                        "",
                    )

            for eidx, (k, i) in enumerate(self.create_list):
                self.md += (self.create[t, eidx] <= self.sumComp[(k, t)], "")
            for i in range(I):
                if t + 1 < T:
                    self.md += (
                        self.AliveT[t + 1, i]
                        == self.alive[(t, max(_deps_d[i] + _users_d[i]), i)],
                        "",
                    )
                elif i not in self.protected_indices:
                    # in the end of bwd, del every HDN
                    self.md += (
                        self.alive[(t, max(_deps_d[i] + _users_d[i]), i)] == 0,
                        "",
                    )

        def _num_hazards(t, i, k):
            if i in self.protected_indices:
                return _max_num_hazards(t, i, k)
            if t + 1 < T:
                return (
                    1
                    - self.sumComp[(k, t)]
                    + self.AliveT[t + 1, i]
                    + lpSum(self.sumComp[(j, t)] for j in _users_d[i] if j > k)
                )
            return (
                1
                - self.sumComp[(k, t)]
                + lpSum(self.sumComp[(j, t)] for j in _users_d[i] if j > k)
            )

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in _users_d[i] if j > k)
            if t + 1 < T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # delete when not needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += (1 - self.delete[t, eidx] <= _num_hazards(t, i, k), "")

        # don't delete if still needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md += (
                    _max_num_hazards(t, i, k) * (1 - self.delete[t, eidx])
                    >= _num_hazards(t, i, k),
                    "",
                )
                if i in self.protected_indices:
                    self.md += (self.delete[t, eidx] == 0, "")

        self.U = {}
        for t in range(T):
            self.U[(t, 0)] = (
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
                    self.AliveP[j][t, o] * save_mem
                    for j in range(J)
                    for o, save_mem in enumerate(self.saved_mem[j])
                )
                + lpSum(  # if the first fwd operation creates phantoms
                    self.Comp[0][t, o] * self.saved_mem[self.hcn2sub_c[0]][o]
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
                self.U[(t, k)] = (
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
                    self.U[(t, k)] += lpSum(
                        self.Comp[k][t, o] * self.saved_mem[j][o]
                        for o in range(self.nOpts[k])
                    )
                else:
                    if j is None:
                        continue
                    fwd_i = min(self.sub_c2hcn[j])
                    self.U[(t, k)] += lpSum(
                        (
                            self.AliveP[j][t + 1, o]
                            - self.Comp[fwd_i][t, o]
                            - self.AliveP[j][t, o]
                        )
                        * self.saved_mem[j][o]
                        for o in range(self.nOpts[k])
                    )
        for t in range(T):
            for k in range(T):
                if self.enable_offload:
                    # weight_mem = 0
                    weight_mem = lpSum(self.AliveW[t,k,w]*self.weights_size[w]
                                       for w in range(W))
                else:
                    weight_mem = 0
                j = self.hcn2sub_c[k]
                self.md += (self.U[(t, k)] >= 0, "")
                self.md += (self.U[(t, k)] <= self.peak_budget - weight_mem, "")
                if j is None or not accurate_mem:
                    # don't consider correction_term
                    self.md += (
                        self.U[(t, k)]
                        + lpSum(
                            self.Comp[k][t, o] * self.overhead[k][o]
                            for o in range(self.nR[k])
                        )
                        + lpSum(
                            self.mem[i_] * self.delete[t, eidx_d]
                            for eidx_d, (k_, i_) in enumerate(self.delete_list)
                            if k == k_
                        )
                        <= self.peak_budget - weight_mem,
                        "",
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
                                self.U[(t, k)]
                                + self.Comp[k][t, o] * overhead / self.gcd
                                + correction_term
                                + lpSum(
                                    self.mem[i_] * self.delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                                    if k == k_
                                )
                                <= self.peak_budget - weight_mem,
                                "",
                            )
                        if not (
                            op_sched.fwd_overhead_correction
                            if hcn.is_fwd
                            else op_sched.bwd_overhead_correction
                        ):
                            self.md += (
                                self.U[(t, k)]
                                + self.Comp[k][t, o] * self.overhead[k][o]
                                + lpSum(
                                    self.mem[i_] * self.delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                                    if k == k_
                                )
                                <= self.peak_budget - weight_mem,
                                "",
                            )
                if t == self.loss_idx and self.save_budget:
                    self.md += (self.U[(t, k)] <= self.save_budget, "")

    def add_abar_constraint(self, save_budget):
        T = len(self.hgraph.list_hcn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
            self.md += (self.U[(self.loss_idx, k)] <= self.save_budget, "")

    def solve(self, solver=""):
        # self.md.message("\n\nRestarting solve\n\n")
        # solver = get_solver(solver, msg=0)
        # solver = solver or solver_name[0]
        try:
            solver = get_solver(solver, msg=0)
        except:
            avail_solver = listSolvers(onlyAvailable=True)[0]
        #     print(f"Cannot get {solver}, will use {avail_solver}")
            solver = get_solver(avail_solver, msg=0)

        status = self.md.solve(solver)
        self.status = LpStatus[status]  # readable status
        self.feasible = status == 1

        # if self.md.status == 9:
        #     print(
        #         f"GUROBI stopped early for reaching time limit with gap {self.md.MIPGap}"
        #     )
        # # infeasible = self.md.status == GRB.INFEASIBLE
        # if self.md.solCount < 1:
        #     self.feasible = False
        # else:
        #     self.solve_time = self.md.Runtime
        #     self.feasible = True
        if self.feasible:
            self.solve_time = self.md.solutionTime

    def schedule(self, hgraph=None, check_valid=False):
        """
        Given the solution from HILP, we want to translate the result
        to a OpSchedule that can be used in a higher level.
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(hgraph.list_hcn)
        I = len(hgraph.list_hdn)
        J = len(self.list_list_sched)
        if self.enable_offload:
            W = len(self.weights_size)

        def sol(value):
            return value > 0.9999  # inttol

        op_list = []
        ofl_list = []
        prf_list = []
        for t in range(T):
            for k in range(T):
                if t == self.loss_idx and k == self.loss_idx:
                    # loss_idx = len(op_list)
                    # loss_op = Op(K_C_node("loss"))

                    op_list.append(Op(self.hgraph.cluster.loss_kcn))
                j = self.hcn2sub_c[k]
                # if self.sumComp[(k, t)].value() == 1:
                if sol(self.sumComp[(k, t)].value()):
                    hcn = hgraph.list_hcn[k]
                    opt = -1
                    for o in range(self.nOpts[k]):
                        if sol(self.Comp[k][t, o].value()):
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
                            for op in sub_op_list[::-1]:
                                if (
                                    op.is_del
                                    and not op.disabled
                                    and op.kn in phantoms_to_keep
                                ):
                                    # Only the last del should be disabled
                                    op.disabled = True
                                    phantoms_to_keep.remove(op.kn)

                        # translating sub_op_list
                        if hcn.sub_cluster is not hcn.sub_cluster.representee_cluster:
                            sub_op_list = hcn.sub_cluster.translate_op_list(sub_op_list)
                            # translator_re = (
                            #     hcn.sub_cluster.representee_cluster.translator
                            # )
                            # translator = hcn.sub_cluster.translator
                            # for op in sub_op_list:
                            #     if op.is_del:
                            #         ana_kn = (
                            #             translator_re.dict_name_to_ano_triplet[
                            #                 op.kn.name
                            #             ]
                            #         )
                            #         op.kn = translator.dict_ano_triplet_to_kdn[
                            #             ana_kn
                            #         ]
                            #     else:
                            #         ana_kn = (
                            #             translator_re.dict_name_to_ano_triplet[
                            #                 op.kn.name
                            #             ]
                            #         )
                            #         op.kn = translator.dict_ano_triplet_to_kcn[
                            #             ana_kn
                            #         ]

                    else:
                        h_obj = hcn
                        sub_op_list = deepcopy(h_obj.ff_op_list)

                    op_list += sub_op_list

                for eidx, (k_, i) in enumerate(self.delete_list):
                    # print(k_, i)
                    # if k == k_ and self.delete[t, eidx].value()==1:
                    if k == k_ and sol(self.delete[t, eidx].value()):
                        hdn = hgraph.list_hdn[i]
                        op_list.append(Op(hdn.kdn))

                if self.enable_offload:
                    for w in range(W):
                        weight = self.hgraph.list_hcn[self.weight2hcn[w][0]].sub_cluster
                        if self.OflW[(t,k,w)].value()>0:
                            ofl_list.append(OflOp(target=weight.name, 
                                                  fraction=self.OflW[(t,k,w)].value(),
                                                  after=op_list[-1]))
                        if self.PrfW[(t,k,w)].value()>0:
                            prf_list.append(PrfOp(target=weight.name, 
                                                  fraction=self.PrfW[(t,k,w)].value(),
                                                  after=op_list[-1]))


        # interfaces = dict()
        # interfaces["inputs_kdn_data"] = set(
        #     hdn.kdn for hdn in hgraph.inputs_hdn_data
        # )
        # interfaces["outputs_kdn_data"] = set(
        #     hdn.kdn for hdn in hgraph.outputs_hdn_data
        # )
        # interfaces["inputs_kdn_grad"] = set(
        #     hdn.kdn for hdn in hgraph.inputs_hdn_grad
        # )
        # interfaces["outputs_kdn_grad"] = set(
        #     hdn.kdn for hdn in hgraph.outputs_hdn_grad
        # )
        # loss_idx =
        ### debug
        # no_bwd = True
        # for op in op_list:
        #     if "bwd" in op.name:
        #         no_bwd = False
        # if no_bwd:
        #     raise("wrong solution")
        op_sched = OpSchedule(op_list, 
                              ofl_list=ofl_list, 
                              prf_list=prf_list,
                              loss_idx=None, 
                              cluster=self.hgraph.cluster)
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
