# The ILP algorithm is inspired from Checkmate: https://github.com/mllg/checkmate
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

from rkgb.core.hierarchical import HierarchicalGraph
from .ilp_utils import RkLpVariable


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
        **kwargs,
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
        self.single_fwd = False
        self.single_bwd = False
        self.protected_names = protected_names
        self.hgraph = hgraph
        self.md = LpProblem(f"rockmateMILP", LpMinimize)
        self.use_correction_term = accurate_mem

        self.config_ilp()

    def build(self):
        # OVERWRITTING METHOD IN OFFLOAD
        self.add_variables()
        self.add_constraints()
        self.add_objective()

    def config_ilp(self):
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
                ff = True#not hcn.sub_cluster.is_bottom
                if hcn.is_fwd:
                    self.list_list_sched.append(hcn.list_sched)
                    self.sub_clusters.append(hcn.sub_cluster)
                j = self.sub_clusters.index(hcn.sub_cluster)
                self.hcn2sub_c.append(j)
                list_sched = self.list_list_sched[j]  # hcn bwd does not have list_sched
                # self.hcn2sub_c.append(len(self.list_list_sched) - 1)
                self.nComp.append(len(list_sched) + (1 if hcn.is_fwd and ff else 0))
                self.nSched.append(len(list_sched))

                if hcn.is_fwd:
                    # add fast forward to the options (final one)
                    self.time.append(
                        [op_sched.fwd_time for op_sched in list_sched] + 
                        ([hcn.ff_time] if ff else [])
                    )
                    self.overhead.append(
                        [op_sched.fwd_overhead / self.gcd for op_sched in list_sched]
                        + ([hcn.ff_overhead / self.gcd] if ff else [])
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

        self.hcn2idx = {hcn: i for i, hcn in enumerate(self.hgraph.list_HCNs)}
        self.han2idx = {hcn: i for i, hcn in enumerate(self.hgraph.list_HANs)}

        self.protected_indices = []
        self.input_grad_indices = []
        self.input_data_indices = []
        self.han_deps = []
        self.han_users = []
        self.han_users_real = []  # without sched-dependent users

        for i, han in enumerate(self.hgraph.list_HANs):
            if han.anode.name in self.protected_names:
                self.protected_indices.append(i)
            if han in self.hgraph.input_grad_HANs:
                self.input_grad_indices.append(i)
            if han in self.hgraph.input_data_HANs:
                self.input_data_indices.append(i)
            self.han_deps.append([self.hcn2idx[hcn] for hcn in han.deps])
            self.han_users.append(
                [self.hcn2idx[hcn] for hcn in han.users if hcn in self.hcn2idx]
            )  # current sub-graph users

        self.han_users_real = self.han_users.copy()
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
            cluster = self.sub_clusters[self.hcn2sub_c[k]]
            for op_sched in list_sched:
                # for i_ in op_sched.dep_interfaces_data:
                for anode in op_sched.dep_interfaces_data:
                    self_anode = cluster.translate_representee_node(anode)
                    # Without specifying schedule, we assume it's possible to use han here
                    for i in range(self.I):
                        if (
                            self_anode.name == self.hgraph.list_HANs[i].anode.name
                            and k not in self.han_users[i]
                        ):
                            self.han_users[i].append(k)

        ##############################

        self.create_list = [(k, i) for k in range(self.T) for i in self.hcn_users[k]]
        self.delete_list = [
            (k, i) for i in range(self.I) for k in self.han_deps[i] + self.han_users[i]
        ]

        self.Cr = len(self.create_list)
        self.De = len(self.delete_list)
        self.W = len(self.sub_clusters)

    def krange(self, t):
        if self.single_fwd:
            return [t]
        elif self.single_bwd and t > self.loss_idx:
            return list(range(self.loss_idx)) + [t]
        return list(range(t + 1))

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
            end = max(self.krange(t))
            if i < end:
                t_ = t
                i_ = self.krange(t)[self.krange(t).index(i) + 1]
            else:
                i_ = min(self.krange(t + 1))
                t_ = t + 1
        return (t_, i_)

    def add_objective(self, bandwidth_cost=0.01):
        self.md += lpSum(self.Time[t, k] for t in range(self.T) for k in self.krange(t))

    def add_constraints(self):
        self.add_valid_constraints()
        self.add_memory_constrains()

        ##### Time constraints
        for t in range(self.T):
            for k in self.krange(t):
                self.md += self.Time[t, k] >= lpSum(
                    self.Comp[t, k, o] * self.time[k][o] for o in range(self.nComp[k])
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
                self.sumAliveP[t, j] = lpSum(
                    self.AliveP[t, j, o] for o in range(len(self.list_list_sched[j]))
                )

        self.active_stages = dict()
        for i in range(self.I):
            self.active_stages[i] = []
            for t in range(self.T):
                for k in self.krange(t):
                    if k in self.han_deps[i] + self.han_users[i]:
                        self.active_stages[i].append(t)
            if not self.hgraph.list_HANs[i].deps:  # src node
                self.active_stages[i].append(-1)

        # to present whether one saved tensor can be inherited from the last stage
        self.AliveA = RkLpVariable.dicts(
            "AliveA",
            [
                (t, c)
                for t in range(1, self.T)
                for c, (k, i) in enumerate(self.create_list)
                #    if t-1 in self.active_stages[i]
            ],
            cat=self.bin_type,
        )  # activation

        self.AliveT = RkLpVariable.dicts(
            "AliveT",
            [
                (t, i)
                for t in range(self.T)
                for i in range(self.I)
                #    if t-1 in self.active_stages[i]
            ],
            cat=self.bin_type,
        )  # tensor that can be shared by acts

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

        self.Time = RkLpVariable.dicts(
            "Time",
            [(t, k) for t in range(self.T) for k in self.krange(t)],
            lowBound=None,
            upBound=None,
            cat="Continuous",
        )
        self.prefill_compute()
        # if self.with_offload:
        #     add_offload_variables(self)

    def add_valid_constraints(self):
        # In the last stage, every source edge of input_grad should be alive or executed
        for i in self.input_grad_indices:
            for j_, (k_, i_) in enumerate(self.create_list):
                if i_ == i:
                    self.md += (
                        self.AliveA[self.T - 1, j_] + self.sumComp[self.T - 1, k_] == 1
                    )

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
                    self.sumAliveP[t, j] <= 1
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
                    cluster = self.sub_clusters[j]
                    # for i in list_sched[o].dep_interfaces_data:
                    for anode in list_sched[o].dep_interfaces_data:
                        self_anode = cluster.translate_representee_node(anode)
                        hcn = self.hgraph.list_HCNs[bwd_i]
                        # Tensor req_i is required by BWD
                        req_i = [han.anode.name for han in self.hgraph.list_HANs].index(
                            self_anode.name
                        )
                        # req_i = self.han2idx[anode]
                        for j_, (k_, i_) in enumerate(self.create_list):
                            if i_ == req_i:
                                self.md += (
                                    self.Comp[t, bwd_i, o]
                                    <= self.sumComp[t, k_] + self.AliveA[t, j_]
                                )

    def add_activation_deletion(self):
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
                        self.md += self.AliveT[self.T - 1, i] == 0

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

    def add_activation_mem(self):
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
                    for o in range(self.nSched[0])
                )
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

    def save_mem(self, t, k):
        # OVERWRITTING METHOD IN OFFLOAD
        # it represents the saving memory AFTER deletion operations at step (t,k)
        return self.U[t, k]

    def overhead_mem(self, t, k):
        # OVERWRITTING METHOD IN OFFLOAD
        return self.act_overhead(t, k)

    def act_overhead(self, t, k):
        j = self.hcn2sub_c[k]
        if self.use_correction_term and not j is None:
            return self.correction_term_overhead(t, k)
        overhead = lpSum(
            self.Comp[t, k, o] * self.overhead[k][o] for o in range(self.nComp[k])
        )
        overhead += lpSum(
            self.mem[i_] * self.delete[t, eidx_d]
            for eidx_d, (k_, i_) in enumerate(self.delete_list)
            if k == k_
        )
        return [overhead]

    def correction_term_overhead(self, t, k):
        j = self.hcn2sub_c[k]
        cluster = self.sub_clusters[j]
        hcn = self.hgraph.list_HCNs[k]
        overhead_terms = []
        for o, op_sched in enumerate(self.list_list_sched[j]):
            correction_terms = (
                op_sched.fwd_overhead_correction
                if hcn.is_fwd
                else op_sched.bwd_overhead_correction
            )
            if not correction_terms:
                overhead_terms.append(
                    self.Comp[t, k, o] * self.overhead[k][o]
                    + lpSum(
                        self.mem[i_] * self.delete[t, eidx_d]
                        for eidx_d, (k_, i_) in enumerate(self.delete_list)
                        if k == k_
                    )
                )
            for correction in correction_terms:
                correction_term = 0
                overhead = (correction["save"] + correction["overhead"]) - (
                    op_sched.mem if hcn.is_fwd else 0
                )
                for inter_position, inter_mem in correction.items():
                    if inter_position == "save" or inter_position == "overhead":
                        continue

                    i_ = [han.anode.name for han in self.hgraph.list_HANs].index(
                        # self.sub_clusters[j]
                        cluster.translate_representee_node(
                            op_sched.cluster.list_anodes[inter_position[0]]
                        ).name
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

                overhead_terms.append(
                    self.Comp[t, k, o] * overhead / self.gcd
                    + correction_term / self.gcd
                    + lpSum(
                        self.mem[i_] * self.delete[t, eidx_d]
                        for eidx_d, (k_, i_) in enumerate(self.delete_list)
                        if k == k_
                    )
                )
        return overhead_terms

    def add_memory_constrains(self):
        self.add_activation_deletion()
        self.add_activation_mem()
        self.add_abar_constraint(self.save_budget * self.gcd)

        for t in range(self.T):
            for k in self.krange(t):
                self.md += self.save_mem(t, k) >= 0
                self.md += self.save_mem(t, k) <= (self.peak_budget)
                for overhead in self.overhead_mem(t, k):
                    self.md += self.save_mem(t, k) + overhead <= self.peak_budget

    def add_abar_constraint(self, save_budget):
        # self.save_budget = save_budget / self.gcd
        for k in range(self.T):
            self.md += self.U[(self.loss_idx, k)] <= save_budget / self.gcd

    def prefill_compute(self):
        self.active_stages = dict()
        for i in range(self.I):
            self.active_stages[i] = []
            for t in range(self.T):
                for k in self.krange(t):
                    if k in self.han_deps[i] + self.han_users[i]:
                        self.active_stages[i].append(t)
            if not self.hgraph.list_HANs[i].deps:  # src node
                self.active_stages[i].append(-1)

        for j, list_sched in enumerate(self.list_list_sched):
            for o in range(len(list_sched)):
                # if (0,j,o) not in self.AliveP:
                self.AliveP[0, j, o] = 0
                for t in range(1, self.T + 1):
                    # if (t,j,o) not in self.AliveP:
                    if not t - 1 in self.sub_c2hcn[j]:
                        self.AliveP[t, j, o] = self.AliveP[t - 1, j, o]

        for c, (k, i) in enumerate(self.create_list):
            self.AliveA[0, c] = 0
            for t in range(1, self.T):
                if not t - 1 in self.active_stages[i]:
                    self.AliveA[t, c] = self.AliveA[t - 1, c]

        for i in range(self.I):
            if not -1 in self.active_stages[i]:  # src node
                self.AliveT[0, i] = 0
            for t in range(1, self.T):
                if not t - 1 in self.active_stages[i]:
                    self.AliveT[t, i] = self.AliveT[t - 1, i]

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
                    if i in self.protected_indices:
                        continue
                    if k == max(self.han_deps[i] + self.han_users[i]) and k == t:
                        self.delete[t, d] = 1
                        pass

    def solve(self, solver=""):
        # some solvers have no support of 'Time limit reached' status
        print(f"Nb comp: {len(self.Comp)}, T:{self.T}")
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
            # print(f"finished solving in {self.md.solutionTime}")
            # print(f"objective {self.md.objective.value()}")
            self.solve_time = self.md.solutionTime
            print(f"Solved in {self.solve_time}s")
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
