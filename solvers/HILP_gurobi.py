# import logging
# import math
from typing import Dict, Any
import numpy as np
from gurobipy import GRB, Model, quicksum

# from rockmate.def_op import RunOp, DelOp, OpSchedule
from solvers.op_schedule import Op, OpSchedule
from rkgb.Htools import *


class ModelGurobi:
    """
    The Gurobi model will build the ILP model by given Hgraph and budget.
    RN this model will take a rk_chain to solve the solution.
    """

    def __init__(
        self,
        hgraph: H_graph,
        peak_budget: int,
        save_budget: int,
        gurobi_params: Dict[str, Any] = {
            "LogToConsole": 0,
            "IntegralityFocus": 1,
            "TimeLimit": 4 * 60,
        },
        gcd=None,
        accurate_mem=False,
        protected_names=[],
    ):
        self.gcd = gcd if gcd else 1
        self.peak_budget = peak_budget / self.gcd
        self.save_budget = save_budget / self.gcd
        self.gurobi_params = gurobi_params
        self.feasible = None
        self.solve_time = None

        #############################
        self.hgraph = hgraph
        self.hcn2sub_c = []
        self.sub_cs = []
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
                    self.sub_cs.append(hcn.list_sched)
                # self.hcn2sub_c.append(self.sub_cs.index(hcn.sub_cluster))
                self.hcn2sub_c.append(len(self.sub_cs) - 1)
                self.nR.append(len(hcn.list_sched) + (1 if hcn.is_fwd else 0))
                self.nOpts.append(len(hcn.list_sched))

                if hcn.is_fwd:
                    # add fast forward to the options (final one)
                    self.time.append(
                        [op_sched.fwd_time for op_sched in hcn.list_sched]
                        + [hcn.ff_time]
                        if hcn.is_fwd
                        else []
                    )
                    self.overhead.append(
                        [
                            op_sched.fwd_overhead / self.gcd
                            for op_sched in hcn.list_sched
                        ]
                        + [hcn.ff_overhead / self.gcd]
                    )
                else:
                    self.time.append(
                        [op_sched.bwd_time for op_sched in hcn.list_sched]
                    )
                    self.overhead.append(
                        [
                            op_sched.bwd_overhead / self.gcd
                            for op_sched in hcn.list_sched
                        ]
                    )
        self.sub_c2hcn = [
            [] for _ in self.sub_cs
        ]  # index of sub_cluster to index of hcn
        for i, j in enumerate(self.hcn2sub_c):
            if j is None:
                continue
            self.sub_c2hcn[j].append(i)
        self.mem = [hdn.mem / self.gcd for hdn in self.hgraph.list_hdn]
        self.saved_mem = [
            [op_sched.mem / self.gcd for op_sched in sub_c]
            for sub_c in self.sub_cs
        ]

        T = len(self.hgraph.list_hcn)
        I = len(self.hgraph.list_hdn)
        J = len(self.sub_cs)

        self.protected_indices = [
            i
            for i, hdn in enumerate(self.hgraph.list_hdn)
            if hdn.kdn.name in protected_names
        ]
        self.input_grad_indices = [
            self.hgraph.list_hdn.index(hdn)
            for hdn in self.hgraph.inputs_hdn_grad
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
            [
                self.hgraph.list_hdn.index(hdn)
                for hdn in self.hgraph.list_hcn[i].users
            ]
            for i in range(T)
        ]  # outputs of hcn

        #### Update edges based on .dep_interfaces_data
        #### In certain schedules, BWD depends on input/output data
        for i in range(I):
            for k, hcn in enumerate(self.hgraph.list_hcn):
                if hcn.sub_cluster is None:
                    continue
                for op_sched in hcn.list_sched:
                    # Without specifying schedule, we assume it's possible to use hdn here
                    for i_ in op_sched.dep_interfaces_data:
                        if (
                            op_sched.list_kdn[i_]
                            == self.hgraph.list_hdn[i].kdn.name
                            and k not in _users_d[i]
                        ):
                            _users_d[i].append(k)

        ##############################

        self.md = Model(f"rockmateMILP_{T}_{peak_budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.md.Params, k, v)

        self.create_list = [(k, i) for k in range(T) for i in _users_c[k]]
        self.delete_list = [
            (k, i) for i in range(I) for k in _deps_d[i] + _users_d[i]
        ]

        Cr = len(self.create_list)
        De = len(self.delete_list)
        # print(Cr, De, I)
        # ======build variables======
        # For every HCN[i], R[i] is of size T*nR[i]
        self.R = [
            self.md.addVars(
                T,
                self.nR[i],
                name=f"R{i}",
                vtype=GRB.BINARY,
            )
            for i in range(T)
        ]

        self.sumR = {}
        for i in range(T):
            for t in range(T):
                self.sumR[(i, t)] = quicksum(
                    self.R[i][t, o] for o in range(self.nR[i])
                )

        # Sp for saved Phantoms, option-related
        self.Sp = [
            self.md.addVars(
                T + 1,
                len(sub_c),
                name=f"Sp{j}",
                vtype=GRB.BINARY,
            )
            for j, sub_c in enumerate(self.sub_cs)
        ]
        self.sumSp = {}
        for j in range(J):
            for t in range(T + 1):
                self.sumSp[(j, t)] = quicksum(
                    self.Sp[j][t, o] for o in range(len(self.sub_cs[j]))
                )

        # to present whether one saved tensor can be inheritaged from the last stage
        self.S = self.md.addVars(T, Cr, name="S", vtype=GRB.BINARY)
        self.P = self.md.addVars(T, I, name="P", vtype=GRB.BINARY)
        self.create = self.md.addVars(T, Cr, name="create", vtype=GRB.BINARY)
        self.delete = self.md.addVars(T, De, name="delete", vtype=GRB.BINARY)

        # define objective function
        self.md.setObjective(
            quicksum(
                self.R[i][t, o] * self.time[i][o]
                for i in range(T)
                for t in range(T)
                for o in range(self.nR[i])
            ),
            GRB.MINIMIZE,
        )

        # ======Boundary constraints======
        self.md.addLConstr(
            quicksum(
                self.sumR[(i, t)] for t in range(T) for i in range(t + 1, T)
            ),
            GRB.EQUAL,
            0,
        )
        self.md.addLConstr(
            quicksum(
                self.sumSp[(self.hcn2sub_c[i], t)]
                for t in range(self.loss_idx)
                for i in range(t + 1, self.loss_idx)
                if self.hcn2sub_c[i]
            ),
            GRB.EQUAL,
            0,
        )
        self.md.addLConstr(
            quicksum(
                self.S[t, j]
                for j in range(Cr)
                for t in range(self.create_list[j][0] + 1)
            ),
            GRB.EQUAL,
            0,
        )
        for i in range(I):
            if _deps_d[i]:
                self.md.addLConstr(
                    quicksum(self.P[t, i] for t in range(min(_deps_d[i]) + 1)),
                    GRB.EQUAL,
                    0,
                )

        # ======Correction constraints======

        # In the last stage, every source edge of input_grad should be alive or executed
        for i in self.input_grad_indices:
            for j_, (k_, i_) in enumerate(self.create_list):
                if i_ == i:
                    self.md.addLConstr(
                        self.S[T - 1, j_] + self.sumR[(k_, T - 1)], GRB.EQUAL, 1
                    )

        for j in range(J):
            bwd_i = max(self.sub_c2hcn[j])
            # Forward start with no phantoms
            self.md.addLConstr(
                quicksum(
                    (self.Sp[j][0, o])  # - self.R[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                ),
                GRB.EQUAL,
                0,
            )
            # in the end of bwd, del every phantoms
            self.md.addLConstr(
                quicksum(
                    (self.Sp[j][T, o])  # - self.R[bwd_i][T - 1, o])
                    for o in range(self.nOpts[bwd_i])
                ),
                GRB.EQUAL,
                0,
            )

        # options don't conflict
        for i in range(T):
            for t in range(T):
                self.md.addLConstr(
                    self.sumR[(i, t)],
                    GRB.LESS_EQUAL,
                    1,
                )
        for j in range(J):
            for t in range(T + 1):
                self.md.addLConstr(
                    self.sumSp[(j, t)],
                    GRB.LESS_EQUAL,
                    1,
                )  # assuming two copies of saved tensors won't be kept at the same time

        #### Option-free constraints: from rk-checkmate
        self.md.addLConstr(
            quicksum(self.sumR[(t, t)] for t in range(T)),
            GRB.EQUAL,
            T,
        )  # diagonal should be executed
        self.md.addLConstr(
            quicksum(self.sumR[(self.loss_idx, t)] for t in range(T)),
            GRB.EQUAL,
            1,
        )  # loss should be executed exactly once

        for t in range(T):
            for j in range(Cr):
                self.md.addLConstr(
                    self.S[t, j],
                    GRB.LESS_EQUAL,
                    self.P[t, self.create_list[j][1]],
                )  # one edge created, memory is occupied
        for t in range(T - 1):
            for j in range(Cr):
                src_i = self.create_list[j][0]
                self.md.addLConstr(
                    self.S[t + 1, j],
                    GRB.LESS_EQUAL,
                    self.S[t, j] + self.sumR[(src_i, t)],
                )
        for t in range(T):
            for j, (k, i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md.addLConstr(
                        self.sumR[(k_, t)],
                        GRB.LESS_EQUAL,
                        self.sumR[(k, t)] + self.S[t, j],
                    )

        #### Options-related constraints
        for j in range(J):
            fwd_i = min(self.sub_c2hcn[j])
            bwd_i = max(self.sub_c2hcn[j])
            for t in range(T):
                for o in range(self.nOpts[fwd_i]):
                    self.md.addLConstr(
                        self.Sp[j][t + 1, o],
                        GRB.LESS_EQUAL,
                        self.Sp[j][t, o] + self.R[fwd_i][t, o],
                    )  # phantoms can only be generated by fwd
                    self.md.addLConstr(
                        self.Sp[j][t + 1, o],
                        GRB.GREATER_EQUAL,
                        self.Sp[j][t, o]
                        - self.R[bwd_i][t, o]
                        + self.R[fwd_i][t, o],
                    )  # phantoms can only be deleted by bwd
                    self.md.addLConstr(
                        self.R[bwd_i][t, o],
                        GRB.LESS_EQUAL,
                        self.Sp[j][t, o] + self.R[fwd_i][t, o],
                    )

                    sub_c = self.sub_cs[j]
                    for i in sub_c[o].dep_interfaces_data:
                        name = sub_c[o].list_kdn[i].name
                        # Tensor req_i is required by BWD
                        req_i = [
                            hdn.kdn.name for hdn in self.hgraph.list_hdn
                        ].index(name)
                        for j_, (k_, i_) in enumerate(self.create_list):
                            if i_ == req_i:
                                self.md.addLConstr(
                                    self.R[bwd_i][t, o],
                                    GRB.LESS_EQUAL,
                                    self.sumR[(k_, t)] + self.S[t, j_],
                                )

        # ======Memory constraints======
        # we don't keep eyes on the alive status all the time
        # only the steps when changes can happen
        self.alive = {}
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.alive[(t, k, i)] = self.P[t, i]
                self.alive[(t, k, i)] += quicksum(
                    self.create[t, eidx_c]
                    for eidx_c, (k_, i_) in enumerate(self.create_list)
                    if i_ == i and k_ <= k
                )
                self.alive[(t, k, i)] -= quicksum(
                    self.delete[t, eidx_d]
                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                    if i_ == i and k_ <= k
                )
                self.md.addLConstr(self.alive[(t, k, i)], GRB.GREATER_EQUAL, 0)
                self.md.addLConstr(self.alive[(t, k, i)], GRB.LESS_EQUAL, 1)
                if (k, i) in self.create_list:
                    didx = self.delete_list.index((k, i))
                    self.md.addLConstr(
                        self.alive[(t, k, i)] + self.delete[t, didx],
                        GRB.GREATER_EQUAL,
                        self.sumR[(k, t)],
                    )

            for eidx, (k, i) in enumerate(self.create_list):
                self.md.addLConstr(
                    self.create[t, eidx], GRB.LESS_EQUAL, self.sumR[(k, t)]
                )
            for i in range(I):
                if t + 1 < T:
                    self.md.addLConstr(
                        self.P[t + 1, i],
                        GRB.EQUAL,
                        self.alive[(t, max(_deps_d[i] + _users_d[i]), i)],
                    )
                elif i not in self.protected_indices:
                    # in the end of bwd, del every HDN
                    self.md.addLConstr(
                        self.alive[(t, max(_deps_d[i] + _users_d[i]), i)],
                        GRB.EQUAL,
                        0,
                    )

        def _num_hazards(t, i, k):
            if i in self.protected_indices:
                return _max_num_hazards(t, i, k)
            if t + 1 < T:
                return (
                    1
                    - self.sumR[(k, t)]
                    + self.P[t + 1, i]
                    + quicksum(self.sumR[(j, t)] for j in _users_d[i] if j > k)
                )
            return (
                1
                - self.sumR[(k, t)]
                + quicksum(self.sumR[(j, t)] for j in _users_d[i] if j > k)
            )

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in _users_d[i] if j > k)
            if t + 1 < T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # delete when not needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md.addLConstr(
                    1 - self.delete[t, eidx],
                    GRB.LESS_EQUAL,
                    _num_hazards(t, i, k),
                )

        # don't delete if still needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md.addLConstr(
                    _max_num_hazards(t, i, k) * (1 - self.delete[t, eidx]),
                    GRB.GREATER_EQUAL,
                    _num_hazards(t, i, k),
                )

        self.U = {}
        for t in range(T):
            self.U[(t, 0)] = (
                quicksum(self.P[t, i] * self.mem[i] for i in range(I))
                + quicksum(
                    self.create[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.create_list)
                    if k_ == 0
                )
                - quicksum(
                    self.delete[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.delete_list)
                    if k_ == 0
                )
                + quicksum(
                    self.Sp[j][t, o] * save_mem
                    for j in range(J)
                    for o, save_mem in enumerate(self.saved_mem[j])
                )
                + quicksum(  # if the first fwd operation creates phantoms
                    self.R[0][t, o] * self.saved_mem[self.hcn2sub_c[0]][o]
                    for o in range(self.nOpts[0])
                )
                # - quicksum(
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
                    + quicksum(
                        self.create[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.create_list)
                        if k_ == k
                    )
                    - quicksum(
                        self.delete[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.delete_list)
                        if k_ == k
                    )
                )
                # if k < self.loss_idx:
                if self.hgraph.list_hcn[k].is_fwd:
                    self.U[(t, k)] += quicksum(
                        self.R[k][t, o] * self.saved_mem[j][o]
                        for o in range(self.nOpts[k])
                    )
                else:
                    if j is None:
                        continue
                    fwd_i = min(self.sub_c2hcn[j])
                    self.U[(t, k)] += quicksum(
                        (
                            self.Sp[j][t + 1, o]
                            - self.R[fwd_i][t, o]
                            - self.Sp[j][t, o]
                        )
                        * self.saved_mem[j][o]
                        for o in range(self.nOpts[k])
                    )
        for t in range(T):
            for k in range(T):
                j = self.hcn2sub_c[k]
                self.md.addLConstr(self.U[(t, k)], GRB.GREATER_EQUAL, 0)
                self.md.addLConstr(
                    self.U[(t, k)],
                    GRB.LESS_EQUAL,
                    self.peak_budget,
                )
                if j is None or not accurate_mem:
                    # don't consider correction_term
                    self.md.addLConstr(
                        self.U[(t, k)]
                        + quicksum(
                            self.R[k][t, o] * self.overhead[k][o]
                            for o in range(self.nR[k])
                        )
                        + quicksum(
                            self.mem[i_] * self.delete[t, eidx_d]
                            for eidx_d, (k_, i_) in enumerate(self.delete_list)
                            if k == k_
                        ),
                        GRB.LESS_EQUAL,
                        self.peak_budget,
                    )
                else:
                    hcn = self.hgraph.list_hcn[k]
                    for o, op_sched in enumerate(hcn.list_sched):
                        for correction in (
                            op_sched.fwd_overhead_correction
                            if hcn.is_fwd
                            else op_sched.bwd_overhead_correction
                        ):
                            correction_term = 0
                            overhead = (
                                correction["save"] + correction["overhead"]
                            ) - (op_sched.mem if hcn.is_fwd else 0)
                            for inter_position, inter_mem in correction.items():
                                if (
                                    inter_position == "save"
                                    or inter_position == "overhead"
                                ):
                                    continue

                                i_ = [
                                    hdn.kdn.name for hdn in self.hgraph.list_hdn
                                ].index(
                                    op_sched.list_kdn[inter_position[0]].name
                                )
                                if inter_position[1] == "always":
                                    not_kept_alive = 1
                                elif not inter_position[1]:  # ending status
                                    if (k, i_) in self.delete_list:
                                        eidx = self.delete_list.index((k, i_))
                                        not_kept_alive = self.delete[t, eidx]
                                    else:  # when output_data is not deps, but we care about it
                                        # eidx = self.delete_list.index((k, i_))
                                        k_ = max(
                                            [
                                                kk
                                                for kk in _users_d[i_]
                                                if kk < k
                                            ]
                                        )
                                        not_kept_alive = self.alive[(t, k_, i_)]
                                else:  # start status
                                    eidx = self.create_list.index((k, i_))
                                    not_kept_alive = self.create[t, eidx]
                                correction_term += not_kept_alive * inter_mem
                            self.md.addLConstr(
                                self.U[(t, k)]
                                + self.R[k][t, o] * overhead / self.gcd
                                + correction_term
                                + quicksum(
                                    self.mem[i_] * self.delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(
                                        self.delete_list
                                    )
                                    if k == k_
                                ),
                                GRB.LESS_EQUAL,
                                self.peak_budget,
                            )
                        if not (
                            op_sched.fwd_overhead_correction
                            if hcn.is_fwd
                            else op_sched.bwd_overhead_correction
                        ):
                            self.md.addLConstr(
                                self.U[(t, k)]
                                + self.R[k][t, o] * self.overhead[k][o]
                                + quicksum(
                                    self.mem[i_] * self.delete[t, eidx_d]
                                    for eidx_d, (k_, i_) in enumerate(
                                        self.delete_list
                                    )
                                    if k == k_
                                ),
                                GRB.LESS_EQUAL,
                                self.peak_budget,
                            )
                if t == self.loss_idx and self.save_budget:
                    self.md.addLConstr(
                        self.U[(t, k)], GRB.LESS_EQUAL, self.save_budget
                    )

    def add_abar_constraint(self, save_budget):
        T = len(self.hgraph.list_hcn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
            self.md.addLConstr(
                self.U[(self.loss_idx, k)], GRB.LESS_EQUAL, self.save_budget
            )

    def solve(self):
        # self.md.message("\n\nRestarting solve\n\n")
        self.md.optimize()
        if self.md.status == 9:
            print(
                f"GUROBI stopped early for reaching time limit with gap {self.md.MIPGap}"
            )
        # infeasible = self.md.status == GRB.INFEASIBLE
        if self.md.solCount < 1:
            self.feasible = False
        else:
            self.solve_time = self.md.Runtime
            self.feasible = True

    def schedule(self, hgraph=None):
        """
        Given the solution from HILP, we want to translate the result
        to a OpSchedule that can be used in a higher level.
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(hgraph.list_hcn)
        I = len(hgraph.list_hdn)
        J = len(self.sub_cs)

        op_list = []
        for t in range(T):
            for k in range(T):
                if t == self.loss_idx and k == self.loss_idx:
                    # loss_idx = len(op_list)
                    loss_op = Op(K_C_node("loss"))
                    op_list.append(loss_op)
                j = self.hcn2sub_c[k]
                if self.sumR[(k, t)].getValue() == 1:
                    hcn = hgraph.list_hcn[k]
                    opt = -1
                    for o in range(self.nOpts[k]):
                        if self.R[k][t, o].X == 1:
                            opt = o
                            break
                    if opt > -1:
                        h_obj = hcn.list_sched[opt]
                        if hcn.is_fwd:
                            sub_op_list = h_obj.op_list[: h_obj.loss_idx].copy()
                        else:
                            sub_op_list = h_obj.op_list[
                                h_obj.loss_idx + 1 :
                            ].copy()
                            # if self.sumSp[(j, t + 1)].getValue() == 0:
                            # sub_op_list.append()

                    else:
                        h_obj = hcn
                        sub_op_list = h_obj.ff_op_list.copy()

                    if (
                        not hcn.is_fwd and self.sumSp[(j, t + 1)].getValue() > 0
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

                    op_list += sub_op_list

                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k == k_ and self.delete[t, eidx].X == 1:
                        hdn = hgraph.list_hdn[i]
                        op_list.append(Op(hdn.kdn))

        interfaces = dict()
        interfaces["inputs_kdn_data"] = set(
            hdn.kdn for hdn in hgraph.inputs_hdn_data
        )
        interfaces["outputs_kdn_data"] = set(
            hdn.kdn for hdn in hgraph.outputs_hdn_data
        )
        interfaces["inputs_kdn_grad"] = set(
            hdn.kdn for hdn in hgraph.inputs_hdn_grad
        )
        interfaces["outputs_kdn_grad"] = set(
            hdn.kdn for hdn in hgraph.outputs_hdn_grad
        )
        # loss_idx =
        return OpSchedule(op_list, loss_idx=None, interfaces=interfaces)

    # def schedule(self, hgraph=None):
    #     """
    #     Given the solution from HILP, we want to translate the result
    #     to a H_option that can be used in a higher level.
    #     """
    #     hgraph = hgraph if hgraph else self.hgraph
    #     assert self.feasible, "Cannot schedule an infeasible model!"
    #     T = len(hgraph.list_hcn)
    #     I = len(hgraph.list_hdn)
    #     J = len(self.sub_cs)

    #     op_list = []
    #     alive_list = []
    #     # alive_status = np.zeros(I, dtype=bool)
    #     alive_status = {}
    #     sizes = {}
    #     for hdn in hgraph.list_hdn:
    #         alive_status[hdn.name] = (
    #             0 if (hdn in hgraph.inputs_hdn_data) else -1
    #         )
    #         sizes[hdn.name] = [hdn.mem]

    #     for sub_c in self.sub_cs:
    #         alive_status[sub_c.name] = -1  # to represent phantom from sub_c
    #         sizes[sub_c.name] = [op_sched.mem for op_sched in sub_c]

    #     for t in range(T):
    #         for k in range(T):
    #             j = self.hcn2sub_c[k]
    #             if self.sumR[(k, t)].getValue() == 1:
    #                 hcn = hgraph.list_hcn[k]

    #                 opt = -1
    #                 for o in range(self.nOpts[k]):
    #                     if self.R[k][t, o].X == 1:
    #                         opt = o
    #                         break
    #                 # if hcn.is_fwd and self.R[k][t, -1].X == 1:
    #                 #     opt = -1
    #                 if opt > -1:
    #                     h_obj = hcn.list_sched[opt]
    #                 else:
    #                     h_obj = hcn

    #                 for eidx, (k_, i) in enumerate(self.create_list):
    #                     if k == k_ and self.create[t, eidx].X == 1:
    #                         alive_status[hgraph.list_hdn[i].name] = 0

    #                 # phantoms will be created when not ff
    #                 if hcn.is_fwd and j is not None:
    #                     alive_status[self.sub_cs[j].name] = opt

    #                 op_list.append(
    #                     H_op(
    #                         hcn.name,
    #                         h_obj,
    #                         is_fwd=hcn.is_fwd,
    #                         is_del=False,
    #                     )
    #                 )
    #                 alive_list.append(alive_status.copy())

    #                 if (
    #                     not hcn.is_fwd
    #                     and self.sumSp[(j, t + 1)].getValue() == 0
    #                 ):
    #                     op_list.append(
    #                         H_op(
    #                             "Del_" + hcn.sub_cluster.name,
    #                             h_obj,
    #                             is_del=True,
    #                         )
    #                     )  # del hcn.name means del phantom
    #                     alive_status[hcn.sub_cluster.name] = -1
    #                     alive_list.append(alive_status.copy())

    #             for eidx, (k_, i) in enumerate(self.delete_list):
    #                 if k == k_ and self.delete[t, eidx].X == 1:
    #                     hdn = hgraph.list_hdn[i]
    #                     alive_status[hdn.name] = -1
    #                     op_list.append(
    #                         H_op("Del_" + hdn.name, hdn, is_del=True)
    #                     )
    #                     alive_list.append(alive_status.copy())

    #         # At the end of the stage
    #         for j in range(J):
    #             # hcn = hgraph.list_hcn[self.sub_c2hcn[j]]

    #             if (
    #                 alive_status[self.sub_cs[j].name] >= 0
    #                 and self.sumSp[(j, t + 1)].getValue() == 0
    #             ):
    #                 # del phantom happens either after bwd or at the end of stage
    #                 h_obj = self.sub_cs[j].list_sched[
    #                     alive_status[self.sub_cs[j].name]
    #                 ]
    #                 alive_status[self.sub_cs[j].name] = -1
    #                 op_list.append(
    #                     H_op(
    #                         "Del_" + self.sub_cs[j].name,
    #                         h_obj,
    #                         is_del=True,
    #                     )
    #                 )
    #                 alive_list.append(alive_status.copy())

    #     op_sched = H_sched(op_list, alive_list, sizes, hgraph)
    #     # fwd_sched = OpSchedule(
    #     #     op_list[: self.loss_idx + 1],
    #     #     alive_list[: self.loss_idx + 1],
    #     #     hgraph,
    #     # )
    #     # bwd_sched = OpSchedule(
    #     #     op_list[self.loss_idx + 1 :],
    #     #     alive_list[self.loss_idx + 1 :],
    #     #     hgraph,
    #     # )
    #     # fwd_sched.del_input(hgraph)
    #     for i, op in enumerate(op_list):
    #         if "Loss" in op.name:
    #             loss_idx = i
    #             break
    #     op_sched.get_info()
    #     # fwd_sched, bwd_sched = op_sched.split_sched(loss_idx)
    #     # op_schedion = H_option(hgraph, op_sched)
    #     # return fwd_sched, bwd_sched, op_schedion
    #     return op_sched


# def add_hilp_option(hgraph, budget, save_budget):
#     md = ModelGurobi(hgraph, budget, save_budget)
#     md.solve()
#     if md.feasible:
#         op_sched = md.schedule()
#         hgraph.add_sched(op_sched)
#         print(
#             f"Solve Hgraph with {len(hgraph.list_hcn)} nodes takes {md.solve_time:03f}s"
#         )


# def get_hg_budgets(hg, nb_bdg_peak=3, nb_bdg_save=6):
#     # return reasonable budget list
#     budgets = []
#     sizes = []
#     # fwd_hdns = set()
#     for hcn in hg.list_hcn:
#         # if hcn.is_fwd:
#         for hdn in hcn.users:
#             # if hdn not in hg.interfaces:
#             #     fwd_hdns.add(hdn)
#             if not hcn.sub_cluster is None:
#                 sizes.append(hcn.list_sched[0].mem)
#     sizes += [hdn.mem for hdn in hg.list_hdn]

#     overheads = [hcn.sub_cluster.ff_overhead for hcn in hg.list_hcn] + [
#         op_sched.bwd_overhead for op_sched in hg.list_sched
#     ]
#     max_bdg = sum(sizes) + max(overheads)
#     # max_bdg = hg.list_sched[0].mem + max(overheads)

#     # TODO: find the minimum feasible budget
#     # min_bdg = hg.fast_fwd_overhead()[0]
#     min_bdg = min(op_sched.mem for op_sched in hg.list_sched) + max(overheads)

#     l_bd_peak = np.linspace(min_bdg, max_bdg, nb_bdg_peak)
#     for bd_peak in l_bd_peak:
#         l_bd_save = np.linspace(
#             0,
#             min(bd_peak, hg.list_sched[0].mem),
#             nb_bdg_save,
#         ) + sum(hdn.mem for hdn in hg.interfaces)
#         # for bd_save in l_bd_save:
#         #     budgets.append((bd_peak, bd_save))
#         budgets.append((bd_peak, l_bd_save))
#     return budgets


# def solve_hg(hg: H_graph, print_info=False):
#     # print(f"solving hg {hg.name} with {len(hg.list_hcn)} nodes")
#     budgets = get_hg_budgets(hg)

#     for bdg_peak, l_bdg_save in budgets:
#         # print(bdg_peak)

#         md = ModelGurobi(hg, bdg_peak, save_budget=False)
#         # md = ModelGurobi(hg, 1e10, save_budget=False)
#         for bdg_save in np.sort(l_bdg_save)[::-1]:
#             # print(bdg_save)
#             md.add_abar_constraint(bdg_save)
#             md.solve()

#             # add_hilp_option(hg, bdg_peak, bdg_save)
#             if md.feasible:
#                 op_sched = md.schedule_()
#                 # print(op_sched.mem)
#                 hg.add_sched(op_sched)
#                 if print_info:
#                     print(
#                         f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {md.solve_time:03f}s"
#                     )
#     hg.refine_scheds()


# def solve_hg_recursive(hg: H_graph, solve_self=True, print_info=False):
#     for hcn in hg.list_hcn:
#         if hcn.is_fwd and hcn.sub_cluster is not None:
#             sub_c = hcn.sub_cluster
#             if len(sub_c) <= 1:
#                 solve_hg_recursive(sub_c, print_info=print_info)
#     if solve_self and len(hg.list_hcn) >= 1:  # not bottom hgraph
#         # print(f"Try to solve Hgraph with size {len(hg.list_hcn)}")
#         solve_hg(hg, print_info=print_info)
