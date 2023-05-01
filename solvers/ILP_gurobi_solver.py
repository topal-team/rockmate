# import logging
# import math
from typing import Dict, Any
import numpy as np
from gurobipy import GRB, Model, quicksum
from rockmate.def_op import RunOp, DelOp, OpSchedule


class ModelGurobi:
    """
    The Gurobi model will build the ILP model by given Kgraph and budget.
    """

    def __init__(
        self,
        kg,
        budget: int,
        save_budget: int,
        gurobi_params: Dict[str, Any] = {},
        gcd=None,
    ):
        self.kg = kg
        self.time = [kcn.time for kcn in self.kg.list_kcn]
        self.gcd = gcd if gcd else 1
        self.budget = budget / self.gcd
        self.save_budget = save_budget / self.gcd
        self.overhead = [kcn.overhead / self.gcd for kcn in self.kg.list_kcn]
        self.mem = [kdn.mem / self.gcd for kdn in self.kg.list_kdn]
        self.gurobi_params = gurobi_params
        self.feasible = None
        self.solve_time = None

        self.output_indices = [
            self.kg.list_kdn.index(n) for n in [self.kg.output_kdn_grad]
        ]
        #  self.kg.output_kdn_data]]
        self.protected_indices = []
        self.loss_idx = self.kg.list_kcn.index(self.kg.loss_kcn)
        T = len(self.kg.list_kcn)
        I = len(self.kg.list_kdn)

        self.md = Model(f"rockmateMILP_{T}_{budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.md.Params, k, v)

        _deps_d = [
            [self.kg.list_kcn.index(kcn) for kcn in self.kg.list_kdn[i].deps]
            for i in range(I)
        ]
        # _deps_c = [[self.kg.list_kdn.index(kdn)
        #             for kdn in self.kg.list_kcn[i].deps_real] for i in range(T)]
        _users_d = [
            [
                self.kg.list_kcn.index(kcn)
                for kcn in self.kg.list_kdn[i].users_real
                if kcn in self.kg.list_kcn
            ]
            for i in range(I)
        ]
        # TODO: there's user in the next graph?
        # return [self.kg.list_kcn.index(kcn)
        #         for kcn in self.kg.list_kdn[i].users_real]
        _users_c = [
            [self.kg.list_kdn.index(kdn) for kdn in self.kg.list_kcn[i].users]
            for i in range(T)
        ]

        self.create_list = [
            (k, i)
            for k, kcn in enumerate(self.kg.list_kcn)
            for i in _users_c[k]
        ]
        self.delete_list = [
            (k, i)
            for i, kdn in enumerate(self.kg.list_kdn)
            for k in _deps_d[i] + _users_d[i]
        ]

        Cr = len(self.create_list)
        De = len(self.delete_list)
        # ======build varaibles======
        self.R = self.md.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.md.addVars(T, Cr, name="S", vtype=GRB.BINARY)
        self.P = self.md.addVars(T, I, name="P", vtype=GRB.BINARY)
        self.create = self.md.addVars(T, Cr, name="create", vtype=GRB.BINARY)
        self.delete = self.md.addVars(T, De, name="delete", vtype=GRB.BINARY)

        # define objective function
        self.md.setObjective(
            quicksum(
                self.R[t, i] * self.time[i] for t in range(T) for i in range(T)
            )
            # + quicksum(
            #     self.delete[t, k] * (T - t) / T * 0.1 * max(self.time)
            #     for t in range(T)
            #     for k in range(De)
            # )
            ,
            GRB.MINIMIZE,
        )

        # ======build constraints======
        self.md.addLConstr(
            quicksum(self.R[t, i] for t in range(T) for i in range(t + 1, T)),
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
        self.md.addLConstr(
            quicksum(
                self.P[t, i]
                for i in range(I)
                for t in range(min(_deps_d[i]) + 1)
            ),
            GRB.EQUAL,
            0,
        )
        self.md.addLConstr(
            quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T
        )
        self.md.addLConstr(
            quicksum(self.R[t, self.loss_idx] for t in range(T)), GRB.EQUAL, 1
        )  # fwd_loss can only run once

        for t in range(T):
            for j in range(Cr):
                self.md.addLConstr(
                    self.S[t, j],
                    GRB.LESS_EQUAL,
                    self.P[t, self.create_list[j][1]],
                )
        for t in range(T - 1):
            for i in range(Cr):
                self.md.addLConstr(
                    self.S[t + 1, i],
                    GRB.LESS_EQUAL,
                    self.S[t, i] + self.R[t, self.create_list[i][0]],
                )
        # ensure all computations are possible
        for t in range(T):
            for j, (k, i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md.addLConstr(
                        self.R[t, k_],
                        GRB.LESS_EQUAL,
                        self.R[t, k] + self.S[t, j],
                    )

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
                        self.R[t, k],
                    )

            for eidx, (k, i) in enumerate(self.create_list):
                self.md.addLConstr(
                    self.create[t, eidx], GRB.LESS_EQUAL, self.R[t, k]
                )
            for i in range(I):
                if t + 1 < T:
                    self.md.addLConstr(
                        self.P[t + 1, i],
                        GRB.EQUAL,
                        self.alive[(t, max(_deps_d[i] + _users_d[i]), i)],
                    )
                else:  # if i not in self.output_indices:
                    # in the end of bwd, del everything
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
                    - self.R[t, k]
                    + self.P[t + 1, i]
                    + quicksum(self.R[t, j] for j in _users_d[i] if j > k)
                )
            return (
                1
                - self.R[t, k]
                + quicksum(self.R[t, j] for j in _users_d[i] if j > k)
            )

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in _users_d[i] if j > k)
            if t + 1 < T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # delete when not needed
        # for t in range(T):
        #     for eidx, (k, i) in enumerate(self.delete_list):
        #         self.md.addLConstr(1 - self.delete[t, eidx],
        #                             GRB.LESS_EQUAL,
        #                             _num_hazards(t, i, k))

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
                + quicksum(
                    self.delete[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.delete_list)
                    if k_ == 0
                )
            )

        for t in range(T):
            for k in range(1, T):
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
        for t in range(T):
            for k in range(T):
                self.md.addLConstr(self.U[(t, k)], GRB.GREATER_EQUAL, 0)
                self.md.addLConstr(
                    self.U[(t, k)]
                    + self.R[t, k] * self.overhead[k]
                    + quicksum(
                        self.mem[i_] * self.delete[t, eidx_d]
                        for eidx_d, (k_, i_) in enumerate(self.delete_list)
                        if k == k_
                    ),
                    GRB.LESS_EQUAL,
                    self.budget,
                )
                if t == T // 2 and self.save_budget:
                    self.md.addLConstr(
                        self.U[(t, k)], GRB.LESS_EQUAL, self.save_budget
                    )

    def add_abar_constraint(self, save_budget):
        T = len(self.kg.list_kcn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
            t = T // 2
            self.md.addLConstr(self.U[(t, k)], GRB.LESS_EQUAL, self.save_budget)

    def solve(self):

        self.md.message("\n\nRestarting solve\n\n")
        self.md.optimize()

        infeasible = self.md.status == GRB.INFEASIBLE
        if infeasible:
            self.feasible = False
        else:
            if self.md.solCount < 1:
                raise ValueError(
                    "Model status is {}, but solCount is {}".format(
                        self.md.status, self.md.solCount
                    )
                )
            self.feasible = True

    def schedule(self, kg=None):
        kg = kg if kg else self.kg
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(kg.list_kcn)
        I = len(kg.list_kdn)

        op_list = []
        alive_list = []
        alive_status = np.zeros(I + 2, dtype=bool)
        alive_status[-1] = 1  # input_data_kdn
        for t in range(T):
            for k in range(T):
                if self.R[t, k].X == 1:
                    kcn = kg.list_kcn[k]
                    if "loss" in kcn.name:
                        op_list.append(RunOp(kcn))
                        alive_list.append(alive_status.copy())
                        alive_status[kg.list_kdn.index(kg.output_kdn_grad)] = 1
                        # alive_status[kg.list_kdn.index(kg.output_kdn_data)] = 0
                    for eidx, (k_, i) in enumerate(self.create_list):
                        if k == k_ and self.create[t, eidx].X == 1:
                            alive_status[i] = 1
                    op_list.append(RunOp(kcn))
                    alive_list.append(alive_status.copy())
                    # for i in range(I):
                    #     if self.alive[(t,k,i)].getValue():
                    #         alive_list[-1][i] = 1
                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k == k_ and self.delete[t, eidx].X == 1:
                        kdn = kg.list_kdn[i]
                        if "phantom" in kdn.name:
                            alive_status[i] = 0
                            op_list.append(DelOp(kdn))
                            alive_list.append(alive_status.copy())
                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k == k_ and self.delete[t, eidx].X == 1:
                        kdn = kg.list_kdn[i]
                        if "phantom" not in kdn.name:
                            alive_status[i] = 0
                            op_list.append(DelOp(kdn))
                            alive_list.append(alive_status.copy())
        for i, op in enumerate(op_list):
            if "loss" in op.name:
                loss_i = i
                break

        input_kdn = kg.input_kdn_data
        # if "src" not in input_kdn.name:
        #     del_input_op = DelOp(input_kdn)
        #     del_input_idx = len(op_list)
        #     for i, op in enumerate(op_list):
        #         if isinstance(op, RunOp) and input_kdn in op.deps_global:
        #             del_input_idx = i + 1
        #     op_list.insert(del_input_idx, del_input_op)
        #     alive_status = alive_list[del_input_idx - 1]
        #     alive_status[-1] = 0
        #     alive_list.insert(del_input_idx, alive_status)

        fwd_sched = OpSchedule(
            op_list[: loss_i + 1],
            alive_list[: loss_i + 1],
            kg.input_kdn_data,
            kg.input_kdn_grad,
            kg.output_kdn_data,
            kg.list_kdn,
        )
        bwd_sched = OpSchedule(
            op_list[loss_i + 1 :],
            alive_list[loss_i + 1 :],
            kg.input_kdn_data,
            kg.input_kdn_grad,
            kg.output_kdn_data,
            kg.list_kdn,
        )
        # fwd_sched.del_input(kg)
        return fwd_sched, bwd_sched
