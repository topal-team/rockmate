# import logging
# import math
from typing import Dict, Any
import numpy as np
from rockmate.def_op import RunOp, DelOp, OpSchedule
from mip import Model, xsum, maximize, BINARY, minimize, OptimizationStatus
from rkgb.utils.global_vars import solver_name


class ModelMIP:
    """
    The MIP model will build the ILP model by given Kgraph and budget.
    """

    def __init__(
        self,
        kg,
        budget: int,
        save_budget: int,
        gcd=None,
        gurobi_params: Dict[str, Any] = {},
    ):
        self.kg = kg
        self.time = [kcn.time for kcn in self.kg.list_kcn]
        self.gcd = gcd if gcd else 1
        self.budget = budget / self.gcd
        self.save_budget = save_budget / self.gcd
        self.overhead = [kcn.overhead.v / self.gcd for kcn in self.kg.list_kcn]
        self.mem = [kdn.mem.v / self.gcd for kdn in self.kg.list_kdn]
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

        self.md = Model(
            f"rockmateMILP_{T}_{budget}", solver_name=solver_name[0]
        )
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
        self.R = self.md.add_var_tensor((T, T), "R", var_type=BINARY)
        self.S = self.md.add_var_tensor((T, Cr), "S", var_type=BINARY)
        self.P = self.md.add_var_tensor((T, I), "P", var_type=BINARY)
        self.create = self.md.add_var_tensor((T, Cr), "create", var_type=BINARY)
        self.delete = self.md.add_var_tensor((T, De), "delete", var_type=BINARY)

        # define objective function
        self.md.objective = minimize(
            xsum(
                self.R[t, i] * self.time[i] for t in range(T) for i in range(T)
            )
        )

        # ======build constraints======
        self.md.add_constr(
            xsum(self.R[t, i] for t in range(T) for i in range(t + 1, T)) == 0,
        )
        self.md.add_constr(
            xsum(
                self.S[t, j]
                for j in range(Cr)
                for t in range(self.create_list[j][0] + 1)
            )
            == 0,
        )
        self.md.add_constr(
            xsum(
                self.P[t, i]
                for i in range(I)
                for t in range(min(_deps_d[i]) + 1)
            )
            == 0,
        )
        self.md.add_constr(xsum(self.R[t, t] for t in range(T)) == T)
        self.md.add_constr(
            xsum(self.R[t, self.loss_idx] for t in range(T)) == 1
        )  # fwd_loss can only run once

        for t in range(T):
            for j in range(Cr):
                self.md.add_constr(
                    self.S[t, j] <= self.P[t, self.create_list[j][1]],
                )
        for t in range(T - 1):
            for i in range(Cr):
                self.md.add_constr(
                    self.S[t + 1, i]
                    <= self.S[t, i] + self.R[t, self.create_list[i][0]],
                )
        # ensure all computations are possible
        for t in range(T):
            for j, (k, i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md.add_constr(
                        self.R[t, k_] <= self.R[t, k] + self.S[t, j],
                    )

        self.alive = {}
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.alive[(t, k, i)] = self.P[t, i]
                self.alive[(t, k, i)] += xsum(
                    self.create[t, eidx_c]
                    for eidx_c, (k_, i_) in enumerate(self.create_list)
                    if i_ == i and k_ <= k
                )
                self.alive[(t, k, i)] -= xsum(
                    self.delete[t, eidx_d]
                    for eidx_d, (k_, i_) in enumerate(self.delete_list)
                    if i_ == i and k_ <= k
                )
                self.md.add_constr(self.alive[(t, k, i)] >= 0)
                self.md.add_constr(self.alive[(t, k, i)] <= 1)
                if (k, i) in self.create_list:
                    didx = self.delete_list.index((k, i))
                    self.md.add_constr(
                        self.alive[(t, k, i)] + self.delete[t, didx]
                        >= self.R[t, k],
                    )

            for eidx, (k, i) in enumerate(self.create_list):
                self.md.add_constr(self.create[t, eidx] <= self.R[t, k])
            for i in range(I):
                if t + 1 < T:
                    self.md.add_constr(
                        self.P[t + 1, i]
                        == self.alive[(t, max(_deps_d[i] + _users_d[i]), i)],
                    )
                else:  # if i not in self.output_indices:
                    # in the end of bwd, del everything
                    self.md.add_constr(
                        self.alive[(t, max(_deps_d[i] + _users_d[i]), i)] == 0,
                    )

        def _num_hazards(t, i, k):
            if i in self.protected_indices:
                return _max_num_hazards(t, i, k)
            if t + 1 < T:
                return (
                    1
                    - self.R[t, k]
                    + self.P[t + 1, i]
                    + xsum(self.R[t, j] for j in _users_d[i] if j > k)
                )
            return (
                1
                - self.R[t, k]
                + xsum(self.R[t, j] for j in _users_d[i] if j > k)
            )

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in _users_d[i] if j > k)
            if t + 1 < T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

        # delete when not needed
        # for t in range(T):
        #     for eidx, (k, i) in enumerate(self.delete_list):
        #         self.md.add_constr(1 - self.delete[t, eidx]
        #                              <=
        #                             _num_hazards(t, i, k))

        # don't delete if still needed
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md.add_constr(
                    _max_num_hazards(t, i, k) * (1 - self.delete[t, eidx])
                    >= _num_hazards(t, i, k),
                )

        self.U = {}
        for t in range(T):
            self.U[(t, 0)] = (
                xsum(self.P[t, i] * self.mem[i] for i in range(I))
                + xsum(
                    self.create[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.create_list)
                    if k_ == 0
                )
                + xsum(
                    self.delete[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.delete_list)
                    if k_ == 0
                )
            )

        for t in range(T):
            for k in range(1, T):
                self.U[(t, k)] = (
                    self.U[(t, k - 1)]
                    + xsum(
                        self.create[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.create_list)
                        if k_ == k
                    )
                    - xsum(
                        self.delete[t, eidx] * self.mem[i]
                        for eidx, (k_, i) in enumerate(self.delete_list)
                        if k_ == k
                    )
                )
        for t in range(T):
            for k in range(T):
                self.md.add_constr(self.U[(t, k)] >= 0)
                self.md.add_constr(
                    self.U[(t, k)]
                    + self.R[t, k] * self.overhead[k]
                    + xsum(
                        self.mem[i_] * self.delete[t, eidx_d]
                        for eidx_d, (k_, i_) in enumerate(self.delete_list)
                        if k == k_
                    )
                    <= self.budget,
                )
                if t == T // 2 and self.save_budget:
                    self.md.add_constr(self.U[(t, k)] <= self.save_budget)

    def add_abar_constraint(self, save_budget):
        T = len(self.kg.list_kcn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
            t = T // 2
            self.md.add_constr(self.U[(t, k)] <= self.save_budget)

    def solve(self):

        # self.md.message("\n\nRestarting solve\n\n")
        self.md.optimize()

        infeasible = self.md.status == OptimizationStatus.INFEASIBLE
        if infeasible:
            self.feasible = False
        else:
            # if self.md.solCount < 1:
            #     raise ValueError(
            #         "Model status is {}, but solCount is {}".format(
            #             self.md.status, self.md.solCount
            #         )
            #     )
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
                if self.R[t, k] == 1:
                    kcn = kg.list_kcn[k]
                    if "loss" in kcn.name:
                        op_list.append(RunOp(kcn))
                        alive_list.append(alive_status.copy())
                        alive_status[kg.list_kdn.index(kg.output_kdn_grad)] = 1
                        # alive_status[kg.list_kdn.index(kg.output_kdn_data)] = 0
                    for eidx, (k_, i) in enumerate(self.create_list):
                        if k == k_ and self.create[t, eidx] == 1:
                            alive_status[i] = 1
                    op_list.append(RunOp(kcn))
                    alive_list.append(alive_status.copy())
                    # for i in range(I):
                    #     if self.alive[(t,k,i)].getValue():
                    #         alive_list[-1][i] = 1
                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k == k_ and self.delete[t, eidx] == 1:
                        kdn = kg.list_kdn[i]
                        if "phantom" in kdn.name:
                            alive_status[i] = 0
                            op_list.append(DelOp(kdn))
                            alive_list.append(alive_status.copy())
                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k == k_ and self.delete[t, eidx] == 1:
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
