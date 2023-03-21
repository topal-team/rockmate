# import logging
# import math
from typing import Dict, Any
import numpy as np
from gurobipy import GRB, Model, quicksum
from rockmate.def_op import RunOp, DelOp, OpSchedule
from rkgb.Htools import H_op, H_sched, H_option


class ModelGurobi:
    """
    The Gurobi model will build the ILP model by given Hgraph and budget.
    RN this model will take a rk_chain to solve the solution.
    """

    def __init__(
        self,
        hgraph,
        budget: int,
        save_budget: int,
        gurobi_params: Dict[str, Any] = {
            "LogToConsole": 0,
            "IntegralityFocus": 1,
        },
        gcd=None,
    ):
        #############################
        self.hgraph = hgraph
        self.hcn2sub_g = []
        self.sub_gs = []
        self.nOpts = []  # number of opts
        self.nR = []  # number to run R, =nOpts unless fwd
        self.time = []
        self.overhead = []
        for i, hcn in enumerate(self.hgraph.list_hcn):
            if "Loss" in hcn.name:
                self.loss_idx = i
            if hcn.sub_graph is None:
                # only when hcn is fwd with requires_grad=False
                self.hcn2sub_g.append(None)
                self.nR.append(1)
                self.nOpts.append(0)
                self.time.append([hcn.fwd_time])
                self.overhead.append([hcn.fwd_overhead])
            else:
                if hcn.sub_graph not in self.sub_gs:
                    self.sub_gs.append(hcn.sub_graph)
                self.hcn2sub_g.append(self.sub_gs.index(hcn.sub_graph))
                self.nR.append(
                    len(hcn.sub_graph.list_opt) + (1 if hcn.is_fwd else 0)
                )
                self.nOpts.append(len(hcn.sub_graph.list_opt))

                if hcn.is_fwd:
                    # add fast forward to the options
                    self.time.append(
                        [h_opt.fwd_time for h_opt in hcn.sub_graph.list_opt]
                        + [hcn.fwd_time]
                        if hcn.is_fwd
                        else []
                    )
                    self.overhead.append(
                        [h_opt.fwd_overhead for h_opt in hcn.sub_graph.list_opt]
                        + [hcn.fwd_overhead]
                    )
                else:
                    self.time.append(
                        [h_opt.bwd_time for h_opt in hcn.sub_graph.list_opt]
                    )
                    self.overhead.append(
                        [h_opt.bwd_overhead for h_opt in hcn.sub_graph.list_opt]
                    )
        self.sub_g2hcn = [[] for _ in self.sub_gs]
        for i, j in enumerate(self.hcn2sub_g):
            if j is None:
                continue
            self.sub_g2hcn[j].append(i)
        self.mem = [hdn.mem for hdn in self.hgraph.list_hdn]
        self.saved_mem = [
            [h_opt.mem for h_opt in sub_g.list_opt] for sub_g in self.sub_gs
        ]

        T = len(self.hgraph.list_hcn)
        I = len(self.hgraph.list_hdn)
        J = len(self.sub_gs)

        self.protected_indices = []

        _deps_d = [
            [self.hgraph.list_hcn.index(hcn) for hcn in hdn.deps]
            for hdn in self.hgraph.list_hdn
        ]
        _users_d = [
            [
                self.hgraph.list_hcn.index(hcn)
                for hcn in self.hgraph.list_hdn[i].users
                if hcn in self.hgraph.list_hcn
            ]
            for i in range(I)
        ]
        _users_c = [
            [
                self.hgraph.list_hdn.index(hdn)
                for hdn in self.hgraph.list_hcn[i].users
            ]
            for i in range(T)
        ]

        ##############################

        self.gcd = gcd if gcd else 1
        self.budget = budget / self.gcd
        self.save_budget = save_budget / self.gcd
        self.gurobi_params = gurobi_params
        self.feasible = None
        self.solve_time = None

        self.md = Model(f"rockmateMILP_{T}_{budget}")
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
        self.R = [
            self.md.addVars(T, self.nR[i], name=f"R{i}", vtype=GRB.BINARY,)
            for i in range(T)
        ]
        self.sumR = {}
        for i in range(T):
            for t in range(T):
                self.sumR[(i, t)] = quicksum(
                    self.R[i][t, o] for o in range(self.nR[i])
                )
        self.Sp = [
            self.md.addVars(
                T + 1, len(sub_g.list_opt), name=f"Sp{j}", vtype=GRB.BINARY,
            )
            for j, sub_g in enumerate(self.sub_gs)
        ]
        self.sumSp = {}
        for j in range(J):
            for t in range(T + 1):
                self.sumSp[(j, t)] = quicksum(
                    self.Sp[j][t, o]
                    for o in range(len(self.sub_gs[j].list_opt))
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

        # ======boundary constraints======
        self.md.addLConstr(
            quicksum(
                self.sumR[(i, t)] for t in range(T) for i in range(t + 1, T)
            ),
            GRB.EQUAL,
            0,
        )
        self.md.addLConstr(
            quicksum(
                self.sumSp[(self.hcn2sub_g[i], t)]
                for t in range(self.loss_idx)
                for i in range(t + 1, self.loss_idx)
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
        if _deps_d[i]:
            self.md.addLConstr(
                quicksum(
                    self.P[t, i]
                    for i in range(I)
                    for t in range(min(_deps_d[i]) + 1)
                ),
                GRB.EQUAL,
                0,
            )

        # options don't conflict
        for i in range(T):
            for t in range(T):
                self.md.addLConstr(
                    self.sumR[(i, t)], GRB.LESS_EQUAL, 1,
                )
        for j in range(J):
            for t in range(T + 1):
                self.md.addLConstr(
                    self.sumSp[(j, t)], GRB.LESS_EQUAL, 1,
                )  # assuming no keeping two copies of saved tensors at the same time

        # option-free constraints
        self.md.addLConstr(
            quicksum(self.sumR[(t, t)] for t in range(T)), GRB.EQUAL, T,
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

        # options-related constraints
        for j in range(J):
            fwd_i = min(self.sub_g2hcn[j])
            bwd_i = max(self.sub_g2hcn[j])
            for t in range(T):
                for o in range(self.nOpts[fwd_i]):
                    self.md.addLConstr(
                        self.Sp[j][t + 1, o],
                        GRB.LESS_EQUAL,
                        self.Sp[j][t, o] + self.R[fwd_i][t, o],
                    )
                    self.md.addLConstr(
                        self.R[bwd_i][t, o],
                        GRB.LESS_EQUAL,
                        self.Sp[j][t, o] + self.R[fwd_i][t, o],
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
                + quicksum(
                    self.delete[t, eidx] * self.mem[i]
                    for eidx, (k_, i) in enumerate(self.delete_list)
                    if k_ == 0
                )
                + quicksum(
                    self.Sp[j][t, o] * save_mem
                    for j in range(J)
                    for o, save_mem in enumerate(self.saved_mem[j])
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
                j = self.hcn2sub_g[k]
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
                    fwd_i = min(self.sub_g2hcn[j])
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
                j = self.hcn2sub_g[k]

                self.md.addLConstr(self.U[(t, k)], GRB.GREATER_EQUAL, 0)
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
                    self.budget,
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

    def schedule(self, hgraph=None):
        """ 
        Given the solution from HILP, we want to translate the result
        to a H_option that can be used in a higher level.
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(hgraph.list_hcn)
        I = len(hgraph.list_hdn)
        J = len(self.sub_gs)

        op_list = []
        alive_list = []
        # alive_status = np.zeros(I, dtype=bool)
        alive_status = {}
        sizes = {}
        for hdn in hgraph.list_hdn:
            alive_status[hdn.name] = (
                0 if (hdn in hgraph.inputs_hdn_data) else -1
            )
            sizes[hdn.name] = [hdn.mem]

        for sub_g in self.sub_gs:
            alive_status[sub_g.name] = -1  # to represent phantom from sub_g
            sizes[sub_g.name] = [h_opt.mem for h_opt in sub_g.list_opt]

        for t in range(T):
            for k in range(T):
                j = self.hcn2sub_g[k]
                if self.sumR[(k, t)].getValue() == 1:

                    hcn = hgraph.list_hcn[k]

                    opt = -1
                    for o in range(self.nOpts[k]):
                        if self.R[k][t, o].X == 1:
                            opt = o
                            break
                    # if hcn.is_fwd and self.R[k][t, -1].X == 1:
                    #     opt = -1
                    if opt > -1:
                        h_obj = hcn.sub_graph.list_opt[opt]
                    else:
                        h_obj = hcn

                    for eidx, (k_, i) in enumerate(self.create_list):
                        if k == k_ and self.create[t, eidx].X == 1:
                            alive_status[hgraph.list_hdn[i].name] = 0

                    # phantoms will be created when not ff
                    if hcn.is_fwd and j is not None:
                        alive_status[self.sub_gs[j].name] = opt

                    op_list.append(
                        H_op(hcn.name, h_obj, is_fwd=hcn.is_fwd, is_del=False,)
                    )
                    alive_list.append(alive_status.copy())

                    if (
                        not hcn.is_fwd
                        and self.sumSp[(j, t + 1)].getValue() == 0
                    ):
                        op_list.append(
                            H_op(
                                "Del_" + hcn.sub_graph.name, h_obj, is_del=True,
                            )
                        )  # del hcn.name means del phantom
                        alive_status[hcn.sub_graph.name] = -1
                        alive_list.append(alive_status.copy())

                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k == k_ and self.delete[t, eidx].X == 1:
                        hdn = hgraph.list_hdn[i]
                        alive_status[hdn.name] = -1
                        op_list.append(
                            H_op("Del_" + hdn.name, hdn, is_del=True)
                        )
                        alive_list.append(alive_status.copy())

            # At the end of the stage
            for j in range(J):
                # hcn = hgraph.list_hcn[self.sub_g2hcn[j]]

                if (
                    alive_status[self.sub_gs[j].name] >= 0
                    and self.sumSp[(j, t + 1)].getValue() == 0
                ):
                    # del phantom happens either after bwd or at the end of stage
                    h_obj = self.sub_gs[j].list_opt[
                        alive_status[self.sub_gs[j].name]
                    ]
                    alive_status[self.sub_gs[j].name] = -1
                    op_list.append(
                        H_op("Del_" + self.sub_gs[j].name, h_obj, is_del=True,)
                    )
                    alive_list.append(alive_status.copy())

        h_sched = H_sched(op_list, alive_list, sizes)
        # fwd_sched = OpSchedule(
        #     op_list[: self.loss_idx + 1],
        #     alive_list[: self.loss_idx + 1],
        #     hgraph,
        # )
        # bwd_sched = OpSchedule(
        #     op_list[self.loss_idx + 1 :],
        #     alive_list[self.loss_idx + 1 :],
        #     hgraph,
        # )
        # fwd_sched.del_input(hgraph)
        for i, op in enumerate(op_list):
            if "Loss" in op.name:
                loss_idx = i
                break
        fwd_sched, bwd_sched = h_sched.split_sched(loss_idx)
        h_option = H_option(hgraph, op_list, alive_list)
        return fwd_sched, bwd_sched, h_option


def add_option(hgraph, budget, save_budget):
    md = ModelGurobi(hgraph, budget, save_budget)
    md.solve()
    if md.feasible:
        _, _, h_option = md.schedule()
        hgraph.add_option(h_option)
