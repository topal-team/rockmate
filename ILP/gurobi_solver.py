# import logging
# import math
import os
from typing import Dict, Any, Optional

import numpy as np
from gurobipy import GRB, Model, quicksum
from rockmate.def_code import RunOp, DelOp, OpSchedule
from rotor import timing#for debug
import torch
class ModelGurobi:
    """
    The Gurobi model will build the ILP model by given Kgraph and budget.
    
    """
    def __init__(self,
        kg,
        budget: int,
        save_budget: int,
        gurobi_params: Dict[str,Any] = None,
        gcd = None
        ):
        self.kg = kg
        self.time = [kcn.time for kcn in self.kg.list_kcn]
        self.gcd = gcd if gcd else 1
        self.budget = budget//self.gcd
        self.save_budget = save_budget//self.gcd
        self.overhead = [kcn.overhead.v//self.gcd for kcn in self.kg.list_kcn]
        self.mem = [kdn.mem.v//self.gcd for kdn in self.kg.list_kdn]
        self.gurobi_params = gurobi_params
        self.feasible = None
        self.solve_time = None

        self.output_indices = [self.kg.list_kdn.index(n) for n in 
                                    [self.kg.output_kdn_grad]]   
                                    #  self.kg.output_kdn_data]]
        self.protected_indices = []
        self.loss_idx = self.kg.list_kcn.index(self.kg.loss_kcn)
        T = len(self.kg.list_kcn)
        I = len(self.kg.list_kdn)

        self.md = Model(f"rockmateMILP_{T}_{budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.md.Params, k, v)

        ## useful functions
        # def _deps_d(i):
        #     return [self.kg.list_kcn.index(kcn)
        #             for kcn in self.kg.list_kdn[i].deps]
        # def _deps_c(i):
        #     return [self.kg.list_kdn.index(kdn)
        #             for kdn in self.kg.list_kcn[i].deps_real]
        # def _users_d(i):
        #     # TODO: there's user in the next graph?
        #     # return [self.kg.list_kcn.index(kcn)
        #     #         for kcn in self.kg.list_kdn[i].users_real]
        #     return [self.kg.list_kcn.index(kcn)
        #             for kcn in self.kg.list_kdn[i].users_real if kcn in self.kg.list_kcn]
        # def _users_c(i):
        #     return [self.kg.list_kdn.index(kdn)
        #             for kdn in self.kg.list_kcn[i].users]

        _deps_d = [[self.kg.list_kcn.index(kcn)
                    for kcn in self.kg.list_kdn[i].deps] for i in range(I)]
        _deps_c = [[self.kg.list_kdn.index(kdn)
                    for kdn in self.kg.list_kcn[i].deps_real] for i in range(T)]
        _users_d = [[self.kg.list_kcn.index(kcn)
                    for kcn in self.kg.list_kdn[i].users_real 
                    if kcn in self.kg.list_kcn] for i in range(I)]
            # TODO: there's user in the next graph?
            # return [self.kg.list_kcn.index(kcn)
            #         for kcn in self.kg.list_kdn[i].users_real]
        _users_c = [[self.kg.list_kdn.index(kdn)
                    for kdn in self.kg.list_kcn[i].users] for i in range(T)]

        self.create_list = [(k,i) for k,kcn in enumerate(self.kg.list_kcn)
                           for i in _users_c[k]]
        self.delete_list = [(k,i) for i,kdn in enumerate(self.kg.list_kdn)
                           for k in _deps_d[i] + _users_d[i]]

        Cr = len(self.create_list)
        De = len(self.delete_list)
        # ======build varaibles======
        self.R = self.md.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.md.addVars(T, Cr, name="S", vtype=GRB.BINARY)
        self.P = self.md.addVars(T, I, name="P", vtype=GRB.BINARY)
        self.create = self.md.addVars(T, Cr, name="create", vtype=GRB.BINARY)
        self.delete = self.md.addVars(T, De, name="delete", vtype=GRB.BINARY)

        # ======build constraints======
        # with Timer("Gurobi model construction", extra_data={"T": str(T),
        #            "budget": str(budget)}):
        #     with Timer("Objective construction", extra_data={"T": str(T),
        #                "budget": str(budget)}):
                # seed solver with a baseline strategy
                #if self.seed_s is not None:
                #    for x in range(T):
                #        for y in range(T):
                #            if self.seed_s[x, y] < 1:
                #                self.init_constraints.append(
                #                    self.md.addLConstr(self.S[x, y], GRB.EQUAL, 0))
                #    self.md.update()

                # define objective function
        self.md.setObjective(quicksum(self.R[t, i] * self.time[i]
                                        for t in range(T)
                                        for i in range(T)),
                                        GRB.MINIMIZE)
        # timer = timing.make_timer(torch.device("cpu"))
        # timer.start()

            # with Timer("Variable initialization",
            #            extra_data={"T": str(T), "budget": str(budget)}):
        #if self.imposed_schedule == ImposedSchedule.FULL_SCHEDULE:
        self.md.addLConstr(quicksum(self.R[t, i] for t in range(T)
                                    for i in range(t+1, T)),
                            GRB.EQUAL, 0)
        # self.md.addLConstr(quicksum(self.R[t, i] for i in range(T)
        #                             for t in range(T)),
        #                     GRB.EQUAL, 2*T)
        self.md.addLConstr(quicksum(self.S[t, j] for j in range(Cr)
                                    for t in range(self.create_list[j][0] + 1)),
                            GRB.EQUAL, 0)
        self.md.addLConstr(quicksum(self.P[t, i] for i in range(I)
                                    for t in range(min(_deps_d[i]) + 1)),
                            GRB.EQUAL, 0)
        self.md.addLConstr(quicksum(self.R[t, t] for t in range(T)),
                            GRB.EQUAL, T)
        self.md.addLConstr(quicksum(self.R[t, self.loss_idx] for t in range(T)),
                            GRB.EQUAL, 1)#fwd_loss can only run once
                #elif self.imposed_schedule == ImposedSchedule.COVER_ALL_NODES:
                #    self.md.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                #    for i in range(T):
                #        self.md.addLConstr(quicksum(self.R[t, i] for t in range(T)),
                #                                   GRB.GREATER_EQUAL, 1)
                #elif self.imposed_schedule == ImposedSchedule.COVER_LAST_NODE:
                #    self.md.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                #    # note: the integrality gap is very large as this constraint
                #    # is only applied to the last node (last column of self.R).
                #    self.md.addLConstr(quicksum(self.R[t, T - 1] for t in range(T)),
                #                               GRB.GREATER_EQUAL, 1)

            # with Timer("Correctness constraints",
            #            extra_data={"T": str(T), "budget": str(budget)}):
                # ensure all checkpoints are in memory
        for t in range(T):
            for j in range(Cr):
                self.md.addLConstr(self.S[t,j], GRB.LESS_EQUAL,
                                     self.P[t, self.create_list[j][1]])
        for t in range(T - 1):
            for i in range(Cr):
                self.md.addLConstr(self.S[t+1, i], GRB.LESS_EQUAL,
                                   self.S[t, i] + self.R[t,self.create_list[i][0]])
        # ensure all computations are possible
        for t in range(T):
            for j,(k,i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md.addLConstr(self.R[t, k_], GRB.LESS_EQUAL,
                                        self.R[t, k] + self.S[t, j])
        # for i in range(T):
        #     for t in range(T):
        #         for j in range(Cr):
        #             if self.create_list[j][1] in _deps_c[i]:
        #                 self.md.addLConstr(self.R[t, i], GRB.LESS_EQUAL,
        #                                     self.R[t, self.create_list[j][0]]
        #                                     + self.S[t, j])
        # timer.end()
        # print("Correctness constraints: %.4f"%timer.elapsed())
            # define memory constraints
            # with Timer("Presence constraints", extra_data={"T": str(T),
            #            "budget": str(budget)}):
                # ensure all checkpoints are in memory
        # timer.reset()
        # timer.start()
        self.alive = {}
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.alive[(t,k,i)] = self.P[t,i]
                self.alive[(t,k,i)] += quicksum(self.create[t,eidx_c]
                                            for eidx_c, (k_, i_)
                                            in enumerate(self.create_list)
                                            if i_==i and k_<=k)
                self.alive[(t,k,i)] -= quicksum(self.delete[t,eidx_d]
                                            for eidx_d, (k_, i_)
                                            in enumerate(self.delete_list)
                                            if i_==i and k_<=k)
                self.md.addLConstr(self.alive[(t,k,i)], GRB.GREATER_EQUAL, 0)
                self.md.addLConstr(self.alive[(t,k,i)], GRB.LESS_EQUAL, 1)
                if (k,i) in self.create_list:
                    didx = self.delete_list.index((k,i))
                    self.md.addLConstr(self.alive[(t,k,i)] +
                                        self.delete[t, didx],
                                        GRB.GREATER_EQUAL, self.R[t, k])

            for eidx, (k, i) in enumerate(self.create_list):
                self.md.addLConstr(self.create[t,eidx],
                                    GRB.LESS_EQUAL, self.R[t, k])
            for i in range(I):
                if t + 1 < T:
                    self.md.addLConstr(self.P[t+1,i], GRB.EQUAL,
                        self.alive[(t, max(_deps_d[i] + _users_d[i]), i)])
                else:#if i not in self.output_indices:
                    # in the end of bwd, del everything except output grad
                    self.md.addLConstr(self.alive[(t, 
                        max(_deps_d[i] + _users_d[i]), i)], GRB.EQUAL, 0)
        # timer.end()
        # print("Tensor state constraints: %.4f"%timer.elapsed())

        # timer.reset()
        # timer.start()
        def _num_hazards(t, i, k):
            if i in self.protected_indices: return _max_num_hazards(t, i, k)
            if t + 1 < T:
                return (1 - self.R[t, k] + self.P[t + 1, i] +
                        quicksum(self.R[t, j] for j in _users_d[i] if j > k))
            return (1 - self.R[t, k] +
                    quicksum(self.R[t, j] for j in _users_d[i] if j > k))

        def _max_num_hazards(t, i, k):
            num_uses_after_k = sum(1 for j in _users_d[i] if j > k)
            if t + 1 < T:
                return 2 + num_uses_after_k
            return 1 + num_uses_after_k

            # with Timer("Constraint: upper bound for 1 - delete",
            #            extra_data={"T": str(T), "budget": str(budget)}):
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md.addLConstr(1 - self.delete[t, eidx],
                                    GRB.LESS_EQUAL,
                                    _num_hazards(t, i, k))
        # with Timer("Constraint: lower bound for 1 - delete",
        #            extra_data={"T": str(T), "budget": str(budget)}):
        for t in range(T):
            for eidx, (k, i) in enumerate(self.delete_list):
                self.md.addLConstr(
                    _max_num_hazards(t, i, k) *
                    (1 - self.delete[t, eidx]),
                    GRB.GREATER_EQUAL, _num_hazards(t, i, k)
                )

            # with Timer(
            #     "Constraint: initialize memory usage",
            #     extra_data={"T": str(T), "budget": str(budget)},
            #     ):
    #     self.U = self.md.addVars(T, T, name="U", lb=0, ub=self.budget)
    #     for t in range(T):
    #         self.md.addLConstr(
    #             self.U[t, 0],
    #             GRB.EQUAL,
    #             quicksum(self.P[t, i] * self.mem[i] for i in range(I)) +
    #             quicksum(self.create[t, eidx]*self.mem[i]
    #                         for eidx, (k_,i) in
    #                         enumerate(self.create_list) if k_==0) +
    #             quicksum(self.delete[t, eidx]*self.mem[i]
    #                         for eidx, (k_,i) in
    #                         enumerate(self.delete_list) if k_==0)
    #         )
    # # with Timer("Constraint: memory recurrence",
    # #            extra_data={"T": str(T), "budget": str(budget)}):
    #     for t in range(T):
    #         for k in range(1,T):
    #             self.md.addLConstr(
    #                 self.U[t, k], GRB.EQUAL, self.U[t, k-1] +
    #                 quicksum(self.create[t, eidx]*self.mem[i]
    #                             for eidx, (k_,i) in
    #                             enumerate(self.create_list) if k_==k) -
    #                 quicksum(self.delete[t, eidx]*self.mem[i]
    #                             for eidx, (k_,i) in
    #                             enumerate(self.delete_list) if k_==k)
    #             )
    #     for t in range(T):
    #         for k in range(T):
    #             self.md.addLConstr(self.U[t, k], GRB.GREATER_EQUAL, 0)
    #             self.md.addLConstr(self.U[t, k] +
    #             self.R[t ,k]*self.overhead[k] +
    #             quicksum(self.mem[i_] * self.delete[t,eidx_d]
    #                      for eidx_d, (k_, i_) in enumerate(self.delete_list)
    #                      if k==k_), GRB.LESS_EQUAL, self.budget)
    #             if t == T//2 and self.save_budget:
    #                 self.md.addLConstr(self.U[t, k], GRB.LESS_EQUAL, self.save_budget)
        self.U = {}#self.md.addVars(T, T, name="U", lb=0, ub=self.budget)
        for t in range(T):
            self.U[(t, 0)] = (quicksum(self.P[t, i] * self.mem[i] for i in range(I)) +
                quicksum(self.create[t, eidx]*self.mem[i]
                            for eidx, (k_,i) in
                            enumerate(self.create_list) if k_==0) +
                quicksum(self.delete[t, eidx]*self.mem[i]
                            for eidx, (k_,i) in
                            enumerate(self.delete_list) if k_==0))
    # with Timer("Constraint: memory recurrence",
    #            extra_data={"T": str(T), "budget": str(budget)}):
        for t in range(T):
            for k in range(1,T):
                    self.U[(t, k)] = (self.U[(t, k-1)] +
                    quicksum(self.create[t, eidx]*self.mem[i]
                                for eidx, (k_,i) in
                                enumerate(self.create_list) if k_==k) -
                    quicksum(self.delete[t, eidx]*self.mem[i]
                                for eidx, (k_,i) in
                                enumerate(self.delete_list) if k_==k))
        for t in range(T):
            for k in range(T):
                self.md.addLConstr(self.U[(t, k)], GRB.GREATER_EQUAL, 0)
                self.md.addLConstr(self.U[(t, k)] +
                self.R[t ,k]*self.overhead[k] +
                quicksum(self.mem[i_] * self.delete[t,eidx_d]
                         for eidx_d, (k_, i_) in enumerate(self.delete_list)
                         if k==k_), GRB.LESS_EQUAL, self.budget)
                if t == T//2 and self.save_budget:
                    self.md.addLConstr(self.U[(t, k)], GRB.LESS_EQUAL, self.save_budget)
        # timer.end()
        # print("Memory constraints: %.4f"%timer.elapsed())

    def solve(self):
        # with Timer("Gurobi model optimization",
        #            extra_data={"T": str(T), "budget": str(self.budget)}):
            # if self.seed_s is not None:
            #     self.md.Params.TimeLimit = self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT
            #     self.md.optimize()
            #     if self.md.status == GRB.INFEASIBLE:
            #         print("Infeasible ILP seed at budget {:.2E}".format(self.budget))
            #     self.md.remove(self.init_constraints)
        #self.md.Params.TimeLimit = self.gurobi_params.get("TimeLimit", 0)
        self.md.message("\n\nRestarting solve\n\n")
        # with Timer("ILPSolve") as solve_ilp:
        
        self.md.optimize()
        #self.solve_time = solve_ilp.elapsed

        infeasible = self.md.status == GRB.INFEASIBLE
        if infeasible:
            self.feasible = False
            # return (None, None, None, None)
            # raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")
        else:
            if self.md.solCount < 1:
                raise ValueError("Model status is {}, but solCount is {}".format(
                                 self.md.status, self.md.solCount))
            self.feasible = True

    def schedule(self):
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(self.kg.list_kcn)
        I = len(self.kg.list_kdn)

        op_list = []
        alive_list = []
        alive_status = np.zeros(I, dtype=bool)
        for t in range(T):
            for k in range(T):
                if self.R[t, k].X:
                    kcn = self.kg.list_kcn[k]
                    for eidx, (k_, i) in enumerate(self.create_list):
                        if k==k_ and self.create[t, eidx].X:
                            alive_status[i] = 1
                    op_list.append(RunOp(kcn))
                    alive_list.append(alive_status.copy())
                    # for i in range(I):
                    #     if self.alive[(t,k,i)].getValue(): 
                    #         alive_list[-1][i] = 1
                for eidx, (k_, i) in enumerate(self.delete_list):
                    if k==k_ and self.delete[t, eidx].X:
                        kdn = self.kg.list_kdn[i]
                        alive_status[i] = 0
                        op_list.append(DelOp(kdn))
                        alive_list.append(alive_status.copy())
        for i,op in enumerate(op_list):
            if "loss" in op.name:
                loss_i = i
                break
        fwd_sched = OpSchedule(op_list[:loss_i+1], alive_list[:loss_i+1],
                               self.kg.list_kdn, output=self.kg.output_kdn_data)
        bwd_sched = OpSchedule(op_list[loss_i+1:], alive_list[loss_i+1:],
                               self.kg.list_kdn, output=self.kg.output_kdn_grad)
        return fwd_sched, bwd_sched