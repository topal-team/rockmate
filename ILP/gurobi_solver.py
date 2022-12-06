import logging
import math
import os
from typing import Dict, Any, Optional

import numpy as np
from gurobipy import GRB, Model, quicksum
from rockmate.def_code import RunOp, DelOp

class ModelGurobi:
    def __init__(self,
        kg: K_graph,
        budget: int,
        gurobi_params: Dict[str,Any] = None,
        gcd = None
        ):
        self.kg = kg
        self.time = [kcn.time for kcn in self.kg.list_kcn]
        self.gcd = gcd
        if self.gcd:
            self.budget = budget//self.gcd
            self.mem = [kdn.mem//self.gcd for kdn in self.kg.list_kdn]
        self.gurobi_params = gurobi_params

        T = len(self.kg.list_kcn)
        I = len(self.kg.list_kdn)

        self.md = Model(f"rockmateMILP_{T}_{budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.md.Params, k, v)

        ## useful functions
        def _deps_d(i):
            return [self.kg.list_kcn.index(kcn)
                        for kcn in self.kg.list_kdn[i].deps]
        def _deps_c(i):
            return [self.kg.list_kdn.index(kdn)
                        for kdn in self.kg.list_kcn[i].deps]
        def _users_d(i):
            return [self.kg.list_kcn.index(kcn)
                        for kcn in self.kg.list_kdn[i].users]
        def _users_c(i):
            return [self.kg.list_kdn.index(kdn)
                        for kdn in self.kg.list_kcn[i].users]

        self.CreateList = [(k,i) for k,kcn in enumerate(self.kg.list_kcn)
                           for i in _users_c(k)]
        self.DeleteList = [(k,i) for k,kcn in enumerate(self.kg.list_kcn)  
                           _deps_c(k) + _users_c(k)]

        Cr = len(self.CreateSet)
        De = len(self.DeleteSet)
        # ======build varaibles======
        self.R = self.md.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.md.addVars(T, Cr, name="S", vtype=GRB.BINARY)
        self.P = self.md.addVars(T, I, name="P", vtype=GRB.BINARY)
        self.create = self.md.addVars(T, Cr, name="create", vtype=GRB.BINARY)
        self.delete = self.md.addVars(T, De, name="delete", vtype=GRB.BINARY)

        self.U = self.md.addVars(T, T, name="U", lb=0, ub=self.budget)
        for t in range(T):
            for k in range(T):
                self.md.addLConstr(self.U[t, k], GRB.GREATER_EQUAL, 0)
                self.md.addLConstr(self.U[t, k] +
                self.R[t ,k]*self.overhead[k] +
                quicksum(self.mem[i_]*self.delete[t,eidx_d] for eidx_d, (k_, i_) 
                    in enumerate(self.DeleteList) if k==k_,
                GRB.LESS_EQUAL, self.budget)

        # ======build constraints======
        with Timer("Gurobi model construction", extra_data={"T": str(T),
                   "budget": str(budget)}):
            with Timer("Objective construction", extra_data={"T": str(T),
                       "budget": str(budget)}):
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

            with Timer("Variable initialization", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                if self.imposed_schedule == ImposedSchedule.FULL_SCHEDULE:
                    self.md.addLConstr(quicksum(self.R[t, i] for t in range(T) 
                                               for i in range(t+1, T)), 
                                      GRB.EQUAL, 0)
                    self.md.addLConstr(quicksum(self.S[t, j] for j in range(J) 
                                               for t in range(self.CreateList(j)[0])), T),
                                      GRB.EQUAL, 0) 
                    self.md.addLConstr(quicksum(self.R[t, t] for t in range(T)), 
                                      GRB.EQUAL, T)
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

            with Timer("Correctness constraints", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                # ensure all checkpoints are in memory
                for t in range(T - 1):
                    for i in range(T):
                        self.md.addLConstr(self.S[t+1, i], GRB.LESS_EQUAL, 
                                          self.S[t, i] + self.R[t, 
                                          self.CreateList(j)[0]])
                # ensure all computations are possible
                for i in range(T):
                    for t in range(T):
                        for j in _deps_c(i):
                            self.md.addLConstr(self.R[t, i], GRB.LESS_EQUAL, 
                                              self.R[t, self.CreateList(j)[0]]
                                            + self.S[t, j])
                            
            # define memory constraints
            with Timer("Presence constraints", extra_data={"T": str(T), 
                       "budget": str(budget)}):
                # ensure all checkpoints are in memory
                alive = {}
                for t in range(T):
                    for eidx, (k, i) in enumerate(self.DeleteList):
                        alive[(t,k,i)] = self.P[t,i]
                        alive[(t,k,i)] += quicksum(self.create[t,eidx_c] 
                                                   for eidx_c, (k_, i_) 
                                                   in enumerate(self.CreateList) 
                                                   if i_==i and k_<=k)
                        alive[(t,k,i)] -= quicksum(self.delete[t,eidx_d] 
                                                   for eidx_d, (k_, i_) 
                                                   in enumerate(self.DeleteList) 
                                                   if i_==i and k_<=k)
                        self.md.addLConstr(alive[(t,k,i)], GRB.GREATER_EQUAL, 0)
                        self.md.addLConstr(alive[(t,k,i)], GRB.LESS_EQUAL, 1)
                        if (k,i) in self.CreateList:
                            didx = self.DeleteList.index((k,i))
                            self.md.addLConstr(alive[(t,k,i)] +
                                              self.delete[t, didx], 
                                              GRB.GREATER_EQUAL, self.R[t, k])

                    for eidx, (k, i) in enumerate(self.CreateList):
                            self.md.addLConstr(self.create[t,eidx], 
                                              GRB.LESS_EQUAL, 
                                              self.R[t, k])

            def _num_hazards(t, i, k):
                if t + 1 < T:
                    return 1 - self.R[t, k] + self.P[t + 1, i] + 
                           quicksum(self.R[t, j] for j in _users_d(i) if j > k)
                return 1 - self.R[t, k] + 
                       quicksum(self.R[t, j] for j in _users_d(i) if j > k)

            def _max_num_hazards(t, i, k):
                num_uses_after_k = sum(1 for j in _users_d(i) if j > k)
                if t + 1 < T:
                    return 2 + num_uses_after_k
                return 1 + num_uses_after_k

            with Timer("Constraint: upper bound for 1 - delete", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for eidx, (k, i) in enumerate(self.DeleteList):
                        self.md.addLConstr(1 - self.delete[t, eidx], 
                                          GRB.LESS_EQUAL, 
                                          _num_hazards(t, i, k))
            with Timer("Constraint: lower bound for 1 - delete", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for eidx, (k, i) in enumerate(self.DeleteList):
                        self.md.addLConstr(
                            _max_num_hazards(t, i, k) * 
                            (1 - self.delete[t, eidx]), 
                            GRB.GREATER_EQUAL, _num_hazards(t, i, k)
                        )

            with Timer(
                "Constraint: initialize memory usage",
                extra_data={"T": str(T), "budget": str(budget)},
                ):
                for t in range(T):
                    self.md.addLConstr(
                        self.U[t, 0],
                        GRB.EQUAL,
                        quicksum(self.P[t, i] * self.mem[i] for i in range(I)) +
                        quicksum(self.create[t, edix]*self.mem[i] 
                                 for eidx, (k_,i) in 
                                 enumerate(self.CreateList) if k_==0)
                        quicksum(self.delete[t, edix]*self.mem[i] 
                                 for eidx, (k_,i) in 
                                 enumerate(self.DeleteList) if k_==0)
                    )
            with Timer("Constraint: memory recurrence", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for k in range(1,T):
                        self.md.addLConstr(
                            self.U[t, k], GRB.EQUAL, self.U[t, k-1] + 
                            quicksum(self.create[t, edix]*self.mem[i] 
                                     for eidx, (k_,i) in 
                                     enumerate(self.CreateList) if k_==k) -
                            quicksum(self.delete[t, edix]*self.mem[i] 
                                     for eidx, (k_,i) in 
                                     enumerate(self.DeleteList) if k_==k)
                        )
    
    def solve(self):
        T = len(self.kg.list_kcn)
        with Timer("Gurobi model optimization", 
                   extra_data={"T": str(T), "budget": str(self.budget)}):
            # if self.seed_s is not None:
            #     self.md.Params.TimeLimit = self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT
            #     self.md.optimize()
            #     if self.md.status == GRB.INFEASIBLE:
            #         print("Infeasible ILP seed at budget {:.2E}".format(self.budget))
            #     self.md.remove(self.init_constraints)
            self.md.Params.TimeLimit = self.gurobi_params.get("TimeLimit", 0)
            self.md.message("\n\nRestarting solve\n\n")
            with Timer("ILPSolve") as solve_ilp:
                self.md.optimize()
            self.solve_time = solve_ilp.elapsed

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
        assert self.feasible, f"Cannot schedule an infeasible model!"
        op_sched = []
        T = len(self.kg.list_kcn)
        for t in range(T):
            for k in range(T):
                if self.R[t, k].X: 
                    kcn = self.kg.list_kcn[k]
                    op_sched.append(RunOp(kcn))
                for eidx, (k, i) in enumerate(self.DeleteList):
                    if self.delete[t, eidx]:
                        kdn = self.kg.list_kdn[i]
                        op_sched.append(DelOp(kdn))
        return op_sched