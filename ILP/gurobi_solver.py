import logging
import math
import os
from typing import Dict, Any, Optional

import numpy as np
from gurobipy import GRB, Model, quicksum

class ModelGurobi:
    def __init__(self,
        g: CDGraph,
        budget: int,
        gurobi_params: Dict[str,Any] = None,
        gcd = None
        ):
        self.g = g
        self.gcd = gcd
        if self.gcd:
            self.budget = budget//self.gcd
            self.mem = [m//self.gcd for m in self.g.mem]
        self.gurobi_params = gurobi_params

        self.m = Model(f"rockmateMILP_{self.g.size}_{budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        T = len(self.g.CompInputData)
        J = len(self.g.DataInputComp)
        I = len(self.g.TensorToData)

        self.CreateList = [(k,i) for k in range(T) for i in self.g.CompOutputTensors[k]]
        self.DeleteList = [(k,i) for k in range(T) for i in 
                           self.g.CompInputTensors[k]+self.g.CompOutputTensors[k]]

        Cr = len(self.CreateSet)
        De = len(self.DeleteSet)
        # ======build varaibles======
        self.R = self.m.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.m.addVars(T, J, name="S", vtype=GRB.BINARY)
        self.P = self.m.addVars(T, I, name="P", vtype=GRB.BINARY)
        self.create = self.m.addVars(T, Cr, name="create", vtype=GRB.BINARY)
        self.delete = self.m.addVars(T, De, name="delete", vtype=GRB.BINARY)

        self.U = self.m.addVars(T, T, name="U", lb=0, ub=self.budget)
        # for x in range(T):
        #     for y in range(T):
        #         self.m.addLConstr(self.U[x, y], GRB.GREATER_EQUAL, 0)
        #         self.m.addLConstr(self.U[x, y], GRB.LESS_EQUAL, self.budget)

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
                #                    self.m.addLConstr(self.S[x, y], GRB.EQUAL, 0))
                #    self.m.update()

                # define objective function
                self.m.setObjective(quicksum(self.R[t, i] * self.g.time[i]
                                             for t in range(T) for i in range(T)), GRB.MINIMIZE)

            with Timer("Variable initialization", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                if self.imposed_schedule == ImposedSchedule.FULL_SCHEDULE:
                    self.m.addLConstr(quicksum(self.R[t, i] for t in range(T) 
                                               for i in range(t+1, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.S[t, j] for j in range(J) for t 
                                               in range(self.g.DataInputComp(j), T), GRB.EQUAL, 0) 
                    self.m.addLConstr(quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T)
                #elif self.imposed_schedule == ImposedSchedule.COVER_ALL_NODES:
                #    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                #    for i in range(T):
                #        self.m.addLConstr(quicksum(self.R[t, i] for t in range(T)), 
                #                                   GRB.GREATER_EQUAL, 1)
                #elif self.imposed_schedule == ImposedSchedule.COVER_LAST_NODE:
                #    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                #    # note: the integrality gap is very large as this constraint
                #    # is only applied to the last node (last column of self.R).
                #    self.m.addLConstr(quicksum(self.R[t, T - 1] for t in range(T)), 
                #                               GRB.GREATER_EQUAL, 1)

            with Timer("Correctness constraints", extra_data={"T": str(T), "budget": str(budget)}):
                # ensure all checkpoints are in memory
                for t in range(T - 1):
                    for i in range(T):
                        self.m.addLConstr(self.S[t+1, i], GRB.LESS_EQUAL, 
                                          self.S[t, i] + self.R[t, self.g.DataInputComp(j)])
                # ensure all computations are possible
                for i in range(T):
                    for t in range(T):
                        for j in self.g.CompInputData(i):
                            self.m.addLConstr(self.R[t, i], GRB.LESS_EQUAL, 
                                              self.R[t, self.g.DataInputComp(j)] + self.S[t, j])
                            
            # define memory constraints
            with Timer("Presence constraints", extra_data={"T": str(T), "budget": str(budget)}):
                # ensure all checkpoints are in memory
                alive = {}
                for t in range(T):
                    for eidx, (k, i) in enumerate(self.DeleteList):
                        alive[(t,k,i)] = self.P[t,i]
                        alive[(t,k,i)] += quicksum(self.create[t,eidx_c] for eidx_c, (k_, i_) 
                                                   in enumerate(self.CreateList) if i_==i and k_<=k)
                        alive[(t,k,i)] -= quicksum(self.delete[t,eidx_c] for eidx_c, (k_, i_) 
                                                   in enumerate(self.DeleteList) if i_==i and k_<=k)
                        self.m.addLConstr(alive[(t,k,i)], GRB.GREATER_EQUAL, 0)
                        self.m.addLConstr(alive[(t,k,i)], GRB.LESS_EQUAL, 1)
                        if (k,i) in self.CreateList:
                            self.m.addLConstr(alive[(t,k,i)]+self.delete[t, self.DeleteList.index((k,i))], GRB.GREATER_EQUAL, self.R[t, k])

                    for eidx, (k, i) in enumerate(self.CreateList):
                            self.m.addLConstr(self.create[t,eidx], GRB.LESS_EQUAL, self.R[t, k])

            def _num_hazards(t, i, k):
                if t + 1 < T:
                    return 1 - self.R[t, k] + self.P[t + 1, i] + 
                           quicksum(self.R[t, j] for j in self.g.TensorInputComp[i] if j > k)
                return 1 - self.R[t, k] + 
                       quicksum(self.R[t, j] for j in self.g.TensorInputComp(i) if j > k)

            def _max_num_hazards(t, i, k):
                num_uses_after_k = sum(1 for j in self.g.TensorInputComp[i] if j > k)
                if t + 1 < T:
                    return 2 + num_uses_after_k
                return 1 + num_uses_after_k

            with Timer("Constraint: upper bound for 1 - delete", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for eidx, (k, i) in enumerate(self.DeleteList):
                        self.m.addLConstr(1 - self.delete[t, eidx], 
                                          GRB.LESS_EQUAL, _num_hazards(t, i, k))
            with Timer("Constraint: lower bound for 1 - delete", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for eidx, (k, i) in enumerate(self.DeleteList):
                        self.m.addLConstr(
                            _max_num_hazards(t, i, k) * (1 - self.delete[t, eidx]), 
                            GRB.GREATER_EQUAL, _num_hazards(t, i, k)
                        )

            with Timer(
                "Constraint: initialize memory usage (includes spurious checkpoints)",
                extra_data={"T": str(T), "budget": str(budget)},
                ):
                for t in range(T):
                    self.m.addLConstr(
                        self.U[t, 0],
                        GRB.EQUAL,
                        quicksum(self.P[t, i] * self.mem[i] for i in range(I)) +
                        quicksum(self.create[t, edix]*self.mem[i] 
                                 for eidx, (k_,i) in enumerate(self.CreateList) if k_==0)
                        quicksum(self.delete[t, edix]*self.mem[i] 
                                 for eidx, (k_,i) in enumerate(self.DeleteList) if k_==0)
                    )
            with Timer("Constraint: memory recurrence", 
                       extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for k in range(1,T):
                        self.m.addLConstr(
                            self.U[t, k], GRB.EQUAL, self.U[t, k-1] + 
                            quicksum(self.create[t, edix]*self.mem[i] 
                                     for eidx, (k_,i) in enumerate(self.CreateList) if k_==k) -
                            quicksum(self.delete[t, edix]*self.mem[i] 
                                     for eidx, (k_,i) in enumerate(self.DeleteList) if k_==k)
                        )
    
    def solve(self):
        T = len(self.g.CompInputData)
        with Timer("Gurobi model optimization", extra_data={"T": str(T), "budget": str(self.budget)}):
            # if self.seed_s is not None:
            #     self.m.Params.TimeLimit = self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT
            #     self.m.optimize()
            #     if self.m.status == GRB.INFEASIBLE:
            #         print("Infeasible ILP seed at budget {:.2E}".format(self.budget))
            #     self.m.remove(self.init_constraints)
            self.m.Params.TimeLimit = self.gurobi_params.get("TimeLimit", 0)
            self.m.message("\n\nRestarting solve\n\n")
            with Timer("ILPSolve") as solve_ilp:
                self.m.optimize()
            self.solve_time = solve_ilp.elapsed

        infeasible = self.m.status == GRB.INFEASIBLE
        if infeasible:
            self.feasible = False
            # return (None, None, None, None)
            # raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")
        else:
            if self.m.solCount < 1:
                raise ValueError("Model status is {} (not infeasible), but solCount is {}".format(self.m.status, self.m.solCount))
            self.feasible = True

    # def schedule(self):
    #     assert self.feasible, f"Cannot schedule an infeasible model!"
    #     sched = []
    #     T = len(self.g.CompInputData)
    #     for t in range(T):
    #         for k in range(T):
    #             if self.m.R[t, k].X: sched.append()