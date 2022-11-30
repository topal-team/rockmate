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
        ):
        self.g = g
        self.budget = budget
        self.gurobi_params = gurobi_params

        self.m = Model(f"rockmateMILP_{self.g.size}_{budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        T = self.g.size
        # ======build varaibles======
        self.R = self.m.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.m.addVars(T, T, name="S", vtype=GRB.BINARY)
        self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.BINARY)
        


        # ======build constraints======
        with Timer("Gurobi model construction", extra_data={"T": str(T), "budget": str(budget)}):
            with Timer("Objective construction", extra_data={"T": str(T), "budget": str(budget)}):
                # seed solver with a baseline strategy
                if self.seed_s is not None:
                    for x in range(T):
                        for y in range(T):
                            if self.seed_s[x, y] < 1:
                                self.init_constraints.append(self.m.addLConstr(self.S[x, y], GRB.EQUAL, 0))
                    self.m.update()

                # define objective function
                self.m.setObjective(quicksum(self.R[t, i] * permute_cpu[i] for t in range(T) for i in range(T)), GRB.MINIMIZE)

            with Timer("Variable initialization", extra_data={"T": str(T), "budget": str(budget)}):
                if self.imposed_schedule == ImposedSchedule.FULL_SCHEDULE:
                    self.m.addLConstr(quicksum(self.R[t, i] for t in range(T) for i in range(t + 1, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.S[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T)
                elif self.imposed_schedule == ImposedSchedule.COVER_ALL_NODES:
                    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                    for i in range(T):
                        self.m.addLConstr(quicksum(self.R[t, i] for t in range(T)), GRB.GREATER_EQUAL, 1)
                elif self.imposed_schedule == ImposedSchedule.COVER_LAST_NODE:
                    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                    # note: the integrality gap is very large as this constraint
                    # is only applied to the last node (last column of self.R).
                    self.m.addLConstr(quicksum(self.R[t, T - 1] for t in range(T)), GRB.GREATER_EQUAL, 1)

            with Timer("Correctness constraints", extra_data={"T": str(T), "budget": str(budget)}):
                # ensure all checkpoints are in memory
                for t in range(T - 1):
                    for i in range(T):
                        self.m.addLConstr(self.S[t + 1, i], GRB.LESS_EQUAL, self.S[t, i] + self.R[t, i])
                # ensure all computations are possible
                for (u, v) in self.g.edge_list:
                    for t in range(T):
                        self.m.addLConstr(self.R[t, v], GRB.LESS_EQUAL, self.R[t, u] + self.S[t, u])
