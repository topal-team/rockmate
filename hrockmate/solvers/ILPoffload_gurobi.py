# import logging
# import math
from typing import Dict, Any
import numpy as np
from copy import deepcopy
from gurobipy import GRB, Model, quicksum
from gurobipy.GRB import GREATER_EQUAL as GEQ
from gurobipy.GRB import LESS_EQUAL as LEQ
from gurobipy.GRB import EQUAL as EQ


# from hrockmate.def_op import RunOp, DelOp, OpSchedule
from .op_schedule import Op, OpSchedule
from hrockmate.rkgb.Htools import *


class ModelGurobi:
    """
    The Gurobi model will build the ILP model by given Hgraph and budget.
    RN this model will take a rk_chain to solve the solution.
    """

    def __init__(
        self,
        hgraph: H_graph,
        peak_budget: int,
        save_budget=None,
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
        if save_budget:
            self.save_budget = save_budget / self.gcd
        else:
            self.save_budget = peak_budget / self.gcd

        self.gurobi_params = gurobi_params
        self.feasible = None
        self.solve_time = None


        #############################
        self.hgraph = hgraph

        T = len(self.hgraph.list_hcn)
        L = len(self.hgraph.list_hdn)

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
            for i in range(L)
        ]  # outputs of hdn
        _users_c = [
            [
                self.hgraph.list_hdn.index(hdn)
                for hdn in self.hgraph.list_hcn[i].users
            ]
            for i in range(T)
        ]  # outputs of hcn


        self.md = Model(f"rockmateMILP_{T}_{peak_budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.md.Params, k, v)

        self.create_list = [(k, i) for k in range(T) for i in _users_c[k]]
        self.delete_list = [
            (k, i) for i in range(L) for k in _deps_d[i] + _users_d[i]
        ]

        E = len(self.create_list)

        self.contributions = [[] for _ in range(L)]
        for (k,i) in self.create_list:
            self.contributions[i].append(k)
        self.sizes = [hdn.mem / self.gcd for hdn in self.hgraph.list_hdn]
        self.overhead = [hcn.ff_overhead / self.gcd for hcn in self.hgraph.list_hcn]#placeholder
        self.comp_time = [hcn.ff_time for hcn in self.hgraph.list_hcn]#placeholder

        self.Comp = self.md.addVars(T, T, name="Comp", vtype=GRB.BINARY)
        self.Alive = self.md.addVars(T, T, E, name="Alive", vtype=GRB.BINARY)
        self.Ocp = self.md.addVars(T, T, L, name="Ocp", vtype=GRB.BINARY)
        self.Time = self.md.addVars(T, T, name="Time", vtype=GRB.CONTINUOUS)
        self.Ofl = self.md.addVars(T, T, E, lb=0, ub=1, name="Ofl", vtype=GRB.CONTINUOUS)
        self.Prf = self.md.addVars(T, T, E, lb=0, ub=1, name="Prf", vtype=GRB.CONTINUOUS)
        self.PrfEnd = self.md.addVars(T, T, E, name="PrfEnd", vtype=GRB.BINARY)
        self.PrfProg = self.md.addVars(T, E, lb=0, ub=1, name="PrfProg", vtype=GRB.CONTINUOUS)
        

        # define objective function
        self.md.setObjective(
            quicksum(
                self.Time[i, t]
                for i in range(T)
                for t in range(T)
            ),
            GRB.MINIMIZE,
        )

        # ======Boundary constraints======
        self.md.addLConstr(
            quicksum(
                self.Comp[i, t] for t in range(T) for i in range(t + 1, T)
            ),
            EQ,
            0,
        )
        self.md.addLConstr(
            quicksum(
                self.Alive[t, i, j]
                for j in range(E)
                for t in range(self.create_list[j][0] + 1)
                for i in range(self.create_list[j][0] + 1)
            ),
            EQ,
            0,
        )

        # ======Step Constraints======
        for i in range(T):
            for j in range(T):
                self.md.addLConstr(self.Time[i,j], GEQ, self.Comp[i,j] * self.comp_time[j])
                self.md.addLConstr(self.Time[i,j], GEQ, quicksum(
                    quicksum(self.Ofl[i,j,k]*self.sizes[l] for k in self.contributions[l])/
                    quicksum(self.Alive[i,j,k] for k in self.contributions[l])
                    for l in range(L)
                ))
                self.md.addLConstr(self.Time[i,j], GEQ, quicksum(
                    quicksum(self.Prf[i,j,k]*self.sizes[l] for k in self.contributions[l])/
                    quicksum(self.Ofl[i,j,k] for k in self.contributions[l])
                    for l in range(L)
                ))
                self.md.addLConstr(self.Prf[i,j], GEQ, self.Comp[i,j])
                self.md.addLConstr(self.Ofl[i,j], GEQ, self.Comp[i,j])

                if j<T-1:
                    self.md.addLConstr(self.Alive[i,j+1], GEQ,
                                    self.Alive[i,j] - self.Comp[i,j])
                    self.md.addLConstr(self.Alive[i,j+1], LEQ,
                                    self.Alive[i,j] + self.Comp[i,j])
                    for k in range(E):
                        self.md.addLConstr(self.Alive[i,j+1,k], LEQ, 
                                        self.Alive[i,j,k] + 
                                        self.PrfEnd(i,j,k) +
                                        self.Comp(i,j) * self.create_list[k][0]==j)

                for k in range(E):
                    self.md.addLConstr(self.Ofl[i,j,k], LEQ, self.Alive[i,j,k])
                    self.md.addLConstr(self.Prf[i,j,k], LEQ, 
                                       quicksum(self.Ofl[ii,jj,k] for ii in range(i) for jj in range(ii))+
                                       quicksum(self.Ofl[i,jj,k] for jj in range(j)))
                    self.md.addLConstr(self.PrfEnd[i,j,k], LEQ, 
                                       self.PrfProg[i,k] + 
                                       quicksum(self.Prf[i,jj,k]-self.PrfEnd[i,jj,k] for jj in range(j)))
                    self.md.addLConstr(self.PrfEnd[i,j,k], GEQ, 
                                       self.PrfProg[i,k] + 
                                       quicksum(self.Prf[i,jj,k]-self.PrfEnd[i,jj,k] for jj in range(j)) - 
                                       0.9999)
                for l in range(L):
                    for k in self.contributions[l]:
                        self.md.addLConstr(self.Prf[i,j,k], LEQ, 
                                           self.Ocp(i,j,l))
                        self.md.addLConstr(self.Alive[i,j,k], LEQ, self.Ocp[i,j,l])
                    self.md.addLConstr(quicksum(self.Ocp[i,j,l]*self.sizes[l] for l in range(L))+
                                       self.Comp[i,j] * self.overhead[j],
                                       LEQ,
                                       self.save_budget)

        for i in range(T-1):
            for k in range(E):
                self.md.addLConstr(self.PrfProg[i+1,k], EQ, 
                                    self.PrfProg[i,k] + quicksum(self.Prf[i,jj,k]-self.PrfEnd[i,jj,k] for jj in range(i)))
        

    def add_abar_constraint(self, save_budget):
        T = len(self.hgraph.list_hcn)
        L = len(self.hgraph.list_hdn)
        self.save_budget = save_budget / self.gcd
        for k in range(T):
            self.md.addLConstr(
                quicksum(self.Ocp[self.loss_idx, self.loss_idx, l] for l in range(L)), 
                         LEQ, self.save_budget
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
