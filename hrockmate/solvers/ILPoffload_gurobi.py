# import logging
# import math
from typing import Dict, Any
import numpy as np
from copy import deepcopy
from gurobipy import GRB, Model, quicksum
from hrockmate.solvers.op_schedule import PrfOp, OflOp

# import gurobipy.GRB.GREATER_EQUAL as GEQ
# import gurobipy.GRB.LESS_EQUAL as LEQ
# import gurobipy.GRB.EQUAL as EQ


GEQ = GRB.GREATER_EQUAL
LEQ = GRB.LESS_EQUAL
EQ = GRB.EQUAL


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
            "TimeLimit": 20 * 60,
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
            [self.hgraph.list_hdn.index(hdn) for hdn in self.hgraph.list_hcn[i].users]
            for i in range(T)
        ]  # outputs of hcn

        self.md = Model(f"rockmateMILP_{T}_{peak_budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.md.Params, k, v)

        self.create_list = [(k, i) for k in range(T) for i in _users_c[k]]

        E = len(self.create_list)

        self.contributions = [[] for _ in range(L)]
        for j, (k, i) in enumerate(self.create_list):
            self.contributions[i].append(j)
        self.sizes = [hdn.mem / self.gcd for hdn in self.hgraph.list_hdn]
        self.overhead = [
            hcn.ff_overhead / self.gcd for hcn in self.hgraph.list_hcn
        ]  # placeholder
        self.comp_time = [hcn.ff_time for hcn in self.hgraph.list_hcn]  # placeholder
        self.bandwidthOfl = 1 * 1024**2  # byte/ms
        self.bandwidthPrf = 1 * 1024**2  # byte/ms
        for i, hcn in enumerate(self.hgraph.list_hcn):
            if "Loss" in hcn.name:
                self.loss_idx = i

        self.Comp = self.md.addVars(T, T, name="Comp", vtype=GRB.BINARY)
        self.Alive = self.md.addVars(T, T, E, name="Alive", vtype=GRB.BINARY)
        self.Ocp = self.md.addVars(T, T, L, name="Ocp", vtype=GRB.BINARY)
        self.Time = self.md.addVars(T, T, name="Time", vtype=GRB.CONTINUOUS)
        self.Ofl = self.md.addVars(
            T, T, E, lb=0, ub=1, name="Ofl", vtype=GRB.CONTINUOUS
        )
        self.Prf = self.md.addVars(
            T, T, E, lb=0, ub=1, name="Prf", vtype=GRB.CONTINUOUS
        )
        self.PrfEnd = self.md.addVars(T, T, E, name="PrfEnd", vtype=GRB.BINARY)
        self.PrfProg = self.md.addVars(
            T, E, lb=0, ub=1, name="PrfProg", vtype=GRB.CONTINUOUS
        )

        # define objective function
        self.md.setObjective(
            quicksum(self.Time[t, i] for i in range(T) for t in range(T)),
            GRB.MINIMIZE,
        )

        # ======Boundary constraints======
        self.md.addLConstr(
            quicksum(self.Comp[t, i] for t in range(T) for i in range(t + 1, T)),
            EQ,
            0,
        )
        self.md.addLConstr(
            quicksum(self.Alive[0, 0, j] for j in range(E)),
            EQ,
            0,
        )

        self.md.addLConstr(
            quicksum(self.Comp[t, t] for t in range(T)),
            GRB.EQUAL,
            T,
        )

        self.md.addLConstr(
            quicksum(self.PrfEnd[0, 0, j] for j in range(E)),
            GRB.EQUAL,
            0,
        )

        # ======Step Constraints======
        for t in range(T):
            for i in range(T):
                self.md.addLConstr(
                    self.Time[t, i], GEQ, self.Comp[t, i] * self.comp_time[i]
                )
                self.md.addLConstr(
                    self.Time[t, i],
                    GEQ,
                    quicksum(
                        quicksum(
                            self.Ofl[t, i, j] * self.sizes[l]
                            for j in self.contributions[l]
                        )
                        / self.bandwidthOfl  # quicksum(self.Alive[t,i,j] for j in self.contributions[l])
                        for l in range(L)
                    ),
                )
                self.md.addLConstr(
                    self.Time[t, i],
                    GEQ,
                    quicksum(
                        quicksum(
                            self.Prf[t, i, j] * self.sizes[l]
                            for j in self.contributions[l]
                        )
                        / self.bandwidthPrf  # quicksum(self.Ofl[t,i,j] for j in self.contributions[l])
                        for l in range(L)
                    ),
                )

                if i < T - 1:
                    for j in range(E):
                        self.md.addLConstr(
                            self.Alive[t, i + 1, j],
                            GEQ,
                            self.Alive[t, i, j] - self.Comp[t, i],
                        )
                        self.md.addLConstr(
                            self.Alive[t, i + 1, j],
                            LEQ,
                            self.Alive[t, i, j] + self.Comp[t, i],
                        )
                        if self.create_list[j][0] == i:
                            self.md.addLConstr(
                                self.Alive[t, i + 1, j],
                                LEQ,
                                self.Alive[t, i, j]
                                + self.PrfEnd[t, i, j]
                                + self.Comp[t, i],
                            )
                        else:
                            self.md.addLConstr(
                                self.Alive[t, i + 1, j],
                                LEQ,
                                self.Alive[t, i, j] + self.PrfEnd[t, i, j],
                            )
                for j in range(E):
                    self.md.addLConstr(self.Ofl[t, i, j], LEQ, self.Alive[t, i, j])
                    prog = (
                        self.PrfProg[t, j]
                        + quicksum(self.Prf[t, ii, j] for ii in range(i + 1))
                        - (quicksum(self.PrfEnd[t, ii, j] for ii in range(i)))
                    )

                    self.md.addLConstr(
                        prog,
                        LEQ,
                        quicksum(
                            self.Ofl[tt, ii, j]
                            for tt in range(t)
                            for ii in range(tt + 1)
                        )
                        + quicksum(self.Ofl[t, ii, j] for ii in range(i)),
                    )
                    self.md.addLConstr(0, LEQ, prog)
                    self.md.addLConstr(1, GEQ, prog)
                    self.md.addLConstr(self.Prf[t, i, j], LEQ, self.Comp[t, i])
                    self.md.addLConstr(self.PrfEnd[t, i, j], LEQ, self.Comp[t, i])
                    self.md.addLConstr(self.Ofl[t, i, j], LEQ, self.Comp[t, i])
                for l in range(L):
                    for j in self.contributions[l]:
                        self.md.addLConstr(prog, LEQ, self.Ocp[t, i, l])
                        self.md.addLConstr(self.Alive[t, i, j], LEQ, self.Ocp[t, i, l])

                self.md.addLConstr(
                    quicksum(
                        self.Ocp[t, i, l] * self.sizes[l]
                        for l in range(L)
                        if l not in _users_c[i]
                    )
                    + self.Comp[t, i] * self.overhead[i]
                    + quicksum(self.Ocp[t, i, l] * self.sizes[l] for l in _users_c[i]),
                    LEQ,
                    self.peak_budget,
                )

            for j, (k, i) in enumerate(self.create_list):
                for k_ in _users_d[i]:
                    self.md.addLConstr(self.Comp[t, k_], LEQ, self.Alive[t, k_, j])

        self.md.addLConstr(
            quicksum(self.Comp[t, self.loss_idx] for t in range(T)),
            LEQ,
            1,
        )
        for t in range(T - 1):
            for j in range(E):
                self.md.addLConstr(
                    self.Alive[t + 1, 0, j], LEQ, self.Alive[t, T - 1, j]
                )
                self.md.addLConstr(
                    self.PrfProg[t + 1, j],
                    EQ,
                    self.PrfProg[t, j]
                    + quicksum(
                        self.Prf[t, ii, j] - self.PrfEnd[t, ii, j] for ii in range(T)
                    ),
                )

        for j in range(E):
            self.md.addLConstr(
                quicksum(self.Ofl[t, i, j] for t in range(T) for i in range(T)),
                LEQ,
                quicksum(self.Prf[t, i, j] for t in range(T) for i in range(T)),
            )
            self.md.addLConstr(
                quicksum(self.Ofl[t, i, j] for t in range(T) for i in range(T)), LEQ, 1
            )

    def add_abar_constraint(self, save_budget):
        T = len(self.hgraph.list_hcn)
        L = len(self.hgraph.list_hdn)
        self.save_budget = save_budget / self.gcd
        # for k in range(T):
        self.md.addLConstr(
            quicksum(self.Ocp[self.loss_idx, self.loss_idx, l] for l in range(L)),
            LEQ,
            self.save_budget,
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

    def schedule(self, hgraph=None, print_sched=False):
        """
        Given the solution from HILP, we want to translate the result
        to a OpSchedule that can be used in a higher level.
        """
        hgraph = hgraph if hgraph else self.hgraph
        assert self.feasible, "Cannot schedule an infeasible model!"
        T = len(hgraph.list_hcn)
        # I = len(hgraph.list_hdn)
        # J = len(self.list_list_sched)
        E = len(self.create_list)

        op_list = []
        prf_list = []
        ofl_list = []

        for t in range(T):
            for i in range(T):
                if self.Comp[t, i].X == 1:
                    # hcn = hgraph.list_hcn[i]
                    # if "Loss" in hcn.name:
                    #     op_list.append(Op())
                    # kcn_pair = hcn.sub_cluster.list_kcn
                    # op_list.append(Op(kcn_pair[0] if hcn.is_fwd else kcn_pair[1]))
                    op_list.append(Op(hgraph.cluster.list_kcn[i]))
                    if print_sched:
                        print(f"Compute {hgraph.list_hcn[i]} at stage {t}")

                    for e in range(E):
                        k = self.create_list[e][1]
                        hdn = hgraph.list_hdn[k]
                        if self.Ofl[t, i, e].X > 0:
                            if print_sched:
                                print(f"\tOffload {self.Ofl[t,i,e].X*100}% of {hdn}")
                            ofl_list.append(
                                OflOp(
                                    hdn.kdn,
                                    1,
                                    after=op_list[-1],
                                )
                            )
                        if self.Prf[t, i, e].X > 0:
                            if print_sched:
                                print(f"\tPrefetch {self.Prf[t,i,e].X*100}% of {hdn}")
                            prf_list.append(
                                PrfOp(
                                    hdn.kdn,
                                    1,
                                    after=op_list[-1],
                                )
                            )
                        src = self.create_list[e][0]
                        if i < T - 1:
                            if self.Alive[t, i + 1, e].X < self.PrfEnd[t, i, e].X + (
                                src == i
                            ) + (self.Alive[t, i, e].X):
                                if print_sched:
                                    print(f"\tDelete {hdn}")
                                op_list.append(Op(hdn.kdn))

                        # if PrfEnd[t,j,e].X>0:
                        #     print(f"\tPrefetch done of edge {e}")
        op_sched = OpSchedule(op_list, prf_list, ofl_list, refine=False)

        return op_sched
