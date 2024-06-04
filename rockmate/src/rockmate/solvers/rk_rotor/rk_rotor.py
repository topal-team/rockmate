''' HOW TO RUN ROCKMATE WITH UPDATED rk_rotor.py
from rockmate.solvers.main import add_sched

# to get Fc/Fn schedules for clusters
rkmod.preprocess()

# to get Checkmate schedules for clusters
for hcn in rkmod.rkgb_res.hierarchical_cluster.partitionings[0].list_HCNs:
    print(hcn.sub_cluster)
    if hcn.sub_cluster is None:continue
    if not hcn.is_fwd: continue
    solver = HILP(ilp_solver="PULP_CBC_CMD")
    list_sched = solver(hcn.sub_cluster)
    for sched in list_sched:
        add_sched(hcn.sub_cluster, sched)

# rkmod.get_compiled_fct()

'''

import time
import warnings
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from rkgb.core.backward import ComputationNode

from ..main import Solver, get_cluster_budget
from .def_chain import RK_Chain
from .rotor_solver import seq_builder, solve_dp_functional
from ...op_schedule import OpSchedule, ComputeOp

class RK_rotor(Solver):
    def __init__(
        self,
        mem_unit=1024**2,
        force_python=False
    ):
        self.mem_unit = mem_unit
        self.force_python = force_python

    def is_sequential(self, hg: HierarchicalGraph):
        loss_idx = hg.list_HCNs.index(hg.loss_hcn)
        for i, hcn in enumerate(hg.list_HCNs[:loss_idx]):  # toposorted
            users = set(hcn_user for han in hcn.users 
                        for hcn_user in han.users
                        if hcn_user.is_fwd)
            if len(users)>1:
                return False
        return True

    def solve(
        self,
        cluster,
        budgets=None,
    ):
        if budgets is None:
            budgets = get_cluster_budget(cluster.representee_cluster)
        # self.budget = budgets
        # if isinstance(cluster, RK_Chain) or isinstance(cluster, RK_Chain_):
        #     list_seq = []
        #     for budget in budgets:
        #         list_seq.append(self.solve_rk_chain(cluster, budget))
        #     return list_seq
        if isinstance(cluster, HierarchicalCluster):
            list_seq = []
            # for hg in cluster.possible_hg:
            for hg in cluster.partitionings:
                list_seq.extend(self.solve_hg(hg, budgets))
            return list_seq
        else:
            warnings.warn(f"Unrecognized input type {type(cluster)}")

    def solve_hg(self, hg: HierarchicalGraph, budgets=[]):
        if not self.is_sequential(hg):
            return []
        else:
            chain = RK_Chain(hg, self.mem_unit)
            opt_table = self.solve_rk_chain(chain, max(budgets))

            list_op_sched = []
            for budget in budgets:

                try:
                    ## TODO: understand which value this should have precisely
                    ##       make sure that value passed to solve_dp()
                    ##          is not smaller than value passed to seq_builder
                    seq = seq_builder(
                        chain,
                        self.discretize_budget(chain, budget),
                        opt_table
                    )
                except Exception as e:
                    if not "budget" in str(e):  # not enough budget
                        raise e
                    else:
                        continue
                
                fwd_seq, bwd_seq = seq.cut_fwd_bwd()
                
                fwd_op_list = fwd_seq.get_op_list()
                op_sched = OpSchedule(
                    fwd_op_list + [ComputeOp(ComputationNode("loss"))] + bwd_seq.get_op_list(),
                    ## TODO: why not -1 here ???
                    loss_idx = len(fwd_op_list),
                    cluster=hg.cluster,
                )
                # op_sched.solver = "rk-rotor"
                list_op_sched.append(op_sched)
            return list_op_sched

    def discretize_budget(self, chain, budget):
        ## TODO: check the correct value
        return budget // self.mem_unit - chain.cw[0] - chain.cw[chain.ln]

    # Returns the opt_table
    def solve_rk_chain(self, chain, budget):
        start = time.time()

        ## TODO: understand which value this should have precisely
        ##       make sure that value passed to solve_dp() is not smaller than value passed to seq_builder
        mmax = self.discretize_budget(chain, budget)
        opt_table = solve_dp_functional(chain, mmax, force_python=self.force_python)

        end = time.time()
        self.DP_solve_time = end - start
        return opt_table
