import time
import warnings
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from rkgb.core.backward import ComputationNode
from dataclasses import dataclass

from ..main import Solver, get_cluster_budget
from .def_chain import RK_Chain
from .rotor_solver import seq_builder, solve_dp_functional
from ...op_schedule import OpSchedule, ComputeOp

class RK_rotor(Solver):
    '''rk_rotor solver, adapted to the rockmate framework from the Rotor dynamic-programming based algorithm

    The rotor algorithm is originally implemented in https://gitlab.inria.fr/hiepacs/rotor and described
    in the paper:

    Olivier Beaumont, Lionel Eyraud-Dubois, Julien Herrmann, Alexis Joly, Alena Shilova. Optimal
    checkpointing for heterogeneous chains: how to train deep neural networks with limited memory. ACM
    Transactions on Mathematical Software, In press. https://hal.science/hal-02352969v2

    This version is modified to allow choosing between several rematerialization options for each layer.
    '''
    @dataclass
    class Config:
        mem_unit: int = 1024**2
        force_python: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_sequential(self, hg: HierarchicalGraph):
        loss_idx = hg.list_HCNs.index(hg.loss_hcn)
        for i, hcn in enumerate(hg.list_HCNs[:loss_idx]):  # toposorted
            users = set(hcn_user for han in hcn.users 
                        for hcn_user in han.users
                        if hcn_user.is_fwd)
            if len(users)>1:
                return False
        input_data = hg.input_data_HANs
        users_of_input = set(hcn_user for han in hg.input_data_HANs
                             for hcn_user in han.users
                             if hcn_user.is_fwd)
        if len(users_of_input) > 1:
            return False
        return True

    def solve(
        self,
        cluster,
        budgets=None,
    ):
        if budgets is None:
            budgets = get_cluster_budget(cluster.representee_cluster)
        if isinstance(cluster, HierarchicalCluster):
            list_seq = []
            for hg in cluster.partitionings:
                list_seq.extend(self.solve_hg(hg, budgets))
            return list_seq
        else:
            warnings.warn(f"Unrecognized input type {type(cluster)}")

    def solve_hg(self, hg: HierarchicalGraph, budgets=[]):
        if not self.is_sequential(hg):
            return []
        else:
            chain = RK_Chain(hg, self.config.mem_unit)
            opt_table = self.solve_rk_chain(chain, max(budgets))

            list_op_sched = []
            for budget in budgets:

                try:
                    seq = seq_builder(
                        chain,
                        self.discretize_budget(chain, budget),
                        opt_table
                    )
                except ValueError as e:
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
                list_op_sched.append(op_sched)
            return list_op_sched

    def discretize_budget(self, chain, budget):
        ## Both input and output stay in memory all along
        ##   (this is a Rotor assumption, because Rotor can not remove them from memory)
        ##   (and even Rockmate can not assume too much, because the user has access to them)
        return int(budget // self.config.mem_unit) - chain.cw[0] - chain.cw[chain.ln]

    # Returns the opt_table
    def solve_rk_chain(self, chain, budget):
        start = time.time()

        mmax = self.discretize_budget(chain, budget)
        opt_table = solve_dp_functional(chain, mmax, force_python=self.config.force_python)

        end = time.time()
        self.DP_solve_time = end - start
        return opt_table
