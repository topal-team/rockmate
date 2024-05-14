'''
rkgb.Ktools.K_D_node -> rkgb.core.backward.AllocationNode
rkgb.Ktools.K_graph -> rkgb.core.backward.ForwardAndBackwardGraph

rkgb.Htools.H_graph -> rkgb.core.hierarchical.HierarchicalGraph
rkgb.Htools.H_cluster -> rkgb.core.hierarchical.HierarchicalCluster


from rockmate.solvers.def_op import RunOp, DelOp -> 
   from rockmate.solvers.op_schedule import ComputeOp, DeleteOp

from .def_op import OpSchedule as OpSchedule_old -> ?

K_graph vs. ForwardAndBackwardGraph
    dict_kn -> dict_nodes ?

H_graph vs. HierarchicalGraph
    list_hcn -> list_HCNs
    loss_hcn -> loss_hcn
    cluster -> cluster
    outputs_hdn_data -> output_data_HANs ?

HierarchicalCluster
    get_sched() ->  solvers.main.get_sched(cluster)
    translate_op_list() -> solvers.main.translate(cluster, op_list)
    loss_kcn -> loss_cnode
    loss_kcn.time
    possible_hg ->  partitionings 

H_C_node vs. HierarchicalComputationNode
    deps -> deps
    users -> users
    ff_op_list 
    sub_cluster.get_sched()
    sub_cluster.translate_op_list()
    sub_cluster.loss_kcn
    sub_cluster.loss_kcn.time

H_D_node vs. HierarchicalAllocationNode
    deps -> deps
    users -> users

K_C_node(RK_node) vs. ComputationNode(base.Node)
K_D_node(RK_node) vs. AllocationNode(base.Node)
    
    RK_node
        .main_target: str (Python str which is executed)
    K_C_node
        self.name = f"fwd_{mt}" if is_fwd else f"bwd_{mt}"
    K_D_node
        self.name = f"{main_target} {self.kdn_type}" (e.g., x.data or x.grad)
        (self.kdn_type -> self.alloc_type)

    base.Node
        .main_target: str
    ComputationNode
        self.name = f"FWD[{main_target}]" if is_fwd else f"BWD[{main_target}]"
    AllocationNode
        self.name = f"{main_target} {self.allocation_type}"

Allocation # Activation/Parameter/Buffer
    self.name = f"{self.target_name} {self.alloc_type}"

Activation(Allocation)(self, anode: AllocationNode)
    self.target_name = anode.main_target
    self.kdn = self.anode = anode

Op(kn) vs. ComputeOp(Op)(anode), DeleteOp(Op)(alloc)
    op.is_del = isinstance(kn, K_D_node)
    op.name = op.kn.name -> op.target_name ? op.name?
    op.disabled -> op.disabled
    op.kn -> op.target ?
    op.kn.name -> op.target_name ? (=op.target.name)
    op.kn.main_target -> op.target.main_target ?
    op.op_type -> op.op_type


    Op # Compute/Delete/Mappong/Allocate/Offload/Prefetch
        self.name = f"{self.op_type}({self.target_name})"

    DeleteOp(Op)
        self.target = alloc
        self.target_name = self.target.name = alloc.name

    ComputeOp(Op)(self, cnode: ComputationNode)
        self.target = cnode
        self.target_name = self.target.name = cnode.name

OpSched vs. OpSched
    .op_list -> .op_list
    .fwd_time -> .fwd_time
    .save_mem -> .save_mem
    .fwd_overhead -> .fwd_overhead
    .loss_idx -> .loss_idx
    .interfaces -> .interfaces (nodes connectd to other clusters)
    .solver -> ? (not useful)
'''

import time
import warnings
import rkgb
#from rkgb.utils import np
#from rkgb.Ktools import K_C_node
#from rkgb.Htools import H_graph, H_cluster
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster

import math
from ..main import Solver, get_cluster_budget
from ..main import get_sched, translate
from .def_chain import RK_Chain
#from .def_sequence import SeqBlockBwd, SeqBlockFe, RK_Sequence
from .rotor_solver import seq_builder, solve_dp_functional
from ...op_schedule import OpSchedule, Op
#from .def_op import OpSchedule as OpSchedule_old
# from .def_op import RunOp, DelOp

# can be avoided?
class RK_block_:
    def __init__(self):
        pass

# can be avoided?
class RK_Block_Solution_:
    def __init__(self):
        pass


class RK_Chain_:
    def __init__(self):
        pass


class RK_rotor(Solver):
    def __init__(
        self,
        mem_unit=1024**2,
    ):
        self.mem_unit = mem_unit

    def is_sequential(self, hg: HierarchicalGraph):
        loss_idx = hg.list_HCNs.index(hg.loss_hcn)
        for i, hcn in enumerate(hg.list_HCNs[:loss_idx]):  # toposorted
            if len(hcn.users) > 1 or len(hcn.deps) > 1:
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
        if isinstance(cluster, RK_Chain) or isinstance(cluster, RK_Chain_):
            list_seq = []
            for budget in budgets:
                list_seq.append(self.solve_rk_chain(cluster, budget))
            return list_seq
        elif isinstance(cluster, HierarchicalCluster):
            list_seq = []
            # for hg in cluster.possible_hg:
            for hg in cluster.partitionings:
                list_seq.extend(self.solve_hg(hg, budgets))
            return list_seq

        else:
            warnings.warn(f"Unrecognized input type {type(cluster)}")

    # modify hg_to_rk_chain
    # check def_sequence file
    # check def_chain file
    def solve_hg(self, hg: HierarchicalGraph, budgets=[]):
        if not self.is_sequential(hg):
            return []
        else:
            list_op_sched = []
            for budget in budgets:
                self.chain = self.hg_to_rk_chain(hg)
                try:
                    _ = self.solve_rk_chain(self.chain, budget)
                except Exception as e:
                    if not "budget" in str(e):  # not enough budget
                        raise e
                    else:
                        continue
                self.fwd_seq, self.bwd_seq = self.seq.cut_fwd_bwd()
                self.fwd_op_sched_list = [
                    seq.op_sched for seq in self.fwd_seq.seq
                ]
                self.bwd_op_sched_list = [
                    seq.op_sched for seq in self.bwd_seq.seq
                ]

                self.fwd_op_list = [
                    op
                    for op_sched in self.fwd_op_sched_list
                    for op in op_sched.op_list
                ]
                self.bwd_op_list = [
                    op
                    for op_sched in self.bwd_op_sched_list
                    for op in op_sched.op_list
                ]
                op_sched = OpSchedule(
                    (self.fwd_op_list + self.bwd_op_list).copy(),
                    loss_idx=len(self.fwd_op_list),
                    cluster=hg.cluster,
                )
                # op_sched.solver = "rk-rotor"
                list_op_sched.append(op_sched)
            return list_op_sched

    def solve_rk_chain(self, chain, budget):
        self.opt_table = None
        start = time.time()
        mmax = budget // self.mem_unit - chain.cw[0] - chain.cw[chain.ln]
        self.opt_table = solve_dp_functional(
            chain, mmax, self.opt_table, force_python=True
        )
        self.seq = seq_builder(
            chain,
            budget // self.mem_unit - chain.cw[chain.ln],
            self.opt_table,
        )

        end = time.time()
        self.DP_solve_time = end - start
        return self.seq

    def hg_to_rk_chain(self, hg):
        chain = RK_Chain_()

        chain.body = []

        def discretize(values):
            return [math.ceil(value / self.mem_unit) for value in values]

        no_grad_hcns = []
        loss_idx = hg.list_HCNs.index(hg.loss_hcn)

        def set_op_sched(op_sched):
            setattr(op_sched, "time", op_sched.fwd_time)
            setattr(op_sched, "save", op_sched.save_mem)
            setattr(op_sched, "overhead", op_sched.fwd_overhead)

        for i, hcn in enumerate(hg.list_HCNs[:loss_idx]):  # toposorted
            if len(hcn.users) > 1:
                return False
            if hcn.sub_cluster is None:
                # WARNING: if hcn has no bwd, it has to be merged with the next one
                no_grad_hcns.append(hcn)
            else:
                ff_op_list = []
                for f_hcn in no_grad_hcns:
                    ff_op_list += f_hcn.ff_op_list

                fwd_loss = Op(rkgb.Ktools.K_C_node("loss"), disabled=True)
                fn_op_list = ff_op_list.copy() + hcn.ff_op_list  # + [fwd_loss]
                first_hcn = hcn if not no_grad_hcns else no_grad_hcns[0]
                # assume single input

                input_kdn_data = list(first_hcn.deps)[0].kdn
                output_kdn_data = list(hcn.users)[0].kdn
                for op in ff_op_list:
                    # if op.kn.name == input_kdn_data.name:
                    #     print(op.kn.name)
                    #     op.disabled = True
                    if isinstance(op, DeleteOp) and op.target.anode.name == input_kdn_data.name:
                        print(op.target.anode.name)
                        op.disabled = True
                fc_op_list = ff_op_list.copy() + hcn.ff_op_list  # + [fwd_loss]

                no_grad_hcns = []

                Fn_sched = OpSchedule(
                    fn_op_list,
                    len(fn_op_list) - 1,
                    # cluster=hcn.sub_cluster,
                    refine=False,
                    correct_overhead=False,
                )
                Fc_sched = OpSchedule(
                    fc_op_list,
                    len(fc_op_list) - 1,
                    # cluster=hcn.sub_cluster,
                    refine=False,
                    correct_overhead=False,
                )
                set_op_sched(Fc_sched)
                set_op_sched(Fn_sched)

                sols = []

                # for op_sched in hcn.sub_cluster.get_sched():
                for op_sched in get_sched(hcn.sub_cluster):
                    # fwd_op_list = hcn.sub_cluster.translate_op_list(
                    fwd_op_list = translate(hcn.sub_cluster,
                        ff_op_list + op_sched.op_list[: op_sched.loss_idx]
                    )  # + [Op(K_C_node("loss"))]
                    # bwd_op_list = hcn.sub_cluster.translate_op_list(
                    bwd_op_list = translate(hcn.sub_cluster,
                        op_sched.op_list[op_sched.loss_idx + 1 :]
                    )  # start with loss op
                    for op in bwd_op_list:
                        # By default, bwd does not delete input data/grad
                        #if op.kn.main_target == input_kdn_data.main_target:
                        #    op.disabled = True
                        if isinstance(op, DeleteOp) and op.target.anode.main_target == input_kdn_data.main_target:
                           op.disabled = True

                    Fwd_sched = OpSchedule(
                        fwd_op_list,
                        len(fwd_op_list) - 1,
                        # cluster=hcn.sub_cluster,
                        refine=False,
                        correct_overhead=False,
                    )
                    set_op_sched(Fwd_sched)

                    Bwd_sched = OpSchedule(
                        bwd_op_list,
                        0,
                        # cluster=hcn.sub_cluster,
                        refine=False,
                        correct_overhead=False,
                    )  # so that solution can be read
                    hcn.sub_cluster.loss_kcn.time = 0
                    Full_sched = OpSchedule(
                        fwd_op_list
                        + [Op(hcn.sub_cluster.loss_kcn)]
                        + bwd_op_list,
                        len(fwd_op_list) - 1,
                        # cluster=hcn.sub_cluster,
                        refine=False,
                        correct_overhead=False,
                        interfaces={
                            "inputs_kdn_data": op_sched.interfaces[
                                "inputs_kdn_data"
                            ],
                            "outputs_kdn_data": {output_kdn_data}
                            if i == loss_idx - 1
                            else {},
                            "inputs_kdn_grad": op_sched.interfaces[
                                "inputs_kdn_grad"
                            ],
                            "outputs_kdn_grad": set(),  # For now, no op will generate it
                        },
                    )  # different from op_sched: may merge no_grad hcn's
                    setattr(Bwd_sched, "time", Full_sched.bwd_time)
                    setattr(
                        Bwd_sched,
                        "save",
                        Full_sched.save_mem[Full_sched.loss_idx :],
                    )
                    setattr(Bwd_sched, "overhead", Full_sched.bwd_overhead)

                    sol = RK_Block_Solution_()
                    setattr(sol, "fwd_sched", Fwd_sched)
                    setattr(sol, "bwd_sched", Bwd_sched)
                    setattr(sol, "full_sched", Full_sched)
                    setattr(sol, "size_a_bar", Full_sched.mem)
                    setattr(sol, "time_fwd", Full_sched.fwd_time)
                    setattr(sol, "time_bwd", Full_sched.bwd_time)
                    setattr(sol, "overhead_fwd", Full_sched.fwd_overhead)
                    setattr(
                        sol,
                        "overhead_bwd",
                        # Full_sched.bwd_overhead + Full_sched.save_mem[-1]
                        max(Full_sched.save_mem + Full_sched.overhead)
                        # - input_kdn_data.mem  # input_grad is counted in save_mem
                        - Full_sched.mem,
                    )
                    sols.append(sol)
                block = RK_block_()
                setattr(block, "sols", sols)
                setattr(block, "Fc_sched", Fc_sched)
                setattr(block, "Fn_sched", Fn_sched)
                setattr(block, "mem_inp", input_kdn_data.mem)
                # since interfaces is empty, output is counted in memory
                setattr(
                    block,
                    "overhead_fast_fwd",
                    Fc_sched.fwd_overhead + output_kdn_data.mem,
                )
                setattr(block, "time_fast_fwd", Fc_sched.fwd_time)

                chain.body.append(block)

        mkl = lambda n: [[] for _ in range(n)]
        setattr(chain, "ln", len(chain.body))
        setattr(chain, "fw", mkl(chain.ln + 1))
        setattr(chain, "bw", mkl(chain.ln + 1))
        setattr(chain, "cw", [None] * (chain.ln + 2))
        setattr(chain, "cbw", mkl(chain.ln + 2))
        setattr(chain, "fwd_tmp", mkl(chain.ln + 1))
        setattr(chain, "bwd_tmp", mkl(chain.ln + 1))
        setattr(chain, "ff_fwd_tmp", [None] * (chain.ln + 1))
        setattr(chain, "ff_fw", [None] * (chain.ln + 1))

        chain.nb_sol = []
        for i, b in enumerate(chain.body):
            chain.nb_sol.append(len(b.sols))
            if chain.nb_sol[-1] == 0:
                raise Exception(
                    f"We need at least one solution per block. "
                    f"Here {b.block_name} has no solution"
                )
            for sol in b.sols:
                chain.fw[i].append(sol.time_fwd)
                chain.bw[i].append(sol.time_bwd)
                chain.cbw[i + 1].append(sol.size_a_bar)
                chain.fwd_tmp[i].append(sol.overhead_fwd)
                chain.bwd_tmp[i].append(sol.overhead_bwd)
            chain.cw[i] = b.mem_inp
            chain.ff_fwd_tmp[i] = b.overhead_fast_fwd
            chain.ff_fw[i] = b.time_fast_fwd
        chain.cw[chain.ln] = list(hg.output_data_HANs)[
            0
        ].mem  #  the final output

        chain.cbw[-2] = [(c - chain.cw[chain.ln]) for c in chain.cbw[-2]]

        # for the Loss block :
        chain.nb_sol.append(1)
        chain.fw[-1] = [0]
        chain.bw[-1] = [0]
        chain.cw[-1] = 0
        chain.cbw[-1] = [0]
        chain.fwd_tmp[-1] = [0]
        chain.bwd_tmp[-1] = [0]
        chain.ff_fwd_tmp[-1] = 0
        chain.ff_fw[-1] = 0

        chain.cw = discretize(chain.cw)
        chain.cbw = [discretize(x) for x in chain.cbw]
        chain.fwd_tmp = [discretize(x) for x in chain.fwd_tmp]
        chain.bwd_tmp = [discretize(x) for x in chain.bwd_tmp]
        chain.ff_fwd_tmp = discretize(chain.ff_fwd_tmp)

        return chain

    # def get_new_op_sched(self, op_sched: OpSchedule_old, kg):
    #     op_list = [Op(kg.dict_kn[op.name]) for op in op_sched.op_list] # op.name was op.kn.name
    #     op_sched_new = OpSchedule(op_list)
    #     return op_sched_new

    # def get_old_op_sched(self, op_sched: OpSchedule):
    #     op_list = []
    #     for op in op_sched.op_list:
    #         if op.disabled:
    #             continue
    #         if op.is_del:
    #             op_list.append(DelOp(op.kn))
    #         else:
    #             op_list.append(RunOp(op.kn))

    #     op_sched_new = OpSchedule_old(op_list)
    #     return op_sched_new
