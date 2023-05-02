import time
import warnings
from rkgb.utils import np

import rkgb
import math
from solvers.def_chain import RK_Chain
from solvers.def_sequence import SeqBlockBwd, SeqBlockFe, RK_Sequence
from rockmate.rotor_solver import seq_builder, solve_dp_functional
from solvers.op_schedule import OpSchedule, Op
from solvers.def_op import OpSchedule as OpSchedule_old
from solvers.def_op import RunOp, DelOp


class RK_block_:
    def __init__(self):
        pass


class RK_Block_Solution_:
    def __init__(self):
        pass


class RK_Chain_:
    def __init__(self):
        pass


class RK_rotor:
    def __init__(
        self, mem_unit=1024 ** 2,
    ):
        self.mem_unit = mem_unit

    def solve(
        self, chain, mem_limit,
    ):
        self.mem_limit = mem_limit
        if isinstance(chain, RK_Chain) or isinstance(chain, RK_Chain_):
            return self.solve_rk_chain(chain)
        elif isinstance(chain, rkgb.Htools.H_graph):
            return self.solve_rk_chain(self.hg_to_rk_chain(chain))
        else:
            warnings.warn(f"Unrecognized chain type {type(chain)}")

    def solve_rk_chain(self, chain):
        self.opt_table = None
        start = time.time()
        mmax = self.mem_limit // self.mem_unit - chain.cw[0]
        self.opt_table = solve_dp_functional(
            chain, mmax, self.opt_table, force_python=True
        )
        self.seq = seq_builder(
            chain, self.mem_limit // self.mem_unit, self.opt_table
        )
        end = time.time()
        self.DP_solve_time = end - start

        enable = np.zeros(chain.ln)
        for s in self.seq.seq:
            if isinstance(s, SeqBlockFe):
                enable[s.index] = 1
            if isinstance(s, SeqBlockBwd):
                if not enable[s.index - 1]:
                    s.op_sched.del_input()
        self.fwd_seq, self.bwd_seq = self.seq.cut_fwd_bwd()
        self.fwd_op_sched_list = [seq.op_sched for seq in self.fwd_seq.seq]
        self.bwd_op_sched_list = [seq.op_sched for seq in self.bwd_seq.seq]

        self.fwd_op_list = [
            op for op_sched in self.fwd_op_sched_list for op in op_sched.op_list
        ]
        self.bwd_op_list = [
            op for op_sched in self.bwd_op_sched_list for op in op_sched.op_list
        ]

        # self.fwd_alive_list = []
        # self.bwd_alive_list = []
        # list_kdn = []
        # start_i = []
        # for kg in self.list_kg:
        #     start_i.append(len(list_kdn))
        #     list_kdn += kg.list_kdn
        # alive_status = np.zeros(len(list_kdn) + 2, dtype=bool)
        # alive_status[-1] = 1
        # kdn_names = [kdn.name for kdn in list_kdn] + [
        #     self.list_kg[0].input_kdn_grad.name,
        #     self.list_kg[0].input_kdn_data.name,
        # ]
        # for op_sched in self.fwd_op_sched_list:
        #     index = [kdn_names.index(n) for n in op_sched.kdn_names]
        #     for a in op_sched.alive_list:
        #         alive_status[index] = a
        #         self.fwd_alive_list.append(alive_status.copy())
        # for op_sched in self.bwd_op_sched_list:
        #     index = [kdn_names.index(n) for n in op_sched.kdn_names]
        #     for a in op_sched.alive_list:
        #         alive_status[index] = a
        #         self.bwd_alive_list.append(alive_status.copy())

        # self.fwd_op_sched = OpSchedule_old(
        #     self.fwd_op_list,
        #     self.fwd_alive_list,
        #     self.list_kg[0].input_kdn_data,
        #     self.list_kg[0].input_kdn_grad,
        #     self.output,
        #     list_kdn,
        # )

        # self.bwd_op_sched = OpSchedule_old(
        #     self.bwd_op_list,
        #     self.bwd_alive_list,
        #     self.list_kg[0].input_kdn_data,
        #     self.list_kg[0].input_kdn_grad,
        #     self.output,
        #     list_kdn,
        # )

        # self.op_sched = OpSchedule_old(
        #     self.fwd_op_list + self.bwd_op_list,
        #     self.fwd_alive_list + self.bwd_alive_list,
        #     self.list_kg[0].input_kdn_data,
        #     self.list_kg[0].input_kdn_grad,
        #     self.output,
        #     list_kdn,
        # )

        # setattr(self.op_sched, "loss_idx", len(self.fwd_op_list))

        # self.simulation_time = sum(op.time for op in self.fwd_op_list) + sum(
        #     op.time for op in self.bwd_op_list
        # )

        # self.simulation_overhead = self.simulation_time / sum(
        #     [kcn.time for kg in self.list_kg for kcn in kg.list_kcn]
        # )
        return self.fwd_op_list, self.bwd_op_list

    def hg_to_rk_chain(self, hg):
        chain = RK_Chain_()

        chain.body = []

        def discretize(values):
            return [math.ceil(value / self.mem_unit) for value in values]

        no_grad_hcns = []
        loss_idx = hg.list_hcn.index(hg.loss_hcn)

        def set_op_sched(op_sched):
            setattr(op_sched, "time", op_sched.fwd_time)
            setattr(op_sched, "save", op_sched.save_mem)
            setattr(op_sched, "overhead", op_sched.fwd_overhead)

        for i, hcn in enumerate(hg.list_hcn[:loss_idx]):  # toposorted
            if len(hcn.users) > 1:
                return False
            if hcn.sub_graph is None:
                # WARNING: if hcn has no bwd, it has to be merged with the next one
                no_grad_hcns.append(hcn)
            else:
                ff_op_list = []
                for f_hcn in no_grad_hcns:
                    ff_op_list += f_hcn.ff_op_list
                fn_op_list = ff_op_list.copy() + hcn.ff_op_list
                first_hcn = hcn if not no_grad_hcns else no_grad_hcns[0]
                # assume single input

                input_kdn = list(first_hcn.deps)[0].kdn
                for op in ff_op_list:
                    if op.kn.name == input_kdn.name:
                        op.disabled = True
                fc_op_list = ff_op_list.copy() + hcn.ff_op_list

                no_grad_hcns = []

                Fn_sched = OpSchedule(
                    fn_op_list + hcn.ff_op_list,
                    len(fn_op_list),
                    refine=False,
                    correct_overhead=False,
                )
                Fc_sched = OpSchedule(
                    fc_op_list + hcn.ff_op_list,
                    len(fc_op_list),
                    refine=False,
                    correct_overhead=False,
                )
                set_op_sched(Fc_sched)
                set_op_sched(Fn_sched)

                sols = []
                for op_sched in hcn.sub_graph.list_sched:
                    fwd_op_list = (
                        fc_op_list + op_sched.op_list[: op_sched.loss_idx + 1]
                    )
                    bwd_op_list = op_sched.op_list[
                        op_sched.loss_idx :
                    ]  # start with loss op
                    Fwd_sched = OpSchedule(
                        fwd_op_list,
                        len(fwd_op_list) - 2,
                        refine=False,
                        correct_overhead=False,
                    )
                    set_op_sched(Fwd_sched)

                    Bwd_sched = OpSchedule(
                        bwd_op_list, 0, refine=False, correct_overhead=False
                    )
                    setattr(Bwd_sched, "time", Bwd_sched.bwd_time)
                    setattr(Bwd_sched, "save", Bwd_sched.save_mem)
                    setattr(Bwd_sched, "overhead", Bwd_sched.fwd_overhead)

                    sol = RK_Block_Solution_()
                    setattr(sol, "fwd_sched", Fwd_sched)
                    setattr(sol, "bwd_sched", Bwd_sched)
                    setattr(sol, "size_a_bar", Fwd_sched.mem)
                    setattr(sol, "time_fwd", Fwd_sched.fwd_time)
                    setattr(sol, "time_bwd", Bwd_sched.bwd_time)
                    setattr(sol, "overhead_fwd", Fwd_sched.fwd_overhead)
                    setattr(
                        sol,
                        "overhead_bwd",
                        Bwd_sched.bwd_overhead
                        + Bwd_sched.save_mem[-1]
                        - Fwd_sched.mem,
                    )
                    sols.append(sol)
                block = RK_block_()
                setattr(block, "sols", sols)
                setattr(block, "Fc_sched", Fc_sched)
                setattr(block, "Fn_sched", Fn_sched)
                setattr(block, "mem_inp", input_kdn.mem)
                setattr(block, "overhead_fast_fwd", Fc_sched.fwd_overhead)
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
        for (i, b) in enumerate(chain.body):
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
        chain.cw[chain.ln] = list(hg.outputs_hdn_data)[
            0
        ].mem  #  the final output

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

    def get_new_op_sched(self, op_sched: OpSchedule_old, kg):
        op_list = [Op(kg.dict_kn[op.name]) for op in op_sched.op_list]
        op_sched_new = OpSchedule(op_list)
        return op_sched_new

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

