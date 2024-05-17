# This file is not actually used after updating rk_rotor.py
import time
from hrockmate.rkgb.utils import np

from .def_chain import RK_Chain
from .rotor_solver import seq_builder, solve_dp_functional
from .rk_rotor import RK_rotor
from .def_op import OpSchedule
from .def_sequence import (
    SeqBlockBwd,
    SeqBlockFe,
)


class Rockmate:
    def __init__(self, nb_budget_save=10, nb_budget_peak=2, mem_unit=1024 ** 2):
        self.nb_budgey_save = nb_budget_save
        self.nb_budgey_peak = nb_budget_peak
        self.mem_unit = mem_unit

    def solve(self, rkgb_res, mem_limit):
        self.mem_limit = mem_limit
        start = time.time()
        self.list_kg = rkgb_res.K_graph_list
        self.output = self.list_kg[-1].output_kdn_data

        #  -- use checkmate to solve all the blocks --
        self.rk_chain = RK_Chain(
            self.list_kg,
            rkgb_res.equivalent_classes,
            self.nb_budgey_save,
            self.nb_budgey_peak,
            mem_unit=self.mem_unit,
        )
        end = time.time()
        self.ILP_solve_time = end - start

        self.opt_table = None

        # -- solve the chain like rotor --
        start = time.time()
        # mmax = self.mem_limit // self.mem_unit - self.rk_chain.cw[0]
        # self.opt_table = solve_dp_functional(
        #     self.rk_chain, mmax, self.opt_table
        # )
        # self.seq = seq_builder(
        #     self.rk_chain, self.mem_limit // self.mem_unit, self.opt_table
        # )
        self.seq = RK_rotor().solve(self.rk_chain, mem_limit)
        end = time.time()
        self.DP_solve_time = end - start

        enable = np.zeros(len(self.list_kg))
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

        self.fwd_alive_list = []
        self.bwd_alive_list = []
        list_kdn = []
        start_i = []
        for kg in self.list_kg:
            start_i.append(len(list_kdn))
            list_kdn += kg.list_kdn
        alive_status = np.zeros(len(list_kdn) + 2, dtype=bool)
        alive_status[-1] = 1
        kdn_names = [kdn.name for kdn in list_kdn] + [
            self.list_kg[0].input_kdn_grad.name,
            self.list_kg[0].input_kdn_data.name,
        ]
        for op_sched in self.fwd_op_sched_list:
            index = [kdn_names.index(n) for n in op_sched.kdn_names]
            for a in op_sched.alive_list:
                alive_status[index] = a
                self.fwd_alive_list.append(alive_status.copy())
        for op_sched in self.bwd_op_sched_list:
            index = [kdn_names.index(n) for n in op_sched.kdn_names]
            for a in op_sched.alive_list:
                alive_status[index] = a
                self.bwd_alive_list.append(alive_status.copy())

        self.fwd_op_sched = OpSchedule(
            self.fwd_op_list,
            self.fwd_alive_list,
            self.list_kg[0].input_kdn_data,
            self.list_kg[0].input_kdn_grad,
            self.output,
            list_kdn,
        )

        self.bwd_op_sched = OpSchedule(
            self.bwd_op_list,
            self.bwd_alive_list,
            self.list_kg[0].input_kdn_data,
            self.list_kg[0].input_kdn_grad,
            self.output,
            list_kdn,
        )

        self.op_sched = OpSchedule(
            self.fwd_op_list + self.bwd_op_list,
            self.fwd_alive_list + self.bwd_alive_list,
            self.list_kg[0].input_kdn_data,
            self.list_kg[0].input_kdn_grad,
            self.output,
            list_kdn,
        )

        setattr(self.op_sched, "loss_idx", len(self.fwd_op_list))

        self.simulation_time = sum(op.time for op in self.fwd_op_list) + sum(
            op.time for op in self.bwd_op_list
        )

        self.simulation_overhead = self.simulation_time / sum(
            [kcn.time for kg in self.list_kg for kcn in kg.list_kcn]
        )

        return self.op_sched
