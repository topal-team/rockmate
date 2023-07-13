# ============
# = ROCKMATE =
# ============

import rkgb
from rkgb.main import make_inputs
from rkgb.utils import print_debug, np, irotor
from rkgb.utils.global_vars import ref_verbose
from rkgb.utils.small_fcts import get_device
from rkgb.utils.ast_add_on import ast_to_str
from rockmate.def_op import DelOp, OpSchedule
from rockmate.def_chain import RK_Chain
from rockmate.def_sequence import (
    SeqBlockBwd,
    SeqBlockFc,
    SeqBlockFn,
    SeqBlockFe,
)
from rockmate.rotor_solver import seq_builder, solve_dp_functionnal
from rockmate.translator import Translator, RngState
from rockmate.compiler import Compiler, RK_Storage
import torch
from torch import tensor
import ast
import time
import pickle
from datetime import datetime
import warnings
from os import environ


def print_memsizes(list_kg):
    di = list_kg[-1].dict_info
    for kg in list_kg:
        for n in kg.dict_nodes.values():
            mt = n.main_target
            try:
                print_debug(
                    f"{mt} : memsize {di[mt].memsize} ; " f"fm {n.fgt_mem}",
                    end="",
                )
            except:
                print_debug("\nloss")
    print_debug("\n")


class Rockmate(torch.nn.Module):
    def __init__(
        self,
        original_mod,
        model_inputs,
        budget=None,
        mem_unit=None,
        verbose=False,
        solve=True,
        get_sequence=True,
        get_compiled_fct=True,
        nb_budget_save=10,
        nb_budget_peak=5,
        ilp_solver="gurobi",
    ):
        super().__init__()
        ref_verbose[0] = verbose
        self.device = get_device()
        self.original_mod = original_mod
        self.mem_unit = mem_unit if mem_unit else 1024**2
        # -- use pytorch graph builder to get the list of K_graphs --
        self.rkgb_res = rkgb.make_all_graphs(
            original_mod, model_inputs, verbose=verbose, bool_kg=True
        )  # we don't need the whole K_graph
        self.list_kg = self.rkgb_res.K_graph_list
        self.dict_constants = self.rkgb_res.K_graph_list[0].dict_constants
        self.eq_classes = self.rkgb_res.equivalent_classes
        self.init_code = ast_to_str(self.list_kg[0].init_code)
        self.output = self.list_kg[-1].output_kdn_data
        self.budget = budget
        if solve:
            self.get_chain(nb_budget_save, nb_budget_peak)
            if get_sequence:
                self.get_sequence(budget)
                if get_compiled_fct:
                    self.get_compiled_fct()

    def get_chain(self, nb_budget_save=10, nb_budget_peak=5):
        start = time.time()
        #  -- use checkmate to solve all the blocks --
        self.rk_chain = RK_Chain(
            self.list_kg,
            self.eq_classes,
            nb_budget_save,
            nb_budget_peak,
            mem_unit=self.mem_unit,
        )
        end = time.time()
        self.ILP_solve_time = end - start

        self.opt_table = None

    def get_sequence(self, budget):
        for n, p in self.original_mod.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        self.peak_overhead = max(
            [
                sum(
                    [
                        kdn.mem
                        for kdn in kcn.deps_fake
                        if kdn.main_target != kcn.main_target
                    ]
                )
                for kg in self.list_kg
                for kcn in kg.list_kcn
            ]
        )

        if budget:
            self.budget = budget
        else:
            self.budget = (
                # If not given a budget, we use the current available memory
                torch.cuda.get_device_properties(0).total_memory * 0.9
                - torch.cuda.memory_allocated()
                - self.peak_overhead
            )
        print_debug("budget", self.budget)
        # -- solve the chain like rotor --
        start = time.time()
        mmax = self.budget // self.mem_unit - self.rk_chain.cw[0]
        self.opt_table = solve_dp_functionnal(self.rk_chain, mmax, self.opt_table)
        self.seq = seq_builder(
            self.rk_chain, self.budget // self.mem_unit, self.opt_table
        )
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

        self.simulation_time = sum(op.time for op in self.fwd_op_list) + sum(
            op.time for op in self.bwd_op_list
        )

        self.simulation_overhead = self.simulation_time / sum(
            [kcn.time for kg in self.list_kg for kcn in kg.list_kcn]
        )
        self.storage = RK_Storage(self.device, self.original_mod, self.dict_constants)

    def get_compiled_fct(self):
        self.compiler = Compiler(self.storage)
        self.fct_list = self.compiler.compile(self.op_sched)
        loss_idx = len(self.fwd_op_list)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

    def _exec(self, fct_list, record_mem=False, compiled=False):
        if not compiled:
            warnings.warn("Translator is no longer used!")
        if record_mem:
            torch.cuda.reset_peak_memory_stats()
            self.mem_before = torch.cuda.memory_allocated()
            self.max_before = torch.cuda.max_memory_allocated()
            for fct in fct_list:
                fct()
            allo_mem = torch.cuda.memory_allocated() - self.mem_before
            peak_mem = torch.cuda.max_memory_allocated() - self.max_before
            self.max_mem.append(peak_mem - allo_mem)
            self.allo_mem.append(allo_mem)
        else:
            for fct in fct_list:
                fct()

    def forward(self, *args, record_mem=False, compiled=True, **kwargs):
        if not self.training:
            self.original_mod.eval()
            return self.original_mod(*args, **kwargs)
        model_inputs = make_inputs(self.original_mod, args, kwargs)
        for k, v in model_inputs.items():
            self.storage.add_val(k, v)
        exec(self.init_code, self.storage.gd, self.storage.ld)
        for kg in self.list_kg:
            for kdn in kg.list_kdn:
                self.storage.ld[kdn.main_target] = torch.empty(
                    0, device=self.device, requires_grad=kdn.info.requires_grad
                )
        self.max_mem = []
        self.allo_mem = []
        if compiled:
            for l in self.fwd_fct_list:
                self._exec(l, record_mem, compiled=compiled)
            return self.storage.get_val(self.output.main_target)

        return self.storage.get_val(self.output.main_target)

    def backward(
        self, stop=False, record_mem=False, add_output_grad=True, compiled=True
    ):
        if record_mem:
            self.output_size = irotor.tensorMsize(
                self.storage.ld[self.output.main_target]
            )
            # self.allo_mem[-1] += self.output.info.memsize
            # output grad is generated outside
            loss_idx = len(self.allo_mem)
        if stop:
            for l in self.bwd_fct_list[: stop - len(self.fwd_fct_list)]:
                self._exec(l, record_mem, compiled=compiled)
            return None
        if compiled:
            for l in self.bwd_fct_list:
                self._exec(l, record_mem, compiled=compiled)

        if record_mem and add_output_grad:
            self.allo_mem[loss_idx] += self.output_size

    def expect_time(self):
        # Sum of the measured time of each operation for one batch
        return self.fwd_seq.compute_time() + self.bwd_seq.compute_time()

    def expect_mem(self, overhead=False):
        # Peak mem based on the measured memory/overhead of each operation
        pred_mem = []
        acc_mem = np.zeros(len(self.fwd_seq.seq))
        # alive_dict = {}
        # for kg in self.list_kg:
        #     for kdn in kg.list_kdn:
        #         alive_dict[kdn.name] = (0, kdn.mem)
        for i, seq in enumerate(self.fwd_seq.seq + self.bwd_seq.seq):
            op_sched = seq.op_sched
            for a, op in zip(op_sched.alive_list, op_sched.op_list):
                acc_mem[seq.index] = (
                    np.dot(a, op_sched.mem_sizes) - op_sched.input_size[1]
                )
                # if not op_sched.is_fwd:
                #     acc_mem[seq.index] -= op_sched.output_size[1]
                pred_mem.append(sum(acc_mem))
                if overhead and op.op_type == "Run":
                    pred_mem[-1] += op.overhead
            # for s, op in zip(op_sched.save, op_sched.op_list):
            # for i, op in enumerate(op_sched.op_list):
            # acc_mem[seq.index] = s
            # pred_mem.append(sum(acc_mem))
            # if overhead and op.op_type == "Run":
            #     pred_mem[-1] += op.overhead
        return pred_mem

    def reinit(self):
        self.original_mod.zero_grad()
        self.storage.ld = {}

    def save_to_file(self, path, id=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
        sol = {}
        sol["op_sched"] = self.op_sched
        sol["loss_idx"] = len(self.fwd_op_list)
        with open(f"{path}/{id}_solution.pkl", "wb") as f:
            pickle.dump(sol, f)

    def load_from_file(self, path, id=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
        with open(f"{path}/{id}_solution.pkl", "rb") as f:
            sol = pickle.load(f)
        op_sched = sol["op_sched"]
        loss_idx = sol["loss_idx"]
        self.storage = RK_Storage(self.device, self.original_mod, self.dict_constants)
        self.compiler = Compiler(self.storage)
        self.fct_list = self.compiler.compile(op_sched)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]
