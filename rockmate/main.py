# ============
# = ROCKMATE =
# ============

import rkgb
from rkgb.main import make_inputs
from rkgb.utils import print_debug, np, irotor
from rkgb.utils.global_vars import ref_verbose, solver_name
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
from rockmate.HILP_gurobi import *
from rockmate.rotor_solver import seq_builder, solve_dp_functional
from rockmate.translator import Translator, RngState
from rockmate.compiler import Compiler, RK_Storage, make_gd
import torch
from torch import tensor
import ast
import time
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


class CheckpointedModule(torch.nn.Module):
    compiler = None
    autograd_Function = None
    backward_stop = False
    backward_add_output_grad = True
    def __init__(
        self,
        original_mod,
        dict_inputs,
        mem_limit=None,
        mem_unit=None,
        verbose=False,
        get_chain=True,
        get_sequence=True,
        get_compiled_fct=True,
        nb_budget_abar=10,
        nb_budget_all=2,
        ilp_solver="gurobi",
    ):
        super().__init__()
        ref_verbose[0] = verbose
        solver_name[0] = ilp_solver
        self.device = get_device()
        self.original_mod = original_mod
        self.mem_unit = mem_unit if mem_unit else 1024 ** 2
        # -- use pytorch graph builder to get the list of K_graphs --
        self.rkgb_res = rkgb.make_all_graphs(
            original_mod,
            dict_inputs,
            verbose=verbose,
            dict_wanted_graphs={"Kl", "K"},
        )  # we don't need the whole K_graph
        self.list_kg = self.rkgb_res.K_graph_list
        self.dict_constants = self.rkgb_res.K_graph_list[0].dict_constants
        self.eq_classes = self.rkgb_res.equivalent_classes
        self.init_code = ast_to_str(self.list_kg[0].init_code)
        self.output = self.list_kg[-1].output_kdn_data
        self.mem_limit = mem_limit
        if get_chain:
            self.get_chain(nb_budget_abar, nb_budget_all)
            if get_sequence:
                self.get_sequence(mem_limit)
                if get_compiled_fct:
                    self.get_compiled_fct()
                    self.define_autograd_Function()

    def get_chain(self, nb_budget_abar=10, nb_budget_all=5):
        start = time.time()
        #  -- use checkmate to solve all the blocks --
        self.rk_chain = RK_Chain(
            self.list_kg,
            self.eq_classes,
            nb_budget_abar,
            nb_budget_all,
            mem_unit=self.mem_unit,
        )
        end = time.time()
        self.ILP_solve_time = end - start

        self.opt_table = None

    def get_sequence(self, mem_limit):
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

        if mem_limit:
            self.mem_limit = mem_limit
        else:
            self.mem_limit = (
                # If not given a budget, we use the current available memory
                torch.cuda.get_device_properties(0).total_memory * 0.9
                - torch.cuda.memory_allocated()
                - self.peak_overhead
            )
        print_debug("mem_limit", self.mem_limit)
        # -- solve the chain like rotor --
        start = time.time()
        mmax = self.mem_limit // self.mem_unit - self.rk_chain.cw[0]
        self.opt_table = solve_dp_functional(
            self.rk_chain, mmax, self.opt_table
        )
        self.seq = seq_builder(
            self.rk_chain, self.mem_limit // self.mem_unit, self.opt_table
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
        self.gd = make_gd(self.device, self.original_mod, self.dict_constants)

    def get_compiled_fct(self):
        self.compiler = Compiler(self.gd)
        self.fct_list = self.compiler.compile(self.op_sched)
        loss_idx = len(self.fwd_op_list)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

    def get_compiled_fct_HILP(
        self,
        mem_limit=False,
        recursive=True,
        P_config=None,
        gurobi_params={"LogToConsole": 0, "IntegralityFocus": 1,},
    ):
        # based only on rkgb
        mem_limit = mem_limit or self.mem_limit
        kg = self.rkgb_res.K_graph
        sg = self.rkgb_res.S_graph
        if recursive:
            pg = rkgb.Ptools.S_to_P(sg, config=P_config)
            self.hg = rkgb.Htools.P_and_K_to_H(pg, kg)
            print(f"Size of Hgraph {len(self.hg.list_hcn)}")

            save_all_sched = rkgb.Htools.get_save_all_option(self.hg)
            self.hg.add_sched(save_all_sched)
            solve_hg_recursive(self.hg, solve_self=False)
            print("Low level finished")
        self.md = ModelGurobi(
            self.hg, mem_limit, mem_limit, gurobi_params=gurobi_params
        )
        self.md.solve()
        if not self.md.feasible:
            return "Not feasible solution"
        else:
            print(f"Solution with obj: {self.md.md.getObjective().getValue()}")
        self.h_sched = self.md.schedule()
        self.bottom_op_list = self.h_sched.get_bottom_op_list()
        self.kn_list = rkgb.Htools.get_kn_list(self.bottom_op_list, kg)
        loss_idx = [op.name for op in self.bottom_op_list].index(
            "Loss_hcn_of_Hg_0"
        )
        self.gd = make_gd("cuda", self.original_mod, self.dict_constants)
        # TO CHANGE self.device instead of "cuda"
        self.compiler = Compiler(self.gd)
        fct_list = self.compiler.compile_from_KN_list(self.kn_list)
        self.fwd_fct_list = fct_list[:loss_idx]
        self.bwd_fct_list = fct_list[loss_idx + 1 :]


    def _exec(self, fct_list):
        if self.exec_with_record_mem:
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


    def define_autograd_Function(self):
        # To define properly new module's forward and backward
        # functions we need to make it compatible with Autograd.
        # This method MUST be called to create the forward function.
        # With this the module will be fully compatible with Autograd.
        
        # Rem 1: 
        # Autograd.Function forward function kwargs must be defined,
        # so we cannot use "**kwargs". Which means to do things 
        # properly we would need to extract original_mod's kwargs
        # definition (with default value) and write custom Autograd.Function
        # definition using a string and `exec` (since the definition 
        # of the Function depends on some arguments). 
        # To avoid this issue we do the following : 
        # nn.Module.forward function receives *args and **kwargs
        # and saves everything in a buffer `self(nn.Module).dict_inputs_buffer`
        # then we call Function.forward without giving it the inputs.
        # Instead, Function.forward catches the inputs from the buffer.
        # Note: Function.forward needs at least one input, so
        # we just give a dummy input, and this input must requires_grad
        # (otherwise, if none of Function.forward's inputs req_grad,
        # Autograd thinks it's useless to generate a backward function.
        # Because it only sees the inputs and outputs, and we
        # take care of all the intermediate evaluations, therefore
        # autograd doesn't realize there are some params which 
        # require_grad for instance.)
        
        # Rem 2:
        # Normally Autograd.Function's backward method returns inputs' grad,
        # and Autograd then backward the inputs using these grads.
        # But since we use a buffer to pass the inputs (cf Rem 1). 
        # Autograd cannot see the inputs and therefore backward them once
        # we finished. So instead of returning inputs' grad we trigger
        # inputs' backward.
        
        # Rem 3:
        # To isolate our Module range of action we need to detach the inputs
        # before using them so we won't backward through them when handling
        # the backward of our module. Otherwise the backward operations of the
        # first nodes inside our computation graph will trigger inputs' backward.
        # So we detach the inputs, then we do everything for our part, and once
        # we finished, we trigger inputs' backward (cf Rem 2 -> normally we 
        # would have simply returned inputs' grad).
        
        # Rem 4: TODO REWRITE THIS
        # Our goal is to define a custom backward method for the output
        # of the nn.Module, which mean output.backward() will lead to
        # the following lines, where `output` is the output of the forward
        # function of the Checkpointed Module. But by default it's also the
        # output of the last primitive operation inside the module. And we need
        # to be able to backward the last node using its standard backward
        # function. So we must not overwrite last node's output backward
        # function, otherwise last_kcn.backward will call the following lines.
        # SO we need to return a detached copy of the outputs.
        # Thus, the last node's output backward isn't affected, and
        # we properly redefine CheckpointedModule's output backward.
        
        RkMod = self
        # -> so we can access to it inside the following class definition
        # (when defining a Class inside a Class we cannot use `self`)
        class RK_autograd_Function(torch.autograd.Function):
            # === OUR FORWARD FUNCTION ===
            @staticmethod
            def forward(ctx,dummy_input):
                # *** INITIALIZATION PART ***
                # -> Get the inputs using the buffer (Rem 1)
                dict_inputs = RkMod.dict_inputs_buffer
                RkMod.dict_inputs_buffer = None
                # -> Create the RK_Storage for this run, and store it in ctx
                ctx.RK_Storage = storage = RK_Storage()
                RkMod.compiler.storage = storage
                # -> Detach input tensors (Rem 3) and store all the inputs
                dict_input_tensors_detach = dict() # dict : input -> detached input
                for k,v in dict_inputs.items():
                    if isinstance(v,torch.Tensor):
                        v_d = v.detach().requires_grad_(v.requires_grad)
                        dict_input_tensors_detach[v] = v_d
                        storage.ld[k] = v_d
                    # TODO elif iterables of Tensors ?
                    else:
                        storage.ld[k] = v
                # -> Initialize the storage
                for kg in RkMod.list_kg:
                    for kdn in kg.list_kdn:
                        storage.ld[kdn.main_target] = torch.empty(
                            0, device=RkMod.device, requires_grad=kdn.info.requires_grad
                        )

                # *** EXECUTION PART ***
                # -> Autograd turns off itself before giving use the control.
                # -> But we need it to forward/backward each node.
                with torch.enable_grad():
                    exec(
                        RkMod.init_code, 
                        RkMod.gd,# is compiler.gd
                        storage.ld)
                    for l in RkMod.fwd_fct_list:
                        RkMod._exec(l)
                # -> Get the output
                out = RkMod.compiler.get_val(RkMod.output.main_target)
                # TODO multiple outputs
                # -> Clear the compiler
                RkMod.compiler.storage = None
                # -> Remember that out have been detached from the rest during exec
                return out
                """
                ctx.set_materialize_grads(True) # as the default
                # -> so we don't have to check if grad_output is None
                # -> if outputs' grad is None Autograd fill them with zeros
                """
            # === END OF FORWARD FUNCTION ===
        
            """
            @staticmethod
            def setup_context(ctx,inputs,outputs):
                pass
            # PyTorch prefer to handle the ctx in a separate method,
            # but it makes more sense for us to handle ctx during the forward.
            # (Also, setup_context can only access to inputs and outputs,
            # so they suggest to pass intermediate values as outputs,
            # but outputs can only be tensors)
            # Anyway, it's not important, Autograd is originally designed
            # to handle the ctx during the forward, it's just that PyTorch 2.0
            # now prefers to use this separate method.
            """

            # === OUR BACKWARD FUNCTION ===
            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad_out): # TODO multiple outputs
                # -> Reload the storage and out
                storage = ctx.RK_Storage
                RkMod.compiler.storage = storage
                out = RkMod.compiler.get_val(RkMod.output.main_target)
                # -> Put grad_out in out.grad (Rem 4)
                out.grad = grad_out
                # * record_mem stuff *
                if RkMod.exec_with_record_mem:
                    RkMod.output_size = irotor.tensorMsize(
                        storage.ld[RkMod.output.main_target])
                    loss_idx = len(RkMod.allo_mem)
                    # self.allo_mem[-1] += self.output.info.memsize
                    # output grad is generated outside
                # -> exec bwd_fct_list with early stop or not
                stop = RkMod.backward_stop
                if stop:
                    len_fwd = len(RkMod.fwd_fct_list)
                    for l in RkMod.bwd_fct_list[:(stop-len_fwd)]:
                        RkMod._exec(l)
                else:
                    for l in RkMod.bwd_fct_list:
                        RkMod._exec(l)
                    if (RkMod.exec_with_record_mem
                    and RkMod.backward_add_output_grad):
                        RkMod.allo_mem[loss_idx] += RkMod.output_size
                # -> Clear the compiler (and Autograd clears ctx)
                RkMod.compiler.storage = None
                # -> return grad of dummy input (Rem 1)
                return torch.ones(1)
            # === END OF BACKWARD FUNCTION ===
        
        self.autograd_Function = RK_autograd_Function


    # === nn.module's forward method wrapping self.autograd_Function.forward ===
    def forward(self, *args, record_mem=False, **kwargs):
        self.exec_with_record_mem = record_mem
        self.max_mem  = []
        self.allo_mem = []
        if not self.training:
            self.original_mod.eval()
            return self.original_mod(*args, **kwargs)
        else:
            if self.compiler is None:
                raise Exception(
                    "No schedule compiled, no solution to exec,\n"\
                    "To be able to use the RkMod you first need "\
                    "to call get_compiled_fct(_HILP).")
            elif self.autograd_Function is None:
                raise Exception(
                    "The custom forward and backward functions haven't "\
                    "been generated yet, please call the method : "\
                    "define_autograd_Function")
            # -> Send the inputs to Function.forward via the buffer
            self.dict_inputs_buffer = make_inputs(self.original_mod, args, kwargs)
            return self.autograd_Function.apply(torch.ones(1).requires_grad_())           
    # === end of forward ===

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
    