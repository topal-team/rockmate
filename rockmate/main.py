# ============
# = ROCKMATE =
# ============

import torch
from torch import tensor
import ast
import time
import warnings
from os import environ

from . import rkgb
from .rkgb.main import make_inputs, make_all_graphs
from .rkgb.utils import print_debug, np, irotor
from .rkgb.utils.global_vars import ref_verbose, solver_name
from .rkgb.utils.small_fcts import get_device
from .rkgb.utils.ast_add_on import ast_to_str
from .rkgb import Ptools
from .solvers.def_op import OpSchedule as OpSchedule_old
from .solvers.def_chain import RK_Chain
from .solvers.def_sequence import (
    SeqBlockBwd,
    SeqBlockFe,
)
from .solvers.main import preprocess, solve_recursive
from .solvers.op_schedule import *
from .solvers.hilp import HILP
from .solvers.HILP_gurobi import *
from .solvers.rotor_solver import seq_builder, solve_dp_functional
from .compiler import Compiler, RK_Storage, make_gd


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


class HRemat(torch.nn.Module):
    compiler = None
    autograd_Function = None
    backward_stop = False
    backward_add_output_grad = True
    op_sched = None

    def __init__(
        self,
        original_mod,
        model_inputs,
        budget=None,
        list_solvers=[HILP()],
        solve_sched=True,
        verbose=False,
        ilp_solver="gurobi",
        model_kwargs=None,
        rkgb_kwargs=dict(
            partitioners=[
                Ptools.Partitioner_bottom_to_top_2(),
                # Ptools.Partitioner_seq(),
            ],
        ),
    ):
        super().__init__()
        ref_verbose[0] = verbose
        solver_name[0] = ilp_solver
        self.device = get_device()
        object.__setattr__(self, "original_mod", original_mod)
        dict_inputs = make_inputs(original_mod, model_inputs, model_kwargs)
        #  We don't want to use the default setattr
        # because torch.nn.Module will register it as a submodule
        # -- use pytorch graph builder to get the list of K_graphs --
        self.rkgb_res = make_all_graphs(
            original_mod,
            dict_inputs,
            verbose=verbose,
            wanted_graphs={"H"},
            **rkgb_kwargs,
        )  # we don't need K_graph_list
        self.list_solvers = list_solvers
        self.dict_constants = self.rkgb_res.K_graph.dict_constants
        self.init_code = ast_to_str(self.rkgb_res.K_graph.init_code)
        self.dict_output_viewing_code = dict(
            (out_mt,ast_to_str(view_code)) 
            for (out_mt,view_code) in self.rkgb_res.K_graph.dict_output_viewing_code)
        self.outputs_wrapping_code = \
            ast_to_str(self.rkgb_res.K_graph.outputs_wrapping_code)
        self.output = self.rkgb_res.K_graph.list_outputs_kdn_data[0]
        self.budget = budget
        self.gd = make_gd(self.device, self.original_mod, self.dict_constants)
        self.list_solutions = []
        if solve_sched:
            self.solve_sched()

    def solver_recursive(self, list_solvers=None):
        list_solvers = list_solvers or self.list_solvers
        for cluster in self.rkgb_res.H_cluster.all_clusters:
            if not cluster.is_bottom:
                preprocess(
                    cluster, protect_names=["sources data", "sources grad"]
                )
        solve_recursive(
            self.rkgb_res.H_cluster, list_solvers=list_solvers, skip_self=True
        )

    def solve_sched(self, budget=None, list_solvers=None, rec=True):
        # budget should be a single value.
        # if multiple solvers in list_solvers,
        # will choose the one with minimum time
        budget = budget or self.budget
        list_solvers = list_solvers or self.list_solvers
        for solver in list_solvers:
            if (
                True in [isinstance(solver, HILP) for solver in list_solvers]
                and rec
            ):
                self.solver_recursive()
        list_solutions = []
        for solver in list_solvers:
            if isinstance(solver, HILP):
                solver.config.nb_total_nodes = 30
                list_solutions.extend(
                    solver(self.rkgb_res.H_cluster, [budget], accurate_mem=True)
                )
                solver.config.nb_total_nodes = 20  # in case further usage

            else:
                list_solutions.extend(solver(self.rkgb_res.H_cluster, [budget]))
        if not list_solutions:
            warnings.warn("no feasible schedule is found")
        else:
            self.op_sched = list_solutions[
                np.argmin(sum(op_sched.time) for op_sched in list_solutions)
            ]
            self.get_compiled_fct()
        self.list_solutions.extend(list_solutions)

    def get_compiled_fct(self):
        self.compiler = Compiler(self.gd)
        self.fct_list = self.compiler.compile_from_schedule(self.op_sched)
        loss_idx = self.op_sched.loss_idx

        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]
        self.define_autograd_Function()
        self.inherits_original_mod_attributes_and_methods()

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
        #  To define properly new module's forward and backward
        #  functions we need to make it compatible with Autograd.
        #  This method MUST be called to create the forward function.
        # With this the module will be fully compatible with Autograd.

        # Rem 1:
        # Autograd.Function forward function kwargs must be defined,
        #  so we cannot use "**kwargs". Which means to do things
        #  properly we would need to extract original_mod's kwargs
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
        #  Because it only sees the inputs and outputs, and we
        # take care of all the intermediate evaluations, therefore
        # autograd doesn't realize there are some params which
        # require_grad for instance.)

        #  Rem 2:
        # Normally Autograd.Function's backward method returns inputs' grad,
        # and Autograd then backward the inputs using these grads.
        # But since we use a buffer to pass the inputs (cf Rem 1).
        #  Autograd cannot see the inputs and therefore backward them once
        # we finished. So instead of returning inputs' grad we trigger
        # inputs' backward.

        # Rem 3:
        # To isolate our Module range of action we need to detach the inputs
        # before using them so we won't backward through them when handling
        #  the backward of our module. Otherwise the backward operations of the
        # first nodes inside our computation graph will trigger inputs' backward.
        # So we detach the inputs, then we do everything for our part, and once
        # we finished, we trigger inputs' backward (cf Rem 2 -> normally we
        # would have simply returned inputs' grad).

        # Rem 4: TODO REWRITE THIS
        # ALWAYS DETACH IN CASE OF VIEWS TO PROCESS grad_out
        # SO IT'S A DOUBLE DETACH
        # Our goal is to define a custom backward method for the output
        # of the nn.Module, which mean output.backward() will lead to
        # the following lines, where `output` is the output of the forward
        # function of the Checkpointed Module. But by default it's also the
        # output of the last primitive operation inside the module. And we need
        # to be able to backward the last node using its standard backward
        # function. So we must not overwrite last node's output backward
        # function, otherwise last_kcn.backward will call the following lines.
        # SO we need to return a detached copy of the outputs.
        #  Thus, the last node's output backward isn't affected, and
        # we properly redefine CheckpointedModule's output backward.

        RkMod = self

        # -> so we can access to it inside the following class definition
        #  (when defining a Class inside a Class we cannot use `self`)
        class RK_autograd_Function(torch.autograd.Function):
            # === OUR FORWARD FUNCTION ===
            @staticmethod
            def forward(ctx, dummy_input, *args):
                # *** INITIALIZATION PART ***
                #  -> Get the inputs using the buffer (Rem 1)
                dict_inputs = RkMod.dict_inputs_buffer
                RkMod.dict_inputs_buffer = None
                #  -> Create the RK_Storage for this run, and store it in ctx
                ctx.RK_Storage = storage = RK_Storage()
                RkMod.compiler.storage = storage
                #  -> Store what we need to return inputs' grad (Rem 1)
                ctx.name_of_inputs_which_req_grad = (
                    RkMod.name_of_inputs_which_req_grad_buffer
                )
                RkMod.name_of_inputs_which_req_grad_buffer = None
                #  -> Detach input tensors (Rem 3) and store all the inputs
                dict_input_tensors_detach = (
                    dict()
                )  #  dict : input -> detached input
                for k, v in dict_inputs.items():
                    if isinstance(v, torch.Tensor):
                        v_d = v.detach().requires_grad_(v.requires_grad)
                        dict_input_tensors_detach[v] = v_d
                        storage.ld[k] = v_d
                    #  TODO elif iterables of Tensors ?
                    else:
                        storage.ld[k] = v
                #  -> Initialize the storage
                for kdn in RkMod.rkgb_res.K_graph.list_kdn:
                    storage.ld[kdn.main_target] = torch.empty(
                        0,
                        device=RkMod.device,
                        requires_grad=kdn.info.requires_grad,
                    )

                #  *** EXECUTION PART ***
                # -> Autograd turns off itself before giving use the control.
                # -> But we need it to forward/backward each node.
                with torch.enable_grad():
                    exec(
                        RkMod.init_code, RkMod.gd, storage.ld  # is compiler.gd
                    )
                    for l in RkMod.fwd_fct_list:
                        RkMod._exec(l)
                # -> Get the output
                outs = [RkMod.compiler.get_val(out_mt) 
                        for out_mt in RkMod.rkgb_res.S_graph.outputs]
                # -> Remember that out have been detached from the rest during exec
                return outs
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
            def backward(ctx, *grad_outs):  #  TODO multiple outputs
                #  -> Reload the storage and out
                storage = ctx.RK_Storage
                RkMod.compiler.storage = storage
                # -> Put grad_out in out.grad (Rem 4)
                for out_mt,out_grad in zip(RkMod.rkgb_res.S_graph.outputs,grad_outs):
                    out = RkMod.compiler.get_val(out_mt)
                    out.grad = out_grad.view(out_grad.shape)
                    out_grad.data = torch.empty(0)
                    
                #  * record_mem stuff *
                if RkMod.exec_with_record_mem:
                    RkMod.output_size = irotor.tensorMsize(
                        storage.ld[RkMod.output.main_target]
                    )
                    loss_idx = len(RkMod.allo_mem)
                    # self.allo_mem[-1] += self.output.info.memsize
                    # output grad is generated outside
                # -> exec bwd_fct_list with early stop or not
                stop = RkMod.backward_stop
                if stop:
                    len_fwd = len(RkMod.fwd_fct_list)
                    for l in RkMod.bwd_fct_list[: (stop - len_fwd)]:
                        with torch.enable_grad():
                            RkMod._exec(l)
                else:
                    for l in RkMod.bwd_fct_list:
                        with torch.enable_grad():
                            RkMod._exec(l)
                    if (
                        RkMod.exec_with_record_mem
                        and RkMod.backward_add_output_grad
                    ):
                        RkMod.allo_mem[loss_idx] += RkMod.output_size
                    #  -> return grad of dummy input + inputs' which req grad (Rem 1)
                    grad_inputs = tuple(
                        RkMod.compiler.get_val(inp).grad
                        for inp in ctx.name_of_inputs_which_req_grad
                    )
                    grads = (torch.ones(1),) + grad_inputs
                    #  -> Clear the compiler (and Autograd clears ctx)
                    RkMod.compiler.storage = None
                    return grads

            # === END OF BACKWARD FUNCTION ===

        self.autograd_Function = RK_autograd_Function

    #  === nn.module's forward method wrapping self.autograd_Function.forward ===
    def forward(self, *args, record_mem=False, **kwargs):
        self.exec_with_record_mem = record_mem
        self.max_mem = []
        self.allo_mem = []
        if not self.training:
            self.original_mod.eval()
            return self.original_mod(*args, **kwargs)
        else:
            if self.compiler is None:
                raise Exception(
                    "No schedule compiled, no solution to exec,\n"
                    "To be able to use the RkMod you first need "
                    "to call get_compiled_fct(_HILP)."
                )
            elif self.autograd_Function is None:
                raise Exception(
                    "The custom forward and backward functions haven't "
                    "been generated yet, please call the method : "
                    "define_autograd_Function"
                )
            # -> Send the inputs to Function.forward via the buffer (Rem 1)
            self.dict_inputs_buffer = dict_inputs = make_inputs(
                self.original_mod, args, kwargs
            )
            # -> Pass the inputs which req grad to prepare their backward (Rem 1)
            inputs_which_req_grad = []
            name_of_inputs_which_req_grad = []
            for k, v in dict_inputs.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    inputs_which_req_grad.append(v)
                    name_of_inputs_which_req_grad.append(k)
            self.name_of_inputs_which_req_grad_buffer = (
                name_of_inputs_which_req_grad
            )
            dummy_input = torch.ones(1).requires_grad_()
            output_mt_values = self.autograd_Function.apply(
                dummy_input, *inputs_which_req_grad
            )
            for out_mt,out_mt_value \
                in zip(self.rkgb_res.S_graph.outputs,output_mt_values):
                view_code = self.dict_output_viewing_code[out_mt]
                exec(view_code,self.gd,self.compiler.storage.ld)
                # -> We access to out_mt_value directly in the storage
            exec(self.outputs_wrapping_code,self.gd,self.compiler.storage.ld)
            final_output = self.compiler.get_val(self.rkgb_res.D_graph.outputs[0])
            #  -> Clear the compiler
            self.compiler.storage = None
            return final_output


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

    def reinit(self, set_to_none=False):
        # In our experiments, we set the parameter grad to 0's
        # so that Backward only creates memory for activations
        self.original_mod.zero_grad(set_to_none=set_to_none)

    # === Inherits original_mod Attributes and Methods ===
    """
    @property
    def original_mod(self):
        return self._HideFromPytorch.mod
    @original_mod.setter
    def original_mod(self,original_mod):
        print("SET ORIGINAL MOD")
        class HideFromPytorch():
            def __init__(self,mod):
                self.mod = mod
        self._HideFromPytorch = HideFromPytorch(original_mod)
    #original_mod = property(get_original_mod,set_original_mod)
    """

    def inherits_original_mod_attributes_and_methods(self):
        for k, v in self.original_mod.__dict__.items():
            if (
                not "forward" in k
                and not "backward" in k
                and k not in ["training"]
            ):
                self.__dict__[k] = v

















class CheckpointedModule(torch.nn.Module):
    compiler = None
    autograd_Function = None
    backward_stop = False
    backward_add_output_grad = True

    def __init__(
        self,
        original_mod,
        model_inputs,
        mem_limit=None,
        mem_unit=None,
        verbose=False,
        get_chain=True,
        get_sequence=True,
        get_compiled_fct=True,
        nb_budget_abar=10,
        nb_budget_all=2,
        ilp_solver="gurobi",
        model_kwargs=None,
    ):
        super().__init__()
        ref_verbose[0] = verbose
        solver_name[0] = ilp_solver
        self.device = get_device()
        object.__setattr__(self, "original_mod", original_mod)
        dict_inputs = make_inputs(original_mod, model_inputs, model_kwargs)
        #  We don't want to use the default setattr
        # because torch.nn.Module will register it as a submodule
        self.mem_unit = mem_unit if mem_unit else 1024**2
        # -- use pytorch graph builder to get the list of K_graphs --

        self.rkgb_res = rkgb.make_all_graphs(
            original_mod,
            dict_inputs,
            verbose=verbose,
            wanted_graphs={"Kl", "H"},
        )  # we don't need the whole K_graph
        self.list_kg = self.rkgb_res.K_graph_list
        self.list_kg[0].fake_input_kdn_grad()
        self.dict_constants = self.rkgb_res.K_graph.dict_constants
        self.eq_classes = self.rkgb_res.equivalent_classes
        self.init_code = ast_to_str(self.rkgb_res.K_graph.init_code)
        self.dict_output_viewing_code = dict(
            (out_mt,ast_to_str(view_code)) 
            for (out_mt,view_code) in self.rkgb_res.K_graph.dict_output_viewing_code)
        self.output = self.rkgb_res.K_graph.list_outputs_kdn_data[0]
        self.mem_limit = mem_limit
        self.gd = make_gd(self.device, self.original_mod, self.dict_constants)
        if get_chain:
            self.get_chain(nb_budget_abar, nb_budget_all)
            if get_sequence:
                self.get_sequence(mem_limit)
                if get_compiled_fct:
                    self.get_compiled_fct()
                    self.define_autograd_Function()
                    self.inherits_original_mod_attributes_and_methods()

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

        self.fwd_op_sched = OpSchedule_old(
            self.fwd_op_list,
            self.fwd_alive_list,
            self.list_kg[0].input_kdn_data,
            self.list_kg[0].input_kdn_grad,
            self.output,
            list_kdn,
        )

        self.bwd_op_sched = OpSchedule_old(
            self.bwd_op_list,
            self.bwd_alive_list,
            self.list_kg[0].input_kdn_data,
            self.list_kg[0].input_kdn_grad,
            self.output,
            list_kdn,
        )

        self.op_sched = OpSchedule_old(
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

    def get_compiled_fct(self):
        self.compiler = Compiler(self.gd)
        if hasattr(self.op_sched, "interfaces"):
            self.fct_list = self.compiler.compile_from_schedule(self.op_sched)
        else:
            self.fct_list = self.compiler.compile(self.op_sched)
        if hasattr(self.op_sched, "loss_idx"):
            loss_idx = self.op_sched.loss_idx
        else:
            loss_idx = len(self.fwd_op_list)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]
        self.define_autograd_Function()
        # TODO only one get_compiled

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
        #  To define properly new module's forward and backward
        #  functions we need to make it compatible with Autograd.
        #  This method MUST be called to create the forward function.
        # With this the module will be fully compatible with Autograd.

        # Rem 1:
        # Autograd.Function forward function kwargs must be defined,
        #  so we cannot use "**kwargs". Which means to do things
        #  properly we would need to extract original_mod's kwargs
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
        #  Because it only sees the inputs and outputs, and we
        # take care of all the intermediate evaluations, therefore
        # autograd doesn't realize there are some params which
        # require_grad for instance.)

        #  Rem 2:
        # Normally Autograd.Function's backward method returns inputs' grad,
        # and Autograd then backward the inputs using these grads.
        # But since we use a buffer to pass the inputs (cf Rem 1).
        #  Autograd cannot see the inputs and therefore backward them once
        # we finished. So instead of returning inputs' grad we trigger
        # inputs' backward.

        # Rem 3:
        # To isolate our Module range of action we need to detach the inputs
        # before using them so we won't backward through them when handling
        #  the backward of our module. Otherwise the backward operations of the
        # first nodes inside our computation graph will trigger inputs' backward.
        # So we detach the inputs, then we do everything for our part, and once
        # we finished, we trigger inputs' backward (cf Rem 2 -> normally we
        # would have simply returned inputs' grad).

        # Rem 4: TODO REWRITE THIS
        # ALWAYS DETACH IN CASE OF VIEWS TO PROCESS grad_out
        # SO IT'S A DOUBLE DETACH
        # Our goal is to define a custom backward method for the output
        # of the nn.Module, which mean output.backward() will lead to
        # the following lines, where `output` is the output of the forward
        # function of the Checkpointed Module. But by default it's also the
        # output of the last primitive operation inside the module. And we need
        # to be able to backward the last node using its standard backward
        # function. So we must not overwrite last node's output backward
        # function, otherwise last_kcn.backward will call the following lines.
        # SO we need to return a detached copy of the outputs.
        #  Thus, the last node's output backward isn't affected, and
        # we properly redefine CheckpointedModule's output backward.

        RkMod = self

        # -> so we can access to it inside the following class definition
        #  (when defining a Class inside a Class we cannot use `self`)
        class RK_autograd_Function(torch.autograd.Function):
            # === OUR FORWARD FUNCTION ===
            @staticmethod
            def forward(ctx, dummy_input, *args):
                # *** INITIALIZATION PART ***
                #  -> Get the inputs using the buffer (Rem 1)
                dict_inputs = RkMod.dict_inputs_buffer
                RkMod.dict_inputs_buffer = None
                #  -> Create the RK_Storage for this run, and store it in ctx
                ctx.RK_Storage = storage = RK_Storage()
                RkMod.compiler.storage = storage
                #  -> Store what we need to return inputs' grad (Rem 1)
                ctx.name_of_inputs_which_req_grad = (
                    RkMod.name_of_inputs_which_req_grad_buffer
                )
                RkMod.name_of_inputs_which_req_grad_buffer = None
                #  -> Detach input tensors (Rem 3) and store all the inputs
                dict_input_tensors_detach = (
                    dict()
                )  #  dict : input -> detached input
                for k, v in dict_inputs.items():
                    if isinstance(v, torch.Tensor):
                        v_d = v.detach().requires_grad_(v.requires_grad)
                        dict_input_tensors_detach[v] = v_d
                        storage.ld[k] = v_d
                    #  TODO elif iterables of Tensors ?
                    else:
                        storage.ld[k] = v
                #  -> Initialize the storage
                for kg in RkMod.list_kg:
                    for kdn in kg.list_kdn:
                        storage.ld[kdn.main_target] = torch.empty(
                            0,
                            device=RkMod.device,
                            requires_grad=kdn.info.requires_grad,
                        )

                #  *** EXECUTION PART ***
                # -> Autograd turns off itself before giving use the control.
                # -> But we need it to forward/backward each node.
                with torch.enable_grad():
                    exec(
                        RkMod.init_code, RkMod.gd, storage.ld  # is compiler.gd
                    )
                    for l in RkMod.fwd_fct_list:
                        RkMod._exec(l)
                # -> Get the output
                out = RkMod.compiler.get_val(RkMod.rkgb_res.D_graph.outputs[0])
                out_d = out.detach().requires_grad_(out.requires_grad)
                # TODO multiple outputs
                #  -> Clear the compiler
                RkMod.compiler.storage = None
                # -> Remember that out have been detached from the rest during exec
                return out_d
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
            def backward(ctx, grad_out_d):  #  TODO multiple outputs
                #  -> Reload the storage and out
                storage = ctx.RK_Storage
                RkMod.compiler.storage = storage
                # -> Put grad_out in out.grad (Rem 4)
                out = RkMod.compiler.get_val(RkMod.rkgb_res.D_graph.outputs[0])
                out.backward(grad_out_d)  #  -> set out.grad cleanly
                # remember that forward returned out_d not out

                #  * record_mem stuff *
                if RkMod.exec_with_record_mem:
                    RkMod.output_size = irotor.tensorMsize(
                        storage.ld[RkMod.output.main_target]
                    )
                    loss_idx = len(RkMod.allo_mem)
                    # self.allo_mem[-1] += self.output.info.memsize
                    # output grad is generated outside
                # -> exec bwd_fct_list with early stop or not
                stop = RkMod.backward_stop
                if stop:
                    len_fwd = len(RkMod.fwd_fct_list)
                    for l in RkMod.bwd_fct_list[: (stop - len_fwd)]:
                        with torch.enable_grad():
                            RkMod._exec(l)
                else:
                    for l in RkMod.bwd_fct_list:
                        with torch.enable_grad():
                            RkMod._exec(l)
                    if (
                        RkMod.exec_with_record_mem
                        and RkMod.backward_add_output_grad
                    ):
                        RkMod.allo_mem[loss_idx] += RkMod.output_size
                    #  -> return grad of dummy input + inputs' which req grad (Rem 1)
                    grad_inputs = tuple(
                        RkMod.compiler.get_val(inp).grad
                        for inp in ctx.name_of_inputs_which_req_grad
                    )
                    grads = (torch.ones(1),) + grad_inputs
                    #  -> Clear the compiler (and Autograd clears ctx)
                    RkMod.compiler.storage = None
                    return grads

            # === END OF BACKWARD FUNCTION ===

        self.autograd_Function = RK_autograd_Function

    #  === nn.module's forward method wrapping self.autograd_Function.forward ===
    def forward(self, *args, record_mem=False, **kwargs):
        self.exec_with_record_mem = record_mem
        self.max_mem = []
        self.allo_mem = []
        if not self.training:
            self.original_mod.eval()
            return self.original_mod(*args, **kwargs)
        else:
            if self.compiler is None:
                raise Exception(
                    "No schedule compiled, no solution to exec,\n"
                    "To be able to use the RkMod you first need "
                    "to call get_compiled_fct(_HILP)."
                )
            elif self.autograd_Function is None:
                raise Exception(
                    "The custom forward and backward functions haven't "
                    "been generated yet, please call the method : "
                    "define_autograd_Function"
                )
            # -> Send the inputs to Function.forward via the buffer (Rem 1)
            self.dict_inputs_buffer = dict_inputs = make_inputs(
                self.original_mod, args, kwargs
            )
            # -> Pass the inputs which req grad to prepare their backward (Rem 1)
            inputs_which_req_grad = []
            name_of_inputs_which_req_grad = []
            for k, v in dict_inputs.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    inputs_which_req_grad.append(v)
                    name_of_inputs_which_req_grad.append(k)
            self.name_of_inputs_which_req_grad_buffer = (
                name_of_inputs_which_req_grad
            )
            dummy_input = torch.ones(1).requires_grad_()
            return self.autograd_Function.apply(
                dummy_input, *inputs_which_req_grad
            )

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

    def reinit(self, set_to_none=False):
        # In our experiments, we set the parameter grad to 0's
        # so that Backward only creates memory for activations
        self.original_mod.zero_grad(set_to_none=set_to_none)

    # === Inherits original_mod Attributes and Methods ===
    """
    @property
    def original_mod(self):
        return self._HideFromPytorch.mod
    @original_mod.setter
    def original_mod(self,original_mod):
        print("SET ORIGINAL MOD")
        class HideFromPytorch():
            def __init__(self,mod):
                self.mod = mod
        self._HideFromPytorch = HideFromPytorch(original_mod)
    #original_mod = property(get_original_mod,set_original_mod)
    """

    def inherits_original_mod_attributes_and_methods(self):
        for k, v in self.original_mod.__dict__.items():
            if (
                not "forward" in k
                and not "backward" in k
                and k not in ["training"]
            ):
                self.__dict__[k] = v
