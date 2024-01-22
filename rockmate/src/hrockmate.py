# ============
# = ROCKMATE =
# ============
import pickle
import torch
from torch import tensor
import time
from datetime import datetime
import warnings

import rkgb
# from rkgb.main import make_inputs, make_all_graphs, make_late_partitioning
# from rkgb.utils import print_debug, np, irotor
# from rkgb.utils.global_vars import (
#     ref_verbose,
#     solver_name,
#     ExceptionModuleDoesNotReqGrad,
# )
# from rkgb.utils.small_fcts import get_device
# from rkgb.utils.ast_add_on import ast_to_str
# from rkgb import Ptools

# TODO: make_all_graphs, make_late_partitioning

from rkgb.lowlevel.preprocess_samples import ExampleInputs
from rkgb.lowlevel.measure import tensor_memory_size
from rkgb.lowlevel.constants import ref_verbose, ExceptionModuleDoesNotReqGrad
from rkgb.lowlevel.ast_add_on import ast_to_str, make_str_list_assign
from rkgb.core.partitioned import PartitionerBottomToTop, PartitionerSequence, Partitioner

from .solvers.main import preprocess, solve_recursive, get_optimize_stats
from .solvers.op_schedule import *
# from .solvers import RK_rotor, HILP, TwRemat, RK_checkmate
from .solvers import HILP
from .solvers.hilp import default_time_limit
# from .solvers.HILP_gurobi import *
from .compiler import Compiler, RK_Storage, make_gd


class HRockmate(torch.nn.Module):
    compiler = None
    autograd_Function = None
    backward_stop = False
    backward_add_output_grad = True
    op_sched = None
    module_does_not_req_grad = False

    def __init__(
        self,
        original_mod,
        model_inputs,
        budget=None,
        list_solvers=[],
        rkgb_res=None,
        solve_sched=True,
        verbose=False,
        ilp_solver="gurobi",
        ilp_time_limit=60 * 60,
        model_kwargs=None,
        partitioners=None,
        max_size_S_graph_for_no_partitioning=40,
        cpu_optim = torch.optim.Adam,
        gpu_optim = torch.optim.Adam,
        optim_kwargs = {},
        minor_param_size = 10*1024,
        # [
        #    Partitioner,
        #    PartitionerBottomToTop(),
        #    PartitionerSequence(),
        # ],
    ):
        super().__init__()
        ref_verbose[0] = verbose
        # solver_name[0] = ilp_solver
        default_time_limit[0] = ilp_time_limit
        list_solvers = list_solvers or [HILP(ilp_solver=ilp_solver)]

        self.device = torch.device("cuda")# Not obtaining from model
        object.__setattr__(self, "original_mod", original_mod)
        self.exec_with_record_mem = False
        # dict_inputs = ExampleInputs(original_mod, model_inputs, model_kwargs)
        self.named_para_shape = dict()
        for n, p in original_mod.named_parameters():
            self.named_para_shape[n] = p.shape
        if partitioners is None:
            partitioners = []
            can_use_rotor = False
            can_use_checkmate = False
            # for solver in list_solvers:
            #     if isinstance(solver, RK_rotor):
            #         can_use_rotor = True
            #     elif isinstance(solver, RK_checkmate):
            #         can_use_checkmate = True
            partitioners = [
                PartitionerBottomToTop(can_use_rotor=can_use_rotor)
            ]
            if can_use_rotor:
                partitioners.append(PartitionerSequence())
            if can_use_checkmate:
                partitioners.append(Partitioner)

        # ensure HILP config match partitioner config
        for partitioner in partitioners:
            if isinstance(partitioner, PartitionerBottomToTop):
                for solver in list_solvers:
                    if isinstance(solver, HILP):
                        solver.config.nb_total_nodes = max(
                            solver.config.nb_total_nodes,
                            partitioner.config.max_estimate_per_sub_graph,
                        )
                        solver.config.nb_total_nodes_top_level = max(
                            solver.config.nb_total_nodes_top_level,
                            partitioner.config.max_estimate_for_main_graph,
                        )
        #  We don't want to use the default setattr
        # because torch.nn.Module will register it as a submodule
        # -- use gkGB --
        try:
            if rkgb_res is None:
                # self.rkgb_res = make_all_graphs(
                #     original_mod,
                #     dict_inputs,
                #     verbose=verbose,
                #     wanted_graphs={"K"},
                #     partitioners=partitioners,
                #     check_device_is_gpu=False
                # )
                self.rkgb_res = rkgb.rkgb.Result(
                    original_mod,
                    # dict_inputs,
                    model_args=model_inputs,
                    model_kwargs=model_kwargs,
                    # verbose=verbose,
                    wanted_graphs={"FB"},
                    partitioners=partitioners,
                    inspection_device=torch.device("cuda")
                    # check_device_is_gpu=False
                )
            else:
                self.rkgb_res = rkgb_res
            if len(self.rkgb_res.simplified_graph.nodes) <= max_size_S_graph_for_no_partitioning:
                # -> No partitioning !
                self.rkgb_res.build_hierarchical(partitioners)
                list_solvers = [
                    HILP(
                        HILP.Config(
                            nb_total_nodes_top_level=max_size_S_graph_for_no_partitioning
                            + 1
                        )
                    )
                ]
            else:
                self.rkgb_res.build_hierarchical(partitioners)
            self.partitioners = partitioners
            self.list_solvers = list_solvers
            self.dict_constants = self.rkgb_res.forward_and_backward_graph.dict_constants
            self.init_code = ast_to_str(self.rkgb_res.forward_and_backward_graph.init_code)
            self.dict_output_viewing_code = dict(
                (out_mt, ast_to_str(view_code))
                for (
                    out_mt,
                    view_code,
                ) in self.rkgb_res.forward_and_backward_graph.dict_output_viewing_code.items()
            )
            # self.outputs_wrapping_code = ast_to_str(
            #     self.rkgb_res.K_graph.outputs_wrapping_code
            # )
            self.output = self.rkgb_res.forward_and_backward_graph.list_output_data_anodes[0]
            self.budget = budget
            p = list(original_mod.parameters())[0]
            optimize_stats = get_optimize_stats(p, 
                                                cpu_optim=cpu_optim, 
                                                gpu_optim=gpu_optim,
                                                optim_kwargs=optim_kwargs)
            optimize_stats["minor_param_size"] = minor_param_size
            self.minor_param_nodes = []
            for pnode in self.rkgb_res.hierarchical_cluster.parameter_nodes:
                if pnode.mem < minor_param_size and not pnode.is_buffer:
                    self.minor_param_nodes.append(pnode)
            self.gd = make_gd(
                self.device, 
                self.original_mod, 
                self.dict_constants,
                cpu_optim, 
                gpu_optim,
                optim_kwargs=optim_kwargs,
                optimize_stats = optimize_stats
            )
            self.list_solutions = []
            if solve_sched:
                self.solve_sched()
        except ExceptionModuleDoesNotReqGrad:
            self.module_does_not_req_grad = True

    def preprocess(self):
        for cluster in self.rkgb_res.hierarchical_structure.all_clusters:
            if not cluster.is_bottom:
                preprocess(
                    cluster,
                    protect_names=[
                        "sources data",
                        "sources grad",
                        self.output.name,
                    ],
                )

    def solver_recursive(self, list_solvers=None, only_preprocess=False):
        list_solvers = list_solvers or self.list_solvers
        self.preprocess()

        solve_recursive(
            self.rkgb_res.hierarchical_cluster, list_solvers=list_solvers, skip_self=True
        )

    def solve_sched(self, budget=None, list_solvers=None, rec=True):
        # budget should be a single value.
        # if multiple solvers in list_solvers,
        # will choose the one with minimum time
        budget = budget or self.budget
        list_solvers = list_solvers or self.list_solvers
        rotor_solver = False
        hilp_solver = False
        checkmate_solver = False
        for solver in list_solvers:
            if isinstance(solver, HILP):
                hilp_solver = True
            # if isinstance(solver, RK_rotor):
            #     rotor_solver = True
            # if isinstance(solver, RK_checkmate):
            #     checkmate_solver = True

        for solver in list_solvers:
            if isinstance(solver, HILP):
                solver.config.protected_names.append(self.output.name)
                if rotor_solver:
                    solver.config.nb_bdg_save = 10
                    solver.config.nb_bdg_peak = 10

        if (
            True
            in [
                isinstance(solver, HILP)# or isinstance(solver, RK_rotor)
                for solver in list_solvers
            ]
            and rec
        ):
            self.solver_recursive()
        # elif (
        #     True in [isinstance(solver, RK_checkmate) for solver in list_solvers]
        #     and rec
        # ):
        #     self.preprocess()

        list_solutions = []
        for solver in list_solvers:
            if isinstance(solver, HILP):
                solver.config.solve_top_level = True
                solver.config.cpu_optimize_kwargs = self.gd["optimize_stats"]
                # print("temporarily changing total_nodes for top level hilp")
                list_solutions.extend(
                    solver(self.rkgb_res.hierarchical_cluster, [budget], accurate_mem=True)
                )
                solver.config.solve_top_level = False  # in case further usage

            else:
                list_solutions.extend(solver(self.rkgb_res.hierarchical_cluster, [budget]))

        self.rkgb_res.hierarchical_cluster.list_schedules.extend(list_solutions)
        if not list_solutions:
            warnings.warn("no feasible schedule is found")
        else:
            # print([sum(op_sched.time) for op_sched in list_solutions])
            # print(len(list_solutions))
            self.op_sched = list_solutions[
                np.argmin([sum(op_sched.time) for op_sched in list_solutions])
            ]
            self.get_compiled_fct()
        self.list_solutions.extend(list_solutions)

    def get_compiled_fct(self, new_compiler=True):
        if new_compiler:
            self.compiler = Compiler(self.gd)
        self.fct_list, self.init_fct_list, self.restore_fct_list = self.compiler.compile_from_schedule(self.op_sched)
        loss_idx = self.op_sched.loss_idx

        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]
        l = [self.compiler.fct_del_var(v) for v in self.output.tensor_targets]
        self.bwd_fct_list.append(l)
        self.minor_parameters = []
        # for n, p in self.original_mod.named_parameters():
        #     if n in self.minor_param_nodes.append(pnode)
        #     # if n not in [kdn.param_name for kdn in self.rkgb_res.hierarchical_cluster.parameter_nodes]:
        #         self.minor_parameters.append(p)
                # p.data = p.data.to("cuda")
        
        if self.minor_param_nodes:
            self.minor_parameters = [self.original_mod.get_parameter(pnode.param_name) 
                                for pnode in self.minor_param_nodes]
            def optimize():
                self.compiler.storage.ld["optimizers"]["minors"].step()
                for p in self.minor_parameters:p.grad=None
            self.bwd_fct_list.append([optimize])
        
        self.define_autograd_Function()
        self.inherits_original_mod_attributes_and_methods()

    def _exec(self, fct_list):
        if self.exec_with_record_mem:
            torch.cuda.reset_peak_memory_stats()
            self.mem_before = torch.cuda.memory_allocated()
            self.max_before = torch.cuda.max_memory_allocated()
            for fct in fct_list:
                fct()
            torch.cuda.synchronize()
            allo_mem = torch.cuda.memory_allocated() - self.mem_before
            peak_mem = torch.cuda.max_memory_allocated() - self.max_before
            self.max_mem.append(peak_mem - allo_mem)
            self.allo_mem.append(allo_mem)
        else:
            for fct in fct_list:
                fct()

    def init_fwd_exec(self):
        #  -> Initialize the storage
        storage = self.compiler.storage
        for kdn in self.rkgb_res.forward_and_backward_graph.allocation_nodes:
            storage.ld[kdn.main_target] = torch.empty(
                0,
                device=self.device,
                requires_grad=kdn.info.requires_grad,
            )

        for out_node in self.rkgb_res.forward_graph.output_nodes:
            storage.ld[f"out_{out_node.main_target}"] = torch.empty(
                0,
                device=self.device,
                requires_grad=out_node.info.requires_grad,
            )
            

        for op in self.op_sched.init_op_list:
            if isinstance(op, MappingOp) and len(op.targets)==1:
                # to create the full size buffer
                target = op.targets[0]
                storage.ld["cpu_"+target.name.strip("cpu_")] = torch.empty(target.size, 
                                                    dtype=target.dtype, 
                                                    device=torch.device("cpu"),
                                                    pin_memory=True)
                
        for pnode in self.minor_param_nodes:
            for t in pnode.view_targets:
                storage.ld[t] = torch.empty(0, requires_grad=pnode.requires_grad)
            target = pnode.get_value(self.original_mod)
            target.data = target.data.to("cuda")
            storage.ld[pnode.param_name] = target
            code = make_str_list_assign(pnode.view_code, suffix=".data")
            exec(code, self.gd, storage.ld)

        for pnode in self.rkgb_res.hierarchical_cluster.parameter_nodes:
            if pnode.is_buffer:
                for t in pnode.view_targets:
                    storage.ld[t] = torch.empty(0, requires_grad=False)
                target = pnode.get_value(self.original_mod)
                target.data = target.data.to("cuda")
                storage.ld[pnode.param_name] = target
                code = make_str_list_assign(pnode.view_code, suffix=".data")
                exec(code, self.gd, storage.ld)
                # print(target.device, pnode.get_code())
        for k,v in self.op_sched.dict_alloc_param.items():
            if v.grad:continue
            # target = self.gd["self"].get_parameter(k.removesuffix(" parameter"))
            target = v.pnode.get_value(self.original_mod)
            target.grad = None
            storage.ld["cpu_"+k] = torch.empty_like(target, 
                                                dtype=target.dtype, 
                                                device=torch.device("cpu"),
                                                pin_memory=True)
            storage.ld["cpu_"+k].copy_(target.data)
            storage.ld["cpu_"+k].grad = torch.empty_like(storage.ld["cpu_"+k], pin_memory=True)
            # if v.pnode.is_buffer:
            #     storage.ld[k] = target.to("cuda")
            # else:
            storage.ld[k] = target

        # for k, v in self.op_sched.dict_alloc.items():
        #     if isinstance(v, Activation):
        #         continue
                
        #     if isinstance(v, Parameter):
        #         if v.grad:continue
        #         target = self.gd["self"].get_parameter(k.removesuffix(" parameter"))
        #         storage.ld["cpu_"+k] = torch.empty_like(target, 
        #                                             dtype=target.dtype, 
        #                                             device=torch.device("cpu"),
        #                                             pin_memory=True)
        #         storage.ld["cpu_"+k].copy_(self.gd["self"].get_parameter(k.removesuffix(" parameter")).data)
        #         storage.ld["cpu_"+k].grad = torch.empty_like(storage.ld["cpu_"+k], pin_memory=True)
        #         storage.ld[k] = self.gd["self"].get_parameter(k.removesuffix(" parameter"))
                
        #     elif isinstance(v, Buffer):
        #         storage.ld[k] = torch.empty(0, device=self.gd["device"])
                
        #     else:
        #         print(f"Unrecognized type {type(v)}")
        
        storage.ld["optimizers"] = {}
        for op in self.op_sched.op_list:
            if isinstance(op, OptimizeOp):
                optim = self.gd["cpu_optim"] if "cpu" in op.name else self.gd["gpu_optim"]
                storage.ld["optimizers"][op.name] = optim([storage.ld[p] for p in op.list_params], **self.gd["opt_kwargs"])

        if self.minor_parameters:
            storage.ld["optimizers"]["minors"] = self.gd["gpu_optim"](self.minor_parameters, **self.gd["opt_kwargs"])
        

        for l in self.init_fct_list:
            self._exec(l)
        torch.cuda.synchronize()
        with torch.enable_grad():
            exec(self.init_code, self.gd, storage.ld)  # is compiler.gd
            for l,op in zip(self.fwd_fct_list, self.op_sched.op_list[:self.op_sched.loss_idx+1]):
                if isinstance(op, OffloadOp) and op.grad:continue#first iteration without grad offload
                if isinstance(op, OptimizeOp):continue#first iteration no need to optimize
                self._exec(l)
        
    def restore_exec(self, keep_grad=False):
        self.zero_grad()
        for l in self.restore_fct_list:
            self._exec(l)
        for p in self.minor_parameters:
            p.data = p.data.to("cpu")

        for k,v in self.op_sched.dict_alloc_param.items():
            if v.grad:continue
            # target = self.gd["self"].get_parameter(k.removesuffix(" parameter"))
            target = v.pnode.get_valule()
            target.data = self.compiler.storage.ld["cpu_"+k].data
            if keep_grad:
                target.grad = self.compiler.storage.ld["cpu_"+k].grad

        self.compiler.storage = None
        torch.cuda.synchronize()

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
        # requires_grad for instance.)

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
        # we properly redefine HRockmate's output backward.

        RkMod = self

        # -> so we can access to it inside the following class definition
        #  (when defining a Class inside a Class we cannot use `self`)
        class RK_autograd_Function(torch.autograd.Function):
            # === OUR FORWARD FUNCTION ===
            @staticmethod
            def forward(ctx, dummy_input, *args):
                if RkMod.compiler.storage is not None:
                    ctx.RK_Storage = storage = RkMod.compiler.storage
                    ctx.name_of_inputs_which_req_grad = (
                        RkMod.name_of_inputs_which_req_grad_buffer
                    )
                    with torch.enable_grad():
                        exec(RkMod.init_code, RkMod.gd, storage.ld)  # is compiler.gd
                        for l in RkMod.fwd_fct_list:
                            RkMod._exec(l)
                else:
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
                    # RkMod.name_of_inputs_which_req_grad_buffer = None
                    #  -> Detach input tensors (Rem 3) and store all the inputs
                    dict_input_tensors_detach = dict()  #  dict : input -> detached input
                    for k, v in dict_inputs.items():
                        if isinstance(v, torch.Tensor):
                            v_d = v.detach().requires_grad_(v.requires_grad)
                            dict_input_tensors_detach[v] = v_d
                            storage.ld[k] = v_d
                        #  TODO elif iterables of Tensors ?
                        else:
                            storage.ld[k] = v
                    

                    torch.cuda.synchronize()
                    with torch.enable_grad():
                        self.init_fwd_exec()
                    torch.cuda.synchronize()

                #  *** EXECUTION PART ***
                # -> Autograd turns off itself before giving use the control.
                # -> But we need it to forward/backward each node.
                
                # -> Get the output
                # outs = [
                #     RkMod.compiler.get_val(out_node.main_target).detach().requires_grad_()
                #     for out_node in RkMod.rkgb_res.forward_and_backward_graph.list_output_data_anodes
                # ]

                outs = []
                for out_node in self.rkgb_res.forward_graph.output_nodes:
                    # print(kdn)
                    RkMod.compiler.get_val(f"out_{out_node.main_target}").data = RkMod.compiler.get_val(out_node.main_target)
                    o = RkMod.compiler.get_val(f"out_{out_node.main_target}")#.detach().requires_grad_()
                    # print(o.grad_fn)
                    outs.append(o)

                if len(outs) == 1:
                    return outs[0]
                else:
                    return tuple(outs)
                # -> Remember that out have been detached from the rest during exec
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
                # print(grad_outs)
                # for out_mt, out_grad in zip(RkMod.rkgb_res.S_graph.outputs, grad_outs):
                for out_node, out_grad in zip(
                    self.rkgb_res.forward_and_backward_graph.list_output_data_anodes,
                    grad_outs):
                    out_mt = out_node.main_target
                    out = RkMod.compiler.get_val(out_mt)
                    # out.grad = out_grad.view(out.shape)
                    out.grad = out_grad.data.as_strided_(
                                out.shape, out.stride(), out.storage_offset()
                            )
                    out_grad.data = torch.empty(0)

                #  * record_mem stuff *
                if RkMod.exec_with_record_mem:
                    RkMod.output_size = tensor_memory_size(
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
                    if RkMod.exec_with_record_mem and RkMod.backward_add_output_grad:
                        RkMod.allo_mem[loss_idx] += RkMod.output_size
                    #  -> return grad of dummy input + inputs' which req grad (Rem 1)
                    grad_inputs = tuple(
                        RkMod.compiler.get_val(inp).grad
                        for inp in ctx.name_of_inputs_which_req_grad
                    )
                    grads = (torch.ones(1),) + grad_inputs
                    #  -> Clear the compiler (and Autograd clears ctx)
                    # RkMod.compiler.storage = None
                    for out_node in self.rkgb_res.forward_graph.output_nodes:
                        # print(kdn)
                        RkMod.compiler.get_val(f"out_{out_node.main_target}").data = torch.empty(0)
                        
                    return grads

            # === END OF BACKWARD FUNCTION ===

        self.autograd_Function = RK_autograd_Function

    #  === nn.module's forward method wrapping self.autograd_Function.forward ===
    def forward(self, *args, record_mem=False, **kwargs):
        if self.module_does_not_req_grad:
            return self.original_mod(*args, **kwargs)
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
            # self.dict_inputs_buffer = dict_inputs = ExampleInputs(
            #     self.original_mod, args, kwargs
            # )
            self.inputs_buffer = ExampleInputs(
                self.original_mod, args, kwargs
            )
            self.dict_inputs_buffer = dict_inputs = self.inputs_buffer.dict
            # -> Pass the inputs which req grad to prepare their backward (Rem 1)
            inputs_which_req_grad = []
            name_of_inputs_which_req_grad = []
            for k, v in dict_inputs.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    inputs_which_req_grad.append(v)
                    name_of_inputs_which_req_grad.append(k)
            self.name_of_inputs_which_req_grad_buffer = name_of_inputs_which_req_grad
            dummy_input = torch.ones(1).requires_grad_()
            output_mt_values = self.autograd_Function.apply(
                dummy_input, *inputs_which_req_grad
            )
            # for out_mt, out_mt_value in zip(
            #     self.rkgb_res.S_graph.outputs, output_mt_values
            # ):
            for out_node in self.rkgb_res.forward_and_backward_graph.list_output_data_anodes:
                out_mt = out_node.main_target
                view_code = self.dict_output_viewing_code[out_mt]
                exec(view_code, self.gd, self.compiler.storage.ld)
                # -> We access to out_mt_value directly in the storage
            # exec(self.outputs_wrapping_code, self.gd, self.compiler.storage.ld)
            output_nodes = self.rkgb_res.forward_graph.output_nodes
            # final_output = self.compiler.get_val(
            #     self.rkgb_res.D_graph.whole_module_output
            # )
            final_outputs = [self.compiler.get_val(f"out_{output.main_target}") for output in output_nodes]
            #  -> Clear the compiler
            # self.compiler.storage = None
            return tuple(final_outputs)

    # === end of forward ===

    def expect_time(self):
        # Sum of the measured time of each operation for one batch
        return self.fwd_seq.compute_time() + self.bwd_seq.compute_time()
    
    def expect_md_result(self):
        md = self.list_solvers[0].md
        if md:
            expect_time = md.md.objective.value()/sum(t[0] for t in md.time)
            expect_mem = md.peak_budget
            return print(f"Expect overhead: {(expect_time-1):.2%}, mem_limit: {(expect_mem/1024**2):.2f}MB")

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

    def zero_grad(self, set_to_none=True, remain_for_offload=True):
        # if self.compiler.storage:
            # if set_to_none:
            #     for k,v in self.compiler.storage.ld.items():
            #         if "cpu_" in k:
            #             v.grad = None
            # else:
            # for k,v in self.compiler.storage.ld.items():
            #     if "cpu_" in k:
            #         v.grad = torch.zeros_like(v)
        if remain_for_offload:
            remains = []
            for k,v in self.op_sched.dict_alloc_param.items():
                if v.grad and self.op_sched.alive_list[-1][k]:
                    remains.append(v.pnode.param_name)
            for k,p in self.original_mod.named_parameters():
                if k not in remains:
                    p.grad = None if set_to_none else torch.zeros_like(p)
        else:
            self.original_mod.zero_grad(set_to_none=set_to_none)

    def reinit(self, set_to_none=False):
        # In our experiments, we set the parameter grad to 0's
        # so that Backward only creates memory for activations
        self.original_mod.zero_grad(set_to_none=set_to_none)

    def print_sched_results(self):
        t = sum(kcn.time for kcn in self.rkgb_res.hierarchical_cluster.list_cnodes if kcn.time)
        print(f"Original module iter time {t}")
        t = sum(step.time for step in self.op_sched.steps)
        print(f"Schedule: total time {t}")
        t = sum(step.comp_ops.time for step in self.op_sched.steps)
        print(f"Schedule: total compute time {t}")
        t = sum(step.ofl_ops.time for step in self.op_sched.steps)
        print(f"Schedule: total offload time {t}")
        t = sum(step.prf_ops.time for step in self.op_sched.steps)
        print(f"Schedule: total prefetch time {t}")
        t = sum(step.opt_ops.time for step in self.op_sched.steps)
        print(f"Schedule: total cpu optimize time {t}")
        t = sum((step.time - step.comp_ops.time) for step in self.op_sched.steps if step.time == step.opt_ops.time)
        print(f"Schedule: time from waiting cpu optimize {t}")
        t = sum((step.time - step.max2nd()) for step in self.op_sched.steps if step.time == step.ofl_ops.time)
        print(f"Schedule: time from waiting offload {t}")
        t = sum((step.time - step.max2nd()) for step in self.op_sched.steps if step.time == step.prf_ops.time)
        print(f"Schedule: time from waiting prefetch {t}")


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
            if not "forward" in k and not "backward" in k and k not in ["training"]:
                self.__dict__[k] = v

    def save_to_local(self, path, id=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")):
        with open(f"{path}/{id}_rkgb_res.pkl", "wb") as f:
            pickle.dump(self.rkgb_res, f)

    def save_sched_to_local(
        self, path, id=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ):
        with open(f"{path}/{id}_sched.pkl", "wb") as f:
            pickle.dump(self.op_sched, f)

    def load_from_local(self, path, id, load_sched=True):
        with open(f"{path}/{id}_rkgb_res.pkl", "rb") as f:
            self.rkgb_res = pickle.load(f)
            if load_sched:
                self.list_solutions = self.rkgb_res.hierarchical_cluster.list_schedules
                try:
                    with open(f"{path}/{id}_sched.pkl", "rb") as f_sched:
                        self.op_sched = pickle.load(f_sched)
                except Exception:
                    self.op_sched = self.list_solutions[
                        np.argmin(
                            [sum(op_sched.time) for op_sched in self.list_solutions]
                        )
                    ]
                self.get_compiled_fct()
