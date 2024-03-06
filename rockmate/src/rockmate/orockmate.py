# ============
# = ROCKMATE =
# ============
import pickle
import torch
from torch import tensor
import time
from datetime import datetime
import warnings
import gc
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
from rkgb.lowlevel.constants import init_target_string

from .op_schedule import *
from .solvers.main import preprocess, solve_recursive, get_optimize_metrics
# from .solvers import RK_rotor, HILP, TwRemat, RK_checkmate
from .solvers import HILP
from .solvers.hilp import default_time_limit
# from .solvers.HILP_gurobi import *
from .compiler import Compiler, RK_Storage, make_gd
import psutil

class ORockmate(torch.nn.Module):
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
        ilp_time_limit=1 * 60,
        ilp_time_limit_top=10 * 60,
        model_kwargs=None,
        partitioners=None,
        max_size_S_graph_for_no_partitioning=40,
        cpu_optim = torch.optim.Adam,
        gpu_optim = torch.optim.Adam,
        optim_kwargs = {},
        minor_param_size = 10*1024,
    ):
        super().__init__()
        ref_verbose[0] = verbose
        default_time_limit[0] = ilp_time_limit
        self.ilp_time_limit_top = ilp_time_limit_top
        list_solvers = list_solvers or [HILP(ilp_solver=ilp_solver)]
        
        self.partitioners = partitioners
        self.list_solvers = list_solvers
        self.device = torch.device("cuda")# Not obtaining from model
        self.exec_with_record_mem = False
        self.budget = budget
        self.list_solutions = []
        object.__setattr__(self, "original_mod", original_mod)
        self.config_partitioner()
        
        # -- use rkGB --
        try:
            if rkgb_res is None:
                self.rkgb_res = rkgb.rkgb.Result(
                    original_mod,
                    # dict_inputs,
                    model_args=model_inputs,
                    model_kwargs=model_kwargs,
                    # verbose=verbose,
                    wanted_graphs={"FB"},
                    partitioners=partitioners,
                    inspection_device=torch.device("cuda"),
                    print_time_in_each_stage=True
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
                        ),
                        ilp_solver=ilp_solver
                    )
                ]
            else:
                self.rkgb_res.build_hierarchical(partitioners)
        except ExceptionModuleDoesNotReqGrad:
            self.module_does_not_req_grad = True
    
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
        p = list(original_mod.parameters())[0]
        optimize_metrics = get_optimize_metrics(p, 
                                            cpu_optim=cpu_optim, 
                                            gpu_optim=gpu_optim,
                                            optim_kwargs=optim_kwargs)
        optimize_metrics["minor_param_size"] = minor_param_size

        self.minor_param_nodes = []
        for pnode in self.rkgb_res.hierarchical_cluster.parameter_nodes:
            if pnode.mem < minor_param_size and not pnode.is_buffer:
                self.minor_param_nodes.append(pnode)
        self.minor_param_nodes += self.rkgb_res.S.init_node.required_parameter_nodes
        self.gd = make_gd(
            self.device, 
            self.original_mod, 
            self.dict_constants,
            cpu_optim, 
            gpu_optim,
            optim_kwargs=optim_kwargs,
            optimize_metrics = optimize_metrics
        )

        if solve_sched:
            self.solve_sched()
            self.get_compiled_fct()
    
    def config_partitioner(self):
        if self.partitioners is None:
            self.partitioners = [
                PartitionerBottomToTop(can_use_rotor=False)
            ]
        # ensure HILP config match partitioner config
        for partitioner in self.partitioners:
            if isinstance(partitioner, PartitionerBottomToTop):
                for solver in self.list_solvers:
                    if isinstance(solver, HILP):
                        solver.config.nb_total_nodes = max(
                            solver.config.nb_total_nodes,
                            partitioner.config.max_estimate_per_sub_graph,
                        )
                        solver.config.nb_total_nodes_top_level = max(
                            solver.config.nb_total_nodes_top_level,
                            partitioner.config.max_estimate_for_main_graph,
                        )
        

    def preprocess(self):
        for cluster in self.rkgb_res.hierarchical_structure.all_clusters:
            if not cluster.is_bottom:
                preprocess(
                    cluster,
                    protect_names=[
                        f"{init_target_string} data", 
                        f"{init_target_string} grad",
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
        budget -= self.minor_size
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
                solver.config.time_limit_top = self.ilp_time_limit_top
                solver.config.cpu_optimize_kwargs = self.gd["optimize_metrics"]
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
            self.op_sched = list_solutions[
                np.argmin([sum(op_sched.time) for op_sched in list_solutions])
            ]
            self.op_list = self.op_sched.op_list
        self.list_solutions.extend(list_solutions)

    def get_compiled_fct(self, new_compiler=True):
        if new_compiler:
            storage = RK_Storage()
            storage.init(self.gd)
            self.compiler = Compiler(storage)
        self.compiler.compile_sched(self.op_sched)
        self.minor_parameters = []
        
        if self.minor_param_nodes:
            self.minor_parameters = [self.original_mod.get_parameter(pnode.param_name) 
                                for pnode in self.minor_param_nodes]
            def optimize():
                self.compiler.storage.ld["optimizer_minors"].step()
                for p in self.minor_parameters:p.grad=None
                for pnode in self.minor_param_nodes:
                    code = make_str_list_assign(pnode.view_code, suffix=".data")
                    exec(code, self.gd, self.compiler.storage.ld)
            self.op_list[-1].add_fct(optimize)
        
        self.autograd_Function = define_autograd_Function(self)
        self.inherits_original_mod_attributes_and_methods()
        self.compiler.compile_preparation(self.rkgb_res.hierarchical_cluster,
                                          self.op_sched,
                                          self.minor_param_nodes,
                                        #   self.rkgb_res.forward_graph.output_nodes
                                          )

    def _exec(self, op:Op):
        try:
            if self.exec_with_record_mem:
                torch.cuda.reset_peak_memory_stats()
                self.mem_before = torch.cuda.memory_allocated()
                self.max_before = torch.cuda.max_memory_allocated()
                op()
                torch.cuda.synchronize()
                allo_mem = torch.cuda.memory_allocated() - self.mem_before
                peak_mem = torch.cuda.max_memory_allocated() - self.max_before
                self.max_mem.append(peak_mem - allo_mem)
                self.allo_mem.append(allo_mem)
            else:
                op()
        except Exception as e:
            print(f"Failed to execute {op}")
            raise e
            
    def init_fwd_exec(self):
        for op in self.op_sched.init_op_list:
            self._exec(op)
        torch.cuda.synchronize()
        with torch.enable_grad():
            exec(self.init_code, self.gd, self.compiler.storage.ld)  # is compiler.gd
            for op in self.op_list[:self.op_sched.loss_idx+1]:
                if isinstance(op, OffloadOp) and isinstance(op.target, Parameter):
                    if op.target.is_grad or op.target.is_optim_states:continue#first iteration without grad offload
                if isinstance(op, OptimizeOp):continue#first iteration no need to optimize
                self._exec(op)
                torch.cuda.synchronize()
        
    def restore_exec(self, keep_grad=False):
        if self.compiler.storage.ld == {}:return None
        self.zero_grad()
        for l in self.restore_fct_list:
            self._exec(l)
        for p in self.minor_parameters:
            p.data = p.data.to("cpu")

        for k,v in self.op_sched.dict_alloc_param.items():
            if v.pnode.is_buffer:
                target = v.pnode.get_value(self.original_mod)
                target.data = target.data.to("cpu")
                continue

            if v.pnode in self.minor_param_nodes:continue
            if v.grad:continue
            # target = self.gd["self"].get_parameter(k.removesuffix(" parameter"))
            target = v.pnode.get_value(self.original_mod)
            target.data = self.compiler.storage.ld[v.target_name].data
            if keep_grad:
                target.grad = self.compiler.storage.ld[v.target_name].grad

        self.compiler.storage.ld = {}
        torch.cuda.synchronize()

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

            # To output what's defined in module
            output_nodes = self.rkgb_res.forward_graph.output_nodes
            final_outputs = [self.compiler.get_val(f"{output.main_target}") for output in output_nodes]
            return tuple(final_outputs)

    # === end of forward ===

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
                if v.is_grad and self.op_sched.alive_list[-1][k]:
                # if v.is_grad and self.op_sched.init_alive_status[k]:
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
        t = sum(cnode.time for cnode in self.rkgb_res.hierarchical_cluster.list_cnodes if cnode.time)
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

    @property
    def minor_size(self):
        return (sum([pnode.mem for pnode in self.minor_param_nodes]) * 
                   (self.gd["optimize_metrics"]["optimizer_states_size"]+1))

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


def define_autograd_Function(RkMod):
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
    # function, otherwise last_cnode.backward will call the following lines.
    # SO we need to return a detached copy of the outputs.
    #  Thus, the last node's output backward isn't affected, and
    # we properly redefine HRockmate's output backward.

    class RK_autograd_Function(torch.autograd.Function):
        # === OUR FORWARD FUNCTION ===
        @staticmethod
        def forward(ctx, dummy_input, *args):
            if RkMod.compiler.storage.ld != {}:
                ctx.RK_Storage = storage = RkMod.compiler.storage
                ctx.name_of_inputs_which_req_grad = (
                    RkMod.name_of_inputs_which_req_grad_buffer
                )
                with torch.enable_grad():
                    exec(RkMod.init_code, RkMod.gd, storage.ld)  # is compiler.gd
                    for op in RkMod.op_list[:RkMod.op_sched.loss_idx]:
                        RkMod._exec(op)
            else:
                # *** INITIALIZATION PART ***
                #  -> Get the inputs using the buffer (Rem 1)
                dict_inputs = RkMod.dict_inputs_buffer
                RkMod.dict_inputs_buffer = None
                
                #  -> Create the RK_Storage for this run, and store it in ctx
                ctx.RK_Storage = storage = RK_Storage()
                storage.init(RkMod.gd)
                RkMod.compiler.storage = storage
                
                #  -> Store what we need to return inputs' grad (Rem 1)
                ctx.name_of_inputs_which_req_grad = (
                    RkMod.name_of_inputs_which_req_grad_buffer
                )

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
                    RkMod.init_fwd_exec()
                torch.cuda.synchronize()

            #  *** EXECUTION PART ***
            # -> Autograd turns off itself before giving use the control.
            # -> But we need it to forward/backward each node.
            
            # -> Get the output
            outs = []
            for out_node in RkMod.rkgb_res.forward_and_backward_graph.list_output_data_anodes:
                out_mt = out_node.main_target
                if out_mt == RkMod.rkgb_res.forward_graph.nodes[-1].main_target:
                    # Last output node; its grad_fn will not be called again
                    outs.append(RkMod.compiler.get_val(out_mt))
                    continue
                RkMod.compiler.get_val(f"out_{out_mt}").data = RkMod.compiler.get_val(out_mt)
                o = RkMod.compiler.get_val(f"out_{out_mt}")
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
            for out_node, out_grad in zip(
                RkMod.rkgb_res.forward_and_backward_graph.list_output_data_anodes,
                grad_outs):
                out = RkMod.compiler.get_val(out_node.main_target)
                out.grad = out_grad.data
                
                out_grad.data = torch.empty(0)
            for out_node in RkMod.rkgb_res.forward_and_backward_graph.list_output_data_anodes:
                    RkMod.compiler.get_val(f"out_{out_node.main_target}").data = torch.empty(0)
                
            loss_idx = RkMod.op_sched.loss_idx
            #  * record_mem stuff *
            if RkMod.exec_with_record_mem:
                RkMod.output_size = tensor_memory_size(
                    storage.ld[RkMod.output.main_target]
                )
                # TODO: record output grad if needed

            stop = RkMod.backward_stop
            if stop:
                for op in RkMod.op_list[loss_idx+1:loss_idx+stop+1]:
                    with torch.enable_grad():
                        RkMod._exec(op)
            else:
                for op in RkMod.op_list[loss_idx+1:]:
                    RkMod._exec(op)
                if RkMod.exec_with_record_mem and RkMod.backward_add_output_grad:
                    RkMod.allo_mem[loss_idx] += RkMod.output_size
                #  -> return grad of dummy input + inputs' which req grad (Rem 1)
                grad_inputs = tuple(
                    RkMod.compiler.get_val(inp).grad
                    for inp in ctx.name_of_inputs_which_req_grad
                )
                grads = (torch.ones(1),) + grad_inputs
                    
                return grads

        # === END OF BACKWARD FUNCTION ===

    return RK_autograd_Function
