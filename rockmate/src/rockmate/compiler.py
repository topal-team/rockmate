from rkgb.lowlevel.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.core.backward import ComputationNode, AllocationNode
import torch
import numpy as np
from .solvers.op_schedule import (
    Allocation,
    Activation,
    Parameter,
    Buffer,
    Op,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    OpSchedule,
    ExecCodeOp
)
import psutil


class RngState:
    """
    We take care of random operations,
    in particular to be able to recompute a random function deterministically,
    we store random states on the first computation and restore them when needed.
    """

    def __init__(self):
        self.cpu_states = {}
        self.gpu_states = {}

    def get(self, op_name):
        if op_name not in self.cpu_states.keys():
            self.cpu_states[op_name] = torch.get_rng_state()
            self.gpu_states[op_name] = torch.cuda.get_rng_state()

    def restore(self, op_name):
        # pass
        torch.set_rng_state(self.cpu_states[op_name])
        torch.cuda.set_rng_state(self.gpu_states[op_name])


def make_gd(device, 
            nn_mod, 
            dict_constants,
            cpu_optim,
            gpu_optim,
            optim_kwargs={},
            optimize_stats={}):
    return {
        **globals(),
        **dict_constants,
        # "original_mod": nn_mod,
        "self": nn_mod,
        "device": device,
        "torch": torch,
        "meta": torch.ones(1).to(device),
        "cmeta": torch.view_as_complex(torch.ones(2)).to(device),
        "cpu_optim": cpu_optim,
        "gpu_optim": gpu_optim,
        "opt_kwargs": optim_kwargs,
        "optimize_stats": optimize_stats,
        "main_stream": torch.cuda.current_stream(),
        # "prefetch_stream": torch.cuda.current_stream(),
        # "offload_stream": torch.cuda.current_stream(),
        "prefetch_stream": torch.cuda.Stream(device),
        "offload_stream": torch.cuda.Stream(device),
    }

class RK_Storage:
    """
    There can only be one RK_Storage.
    All the changes will be made correspond to it.
    All the RK_Fct share the storage.
    """
    _instance = None
    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def init(self, gd):
        self.ld = dict()
        self.shapes = dict()
        self.dtypes = dict()
        self.rng_state = RngState()
        self.gd = gd

    def add_val(self, val, x):
        self.ld[val] = x


class Compiler:
    """
    The compiler takes the full operation schedule as input,
    return the lists of Python functions.
    Each list corresponds to one operation.
    """

    gd: dict = None
    storage: RK_Storage = None

    def __init__(self, gd):
        self.gd = gd
        self.no_save_dict = {}

    def get_val(self, val):
        if val in self.storage.ld:
            return self.storage.ld[val]
        elif val in self.gd:
            return self.gd[val]
        else:
            raise Exception(f"{val} not defined in executing RK_Env")

    def _is_alive(self, kdn_name, i):
        if not self.op_sched:
            return self.alive_list[i][kdn_name]
        if kdn_name in self.op_sched.kdn_names:
            return self.alive_list[i][self.op_sched.kdn_names.index(kdn_name)]
        else:
            return True

    def _get_names(self, kdn_list):
        """return the list of data node' names"""
        return [kdn.name for kdn in kdn_list]

    def find_target_next_occurance(elements, target, start):
        return start + elements[start:].index(target)

    def get_fwd(self, kn, i, detach=True):
        """get forward part functions
        A K_C_node node consists of a main code that creates the .data, and a body code that contains secondary
        statements about shapes, views, and in-place operations.
        To prevent autograd from creating the whole computational graph in output's grad fn,
        rk-Exec detach each tensor after computing it, so that grad fn only keeps track of the last operation.
        We always name with an underscore the variable before detaching, we call it the proxy.


        Parameters: kn: computation/data node in H Graph at operation i
                    i : index of the operation in the op schedule
                    detach : detach the tensor from computational graph
        """

        if "loss" in kn.main_target:
            return [self.fct_run_forward_no_grad("")]

        # if we find the same operation before current operation
        recomputation = kn.name in self.op_name_list[:i]

        if not kn.proxy or (
            kn.name.replace("FWD", "BWD") not in self.op_name_list[i:]
        ):  # not prepared for BWD
            last_before_bwd = False  # if the forward operation is the last one before backward operations
            # print(kn.name.replace("fwd", "bwd"))
        else:
            next_bwd_idx = i + self.op_name_list[i:].index(
                kn.name.replace("FWD", "BWD")
            )
            last_before_bwd = not (kn.name in self.op_name_list[i + 1 : next_bwd_idx])
        function_list = []

        if kn.is_rand:
            if not recomputation:
                function_list.append(self.fct_get_rng_state(kn.name))
            else:
                function_list.append(self.fct_restore_rng_state(kn.name))

        if not kn.proxy:
            if hasattr(kn, "ff_code"):  # ff_code is in old version code
                function_list = [self.fct_run_forward_with_grad(kn.ff_code)]
            else:
                function_list = [self.fct_run_forward_with_grad(kn.get_code())]
        else:
            # compile inplace code
            inplace_code = make_str_list_assign(
                kn.inplace_code, force_special_kwargs=recomputation
            )
            # compile body code
            body_code = ""
            for bc in kn.body_code:
                suffix = ""
                if recomputation and (bc[0] in kn.tensor_targets):
                    suffix = ".data"
                body_code += (
                    make_str_assign(
                        bc, suffix=suffix, force_special_kwargs=recomputation
                    )
                    + "\n"
                )

            # compile main code
            suffix = ""
            main_code = (
                make_str_assign(
                    kn.main_code, suffix=suffix, force_special_kwargs=recomputation
                )
                + "\n"
            )
            main_code = main_code.replace(kn.main_target, f"_{kn.main_target}")

            # if the operation is not the last one before bwd process, we run the main code in no_grad mode
            if not last_before_bwd:
                for target in kn.tensor_targets:
                    inplace_code = inplace_code.replace(target, "_" + target)

                function_list.append(
                    self.fct_run_forward_no_grad(
                        main_code
                        # .replace("self.", "original_mod.").replace(
                        #     "self[", "original_mod["
                        # ),
                    )
                )
            else:
                # control saved_tensors in autograd : manual check to avoid pytorch to save some tensors in with_grad mode
                no_save_list = []
                candidates = list(kn.deps_real) + list(kn.users)
                candidates = self._get_names(candidates)
                for kdn_name in candidates:
                    if "Delete_"+kdn_name in self.op_name_list[i:next_bwd_idx]:
                        no_save_list.append(kdn_name.split(" ")[0])
                for pnode in kn.required_parameter_nodes_real|kn.required_parameter_nodes_fake:
                    no_save_list.append(pnode.param_name)
                    
                self.no_save_dict[kn.name] = no_save_list

                for (
                    target
                ) in kn.tensor_targets:  # kn.tensor_targets is Tensor of pytorch
                    inplace_code = inplace_code.replace(target, "_" + target)
                # print(kn, no_save_list)
                function_list.append(
                    self.fct_run_forward_with_grad(
                        main_code
                        # .replace("self.", "original_mod.").replace(
                        #     "self[", "original_mod["
                        # ),
                        ,
                        no_save_list=no_save_list,
                    )
                )
            # the detach operation must be performed after in-place operations because they impact data, even though
            # these in-place operations can be applied to views and not directly to the original tensor
            function_list.append(
                self.fct_run_forward_with_grad(
                    inplace_code
                    # .replace("self.", "original_mod.").replace(
                    #     "self[", "original_mod["
                    # ),
                )
            )

            # inplace_targets: the in-place operation variable
            for inplace_target in kn.inplace_targets:
                if inplace_target != kn.main_target:
                    function_list.append(
                        self.fct_del_var(
                            f"_{inplace_target}",
                        )
                    )
            # This detach operation must take place before the creation of various independent views,
            # otherwise we would have to detach each view independently, which is impossible in PyTorch
            if detach:
                function_list.append(self.fct_run_detach(kn.main_target))
            else:
                function_list.append(self.fct_fake_detach(kn.main_target))

            # add body_code functions in the list
            function_list.append(
                self.fct_run_forward_with_grad(
                    body_code
                    # .replace("self.", "original_mod.").replace(
                    #     "self[", "original_mod["
                    # ),
                )
            )

        # get the shape of tensors
        if not recomputation:
            if kn.proxy:
                function_list.append(self.fct_get_shapes(f"_{kn.main_target}"))
            for target in kn.tensor_targets:
                function_list.append(self.fct_get_shapes(target))
        return function_list

    def get_bwd(self, kn, i, detach=True):
        """get backward part functions

        Parameters: kn: computation/data node in H Graph at operation i
                    i : index of the operation in the op schedule
                    detach : if detach is false, we use pytorch's backward
        """
        if not detach:
            return []

        recomputation = kn.name in self.op_name_list[:i]
        last = True
        if kn.name in self.op_name_list[i + 1 :]:  # not the last bwd
            next_bwd_idx = i + 1 + self.op_name_list[i + 1 :].index(kn.name)
            no_fwd_before_bwd = not (
                kn.name.replace("bwd", "fwd") in self.op_name_list[i + 1 : next_bwd_idx]
            )
            if no_fwd_before_bwd:
                last = False

        backward_function_list = []
        delete_tensor_function_list = []

        if kn.is_rand:
            if not recomputation:
                backward_function_list.append(self.fct_get_rng_state(kn.name))
            else:
                backward_function_list.append(self.fct_restore_rng_state(kn.name))

        temporary_tensor_names = [
            kdn_name.split(" ")[0]
            for kdn_name in self._get_names(kn.deps_fake)
            if not self._is_alive(kdn_name, i)
        ]
        if kn.main_target in temporary_tensor_names:
            temporary_tensor_names.append(f"_{kn.main_target}")
        for tensor_name in temporary_tensor_names:
            backward_function_list.append(self.fct_generate_fake_data(tensor_name))
            delete_tensor_function_list.append(self.fct_del_tensor_data(tensor_name))

        if recomputation:
            prev_i = i - self.op_name_list[:i][::-1].index(kn.name) - 1
            input_names = []
            for kdn in kn.users_global:
                if f"{kdn.name}" in self.op_name_list[prev_i:i]:
                    input_names.append(kdn.name.split(" ")[0])
            if input_names:
                backward_function_list.append(
                    self.fct_run_backward_with_inputs(
                        kn.main_target,
                        retain_graph=(not last),
                        input_names=input_names,
                    )
                )
        else:
            backward_function_list.append(
                self.fct_run_backward(kn.main_target, retain_graph=(not last))
            )

        return backward_function_list + delete_tensor_function_list

    def get_del_data(self, kn, i):
        function_list = []
        function_list.append(self.fct_del_tensor_data(kn.main_target))
        if kn.info is not None and kn.info.requires_grad:
            function_list.append(self.fct_del_tensor_data(f"_{kn.main_target}"))
        if kn.has_attribute__base:
            function_list.append(self.fct_del_tensor_base(kn.main_target))
        for v in kn.tensor_targets:
            function_list.append(self.fct_del_tensor_data(v))
        for v in kn.container_targets:
            function_list.append(self.fct_del_var(v))
        # l.append(self.fct_del_var(f"_{op.main_target}"))

        return function_list

    def get_del_grad(self, target, i):
        return [self.fct_del_tensor_grad(target)]

    def get_del_parameter(self, alloc, i):
        del_ops = [self.fct_del_tensor_data(alloc.param_name)]
        for view_target in alloc.view_targets:
            del_ops.append(self.fct_del_tensor_data(view_target))
        return del_ops

    def get_del_buffer(self, alloc, i):
        return [self.fct_del_tensor_data(alloc.param_name)]

    def get_mapping(self, sources, targets, i, copy=False):
        return [self.fct_mapping(sources=sources, targets=targets, copy=copy)]

    def get_allocation(self, alloc):
        function_list = []
        if isinstance(alloc, Parameter) or isinstance(alloc, Buffer):
            function_list.append(
                self.fct_mem_alloc(
                    alloc.param_name,
                    shape=alloc.info.tensor_size
                    if isinstance(alloc, Parameter)
                    else alloc.size,
                    dtype=alloc.dtype,
                    gd=False,
                )
            )
            # function_list.append(self.fct_synchronize())
            # function_list.append(
            #     self.fct_wait_stream(self.gd["prefetch_stream"], self.gd["main_stream"])
            # )
        else:
            function_list.append(
                self.fct_mem_alloc(
                    alloc.param_name,
                    shape=alloc.size,
                    dtype=alloc.dtype,
                    gd=False,
                )
            )
        return function_list

    def get_prefetch(self, op: PrefetchOp, before_idx=None, after_idx=None):
        function_list = []
        # function_list.append(self.fct_mem_alloc(kn.main_target))
        function_list.append(self.fct_prefetch(op.target.param_name, 
                                               after_idx=after_idx, 
                                               indices=op.indices,
                                               view_code=op.target.pnode.get_code(),
                                               stream=self.gd["prefetch_stream"],
                                               is_optim_states=op.is_optim_states))
        return function_list

    def get_offload(self, op: OffloadOp, before_idx=None, after_idx=None):
        function_list = []
        function_list.append(self.fct_offload(op.target.param_name, 
                                              after_idx=after_idx, 
                                              indices=op.indices, 
                                              grad=op.grad,
                                            #   stream=self.gd["main_stream"],
                                              stream=self.gd["offload_stream"],
                                              is_optim_states=op.is_optim_states
                                              ))
        return function_list
    
    def get_exec_code(self, op: ExecCodeOp, wait_stream=None):
        function_list = []
        if wait_stream:
            function_list.append(self.fct_wait_stream(self.gd["main_stream"],
                                                      wait_stream))
        function_list.append(self.fct_exec_code(op.code))
        return function_list

    
    def compile_all_prefetch(self):
        fct_list = []
        for p in self.parameters:
            fct_list.append(self.fct_mem_alloc(p, 
                                               shape=self.parameters[p].info.tensor_size,
                                               dtype=self.parameters[p].dtype, ))
            fct_list.append(self.fct_prefetch(p))
        return fct_list

    def compile(self, op_sched):
        self.op_sched = op_sched
        self.op_name_list = op_sched.op_name_list
        self.alive_list = op_sched.alive_list

        fct_list = []
        for i, op in enumerate(op_sched.op_list):
            if "fwd" in op.name:
                setattr(op.kcn, "proxy", op.proxy)
                fct_list.append(self.get_fwd(op.kcn, i))
            elif "bwd" in op.name:
                fct_list.append(self.get_bwd(op.kcn, i))
            elif "data" in op.name:
                fct_list.append(self.get_del_data(op.kdn, i))
            elif "grad" in op.name:
                fct_list.append(self.get_del_grad(op.kdn, i))
            else:
                fct_list.append([])
        return fct_list

    # H-rockmate
    def compile_from_schedule(self, op_sched):
        # for k, v in op_sched.dict_alloc.items():
        #     if isinstance(v, Activation):
        #         continue
        #         # self.storage.ld[v.kdn.main_target] = torch.empty(
        #         #     0,
        #         #     device=self.gd["device"],
        #         #     requires_grad=v.kdn.info.requires_grad,
        #         # )
        #     if isinstance(v, Parameter):
        #         self.storage.ld[k] = self.gd["original_mod"].get_parameter(k.removesuffix(" parameter"))
        #         # self.storage.shapes[v.kdn.main_target] = self.gd[k].shape
        #         # self.storage.dtypes[v.kdn.main_target] = self.gd[k].dtypes
        #     elif isinstance(v, Buffer):
        #         self.storage.ld[k] = torch.empty(0, device=self.gd["device"])
        #         self.storage.ld["cpu_" + k] = torch.empty(0, device=torch.device("cpu"))
        #     else:
        #         print(f"Unrecognized type {type(v)}")

        # self.op_name_list = op_sched.op_name_list
        self.op_name_list = [
            (str(op) if not op.disabled else "") for op in op_sched.op_list
        ]
        # if op_sched.alive_list == []:
        # op_sched.alive_list = op_sched.create_alive_list()
        # self.alive_list = op_sched.alive_list
        self.parameters = {k:alloc for k, alloc in op_sched.dict_alloc.items() if (
                           isinstance(alloc, Parameter) and not alloc.grad 
                           and not alloc.is_optim_states)}
        # print(self.parameters)
        # self.prf_list = op_sched.prf_list
        # self.ofl_list = op_sched.ofl_list
        self.op_sched = False

        # init_fct = prf_fct[None] + ofl_fct[None]
        def op_to_fct(op_list):
            fct_list = []
            for i, op in enumerate(op_list):
                # print(i, len(fct_list))
                if op.disabled:
                    fct_list.append([])
                    continue

                if isinstance(op, ComputeOp):
                    if "FWD" in op.kcn.name:
                        for kdn in op.kcn.users:
                            if kdn.allocation_type != "data":
                                continue
                            setattr(op.kcn, "proxy", kdn.info.requires_grad)

                        fct_list.append(self.get_fwd(op.kcn, i, detach=op.detach))
                    elif "BWD" in op.kcn.name:
                        fct_list.append(self.get_bwd(op.kcn, i, detach=op.detach))
                    else:
                        fct_list.append([])

                elif isinstance(op, DeleteOp):
                    if isinstance(op.target, Activation):
                        if "data" in op.target.kdn.name:
                            fct_list.append(self.get_del_data(op.target.kdn, i))
                        elif "grad" in op.target.kdn.name:
                            fct_list.append(self.get_del_grad(op.target.kdn.main_target, i))
                        else:  # phantom
                            fct_list.append([])
                    elif isinstance(op.target, Parameter):
                        if op.grad:
                            fct_list.append(self.get_del_grad(op.target.pnode.param_name, i))
                        elif op.is_optim_states:
                            fct_list.append([self.fct_del_optimizer_states(op.target.pnode.param_name)])
                        else:
                            fct_list.append(self.get_del_parameter(op.target.pnode, i))
                    elif isinstance(op.target, Buffer):
                        fct_list.append(self.get_del_buffer(op.target, i))
                    else:
                        fct_list.append([])
                elif isinstance(op, MappingOp):
                    fct_list.append(
                        self.get_mapping(op.sources, op.targets, i, op.copy)
                    )  # TODO
                    # if "merge" in op.name:
                    #     fct_list[-1].append(self.fct_synchronize())
                elif isinstance(op, AllocateOp):
                    if op.is_optim_states:
                        fct_list.append([self.fct_mem_alloc(op.target.pnode.param_name,
                                                            shape=op.target.info.tensor_size,
                                                            dtype=op.target.dtype, 
                                                            is_optim_states=True)])
                    else:
                        fct_list.append(self.get_allocation(op.target))
                elif isinstance(op, PrefetchOp):
                    fct_list.append(self.get_prefetch(op))
                    # fct_list[-1].append(self.fct_wait_stream(self.gd["main_stream"],
                    #                                      self.gd["prefetch_stream"],
                    #     ))

                elif isinstance(op, OffloadOp):
                    # fct_list[-1].append(self.fct_wait_stream(self.gd["offload_stream"],
                    #         self.gd["main_stream"]
                    #     ))
                    fct_list.append(self.get_offload(op))
                    # fct_list[-1].append(self.fct_wait_stream(self.gd["prefetch_stream"],
                    #                                      self.gd["offload_stream"]
                    #     ))
                elif isinstance(op, SynchronizeOp):
                    fct_list.append([self.fct_synchronize()])
                elif isinstance(op, OptimizeOp):
                    fct_list.append([self.fct_optimize(op)])
                elif isinstance(op, ExecCodeOp):
                    fct_list.append(self.get_exec_code(op, wait_stream=self.gd["prefetch_stream"]))
                else:
                    fct_list.append([])

                # if isinstance(op, PrefetchOp):
                #     # wait_fct = self.fct_wait_stream(self.gd["main_stream"],
                #     #                                      self.gd["prefetch_stream"],
                #     #     )
                #     # fct_list[-1].append(wait_fct)
                #     pass
                # elif isinstance(op, OffloadOp):
                #     # wait_fct = self.fct_wait_stream(self.gd["prefetch_stream"],
                #     #                                      self.gd["offload_stream"]
                #     #     )
                #     # fct_list[-1].append(wait_fct)
                #     pass
                # else:
                #     # wait_fct = self.fct_wait_stream(self.gd["offload_stream"],
                #     #                                     self.gd["main_stream"]
                #     #     )

                #     # fct_list[-1].append(self.fct_wait_stream(self.gd["main_stream"],
                #     #                                         self.gd["prefetch_stream"],
                #     #         ))
                #     # fct_list[-1].append(self.fct_wait_stream(self.gd["offload_stream"],
                #     #                                         self.gd["main_stream"]
                #     #         ))
                # # elif isinstance(op, AllocateOp) and "prefetch" in op.name:

                # #     fct_list[-1].append(self.fct_wait_stream(self.gd["prefetch_stream"],
                # #                                             self.gd["main_stream"]
                # #             ))
                #     pass
                #     # fct_list[-1].append(self.fct_wait_stream(self.gd["prefetch_stream"],
                #     #                                         self.gd["offload_stream"]
                #     #         ))
            return fct_list

        init_fct_list = op_to_fct(op_sched.init_op_list)
        restore_fct_list = op_to_fct(op_sched.restore_op_list)
        fct_list = op_to_fct(op_sched.op_list)
        # if op in wait_op:
        #     fct_list[-1].insert(
        #         0,
        #         self.fct_wait_stream(
        #             self.gd["main_stream"], self.gd["prefetch_stream"]
        #         ),
        #     )

        # fct_list[-1].append(self.fct_record_cuda(i))
        # for op in prf_fct[i]:
        #     stream = self.gd["prefetch_stream"]
        #     fct_list[-1].append(self.fct_wait_stream(stream,
        #                 self.gd["main_stream"]
        #             ))
        #     fct_list[-1].append(op)

        # for op in ofl_fct[i]:
        #     stream = self.gd["offload_stream"]
            # fct_list[-1].append(self.fct_wait_stream(stream,
            #             self.gd["main_stream"]
        #             ))
        #     fct_list[-1].append(op)
        # fct_list[-1].extend(page_fct[i])
        # fct_list[-1].append(self.fct_record_cuda(i, stream=self.gd["prefetch_stream"]))
        # fct_list[-1].append(self.fct_record_cuda(i, stream=self.gd["offload_stream"]))

        return fct_list, init_fct_list, restore_fct_list

    def fct_synchronize(self):
        def fct():
            torch.cuda.synchronize()
            # self.gd["prefetch_stream"].synchronize()
            # self.gd["offload_stream"].synchronize()
            # self.gd["main_stream"].synchronize()
            # self.gd["prefetch_stream"].wait_stream(self.gd["offload_stream"])
            # self.gd["prefetch_stream"].wait_stream(self.gd["main_stream"])
            # self.gd["offload_stream"].wait_stream(self.gd["main_stream"])
            # self.gd["offload_stream"].wait_stream(self.gd["prefetch_stream"])
            # self.gd["main_stream"].wait_stream(self.gd["prefetch_stream"])
            # self.gd["main_stream"].wait_stream(self.gd["offload_stream"])

        return fct

    def fct_record_cuda(self, i, stream=None):
        stream = stream or self.gd["main_stream"]

        def fct():
            # assert i not in self.storage.ld["events"]
            self.storage.ld["events"][i] = stream.record_event()

        return fct

    def fct_wait_stream(self, stream, wait_stream):
        def fct():
            # stream.wait_stream(wait_stream)
            wait_stream.synchronize()

        return fct
    
    def fct_exec_code(self, code):
        def exec_code():
            exec(code, self.gd, self.storage.ld)
        return exec_code

    def fct_prefetch(self, var_name, after_idx=None, stream=None, indices=[0,None],view_code="",
                     is_optim_states=False):
        indices = indices
        device = self.gd["device"]
        stream = stream or self.gd["prefetch_stream"]
        if is_optim_states:
            def prefetch():
                with torch.cuda.stream(stream):
                    # self.storage.ld[f"cpu_{var_name}"].grad = torch.zeros_like(self.storage.ld[f"cpu_{var_name}"], 
                    #                                                            pin_memory=True)
                    # mem = torch.cuda.memory_allocated()
                    for k,v in self.storage.ld["optimizers"][f"Optimize_{var_name}"].state.items():
                        if self.storage.ld["optimizers"][f"exp_avg_{var_name}"].mean() == 0:continue
                        # v["exp_avg"].data = torch.zeros_like(self.storage.ld["optimizers"][f"exp_avg_{var_name}"], device=self.gd["device"])
                        # v["exp_avg_sq"].data = torch.zeros_like(self.storage.ld["optimizers"][f"exp_avg_sq_{var_name}"], device=self.gd["device"])
                        v["exp_avg"].copy_(self.storage.ld["optimizers"][f"exp_avg_{var_name}"], non_blocking=True)
                        v["exp_avg_sq"].copy_(self.storage.ld["optimizers"][f"exp_avg_sq_{var_name}"], non_blocking=True)
                    #     x = self.storage.ld["optimizers"][f"exp_avg_{var_name}"]
                    #     assert (torch.cuda.memory_allocated()-mem) == x.element_size()*x.numel()*2
                    
                    # print(f"{var_name}, {self.storage.ld[var_name].grad[0,0]}")
                    pass
            return prefetch
        def prefetch():
            # if after_idx:
            #     stream.wait_event(self.storage.ld["events"][after_idx])
            with torch.cuda.stream(stream):
                # stream.wait_stream(self.gd["offload_stream"])
                # self.storage.ld[var_name][indices[0]: indices[1]].data = self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].to(device)
                self.storage.ld[var_name].data.copy_(
                    self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"],
                    non_blocking=True,
                )
                # exec(view_code, self.gd, self.storage.ld)
                # self.storage.ld[f"_{var_name}"].data = self.storage.ld[
                #     f"{var_name}"
                # ].data
                # if self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].mean()<1e-7:
                # assert torch.allclose(self.storage.ld[var_name][indices[0]: indices[1]].data, self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].to(device))
                # print(f"cpu_{var_name.removesuffix('_prefetch')}", self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].mean()
                # ,self.storage.ld[var_name][indices[0]: indices[1]].data.mean())
            # self.gd["main_stream"].wait_stream(stream)

        return prefetch

    def fct_offload(self, var_name, after_idx=None, stream=None, indices=[0,None], 
                    grad=False, is_optim_states=False):
        indices = indices
        indices_ = [0, None] if "offload" in var_name else indices
        device = self.gd["device"]
        stream = stream or self.gd["offload_stream"]
        var_name = var_name.removesuffix('_offload')
        if grad:
            def offload():
                with torch.cuda.stream(stream):
                    # self.storage.ld[f"cpu_{var_name}"].grad = torch.zeros_like(self.storage.ld[f"cpu_{var_name}"], 
                    #                                                            pin_memory=True)
                    self.storage.ld[f"cpu_{var_name}"].grad.data.copy_(
                        self.storage.ld[var_name].grad,
                        non_blocking=True,
                    )
                    # print(f"{var_name}, {self.storage.ld[var_name].grad[0,0]}")
                    pass
            return offload
        
        elif is_optim_states:
            def offload():
                with torch.cuda.stream(stream):
                    # self.storage.ld[f"cpu_{var_name}"].grad = torch.zeros_like(self.storage.ld[f"cpu_{var_name}"], 
                    #                                                            pin_memory=True)
                    # torch.cuda.synchronize()
                    # mem = torch.cuda.memory_allocated()
                    for k,v in self.storage.ld["optimizers"][f"Optimize_{var_name}"].state.items():
                        self.storage.ld["optimizers"][f"exp_avg_{var_name}"].copy_(v["exp_avg"], non_blocking=True)
                        self.storage.ld["optimizers"][f"exp_avg_sq_{var_name}"].copy_(v["exp_avg_sq"], non_blocking=True)
                        # v["exp_avg"].data = torch.empty(0)
                        # v["exp_avg_sq"].data = torch.empty(0)
                    # torch.cuda.synchronize()
                    # x = self.storage.ld["optimizers"][f"exp_avg_{var_name}"]
                    # assert self.storage.ld["optimizers"][f"exp_avg_{var_name}"].numel()>0
                    # assert (mem - torch.cuda.memory_allocated()) == x.element_size()*x.numel()*2
                    # print(f"{var_name}, {self.storage.ld[var_name].grad[0,0]}")
                    pass
            return offload

        def offload():
            # if after_idx:
            #     stream.wait_event(self.storage.ld["events"][after_idx])
            with torch.cuda.stream(stream):
                # stream.wait_stream(self.gd["main_stream"])
                # assert self.storage.ld[f"cpu_{var_name}"].data.stride() == self.storage.ld[var_name].data.stride()
                self.storage.ld[f"cpu_{var_name}"].data.copy_(
                    self.storage.ld[var_name].data,
                    non_blocking=True,
                )
                pass
                # print(self.storage.ld[var_name].mean())
                # if self.storage.ld[f"cpu_{var_name.removesuffix('_offload')}"].mean()<1e-7:
                # print(f"cpu_{var_name.removesuffix('_offload')}", self.storage.ld[f"cpu_{var_name.removesuffix('_offload')}"].mean())
                # torch.cuda.synchronize()

        return offload

    def fct_mem_alloc(self, var_name, shape, dtype, stream=None, gd=False,
                      is_optim_states=False):
        stream = stream or self.gd["main_stream"]
        if gd:

            def mem_alloc():
                with torch.cuda.stream(stream):
                    self.gd[var_name].data = torch.empty(
                        shape, dtype=dtype, device=self.gd["device"]
                    )
        elif is_optim_states:
            def mem_alloc():
                for k,v in self.storage.ld["optimizers"][f"Optimize_{var_name}"].state.items():
                    v["exp_avg"].data = torch.zeros_like(self.storage.ld["optimizers"][f"exp_avg_{var_name}"], device=self.gd["device"])
                    v["exp_avg_sq"].data = torch.zeros_like(self.storage.ld["optimizers"][f"exp_avg_sq_{var_name}"], device=self.gd["device"])

        else:

            def mem_alloc():
                with torch.cuda.stream(stream):
                    self.storage.ld[var_name].data = torch.empty(
                        shape, dtype=dtype, device=self.gd["device"]
                    )

        return mem_alloc

    def fct_mem_dealloc(self, var_name, stream=None):
        stream = stream or self.gd["main_stream"]

        def mem_dealloc():
            with torch.cuda.stream(stream):
                self.storage.ld[var_name] = torch.empty(0)

        return mem_dealloc

    def fct_mapping(self, sources, targets, stream=None, gd=True, copy=False):
        stream = stream or self.gd["main_stream"]
        if len(sources) == 1:
            targets_name = {}
            start = 0
            for target in targets:
                shape = target.info.tensor_size if isinstance(target, Parameter) else -1
                size = target.size
                targets_name[target.name] = (start, start + size, shape)
                start += size

            # assert start == round(sources[0].mem/4)
            def mapping():
                with torch.cuda.stream(stream):
                    tmp = self.storage.ld[sources[0].name].clone()
                    for k, v in targets_name.items():
                        self.storage.ld[k].data = tmp[v[0] : v[1]].view(v[2]).clone()
                        # print("split", k, self.storage.ld[k].data.shape, v)
                        # print("split", k, self.storage.ld[k].data.mean())
                    tmp.data = torch.empty(0)
                    del tmp

        elif len(targets) == 1:
            if copy:  # assume same size

                def mapping():
                    with torch.cuda.stream(stream):
                        self.storage.ld[targets[0].name].copy_(
                            torch.cat(
                                tuple(
                                    self.storage.ld[s.name].flatten() for s in sources
                                ),
                                0,
                            ).data
                        )

            else:

                def mapping():
                    with torch.cuda.stream(stream):
                        self.storage.ld[targets[0].name] = torch.cat(
                            tuple(self.storage.ld[s.name].flatten() for s in sources), 0
                        )
                    # print("merge", targets[0].name, self.storage.ld[targets[0].name].data.shape)
                    # print("merge", targets[0].name, self.storage.ld[sources[1].name].data.mean())

        return mapping
    
    def fct_optimize(self, op):
        del_grad = op.list_params if "cpu" in op.name else []
        def optimize():
            # torch.cuda.synchronize()
            # self.gd["offload_stream"].synchronize()

            # optimizer = self.storage.ld["optimizers"][op.name]
            # if "cpu" not in op.name:
            # mem = torch.cuda.memory_allocated()
            # print(f"{op.name}")
            # print(f"{self.storage.ld[op.list_params[0]].grad.mean()}")
            # first = True
            # for k,v in self.storage.ld["optimizers"][op.name].state.items():
            #     if "exp_avg" in v:
            #         first = False
            #         assert v["exp_avg"].mean() != 0
            # print(op.name, psutil.virtual_memory())
            # if  "cpu" in op.name:
            #     torch.cuda.synchronize()
            # if  "cpu" not in op.name:
            self.storage.ld["optimizers"][op.name].step()
            pass
            # print(psutil.virtual_memory())

            # torch.cuda.synchronize()
            # if not first: assert mem == torch.cuda.memory_allocated()
            for p in del_grad:
                # self.storage.ld[p].grad = None
                self.storage.ld[p].grad.zero_()
            #     self.storage.ld[p.removeprefix("cpu_")].grad = None
            # torch.cuda.synchronize()
            pass
        return optimize


    #  ==================================
    #  = ELEMENTARY COMPILING FUNCTIONS =
    #  ==================================
    # region Define Register Hooks
    def fct_get_pack(self, no_save_list, sanity_check=False):
        # for i in range(6):no_save_list.append(f"{i}.weight parameter")
        # no_save_list contains a list of names
        def pack(x):
            for i, c in enumerate(no_save_list):
                if self.storage.ld[c].data_ptr() == x.data_ptr():
                    # print(c)
                    if sanity_check:
                        assert torch.equal(
                            self.storage.ld[c].data.as_strided_(
                                x.shape, x.stride(), x.storage_offset()
                            ),
                            x,
                        )
                    return (
                        c,
                        x.shape,
                        x.stride(),
                        x.storage_offset(),
                        # x.clone(),
                    )
            return x

        return pack

    def fct_get_unpack(self):
        def unpack(x):
            if isinstance(x, tuple):
                return self.storage.ld[x[0]].data.as_strided_(*x[1:4])

            return x

        return unpack

        # endregion

    # region Basic Functions

    def fct_get_shapes(self, tensor_name):
        def fct():
            self.storage.shapes[tensor_name] = self.storage.ld[tensor_name].shape
            self.storage.dtypes[tensor_name] = self.storage.ld[tensor_name].dtype
            # self.storage.gd[f"cpu_{tensor_name}"] = torch.empty(
            #     self.storage.ld[tensor_name].shape, pin_memory=True
            # )
            # assert self.storage.ld[f"cpu_{tensor_name}"].shape == self.storage.ld[tensor_name].shape

        return fct

    def fct_get_rng_state(self, op_name):
        def fct():
            self.storage.rng_state.get(op_name)

        return fct

    def fct_restore_rng_state(self, op_name):
        def fct():
            self.storage.rng_state.restore(op_name)

        return fct

    def fct_run_forward_no_grad(self, code):
        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                with torch.no_grad():
                    exec(code, self.gd, self.storage.ld)

        return fct

    def fct_run_forward_with_grad(self, code, no_save_list=[]):
        # no_save_list.extend(list(self.parameters.keys()))

        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                with torch.autograd.graph.saved_tensors_hooks(
                    self.fct_get_pack(no_save_list), self.fct_get_unpack()
                ):
                    exec(code, self.gd, self.storage.ld)

        return fct

    def fct_run_inplace(self, tensor_name, inplace_code):
        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                exec(inplace_code, self.gd, self.storage.ld)

        return fct

    def fct_run_detach(self, tensor_name):
        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                self.storage.ld[tensor_name].data = self.storage.ld[f"_{tensor_name}"].data

        return fct

    def fct_fake_detach(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name] = self.storage.ld[f"_{tensor_name}"]
            self.storage.ld[f"_{tensor_name}"] = torch.empty(0)

        return fct

    def fct_assign_proxy(self, tensor_name):
        def fct():
            self.storage.ld[f"_{tensor_name}"] = self.storage.ld[tensor_name]

        return fct

    def fct_requires_grad(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].requires_grad_()

        return fct

    def fct_run_backward(self, tensor_name, retain_graph):
        """
        to return function of backward operations
        """

        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                self.storage.ld[f"_{tensor_name}"].backward(
                    self.storage.ld[tensor_name].grad, retain_graph=retain_graph
                )

        return fct

    def fct_run_backward_with_inputs(self, tensor_name, retain_graph, input_names):
        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                inputs = [self.storage.ld[name] for name in input_names]
                self.storage.ld[f"_{tensor_name}"].backward(
                    self.storage.ld[tensor_name].grad,
                    inputs=inputs,
                    retain_graph=retain_graph,
                )

        return fct

    def fct_generate_fake_data(self, tensor_name):
        def fct():
            with torch.cuda.stream(self.gd["main_stream"]):
                m = (
                    self.gd["cmeta"]
                    if self.storage.dtypes[tensor_name].is_complex
                    else self.gd["meta"]
                )
                s = self.storage.shapes[tensor_name]
                if s == torch.Size([]):
                    x = m.sum()  # easy way to obtain a Tensor of shape []
                else:
                    x = m.expand(np.prod(s)).view(s)
                self.storage.ld[tensor_name].data = x

        return fct

    def fct_del_tensor_data(self, tensor_name):
        def fct():
            # assert self.storage.ld[tensor_name].data.numel()>0
            self.storage.ld[tensor_name].data = torch.empty(0, device=self.gd["device"])

        return fct

    def fct_del_tensor_base(self, tensor_name):
        def fct():
            self.storage.ld[f"_{tensor_name}"]._base.data = torch.empty(
                0, device=self.gd["device"]
            )

        return fct

    def fct_del_tensor_grad(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].grad = None

        return fct

    def fct_del_var(self, var_name):
        def fct():
            self.storage.ld[var_name] = torch.empty(0, device=self.gd["device"])

        return fct

    def fct_del_tensor_gd(self, tensor_name):
        def fct():
            self.gd[tensor_name].data = torch.empty(0, device=self.gd["device"])

        return fct

    def fct_del_optimizer_states(self, var_name):
        def fct():
            for k,v in self.storage.ld["optimizers"][f"Optimize_{var_name}"].state.items():
                v["exp_avg"].data = torch.empty(0)
                v["exp_avg_sq"].data = torch.empty(0)
            pass
        return fct

    # endregion


class Compiler:
    """
    New structure:
    RK_storage will contain ld and gd, ld is for the 
    temporary storage like activations and tensors generated
    for execution (including the views of params).
    If not parameter offloading, we assume the activations 
    should be released by the end of each iteration. 
    Therefore ld.clear() can be called after each iteration.
    The parameters should be returned back to the nn.module. 
    Optimizer states are generated during the iteration, 
    but they are used for future potential training thus
    should be allowed to be kept after iterations.
    By default, it should be released during reinit process.
    """

    def __init__(self, storage):
        self.storage = storage
        self.compile_op = {
            "ComputeOp":self.Compute,
            "DeleteOp":self.Delete,
            "OffloadOp":self.Offload,
            "PrefetchOp":self.Prefetch,
            "AllocateOp":self.Allocate,
            "OptimizeOp":self.Optimize,
            "SynchronizeOp":self.Synchronize,
            "Op":lambda x:None
            }

    def get_val(self, val):
        if val in self.storage.ld:
            return self.storage.ld[val]
        elif val in self.storage.gd:
            return self.storage.gd[val]
        else:
            raise Exception(f"{val} not defined in executing RK_Env")

    def compile_sched(self, op_sched:OpSchedule):
        for i, op in enumerate(op_sched.init_op_list):
            self.compile_op[op.__class__.__name__](op)
        for i, op in enumerate(op_sched.op_list):
            if op.disabled: continue
            # pos_info = self.pos_info(op_sched, i)
            op_type = op.__class__.__name__
            if op_type not in self.compile_op:
                raise SyntaxWarning(f"Unrecognized operation type {op_type}")
            op.fct_list = []
            self.compile_op[op_type](op)

    def compile_preparation(self, cluster, op_sched, minor_param_nodes, output_nodes):
        op_list = op_sched.op_list
        init_op_list = op_sched.init_op_list
        storage: RK_Storage = self.storage
        prep_op = Op("Preparation")
        for anode in cluster.list_anodes:
            if not anode.info: continue# source anode
            prep_op.fct_list.append(Fct_mem_alloc(anode.main_target,
                                                  shape=0,
                                                  dtype=torch.float32,
                                                  alloc_mode="tensor",
                                                  requires_grad=anode.info.requires_grad))
            # storage.ld[anode.main_target] = torch.empty(
            #     0,
            #     device=self.device,
            #     requires_grad=anode.info.requires_grad,
            # )
            
        # for anode in cluster.interfaces["output_data_anodes"]:
        for out_node in output_nodes:
            prep_op.fct_list.append(Fct_mem_alloc(f"out_{out_node.main_target}",
                                                  shape = 0,
                                                  dtype=torch.float32,
                                                  alloc_mode="tensor",
                                                  requires_grad=out_node.info.requires_grad))
        # for out_node in self.rkgb_res.forward_graph.output_nodes:
        #     storage.ld[f"out_{out_node.main_target}"] = torch.empty(
        #         0,
        #         device=self.device,
        #         requires_grad=out_node.info.requires_grad,
        #     )
            

        # for op in self.op_sched.init_op_list:
        #     if isinstance(op, MappingOp) and len(op.targets)==1:
        #         # to create the full size buffer
        #         target = op.targets[0]
        #         storage.ld["cpu_"+target.name.strip("cpu_")] = torch.empty(target.size, 
        #                                             dtype=target.dtype, 
        #                                             device=torch.device("cpu"),
        #                                             pin_memory=True)
                
        for pnode in cluster.parameter_nodes:
            ignore = False
            if pnode.is_buffer or pnode in minor_param_nodes:
                device = storage.gd["device"]
                ignore = True
            else:
                device = torch.device("cpu")
            # for t in pnode.view_targets:
            #     storage.ld[t] = torch.empty(0, requires_grad=pnode.requires_grad)
            prep_op.fct_list.append(Fct_to_storage(pnode.param_name,
                                                   pnode,
                                                   device=device))
            
            prep_op.fct_list.append(Fct_run_fwd(pnode.param_name,
                                                code=pnode.get_code()))

            prep_op.fct_list.append(Fct_get_shape(pnode.param_name))
            prep_op.fct_list.append(Fct_RNG_state(pnode.param_name))
            if not ignore:#TODO: check if cpu preparation is necessary for the parameter
                prep_op.fct_list.append(Fct_mem_alloc(f"cpu_{pnode.param_name}",
                                                      alloc_mode="tensor",
                                                      device = torch.device("cpu"),
                                                      pin_memory = True))
                prep_op.fct_list.append(Fct_offload(pnode.param_name,
                                                    pin_memory = True))
                prep_op.fct_list.append(Fct_del(pnode.param_name))
                if pnode.requires_grad:
                    prep_op.fct_list.append(Fct_mem_alloc(f"cpu_{pnode.param_name}",
                                                          device = torch.device("cpu"),
                                                          alloc_mode="grad",
                                                          pin_memory = True))
                    # assume not keeping gradients of parameters in init

            # target = pnode.get_value(self.original_mod)
            # target.data = target.data.to("cuda")
            # storage.ld[pnode.param_name] = target
            # code = make_str_list_assign(pnode.view_code, suffix=".data")
            # exec(pnode.get_code(), self.gd, storage.ld)
        # for k,v in self.op_sched.dict_alloc_param.items():
        #     if v.pnode.mem < self.gd["optimize_stats"]["minor_param_size"]:continue
        #     if v.is_grad:continue
        #     if v.is_optim_states:continue
        #     # target = self.gd["self"].get_parameter(k.removesuffix(" parameter"))
        #     target = v.pnode.get_value(self.original_mod)
        #     storage.shapes[v.pnode.param_name] = target.shape
        #     storage.dtypes[v.pnode.param_name] = target.dtype
        #     target.grad = None
        #     # if (f"Optimize_cpu_{k}" in self.op_sched.op_name_list
        #     #     or f"Offload_{k}" in self.op_sched.op_name_list):
        #     if True:
        #         storage.ld["cpu_"+v.target_name] = torch.empty_like(target, 
        #                                             dtype=target.dtype, 
        #                                             device=torch.device("cpu"),
        #                                             pin_memory=True)
        #         storage.ld["cpu_"+v.target_name].copy_(target.data)
        #         # TODO: get info from op_sched
        #         if v.pnode.requires_grad: #and f"Offload_{k}_grad" in self.op_sched.op_name_list:
        #             storage.ld["cpu_"+v.target_name].grad = torch.empty_like(storage.ld["cpu_"+v.target_name], pin_memory=True)
        #     # if v.pnode.is_buffer:
        #     #     storage.ld[k] = target.to("cuda")
        #     # else:
        #     # target.data = torch.empty(0)
        #     storage.ld[v.target_name] = target
        # print(psutil.virtual_memory())
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
        

        # storage.ld["optimizers"] = {}
        for op in op_list:
            if isinstance(op, OptimizeOp):
                optim = self.storage.gd["cpu_optim"] if "cpu" in op.name else self.storage.gd["gpu_optim"]
                prep_op.fct_list.append(Fct_add_optimizer(op.name, op.list_params, optim, **self.storage.gd["opt_kwargs"]))
                # storage.ld["optimizers"][op.name] = optim([storage.ld[p] for p in op.list_params], **self.gd["opt_kwargs"])
            if isinstance(op, OffloadOp) and isinstance(op.target, Parameter) and op.target.is_optim_states:
                var_name = op.target.param_name
                # CPU optim states are placeholder for offload, they are not necessarily attached to the optimizers
                prep_op.fct_list.append(Fct_mem_alloc(f"exp_avg_{var_name}",
                                                      alloc_mode="data",
                                                      device=torch.device("cpu")
                                                      ))
                prep_op.fct_list.append(Fct_mem_alloc(f"exp_avg_sq_{var_name}",
                                                      alloc_mode="data",
                                                      device=torch.device("cpu")
                                                      ))
        if minor_param_nodes:
            minor_parameters = [pnode.param_name for pnode in minor_param_nodes]
            prep_op.fct_list.append(Fct_add_optimizer("optimizer_minors", 
                                                      minor_parameters, 
                                                      self.storage.gd["gpu_optim"], 
                                                      **self.storage.gd["opt_kwargs"]))

        op_sched.init_op_list = [prep_op] + init_op_list
                # var = storage.ld[f"cpu_{var_name}"] if f"cpu_{var_name}" in storage.ld else storage.ld[var_name]
                # storage.ld["optimizers"][f"exp_avg_{var_name}"] = torch.zeros_like(var, pin_memory=True, device="cpu")
                # storage.ld["optimizers"][f"exp_avg_sq_{var_name}"] = torch.zeros_like(var, pin_memory=True, device="cpu")
        # for k,v in op_sched.dict_alloc_param.items():
        #     target = v.pnode.get_value(self.original_mod)
        #     if v.pnode.mem < self.gd["optimize_stats"]["minor_param_size"]:continue
        #     if v.is_grad:continue
            
        #     target.data = torch.empty(0)
        
        # print(psutil.virtual_memory())

        # if self.minor_parameters:
        #     storage.ld["optimizers"]["minors"] = self.gd["gpu_optim"](self.minor_parameters, **self.gd["opt_kwargs"])
        
        

    # def pos_info(self, op_sched:OpSchedule, i):
    #     """
    #     To get the positional information of the operation in the list.
    #     """
    #     op_list = op_sched.op_list
    #     if not isinstance(op_list[i], ComputeOp):
    #         return dict()
    #     pos_info = {
    #         "index":i,
    #         "first_occurrence":True,
    #         "last_occurrence":True,
    #         }
    #     if op_list[i].target.is_fwd:
    #         pos_info["next_bwd_idx"] = None
    #         pos_info["last_before_bwd"] = False
    #     else:
    #         pos_info["temporary_tensor_names"] = []
    #         pos_info["input_names"] = []
        
    #     return pos_info

    
    def Compute(self, op:ComputeOp):
        if op.target.is_fwd:
            op.fct_list.append(Fct_RNG_state(op.name,
                                            get_state=op.pos_info["first_occurrence"]))
            cnode: ComputationNode = op.target
            if not cnode.info.requires_grad:
                op.fct_list.append(Fct_run_fwd(
                                            target_name=cnode.main_target,
                                            code=cnode.get_code()))
                return None
            
            inplace_code = make_str_list_assign(
                    cnode.inplace_code, force_special_kwargs=not op.pos_info["first_occurrence"]
                )
            body_code = ""
            for bc in cnode.body_code:
                suffix = ""
                if not op.pos_info["first_occurrence"] and (bc[0] in cnode.tensor_targets):
                    suffix = ".data"
                body_code += (
                    make_str_assign(
                        bc, suffix=suffix, force_special_kwargs=not op.pos_info["first_occurrence"]
                    )
                    + "\n"
                )
            # suffix = ""
            main_code = (
                make_str_assign(
                    cnode.main_code, #suffix=suffix, 
                    force_special_kwargs=not op.pos_info["first_occurrence"]
                )
                + "\n"
            )
            main_code = main_code.replace(cnode.main_target, f"_{cnode.main_target}")

            if not op.pos_info["last_before_bwd"]:
                for target in cnode.tensor_targets:
                    inplace_code = inplace_code.replace(target, "_" + target)
                op.fct_list.append(Fct_run_fwd(cnode.main_target,
                                            main_code,
                                            fwd_mode="no_grad"))

            else:
                no_save_list = (op.pos_info["no_save_list"] 
                                if "no_save_list" in op.pos_info else
                                (list(cnode.deps_real) 
                                + list(cnode.users) 
                                + list(cnode.required_parameter_nodes_real)
                                + list(cnode.required_parameter_nodes_fake)))
                    
                for (target) in cnode.tensor_targets:  # cnode.tensor_targets is Tensor of pytorch
                    inplace_code = inplace_code.replace(target, "_" + target)
                
                op.fct_list.append(Fct_run_fwd(cnode.main_target,
                                            main_code,
                                            no_save_list=[anode.main_target if hasattr(anode, "main_target")
                                                          else anode.param_name
                                                          for anode in no_save_list],
                                            fwd_mode="with_grad"))
            op.fct_list.append(Fct_run_fwd(cnode.main_target,inplace_code))
            for inplace_target in cnode.inplace_targets:
                    if inplace_target != cnode.main_target:
                        op.fct_list.append(
                            Fct_del(f"_{inplace_target}",
                                    del_mode="data"
                            )
                        )
            if True:#TODO:fake detach
                op.fct_list.append(Fct_detach(cnode.main_target))
            op.fct_list.append(Fct_run_fwd(cnode.main_target, body_code))
            if op.pos_info["first_occurrence"]:
                # op.fct_list.append(Fct_get_shape(cnode.main_target))
                for target_name in cnode.tensor_targets:
                    op.fct_list.append(Fct_get_shape(target_name))
        else:
            cnode: ComputationNode = op.target
            delete_tensor_function_list = []
            op.fct_list.append(Fct_RNG_state(
                                            op.name,
                                            get_state=op.pos_info["first_occurrence"]))
            
            for target_name in op.pos_info["temporary_tensor_names"]:
                op.fct_list.append(Fct_gen_fake_data(target_name, 
                                                     with_proxy=(cnode.main_target==target_name)))
                delete_tensor_function_list.append(Fct_del(target_name, del_mode="data"))
    
            op.fct_list.append(
                Fct_run_bwd(
                        target_name=cnode.main_target,
                        retain_graph=(not op.pos_info["last_occurrence"]),
                        input_names=op.pos_info["input_names"],
                    )
            )

    def Delete(self, op:DeleteOp):
        if isinstance(op.target, Activation):
            alloc:Activation = op.target
            if alloc.anode.allocation_type == "grad":
                op.fct_list.append(Fct_del(
                                    target_name=alloc.anode.main_target,
                                    del_mode=alloc.anode.allocation_type
                                    ))
            elif alloc.anode.allocation_type == "data":
                op.fct_list.append(Fct_del(
                                    target_name=alloc.anode.main_target,
                                    del_mode=alloc.anode.allocation_type
                                    ))
                if alloc.anode.info is not None and alloc.anode.info.requires_grad:
                    op.fct_list.append(Fct_del(
                                    target_name=f"_{alloc.anode.main_target}",
                                    del_mode="data"
                                    ))
                if alloc.anode.has_attribute__base:
                    op.fct_list.append(Fct_del(
                                    target_name=alloc.anode.main_target,
                                    del_mode="base"
                                    ))
                for v in alloc.anode.tensor_targets:
                    op.fct_list.append(Fct_del(
                                    target_name=v,
                                    del_mode="data"
                                    ))
                for v in alloc.anode.container_targets:
                    op.fct_list.append(Fct_del(
                                    target_name=v,
                                    del_mode="var"
                                    ))
        elif isinstance(op.target, Parameter):
            alloc:Parameter = op.target
            del_mode = "data"
            if alloc.is_grad: del_mode = "grad"
            if alloc.is_optim_states: del_mode = "optim_states"
            op.fct_list.append(Fct_del(
                                    target_name=alloc.param_name,
                                    del_mode=del_mode
                                    ))

    def Offload(self, op: OffloadOp):
        target: Parameter = op.target
        offload_mode = "param"
        if target.is_grad:offload_mode = "grad"
        if target.is_optim_states:offload_mode = "optim_states"
        op.fct_list.append(Fct_offload(
                                       target.param_name,
                                       offload_mode=offload_mode))

    def Prefetch(self, op: PrefetchOp):
        target: Parameter = op.target
        prefetch_mode = "param"
        if target.is_optim_states:prefetch_mode = "optim_states"
        op.fct_list.append(Fct_prefetch(
                                       target,
                                       prefetch_mode=prefetch_mode))
        pass

    def Allocate(self, op: AllocateOp):
        target: Parameter = op.target
        alloc_mode = "data"
        if target.is_optim_states:alloc_mode = "optim_states"
        op.fct_list.append(Fct_mem_alloc(
                                       target.param_name,
                                       alloc_mode=alloc_mode))

    def Optimize(self, op:OptimizeOp):
        op.fct_list.append(Fct_optimize(
                                        op.name,
                                        del_grad_list=op.list_params if op.is_cpu else []))
        
    def Synchronize(self, op:SynchronizeOp):
        op.fct_list.append(Fct_synchronize())


class RK_Fct:
    storage = RK_Storage()
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Fct_del(RK_Fct):
    def __init__(self, target_name: str, del_mode="data"):
        super().__init__(target_name=target_name)
        self.target_name = target_name
        self.del_fcts = {
            "data":self.del_data,
            "base":self.del_base,
            "grad":self.del_grad,
            "var":self.del_var,
            "optim_states":self.del_optim_states}
        self.del_mode = del_mode
    
    def __call__(self):
        self.del_fcts[self.del_mode]()

    def del_data(self):
        self.storage.ld[self.target_name].data = torch.empty(0)
        # pass
    def del_grad(self):
        self.storage.ld[self.target_name].grad = None
    def del_base(self):
        self.storage.ld[self.target_name]._base.data = torch.empty(0)
    def del_var(self):
        self.storage.ld[self.target_name] = torch.empty(0)
    def del_optim_states(self):
            for k,v in self.storage.ld[f"Optimize_{self.target_name}"].state.items():
                v["exp_avg"].data = torch.empty(0)
                v["exp_avg_sq"].data = torch.empty(0)

class Fct_gen_fake_data(RK_Fct):
    def __init__(self, target_name: str, with_proxy=False):
        super().__init__(target_name=target_name)
        self.target_name = target_name
        self.with_proxy = with_proxy
    
    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            m = (
                self.storage.gd["cmeta"]
                if self.storage.dtypes[self.target_name].is_complex
                else self.storage.gd["meta"]
            )
            s = self.storage.shapes[self.target_name]
            if s == torch.Size([]):
                x = m.sum()  # easy way to obtain a Tensor of shape []
            else:
                x = m.expand(np.prod(s)).view(s)
            self.storage.ld[self.target_name].data = x
            if self.with_proxy:
                self.storage.ld[f"_{self.target_name}"].data = x

class Fct_detach(RK_Fct):
    def __init__(self, target_name: str):
        super().__init__(target_name=target_name)
        self.target_name = target_name

    def fake_detach(self):
        """
        For certain in-place operations, detach could lead to wrong memory usage.
        Fake detach is to solve it by merging the operations.
        TODO: test on ResNet
        """
        self.storage.ld[self.target_name] = self.storage.ld[f"_{self.target_name}"]
        self.storage.ld[f"_{self.target_name}"] = torch.empty(0)
    
    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.storage.ld[self.target_name].data = self.storage.ld[f"_{self.target_name}"].data

class Fct_run_bwd(RK_Fct):
    def __init__(self,
                 target_name: str, 
                 retain_graph=False, 
                 input_names=[], 
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        self.retain_graph = retain_graph
        self.input_names = input_names

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            inputs = [self.storage.ld[name] for name in self.input_names]
            if not inputs:
                inputs = None
            self.storage.ld[f"_{self.target_name}"].backward(
                self.storage.ld[self.target_name].grad,
                inputs=inputs,
                retain_graph=self.retain_graph,
            )

class Fct_run_fwd(RK_Fct):
    def __init__(self,
                 target_name: str, 
                 code,
                 no_save_list=[],
                 fwd_mode = "with_grad",
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        self.code = code
        self.no_save_list = no_save_list
        self.fwd_fct = {"with_grad":self.fwd_with_grad,
                        "no_grad":self.fwd_no_grad}
        self.fwd_mode = fwd_mode
    
    def fwd_with_grad(self):
        with torch.autograd.graph.saved_tensors_hooks(
                self.fct_get_pack(self.no_save_list), self.fct_get_unpack()
            ):
            exec(self.code, self.storage.gd, self.storage.ld)

    def fwd_no_grad(self):
        with torch.no_grad():
            exec(self.code, self.storage.gd, self.storage.ld)
    
    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.fwd_fct[self.fwd_mode]()

    def fct_get_pack(self, no_save_list, sanity_check=False):
        # no_save_list contains a list of names
        def pack(x):
            for i, c in enumerate(no_save_list):
                if self.storage.ld[c].data_ptr() == x.data_ptr():
                    # print(c)
                    if sanity_check:
                        assert torch.equal(
                            self.storage.ld[c].data.as_strided_(
                                x.shape, x.stride(), x.storage_offset()
                            ),
                            x,
                        )
                    return (
                        c,
                        x.shape,
                        x.stride(),
                        x.storage_offset(),
                        # x.clone(),
                    )
            return x

        return pack

    def fct_get_unpack(self):
        def unpack(x):
            if isinstance(x, tuple):
                return self.storage.ld[x[0]].data.as_strided_(*x[1:4])
            return x

        return unpack


class Fct_get_shape(RK_Fct):
    def __init__(self,
                 target_name: str, 
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.storage.shapes[self.target_name] = self.storage.ld[self.target_name].shape
            self.storage.shapes[f"cpu_{self.target_name}"] = self.storage.ld[self.target_name].shape
            self.storage.dtypes[self.target_name] = self.storage.ld[self.target_name].dtype
            self.storage.dtypes[f"cpu_{self.target_name}"] = self.storage.ld[self.target_name].dtype

class Fct_optimize(RK_Fct):
    def __init__(self,
                 target_name: str, 
                 del_grad_list: list = [],
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        self.del_grad_list = del_grad_list

    def __call__(self):
        self.storage.ld[self.target_name].step()
        # for p in self.del_grad_list:
        #     self.storage.ld[p].grad.zero_()

class Fct_mem_alloc(RK_Fct):
    def __init__(self,
                 target_name: str, 
                 shape = None,
                 dtype = None,
                 alloc_mode = "data",
                 device = None,
                 **kwargs):
        super().__init__(**kwargs)
        # self.target = target
        self.target_name = target_name
        self.alloc_fct = {"data":self.alloc_data,
                          "grad":self.alloc_grad,
                          "tensor":self.alloc_tensor,
                          "optim_states":self.alloc_optim_states}
        self.alloc_mode = alloc_mode
        if device is None:
            self.device = self.storage.gd["device"]
        else:
            self.device = device
        self.kwargs = kwargs
        self.shape = shape
        self.dtype = dtype
    
    def alloc_optim_states(self):
        for k,v in self.storage.ld[f"Optimize_{self.target_name}"].state.items():
            v["exp_avg"].data = torch.empty_like(self.storage.ld[f"exp_avg_{self.target_name}"], device=self.device)
            v["exp_avg_sq"].data = torch.empty_like(self.storage.ld[f"exp_avg_sq_{self.target_name}"], device=self.device)

    def alloc_grad(self):
        shape = self.storage.shapes[self.target_name] if self.shape is None else self.shape
        dtype = self.storage.dtypes[self.target_name] if self.dtype is None else self.dtype
        self.storage.ld[self.target_name].grad = torch.empty(
                    shape, 
                    dtype=dtype, 
                    device=self.device,
                    **self.kwargs
                )

    def alloc_data(self):
        shape = self.storage.shapes[self.target_name] if self.shape is None else self.shape
        dtype = self.storage.dtypes[self.target_name] if self.dtype is None else self.dtype
        self.storage.ld[self.target_name].data = torch.empty(
                    shape, 
                    dtype=dtype, 
                    device=self.device,
                    **self.kwargs
                )
        
    def alloc_tensor(self):
        shape = self.storage.shapes[self.target_name] if self.shape is None else self.shape
        dtype = self.storage.dtypes[self.target_name] if self.dtype is None else self.dtype
        self.storage.ld[self.target_name] = torch.empty(
                    shape, 
                    dtype=dtype, 
                    device=self.device,
                    **self.kwargs
                )
        
    def __call__(self):
         with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.alloc_fct[self.alloc_mode]()


class Fct_offload(RK_Fct):
    def __init__(self, 
                 target_name: str, 
                 offload_mode = "param",
                 stream = "offload_stream",
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        self.offload_fct = {"param":self.offload_param,
                          "grad":self.offload_grad,
                          "optim_states":self.offload_optim_states}
        self.offload_mode = offload_mode
        self.stream = stream

    def offload_param(self):
        self.storage.ld[f"cpu_{self.target_name}"].data.copy_(
                self.storage.ld[self.target_name].data,
                non_blocking=True,
            )
    def offload_grad(self):
        self.storage.ld[f"cpu_{self.target_name}"].grad.data.copy_(
                    self.storage.ld[self.target_name].grad,
                    non_blocking=True,
                )
            
    def offload_optim_states(self):
        for k,v in self.storage.ld[f"Optimize_{self.target_name}"].state.items():
            self.storage.ld[f"exp_avg_{self.target_name}"].copy_(v["exp_avg"], non_blocking=True)
            self.storage.ld[f"exp_avg_sq_{self.target_name}"].copy_(v["exp_avg_sq"], non_blocking=True)

    def __call__(self):
         with torch.cuda.stream(self.storage.gd[self.stream]):
            self.offload_fct[self.offload_mode]()

                    
class Fct_prefetch(RK_Fct):
    def __init__(self, 
                 target: Allocation, 
                 prefetch_mode = "param",
                 stream = "prefetch_stream",
                 **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.target_name = target.target_name
        self.prefetch_fct = {"param":self.prefetch_param,
                          "optim_states":self.prefetch_optim_states}
        self.post_process = {"param":self.post_process_param,
                          "optim_states":self.post_process_optim_states}
        self.prefetch_mode = prefetch_mode
        self.stream = stream
        self.post_process_code = ""
        if isinstance(target, Parameter):
            self.post_process_code = target.pnode.get_code()
    def prefetch_param(self):
        self.storage.ld[self.target_name].data.copy_(
                self.storage.ld[f"cpu_{self.target_name}"].data,
                non_blocking=True,
            )
    # def prefetch_grad(self):
    #     self.storage.ld[f"cpu_{self.target_name}"].grad.data.copy_(
    #                 self.storage.ld[self.target_name].grad,
    #                 non_blocking=True,
    #             )
    def prefetch_optim_states(self):
        for k,v in self.storage.ld[f"Optimize_{self.target_name}"].state.items():
            self.storage.ld[f"exp_avg_{self.target_name}"].copy_(v["exp_avg"], non_blocking=True)
            self.storage.ld[f"exp_avg_sq_{self.target_name}"].copy_(v["exp_avg_sq"], non_blocking=True)

    def post_process_param(self):
        exec(self.post_process_code, self.storage.gd, self.storage.ld)

    def post_process_optim_states(self):
        pass
    
    def __call__(self):
        with torch.cuda.stream(self.storage.gd[self.stream]):
            self.prefetch_fct[self.prefetch_mode]()
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.post_process[self.prefetch_mode]()

class Fct_synchronize(RK_Fct):
    def __init__(self,
                 stream=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.stream = stream

    def wait_stream(self):
        torch.cuda.stream(self.storage.gd[self.stream]).synchronize()

    def sync_all(self):
        torch.cuda.synchronize()

    def __call__(self):
        if self.stream: 
            self.wait_stream()
        else:
            self.sync_all()

class Fct_RNG_state(RK_Fct):
    def __init__(self,
                 target_name=None,
                 get_state = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        self.get_state = get_state

    def __call__(self):
        if self.get_state:
            self.storage.rng_state.get(self.target_name)
        else:
            self.storage.rng_state.restore(self.target_name)

class Fct_to_storage(RK_Fct):
    """
    Get parameter value from nn.module (storage.ld['self']),
    send to the value in storage.ld[target_name].
    By default, module parameters will be kept in CPU (f'cpu_{target_name}')
    and the real value should be empty.
    """
    def __init__(self,
                 target_name,
                 pnode,
                 device=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        self.pnode = pnode
        self.device = device

    def __call__(self):
        self.storage.ld[self.target_name] = self.pnode.get_value(self.storage.gd["self"])
        self.storage.ld[self.target_name].data = self.storage.ld[self.target_name].data.to(self.device)
        
class Fct_add_optimizer(RK_Fct):
    def __init__(self,
                 target_name,
                 list_params,
                 optim,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_name = target_name
        # self.op = op
        self.list_params = list_params
        self.optim = optim
        self.kwargs = kwargs

    def __call__(self):
        self.storage.ld[self.target_name] = self.optim([self.storage.ld[p] for p in self.list_params], **self.kwargs)
        

