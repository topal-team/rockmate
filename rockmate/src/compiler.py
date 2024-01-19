# from rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.lowlevel.ast_add_on import make_str_assign, make_str_list_assign
# from rkgb.utils import np, torch
import torch
import numpy as np
from .solvers.op_schedule import (
    Activation,
    Parameter,
    Buffer,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    OpSchedule,
)

# from .solvers.op_schedule import PrfOp, OflOp


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
            cpu_optimize_stats={}):
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
        "cpu_optimize_stats": cpu_optimize_stats,
        "main_stream": torch.cuda.current_stream(),
        # "prefetch_stream": torch.cuda.current_stream(),
        # "offload_stream": torch.cuda.current_stream(),
        "prefetch_stream": torch.cuda.Stream(device),
        "offload_stream": torch.cuda.Stream(device),
    }


class RK_Storage:
    """ """

    def __init__(self):
        self.ld = {"events": {}}
        self.shapes = dict()
        self.dtypes = dict()
        self.rng_state = RngState()

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
            kn.name.replace("fwd", "bwd") not in self.op_name_list[i:]
        ):  # not prepared for BWD
            last_before_bwd = False  # if the forward operation is the last one before backward operations
            # print(kn.name.replace("fwd", "bwd"))
        else:
            next_bwd_idx = i + self.op_name_list[i:].index(
                kn.name.replace("fwd", "bwd")
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
                candidates = list(kn.deps_global) + list(kn.users_global)
                candidates = self._get_names(candidates)
                for kdn_name in candidates:
                    if "Delete_"+kdn_name in self.op_name_list[i:next_bwd_idx]:
                        no_save_list.append(kdn_name.split(" ")[0])

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
        if kn.includes_base:
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
        return [self.fct_del_tensor_data(alloc.name)]

    def get_del_buffer(self, alloc, i):
        return [self.fct_del_tensor_data(alloc.name)]

    def get_mapping(self, sources, targets, i, copy=False):
        return [self.fct_mapping(sources=sources, targets=targets, copy=copy)]

    def get_allocation(self, alloc):
        function_list = []
        if isinstance(alloc, Parameter) or isinstance(alloc, Buffer):
            function_list.append(
                self.fct_mem_alloc(
                    alloc.name,
                    shape=alloc.info.tsize
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
                    alloc.name,
                    shape=alloc.size,
                    dtype=alloc.dtype,
                    gd=False,
                )
            )
        return function_list

    def get_prefetch(self, op: PrefetchOp, before_idx=None, after_idx=None):
        function_list = []
        # function_list.append(self.fct_mem_alloc(kn.main_target))
        function_list.append(self.fct_prefetch(op.target.name, after_idx=after_idx, indices=op.indices))
        return function_list

    def get_offload(self, op: OffloadOp, before_idx=None, after_idx=None):
        function_list = []
        function_list.append(self.fct_offload(op.target.name, after_idx=after_idx, indices=op.indices, grad=op.grad))
        return function_list
    
    def compile_all_prefetch(self):
        fct_list = []
        for p in self.parameters:
            fct_list.append(self.fct_mem_alloc(p, 
                                               shape=self.parameters[p].info.tsize,
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
        op_sched.alive_list = op_sched.create_alive_list()
        self.alive_list = op_sched.alive_list
        self.parameters = {k:alloc for k, alloc in op_sched.dict_alloc.items() if (
                           isinstance(alloc, Parameter) and not alloc.grad)}
        # print(self.parameters)
        # self.prf_list = op_sched.prf_list
        # self.ofl_list = op_sched.ofl_list
        self.op_sched = False
        # prf_fct = {i: [] for i in range(len(op_sched.op_list))}
        # prf_fct[None] = []
        # ofl_fct = {i: [] for i in range(len(op_sched.op_list))}
        # ofl_fct[None] = []
        # wait_op = []
        # for op in op_sched.prf_list:
        #     if op.disabled:
        #         continue
        #     after_idx = None
        #     if op.after in op_sched.op_list:
        #         after_idx = op_sched.op_list.index(op.after)
        #     prf_fct[after_idx].extend(self.get_prefetch(op, after_idx=after_idx))
        #     wait_op.append(op.before)
        # for op in op_sched.ofl_list:
        #     if op.disabled:
        #         continue
        #     after_idx = None
        #     if op.after in op_sched.op_list:
        #         after_idx = op_sched.op_list.index(op.after)
        #     ofl_fct[after_idx].extend(self.get_offload(op, after_idx=after_idx))
        # if op_sched.ofl_list:
        #     wait_op.append(op_sched.ofl_list[-1].before)

        # init_fct = prf_fct[None] + ofl_fct[None]
        def op_to_fct(op_list):
            fct_list = []
            for i, op in enumerate(op_list):
                # print(i, len(fct_list))
                if op.disabled:
                    fct_list.append([])
                    continue

                if isinstance(op, ComputeOp):
                    if "fwd" in op.kcn.name:
                        for kdn in op.kcn.users:
                            if kdn.kdn_type != "data":
                                continue
                            setattr(op.kcn, "proxy", kdn.info.requires_grad)

                        fct_list.append(self.get_fwd(op.kcn, i, detach=op.detach))
                    elif "bwd" in op.kcn.name:
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
                            fct_list.append(self.get_del_grad(op.target.kdn.name, i))
                        else:
                            fct_list.append(self.get_del_parameter(op.target.kdn, i))
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
            # torch.cuda.synchronize()
            self.gd["prefetch_stream"].synchronize()
            self.gd["offload_stream"].synchronize()
            self.gd["main_stream"].synchronize()
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
            stream.wait_stream(wait_stream)

        return fct

    def fct_prefetch(self, var_name, after_idx=None, stream=None, indices=[0,None]):
        indices = indices
        device = self.gd["device"]
        stream = stream or self.gd["prefetch_stream"]

        def prefetch():
            # if after_idx:
            #     stream.wait_event(self.storage.ld["events"][after_idx])
            with torch.cuda.stream(stream):
                # stream.wait_stream(self.gd["offload_stream"])
                # self.storage.ld[var_name][indices[0]: indices[1]].data = self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].to(device)
                self.storage.ld[var_name].data.copy_(
                    self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"][
                        indices[0] : indices[1]
                    ],
                    non_blocking=True,
                )
                # self.storage.ld[f"_{var_name}"].data = self.storage.ld[
                #     f"{var_name}"
                # ].data
                # if self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].mean()<1e-7:
                # assert torch.allclose(self.storage.ld[var_name][indices[0]: indices[1]].data, self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].to(device))
                # print(f"cpu_{var_name.removesuffix('_prefetch')}", self.storage.ld[f"cpu_{var_name.removesuffix('_prefetch')}"].mean()
                # ,self.storage.ld[var_name][indices[0]: indices[1]].data.mean())
            # self.gd["main_stream"].wait_stream(stream)

        return prefetch

    def fct_offload(self, var_name, after_idx=None, stream=None, indices=[0,None], grad=False):
        indices = indices
        indices_ = [0, None] if "offload" in var_name else indices
        device = self.gd["device"]
        stream = stream or self.gd["offload_stream"]
        var_name = var_name.removesuffix('_offload')
        if grad:
            def offload():
                with torch.cuda.stream(stream):
                    # stream.wait_stream(self.gd["main_stream"])
                    self.storage.ld[f"cpu_{var_name}"].grad = torch.empty_like(self.storage.ld[f"cpu_{var_name}"], 
                                                                               pin_memory=True)
                    self.storage.ld[f"cpu_{var_name}"].grad.data.copy_(
                        self.storage.ld[var_name].grad,
                        non_blocking=True,
                    )
                    pass
            return offload

        def offload():
            # if after_idx:
            #     stream.wait_event(self.storage.ld["events"][after_idx])
            with torch.cuda.stream(stream):
                # stream.wait_stream(self.gd["main_stream"])
                self.storage.ld[f"cpu_{var_name.removesuffix('_offload')}"][
                    indices[0] : indices[1]
                ].data.copy_(
                    self.storage.ld[var_name][indices_[0] : indices_[1]],
                    non_blocking=True,
                )
                # print(self.storage.ld[var_name].mean())
                # if self.storage.ld[f"cpu_{var_name.removesuffix('_offload')}"].mean()<1e-7:
                # print(f"cpu_{var_name.removesuffix('_offload')}", self.storage.ld[f"cpu_{var_name.removesuffix('_offload')}"].mean())
                # torch.cuda.synchronize()

        return offload

    def fct_mem_alloc(self, var_name, shape, dtype, stream=None, gd=False):
        stream = stream or self.gd["main_stream"]
        if gd:

            def mem_alloc():
                with torch.cuda.stream(stream):
                    self.gd[var_name].data = torch.empty(
                        shape, dtype=dtype, device=self.gd["device"]
                    )

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
                shape = target.info.tsize if isinstance(target, Parameter) else -1
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
            self.storage.ld["optimizers"][op.name].step()
            for p in del_grad:
                self.storage.ld[p].grad = None
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
        no_save_list.extend(list(self.parameters.keys()))

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

    # endregion
