from hrockmate.rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
from hrockmate.rkgb.utils import np, torch

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


def make_global_dictionary(device, nn_mod, dict_constants):
    return {
        **globals(),
        **dict_constants,
        "original_mod": nn_mod,
        "device": device,
        "torch": torch,
        "meta": torch.ones(1).to(device),
        "cmeta": torch.view_as_complex(torch.ones(2)).to(device),
        "main_stream": torch.cuda.current_stream(),
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
                        main_code.replace("self.", "original_mod.").replace(
                            "self[", "original_mod["
                        ),
                    )
                )
            else:
                # control saved_tensors in autograd : manual check to avoid pytorch to save some tensors in with_grad mode
                no_save_list = []
                candidates = list(kn.deps_global) + list(kn.users_global)
                candidates = self._get_names(candidates)
                for kdn_name in candidates:
                    if kdn_name in self.op_name_list[i:next_bwd_idx]:
                        no_save_list.append(kdn_name.split(" ")[0])

                for (
                    target
                ) in kn.tensor_targets:  # kn.tensor_targets is Tensor of pytorch
                    inplace_code = inplace_code.replace(target, "_" + target)

                function_list.append(
                    self.fct_run_forward_with_grad(
                        main_code.replace("self.", "original_mod.").replace(
                            "self[", "original_mod["
                        ),
                        no_save_list=no_save_list,
                    )
                )
            # the detach operation must be performed after in-place operations because they impact data, even though
            # these in-place operations can be applied to views and not directly to the original tensor
            function_list.append(
                self.fct_run_forward_with_grad(
                    inplace_code.replace("self.", "original_mod.").replace(
                        "self[", "original_mod["
                    ),
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
                    body_code.replace("self.", "original_mod.").replace(
                        "self[", "original_mod["
                    ),
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

    def get_del_grad(self, kn, i):
        return [self.fct_del_tensor_grad(kn.main_target)]

    def get_prefetch(self, kn, before_idx=None, after_idx=None):
        function_list = []
        # function_list.append(self.fct_mem_alloc(kn.main_target))
        function_list.append(self.fct_prefetch(kn.main_target, after_idx=after_idx))
        return function_list

    def get_offload(self, kn, before_idx=None, after_idx=None):
        function_list = []
        function_list.append(self.fct_offload(kn.main_target, after_idx=after_idx))
        return function_list

    # H-rockmate
    def compile_from_schedule(self, op_sched):
        fct_list = []
        # self.op_name_list = op_sched.op_name_list
        self.op_name_list = [
            (op.name if not op.disabled else "") for op in op_sched.op_list
        ]
        self.op_list = op_sched.op_list
        self.alive_list = op_sched.alive_list
        # self.prf_list = op_sched.prf_list
        # self.ofl_list = op_sched.ofl_list
        self.op_sched = False
        # page_fct = {i: [] for i in range(len(op_sched.op_list))}
        # page_fct[None] = []
        # wait_op = []
        # for op in op_sched.prf_list:
        #     if op.disabled:
        #         continue
        #     after_idx = None
        #     if op.after in self.op_list:
        #         after_idx = self.op_list.index(op.after)
        #     page_fct[after_idx].extend(self.get_prefetch(op.kn, after_idx=after_idx))
        #     wait_op.append(op.before)
        # for op in op_sched.ofl_list:
        #     if op.disabled:
        #         continue
        #     after_idx = None
        #     if op.after in self.op_list:
        #         after_idx = self.op_list.index(op.after)
        #     page_fct[after_idx].extend(self.get_offload(op.kn, after_idx=after_idx))
        # wait_op.append(op.before)

        # fct_list = [page_fct[None]]

        for i, op in enumerate(op_sched.op_list):
            kn = op.kn
            # if op in wait_op:
            #     fct_list.append(
            #         [
            #             self.fct_wait_stream(
            #                 self.gd["main_stream"], self.gd["prefetch_stream"]
            #             )
            #         ]
            #     )
            if op.disabled:
                fct_list.append([])
                continue

            elif "fwd" in op.kn.name:
                for kdn in op.kn.users:
                    if kdn.kdn_type != "data":
                        continue
                    setattr(kn, "proxy", kdn.info.requires_grad)
                fct_list.append(self.get_fwd(kn, i, detach=op.detach))
            elif "bwd" in kn.name:
                fct_list.append(self.get_bwd(kn, i, detach=op.detach))
            elif "data" in kn.name:
                fct_list.append(self.get_del_data(kn, i))
            elif "grad" in kn.name:
                fct_list.append(self.get_del_grad(kn, i))
            else:
                fct_list.append([])

        return fct_list

    def fct_snychronize(self):
        def fct():
            torch.cuda.synchronize()

        return fct

    def fct_record_cuda(self, i, stream=None):
        stream = stream or self.gd["main_stream"]

        def fct():
            assert i not in self.storage.ld["events"]
            self.storage.ld["events"][i] = stream.record_event()

        return fct

    def fct_wait_stream(self, stream, wait_stream):
        def fct():
            stream.wait_stream(wait_stream)

        return fct

    def fct_prefetch(self, var_name, after_idx=None, stream=None, range=[]):
        device = self.gd["device"]
        stream = stream or self.gd["prefetch_stream"]
        range = range or [None, None]

        def prefetch():
            if after_idx:
                stream.wait_event(self.storage.ld["events"][after_idx])
            # stream.wait_stream(self.gd["main_stream"])
            with torch.cuda.stream(stream):
                self.storage.ld[var_name].data = self.storage.ld[f"cpu_{var_name}"].to(
                    device
                )
                self.storage.ld[f"_{var_name}"].data = self.storage.ld[
                    f"{var_name}"
                ].data

        return prefetch

    def fct_offload(self, var_name, after_idx=None, stream=None, range=[]):
        device = self.gd["device"]
        stream = stream or self.gd["offload_stream"]
        range = range or [None, None]

        def offload():
            # stream.wait_stream(self.gd["main_stream"])
            if after_idx:
                stream.wait_event(self.storage.ld["events"][after_idx])
            with torch.cuda.stream(stream):
                self.storage.ld[f"cpu_{var_name}"].copy_(
                    self.storage.ld[var_name], non_blocking=True
                )

        return offload

    def fct_mem_alloc(self, var_name, stream=None):
        stream = stream or self.gd["main_stream"]

        def mem_alloc():
            with torch.cuda.stream(stream):
                self.storage.ld[var_name].data = torch.empty(
                    self.storage.shapes[var_name]
                )

        return mem_alloc

    def fct_mem_dealloc(self, var_name, stream=None):
        stream = stream or self.gd["main_stream"]

        def mem_dealloc():
            with torch.cuda.stream(stream):
                self.storage.ld[var_name] = torch.empty(0)

        return mem_dealloc

    #  ==================================
    #  = ELEMENTARY COMPILING FUNCTIONS =
    #  ==================================
    # region Define Register Hooks
    def fct_get_pack(self, no_save_list, sanity_check=False):
        # no_save_list contains a list of names
        def pack(x):
            for i, c in enumerate(no_save_list):
                if self.storage.ld[c].data_ptr() == x.data_ptr():
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
            self.storage.ld[f"cpu_{tensor_name}"] = torch.empty(
                self.storage.ld[tensor_name].shape
            )

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
            with torch.no_grad():
                exec(code, self.gd, self.storage.ld)

        return fct

    def fct_run_forward_with_grad(self, code, no_save_list=[]):
        def fct():
            # with torch.enable_grad():
            with torch.autograd.graph.saved_tensors_hooks(
                self.fct_get_pack(no_save_list), self.fct_get_unpack()
            ):
                exec(code, self.gd, self.storage.ld)

        return fct

    def fct_run_inplace(self, tensor_name, inplace_code):
        def fct():
            exec(inplace_code, self.gd, self.storage.ld)

        return fct

    def fct_run_detach(self, tensor_name):
        def fct():
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
            self.storage.ld[f"_{tensor_name}"].backward(
                self.storage.ld[tensor_name].grad, retain_graph=retain_graph
            )

        return fct

    def fct_run_backward_with_inputs(self, tensor_name, retain_graph, input_names):
        def fct():
            inputs = [self.storage.ld[name] for name in input_names]
            self.storage.ld[f"_{tensor_name}"].backward(
                self.storage.ld[tensor_name].grad,
                inputs=inputs,
                retain_graph=retain_graph,
            )

        return fct

    def fct_generate_fake_data(self, tensor_name):
        def fct():
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
            self.storage.ld[var_name] = None

        return fct

    # endregion
