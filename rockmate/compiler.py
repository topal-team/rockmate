from rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.utils import np, torch
from rockmate.def_code import DelOp


class RK_Storage:
    def __init__(self, device, nn_mod):
        self.gd = {
            **globals(),
            "original_mod": nn_mod,
            "device": device,
            "torch": torch,
            "meta": torch.ones(1).to(device),
        }
        self.ld = {}
        self.shapes = dict()

    def add_val(self, val, x):
        self.ld[val] = x

    def get_val(self, val):
        try:
            return self.ld[val]
        except:
            try:
                return self.gd[val]
            except:
                raise Exception(f"{val} not in the storage")

    def add_shape(self, target):
        # only for target, thus not consider phantoms
        self.shapes[target] = self.ld[target].shape


class RngState:
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


class Compiler:
    """
    The compiler takes the full operation schedule as input,
    return the Python functions each corresponds to one operation.
    """

    def __init__(self, storage, rngstate):
        self.storage = storage
        self.shapes = storage.shapes
        self.rng_state = rngstate
        self.device = self.storage.gd["device"]

    def _is_alive(self, kdn_name, i):
        if kdn_name in self.op_sched.kdn_names:
            return self.op_sched.alive_list[i][
                self.op_sched.kdn_names.index(kdn_name)
            ]

        else:
            return True

    def find_next_idx(l, target, i):
        return i + l[i:].index(target)

    # region Define Register Hooks
    def get_pack(self, no_save_list, sanity_check=False):
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
                    # return x.detach().clone()
                    return (
                        c,
                        x.shape,
                        x.stride(),
                        x.storage_offset(),
                        # x.clone(),
                    )
            return x

        return pack

    def get_unpack(self, no_save_list):
        def unpack(x):
            if isinstance(x, tuple):
                return self.storage.ld[x[0]].data.as_strided_(*x[1:4])

                # print(
                #     x[0], self.storage.ld[x[0]].data.as_strided_(*x[1:]).shape
                # )
                # if (
                #     self.storage.ld[no_save_list[x[0]]].data.as_strided_(*x[1:])
                #     == None
                # ):
                #     print(f"warning: {no_save_list[x[0]]} is None")
                # with torch.no_grad():
                #     y = self.storage.ld[x[0]].detach().as_strided_(*x[1:]).clone()
                # print(y.grad_fn)
                # return y.detach()
                # if self.storage.ld[x[0]]
                # return self.storage.ld[x[0]].as_strided_(*x[1:])
                # if not torch.equal(
                #     self.storage.ld[x[0]].data.as_strided_(*x[1:4]), x[4],
                # ):

                #     print(
                #         x[0],
                #         torch.sum(
                #             abs(
                #                 self.storage.ld[x[0]].data.as_strided_(*x[1:4])
                #                 - x[4]
                #             )
                #         ),
                #     )  # .data.as_strided_(*x[1:4]))
                # print(x[4])
                # return x[4]
                # return self.storage.ld[x[0]].data.as_strided_(*x[1:4])
            # print(x.shape)
            return x

        return unpack

    # endregion

    # region Basic Functions
    def get_shapes(self, tensor_name):
        def fct():
            self.shapes[tensor_name] = self.storage.ld[tensor_name].shape

        return fct

    def get_rng_state(self, op_name):
        def fct():
            self.rng_state.get(op_name)

        return fct

    def restore_rng_state(self, op_name):
        def fct():
            self.rng_state.restore(op_name)

        return fct

    def run_forward_no_grad(self, code):
        def fct():
            with torch.no_grad():
                exec(code, self.storage.gd, self.storage.ld)

        return fct

    def run_forward_with_grad(self, code, no_save_list=[]):
        def fct():
            with torch.autograd.graph.saved_tensors_hooks(
                self.get_pack(no_save_list), self.get_unpack(no_save_list)
            ):
                exec(code, self.storage.gd, self.storage.ld)

        return fct

    def run_inplace(self, tensor_name, inplace_code):
        def fct():
            # ld = {tensor_name: self.storage.ld[f"_{tensor_name}"]}
            # code = f"{tensor_name} = x\n{inplace_code}"
            # print(tensor_name, inplace_code)
            exec(inplace_code, self.storage.gd, self.storage.ld)

        return fct

    def run_detach(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].data = self.storage.ld[
                f"_{tensor_name}"
            ].data

        return fct

    def assign_proxy(self, tensor_name):
        def fct():
            self.storage.ld[f"_{tensor_name}"] = self.storage.ld[
                tensor_name
            ].data

        return fct

    def requires_grad(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].requires_grad_()

        return fct

    def run_backward(self, tensor_name, retain_graph):
        def fct():
            self.storage.ld[f"_{tensor_name}"].backward(
                self.storage.ld[tensor_name].grad, retain_graph=retain_graph
            )

        return fct

    def run_backward_with_inputs(self, tensor_name, retain_graph, input_names):
        inputs = [self.storage.ld[name] for name in input_names]

        def fct():
            self.storage.ld[f"_{tensor_name}"].backward(
                self.storage.ld[tensor_name].grad,
                inputs=inputs,
                retain_graph=retain_graph,
            )

        return fct

    def generate_fake_data(self, tensor_name):
        def fct():
            x = self.storage.gd["meta"].expand(
                np.prod(self.shapes[tensor_name])
            )
            self.storage.ld[tensor_name].data = x.view(self.shapes[tensor_name])

        return fct

    def del_tensor_data(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].data = torch.empty(0)

        return fct

    def del_tensor_base(self, tensor_name):
        def fct():
            self.storage.ld[f"_{tensor_name}"]._base.data = torch.empty(0)

        return fct

    def del_tensor_grad(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].grad = None

        return fct

    def del_var(self, var_name):
        def fct():
            self.storage.ld[var_name] = None

        return fct

    # endregion

    # region Get Executable Functions

    def get_fwd(self, op, i):
        if "loss" in op.main_target:
            return [self.run_forward_no_grad("")]
        rec = op.name in self.op_sched.op_name_list[:i]
        if not op.proxy:
            last_before_bwd = False
        else:
            next_bwd_idx = i + self.op_sched.op_name_list[i:].index(
                op.name.replace("fwd", "bwd")
            )
            last_before_bwd = not (
                op.name in self.op_sched.op_name_list[i + 1 : next_bwd_idx]
            )
        l = []

        if op.is_rand:
            if not rec:
                l.append(self.get_rng_state(op.name))
            else:
                l.append(self.restore_rng_state(op.name))

        if not last_before_bwd:

            suffix = ".data"
            main_code = (
                make_str_assign(
                    op.main_code, suffix=suffix, force_special_kwargs=rec
                )
                + "\n"
            )

            # compile inplace code
            inplace_code = make_str_list_assign(
                op.inplace_code, force_special_kwargs=rec
            )
            # compile body code
            body_code = ""
            for bc in op.body_code:
                suffix = ""
                if rec and (bc[0] in op.tensor_targets):
                    suffix = ".data"
                body_code += (
                    make_str_assign(bc, suffix=suffix, force_special_kwargs=rec)
                    + "\n"
                )
            ff_code = main_code + inplace_code + "\n" + body_code
            l.append(
                self.run_forward_no_grad(
                    ff_code.replace("self", "original_mod")
                )
            )
            for target in op.tensor_targets:
                l.append(self.requires_grad(target))
            l.append(self.assign_proxy(op.main_target))

        else:
            no_save_list = []
            candidates = list(op.deps_global)  # + list(op.users_global)
            for kdn in candidates:
                # if (
                #     f"{kdn.main_target} data"
                #     in self.op_sched.op_name_list[i:next_bwd_idx]
                # ):
                if True:
                    no_save_list.append(kdn.main_target)
            # no_save_list = []

            # compile main code
            suffix = ""
            main_code = (
                make_str_assign(
                    op.main_code, suffix=suffix, force_special_kwargs=rec
                )
                + "\n"
            )
            main_code = main_code.replace(op.main_target, f"_{op.main_target}")

            # compile inplace code
            inplace_code = make_str_list_assign(
                op.inplace_code, force_special_kwargs=rec
            )
            for target in op.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)

            # compile body code
            body_code = ""
            for bc in op.body_code:
                suffix = ""
                if rec and (bc[0] in op.tensor_targets):
                    suffix = ".data"
                body_code += (
                    make_str_assign(bc, suffix=suffix, force_special_kwargs=rec)
                    + "\n"
                )

            l.append(
                self.run_forward_with_grad(
                    main_code.replace("self", "original_mod"),
                    no_save_list=no_save_list,
                )
            )
            l.append(
                self.run_forward_with_grad(
                    inplace_code.replace("self", "original_mod"),
                )
            )
            l.append(self.run_detach(op.main_target))
            l.append(
                self.run_forward_with_grad(
                    body_code.replace("self", "original_mod")
                )
            )

        # get the shape of tensors
        if not rec:
            l.append(self.get_shapes(f"_{op.main_target}"))
            for target in op.tensor_targets:
                l.append(self.get_shapes(target))
        return l

    def get_bwd(self, op, i):
        rec = op.name in self.op_sched.op_name_list[:i]
        last = False  # not (op.name in self.op_sched.op_name_list[i + 1 :])
        l = []
        l2 = []

        if op.is_rand:
            if not rec:
                l.append(self.get_rng_state(op.name))
            else:
                l.append(self.restore_rng_state(op.name))

        temporary_tensor_names = [
            kdn.main_target
            for kdn in op.deps_fake
            if not self._is_alive(kdn.name, i)
        ]
        if op.main_target in temporary_tensor_names:
            temporary_tensor_names.append(f"_{op.main_target}")
        for tensor_name in temporary_tensor_names:
            l.append(self.generate_fake_data(tensor_name))
            l2.append(self.del_tensor_data(tensor_name))
        if rec:
            prev_i = i - self.op_sched.op_name_list[:i][::-1].index(op.name) - 1
            input_names = []
            for kdn in op.users_global:
                if (
                    f"del {kdn.main_target} data"
                    in self.op_sched.op_name_list[prev_i:i]
                ):
                    input_names.append(kdn.main_target)
            l.append(
                self.run_backward_with_inputs(
                    op.main_target,
                    retain_graph=(not last),
                    input_names=input_names,
                )
            )
        else:
            l.append(self.run_backward(op.main_target, retain_graph=(not last)))

        return l + l2

    def get_del_data(self, op, i):
        l = []
        l.append(self.del_tensor_data(op.main_target))
        if op.proxy:
            l.append(self.del_tensor_data(f"_{op.main_target}"))
        if op.includes_base:
            l.append(self.del_tensor_base(op.main_target))
        for v in op.tensor_targets:
            l.append(self.del_tensor_data(v))
        for v in op.container_targets:
            l.append(self.del_var(v))
        return l

    def get_del_grad(self, op, i):
        return [self.del_tensor_grad(op.main_target)]

    # endregion

    # region
    def compile(self, op_sched):
        self.op_sched = op_sched

        fct_list = []
        for i, op in enumerate(op_sched.op_list):
            if "fwd" in op.name:
                fct_list.append(self.get_fwd(op, i))
            elif "bwd" in op.name:
                fct_list.append(self.get_bwd(op, i))
            elif "data" in op.name:
                fct_list.append(self.get_del_data(op, i))
            elif "grad" in op.name:
                fct_list.append(self.get_del_grad(op, i))
            else:
                fct_list.append([])

        return fct_list

    # endregion
