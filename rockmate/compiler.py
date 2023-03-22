from rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.utils import np, torch
from rockmate.def_op import DelOp

# region Define Register Hooks
def fct_get_pack(storage, no_save_list, sanity_check=False):
    # no_save_list contains a list of names
    def pack(x):
        for i, c in enumerate(no_save_list):
            if storage.ld[c].data_ptr() == x.data_ptr():
                if sanity_check:
                    assert torch.equal(
                        storage.ld[c].data.as_strided_(
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


def fct_get_unpack(storage):
    def unpack(x):
        if isinstance(x, tuple):
            return storage.ld[x[0]].data.as_strided_(*x[1:4])

        return x

    return unpack

    # endregion


# region Basic Functions


def fct_get_shapes(storage, tensor_name):
    def fct():
        storage.shapes[tensor_name] = storage.ld[tensor_name].shape
        storage.dtypes[tensor_name] = storage.ld[tensor_name].dtype

    return fct


def fct_get_rng_state(storage, op_name):
    def fct():
        storage.rng_state.get(op_name)

    return fct


def fct_restore_rng_state(storage, op_name):
    def fct():
        storage.rng_state.restore(op_name)

    return fct


def fct_run_forward_no_grad(storage, code):
    def fct():
        with torch.no_grad():
            exec(code, storage.gd, storage.ld)

    return fct


def fct_run_forward_with_grad(storage, code, no_save_list=[]):
    def fct():
        with torch.autograd.graph.saved_tensors_hooks(
            fct_get_pack(storage, no_save_list), fct_get_unpack(storage)
        ):
            exec(code, storage.gd, storage.ld)

    return fct


def fct_run_inplace(storage, tensor_name, inplace_code):
    def fct():
        # ld = {tensor_name: storage.ld[f"_{tensor_name}"]}
        # code = f"{tensor_name} = x\n{inplace_code}"
        # print(tensor_name, inplace_code)
        exec(inplace_code, storage.gd, storage.ld)

    return fct


def fct_run_detach(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].data = storage.ld[f"_{tensor_name}"].data

    return fct


def fct_assign_proxy(storage, tensor_name):
    def fct():
        storage.ld[f"_{tensor_name}"] = storage.ld[tensor_name]

    return fct


def fct_requires_grad(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].requires_grad_()

    return fct


def fct_run_backward(storage, tensor_name, retain_graph):
    def fct():
        storage.ld[f"_{tensor_name}"].backward(
            storage.ld[tensor_name].grad, retain_graph=retain_graph
        )

    return fct


def fct_run_backward_with_inputs(
    storage, tensor_name, retain_graph, input_names
):
    inputs = [storage.ld[name] for name in input_names]

    def fct():
        storage.ld[f"_{tensor_name}"].backward(
            storage.ld[tensor_name].grad,
            inputs=inputs,
            retain_graph=retain_graph,
        )

    return fct


def fct_generate_fake_data(storage, tensor_name):
    def fct():
        m = (
            storage.gd["cmeta"]
            if storage.dtypes[tensor_name].is_complex
            else storage.gd["meta"]
        )
        x = m.expand(np.prod(storage.shapes[tensor_name]))
        storage.ld[tensor_name].data = x.view(storage.shapes[tensor_name])

    return fct


def fct_del_tensor_data(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].data = torch.empty(
            0, device=storage.gd["device"]
        )

    return fct


def fct_del_tensor_base(storage, tensor_name):
    def fct():
        storage.ld[f"_{tensor_name}"]._base.data = torch.empty(
            0, device=storage.gd["device"]
        )

    return fct


def fct_del_tensor_grad(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].grad = None

    return fct


def fct_del_var(storage, var_name):
    def fct():
        storage.ld[var_name] = None

    return fct

    # endregion


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


class RK_Storage:
    def __init__(self, device, nn_mod, dict_constants):
        self.gd = {
            **globals(),
            **dict_constants,
            "original_mod": nn_mod,
            "device": device,
            "torch": torch,
            "meta": torch.ones(1).to(device),
            "cmeta": torch.view_as_complex(torch.ones(2)).to(device),
        }
        self.ld = {}
        self.shapes = dict()
        self.dtypes = dict()
        self.rng_state = RngState()

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


class Compiler:
    """
    The compiler takes the full operation schedule as input,
    return the lists of Python functions.
    Each list corresponds to one operation.
    """

    def __init__(self, storage):
        self.storage = storage
        self.shapes = storage.shapes
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

    def get_fwd(self, op, i):
        if "loss" in op.main_target:
            return [fct_run_forward_no_grad(self.storage, "")]
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
                l.append(fct_get_rng_state(self.storage, op.name))
            else:
                l.append(fct_restore_rng_state(self.storage, op.name))

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

        # compile main code
        suffix = ""
        main_code = (
            make_str_assign(
                op.main_code, suffix=suffix, force_special_kwargs=rec
            )
            + "\n"
        )
        main_code = main_code.replace(op.main_target, f"_{op.main_target}")

        if not last_before_bwd:

            # inplace_code = inplace_code.replace(
            #     op.main_target, f"_{op.main_target}"
            # )

            for target in op.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)
            l.append(
                fct_run_forward_no_grad(
                    self.storage, main_code.replace("self", "original_mod"),
                )
            )
        else:
            no_save_list = []
            candidates = list(op.deps_global) + list(op.users_global)
            for kdn_name in candidates:
                if kdn_name in self.op_sched.op_name_list[i:next_bwd_idx]:
                    no_save_list.append(kdn_name.split(" ")[0])

            for target in op.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)

            l.append(
                fct_run_forward_with_grad(
                    self.storage,
                    main_code.replace("self", "original_mod"),
                    no_save_list=no_save_list,
                )
            )
        l.append(
            fct_run_forward_with_grad(
                self.storage, inplace_code.replace("self", "original_mod"),
            )
        )
        l.append(fct_run_detach(self.storage, op.main_target))
        l.append(
            fct_run_forward_with_grad(
                self.storage, body_code.replace("self", "original_mod")
            )
        )

        # get the shape of tensors
        if not rec:
            l.append(fct_get_shapes(self.storage, f"_{op.main_target}"))
            for target in op.tensor_targets:
                l.append(fct_get_shapes(self.storage, target))
        return l

    def get_bwd(self, op, i):
        rec = op.name in self.op_sched.op_name_list[:i]
        last = not (op.name in self.op_sched.op_name_list[i + 1 :])
        l = []
        l2 = []

        if op.is_rand:
            if not rec:
                l.append(fct_get_rng_state(self.storage, op.name))
            else:
                l.append(fct_restore_rng_state(self.storage, op.name))

        temporary_tensor_names = [
            kdn_name.split(" ")[0]
            for kdn_name in op.deps_fake
            if not self._is_alive(kdn_name, i)
        ]
        if op.main_target in temporary_tensor_names:
            temporary_tensor_names.append(f"_{op.main_target}")
        for tensor_name in temporary_tensor_names:
            l.append(fct_generate_fake_data(self.storage, tensor_name))
            l2.append(fct_del_tensor_data(self.storage, tensor_name))
        if rec:
            prev_i = i - self.op_sched.op_name_list[:i][::-1].index(op.name) - 1
            input_names = []
            for kdn_name in op.users_global:
                if f"del {kdn_name}" in self.op_sched.op_name_list[prev_i:i]:
                    input_names.append(kdn_name.split(" ")[0])
            l.append(
                fct_run_backward_with_inputs(
                    self.storage,
                    op.main_target,
                    retain_graph=(not last),
                    input_names=input_names,
                )
            )
        else:
            l.append(
                fct_run_backward(
                    self.storage, op.main_target, retain_graph=(not last)
                )
            )

        return l + l2

    def get_del_data(self, op, i):
        l = []
        l.append(fct_del_tensor_data(self.storage, op.main_target))
        if op.info is not None and op.info.requires_grad:
            l.append(fct_del_tensor_data(self.storage, f"_{op.main_target}"))
        if op.includes_base:
            l.append(fct_del_tensor_base(self.storage, op.main_target))
        for v in op.tensor_targets:
            l.append(fct_del_tensor_data(self.storage, v))
        for v in op.container_targets:
            l.append(fct_del_var(self.storage, v))
        # l.append(fct_del_var(self.storage, f"_{op.main_target}"))

        return l

    def get_del_grad(self, op, i):
        return [fct_del_tensor_grad(self.storage, op.main_target)]

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

