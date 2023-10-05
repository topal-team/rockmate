from rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.utils import np, torch
from rkgb.Ktools import K_C_node, K_D_node
from .solvers.def_op import DelOp


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


def make_gd(device, nn_mod, dict_constants):
    return {
        **globals(),
        **dict_constants,
        "original_mod": nn_mod,
        "device": device,
        "torch": torch,
        "meta": torch.ones(1).to(device),
        "cmeta": torch.view_as_complex(torch.ones(2)).to(device),
    }


class RK_Storage:
    def __init__(self):
        self.ld = {}
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

    def _get_names(self, name_list):
        return [kdn.name for kdn in name_list]

        if not self.op_sched:
            return [kdn.name for kdn in name_list]
        else:
            return name_list

    def find_next_idx(l, target, i):
        return i + l[i:].index(target)

    def get_fwd(self, op, i, detach=True):
        if "loss" in op.main_target:
            return [self.fct_run_forward_no_grad("")]
        rec = op.name in self.op_name_list[:i]
        if not op.proxy or (
            op.name.replace("fwd", "bwd") not in self.op_name_list[i:]
        ):  # not prepared for BWD
            last_before_bwd = False
        else:
            next_bwd_idx = i + self.op_name_list[i:].index(
                op.name.replace("fwd", "bwd")
            )
            last_before_bwd = not (
                op.name in self.op_name_list[i + 1 : next_bwd_idx]
            )
        l = []

        if op.is_rand:
            if not rec:
                l.append(self.fct_get_rng_state(op.name))
            else:
                l.append(self.fct_restore_rng_state(op.name))

        if not op.proxy:
            if hasattr(op, "ff_code"):
                l = [self.fct_run_forward_with_grad(op.ff_code)]
            else:
                l = [self.fct_run_forward_with_grad(op.get_code())]
        else:
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
                    self.fct_run_forward_no_grad(
                        main_code.replace("self.", "original_mod.").replace(
                            "self[", "original_mod["
                        ),
                    )
                )
            else:
                no_save_list = []
                candidates = list(op.deps_global) + list(op.users_global)
                candidates = self._get_names(candidates)
                for kdn_name in candidates:
                    if kdn_name in self.op_name_list[i:next_bwd_idx]:
                        no_save_list.append(kdn_name.split(" ")[0])

                for target in op.tensor_targets:
                    inplace_code = inplace_code.replace(target, "_" + target)

                l.append(
                    self.fct_run_forward_with_grad(
                        main_code.replace("self.", "original_mod.").replace(
                            "self[", "original_mod["
                        ),
                        no_save_list=no_save_list,
                    )
                )
            l.append(
                self.fct_run_forward_with_grad(
                    inplace_code.replace("self.", "original_mod.").replace(
                        "self[", "original_mod["
                    ),
                )
            )
            for inplace_target in op.inplace_targets:
                if inplace_target != op.main_target:
                    l.append(
                        self.fct_del_var(
                            f"_{inplace_target}",
                        )
                    )

            if detach:
                l.append(self.fct_run_detach(op.main_target))
            else:
                l.append(self.fct_fake_detach(op.main_target))
            l.append(
                self.fct_run_forward_with_grad(
                    body_code.replace("self.", "original_mod.").replace(
                        "self[", "original_mod["
                    ),
                )
            )

        # get the shape of tensors
        if not rec:
            if op.proxy:
                l.append(self.fct_get_shapes(f"_{op.main_target}"))
            for target in op.tensor_targets:
                l.append(self.fct_get_shapes(target))
        return l

    def get_bwd(self, op, i, detach=True):
        if not detach:
            return []
        rec = op.name in self.op_name_list[:i]
        last = True
        if op.name in self.op_name_list[i + 1 :]:  # not the last bwd
            next_bwd_idx = i + 1 + self.op_name_list[i + 1 :].index(op.name)
            no_fwd_before_bwd = not (
                op.name.replace("bwd", "fwd")
                in self.op_name_list[i + 1 : next_bwd_idx]
            )
            if no_fwd_before_bwd:
                last = False
        l = []
        l2 = []

        if op.is_rand:
            if not rec:
                l.append(self.fct_get_rng_state(op.name))
            else:
                l.append(self.fct_restore_rng_state(op.name))

        temporary_tensor_names = [
            kdn_name.split(" ")[0]
            for kdn_name in self._get_names(op.deps_fake)
            if not self._is_alive(kdn_name, i)
        ]
        if op.main_target in temporary_tensor_names:
            temporary_tensor_names.append(f"_{op.main_target}")
        for tensor_name in temporary_tensor_names:
            l.append(self.fct_generate_fake_data(tensor_name))
            l2.append(self.fct_del_tensor_data(tensor_name))
        if rec:
            prev_i = i - self.op_name_list[:i][::-1].index(op.name) - 1
            input_names = []
            for kdn in op.users_global:
                if f"{kdn.name}" in self.op_name_list[prev_i:i]:
                    input_names.append(kdn.name.split(" ")[0])
            if input_names:
                l.append(
                    self.fct_run_backward_with_inputs(
                        op.main_target,
                        retain_graph=(not last),
                        input_names=input_names,
                    )
                )
        else:
            l.append(
                self.fct_run_backward(op.main_target, retain_graph=(not last))
            )

        return l + l2

    def get_del_data(self, op, i):
        l = []
        l.append(self.fct_del_tensor_data(op.main_target))
        if op.info is not None and op.info.requires_grad:
            l.append(self.fct_del_tensor_data(f"_{op.main_target}"))
        if op.includes_base:
            l.append(self.fct_del_tensor_base(op.main_target))
        for v in op.tensor_targets:
            l.append(self.fct_del_tensor_data(v))
        for v in op.container_targets:
            l.append(self.fct_del_var(v))
        # l.append(self.fct_del_var(f"_{op.main_target}"))

        return l

    def get_del_grad(self, op, i):
        return [self.fct_del_tensor_grad(op.main_target)]

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

    def compile_from_schedule(self, op_sched):
        fct_list = []
        # self.op_name_list = op_sched.op_name_list
        self.op_name_list = [
            (op.name if not op.disabled else "") for op in op_sched.op_list
        ]
        self.alive_list = op_sched.alive_list
        self.op_sched = False
        for i, op in enumerate(op_sched.op_list):
            kn = op.kn
            if op.disabled:
                fct_list.append([])
                continue
            if "fwd" in op.kn.name:
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
            self.storage.shapes[tensor_name] = self.storage.ld[
                tensor_name
            ].shape
            self.storage.dtypes[tensor_name] = self.storage.ld[
                tensor_name
            ].dtype

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
            # ld = {tensor_name: self.storage.ld[f"_{tensor_name}"]}
            # code = f"{tensor_name} = x\n{inplace_code}"
            # print(tensor_name, inplace_code)
            exec(inplace_code, self.gd, self.storage.ld)

        return fct

    def fct_run_detach(self, tensor_name):
        def fct():
            self.storage.ld[tensor_name].data = self.storage.ld[
                f"_{tensor_name}"
            ].data

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
        def fct():
            self.storage.ld[f"_{tensor_name}"].backward(
                self.storage.ld[tensor_name].grad, retain_graph=retain_graph
            )

        return fct

    def fct_run_backward_with_inputs(
        self, tensor_name, retain_graph, input_names
    ):
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
            self.storage.ld[tensor_name].data = torch.empty(
                0, device=self.gd["device"]
            )

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


def kn_list_peak_mem(kn_list, kg, refine_list=False):
    kdn_names = set(
        kn.name for kn in kn_list if isinstance(kn, K_D_node)
    ).union(
        set(
            kdn.name
            for kn in kn_list
            if isinstance(kn, K_C_node)
            for kdn in kn.users
        )
    )

    def refine(kn_list):
        for i, kn in enumerate(kn_list):
            if hasattr(kn, "is_fwd") and "loss" in kn.name:
                kn_list[i] = "Loss"
            if isinstance(kn, K_D_node):
                # try to delete KDN
                src_i = []  # indices of source KCN's after i
                for kcn in kn.deps:
                    if kcn in kn_list[i:]:
                        src_i.append(kn_list[i:].index(kcn) + i)
                    else:
                        src_i.append(len(kn_list))

                next_used_i = len(kn_list)  # the next index to use KDN
                for kcn in kn.users_real:
                    if kcn in kn_list[i:]:
                        next_used_i = min(
                            kn_list[i:].index(kcn) + i, next_used_i
                        )

                if max(src_i) > next_used_i:  # try to use before regenerate
                    kn_list[i] = (
                        "Disabled_" + kn_list[i].name
                    )  # skip this deletion

    if refine_list:
        refine(kn_list)

    alive_list = []
    alive_status = {kdn_name: 0 for kdn_name in kdn_names}
    overhead_list = []

    for kn in kn_list:
        if isinstance(kn, K_C_node):
            for kdn in kn.users:
                alive_status[kdn.name] = 1
        elif isinstance(kn, K_D_node):
            alive_status[kn.name] = 0

        alive_list.append(alive_status.copy())
        if isinstance(kn, K_C_node):
            overhead_list.append(kn.overhead)
        else:
            overhead_list.append(0)

    def optimize(kn_list, alive_list):
        for i, (kn, alive_status) in enumerate(zip(kn_list, alive_list)):
            alive_kn_names = [k for k, v in alive_status.items() if v]
            for kn_name in alive_kn_names:
                kdn = kg.dict_kn[kn_name]
                src_i = []  # indices of source KCN's after i
                for kcn in kdn.deps:
                    if kcn in kn_list[i + 1 :]:
                        src_i.append(kn_list[i + 1 :].index(kcn) + i)
                    else:
                        src_i.append(len(kn_list))

                next_used_i = len(kn_list)  # the next index to use KDN
                for kcn in kdn.users_real:
                    if kcn in kn_list[i:]:
                        next_used_i = min(
                            kn_list[i:].index(kcn) + i, next_used_i
                        )

                if max(src_i) < next_used_i:  # use only after regenerate
                    print(f"{kn_name} is alive at {i}-th place")
                    # kn_list[i] = (
                    #     "Disabled_" + kn_list[i].name
                    # )  # skip this deletion

    # optimize(kn_list, alive_list)

    def _sum_mem(alive_status):
        return sum(kg.dict_kn[k].mem for k, v in alive_status.items() if v)

    return max(
        [
            _sum_mem(alive_status) + overhead
            for alive_status, overhead in zip(alive_list, overhead_list)
        ]
    )


def save_all(kg):
    def _can_del(i, kdn):
        for kcn in kdn.users_real:
            # if "bwd" in kcn.name:
            #     continue
            if kg.list_kcn.index(kcn) > i:
                return False
        return True

    kn_list = []
    alive_list = []
    alive_status = np.zeros(len(kg.list_kdn), dtype=bool)
    alive_status[-1] = True
    for i, kcn in enumerate(kg.list_kcn):
        kn_list.append(kcn)
        for kdn in kcn.users:
            # if "data" not in kdn.kdn_type:
            #     continue
            alive_status[kg.list_kdn.index(kdn)] = 1
        alive_list.append(alive_status.copy())
        for j, kdn in enumerate(kg.list_kdn):
            # if kdn in [kg.output_kdn_data, kg.output_kdn_grad]:
            #     continue
            if alive_status[j] and _can_del(i, kdn):
                kn_list.append(kdn)
                alive_status[j] = 0
                alive_list.append(alive_status.copy())
    return kn_list, alive_list
