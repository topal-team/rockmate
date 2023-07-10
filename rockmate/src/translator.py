from rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.utils import np, torch
from rockmate.def_op import DelOp


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


class Translator:  # to execute Op
    def __init__(self, storage, aggressive=True):
        self.storage = storage
        self.live = {}
        self.fgt = []
        self.code = []
        self.grad = {}
        self.fwd_op_sched = []
        self.bwd_op_sched = []
        self.op_info = []
        self.fwd_code = []
        self.aggressive = aggressive
        if self.aggressive:
            self.alive_global = {}
            self.info_global = {}

    def _estimate_memory(self):
        mem = 0
        for k, v in self.live.items():
            mt, data = k.split(".")
            if v:
                mem += self.mt2op[mt].mem
        return mem

    def translate(self, op_sched, during_fwd=True, first=False):
        if self.aggressive:
            for i, kdn_name in enumerate(op_sched.kdn_names):
                self.alive_global[kdn_name] = op_sched.alive_list[-1][i]
                self.info_global[kdn_name] = op_sched.kdn_info[kdn_name]
        else:
            self.alive_global = {}
        op_name_list = [op.name for op in op_sched.op_list]
        # Fc/Fn cases
        if op_sched.no_grad:
            code_list = []  # ["with torch.no_grad():"]
            for i, op in enumerate(op_sched.op_list):
                if op.op_type == "Run":
                    if "loss" in op.main_target:
                        code_list.append("")
                    else:
                        # code = ast_to_str(make_ast_module([op.main_code]))
                        # code += "\n"+ast_to_str(make_ast_module(op.body_code))
                        # code = op.code
                        # code = "\t".join(code.splitlines(True))
                        if op.is_rand:
                            code = f"rng_state.get('{op.name}');rng_state.restore('{op.name}')\n{op.ff_code}"
                        else:
                            code = op.ff_code
                        # for target in op.tensor_targets:
                        #     code += f"shapes['{target}'] = {target}.shape;"
                        # for phantom_name in op.phantom_names:
                        #     code += (
                        #         f"shapes['{phantom_name}'] = _{phantom_name}.shape;"
                        #     )
                        code_list.append(f"{code}")

                elif op.kdn_type == "data":
                    code = ""
                    if op_sched.del_input_idx == i:
                        for target in op_sched.del_input_op.tensor_targets:
                            # code += f"del {target};"
                            code += (
                                f"{target}.data = torch.empty(0,device=device);"
                            )
                    else:
                        for target in op.tensor_targets:
                            code += (
                                f"{target}.data = torch.empty(0,device=device);"
                            )
                    if op.includes_base:
                        code += f"{op.main_target}._base.data = torch.empty(0,device=device);"

                        # code += f"del {target};"
                    code_list.append(code)
                else:
                    code_list.append("")
                # if op_sched.del_input_idx == i:
                #     code = "\n"
                #     for target in op_sched.del_input_op.tensor_targets:
                #         code += f"{target}.data = torch.empty(0,device=device);"
                #     code_list[-1] += code
            out_target = op_sched.output_size[0]
            code_list[-1] += f"\n{out_target}.requires_grad_();"
            code_list[-1] += f"shapes['{out_target}'] = {out_target}.shape;"
            # if first:
            #     out_op = [
            #         op for op in op_sched.op_list if out_target in op.name
            #     ][0]
            #     for target in out_op.tensor_targets:
            #         if "loss" not in target:
            #             code_list[-1] += f"shapes['{target}'] = {target}.shape;"
            #     for phantom_name in out_op.phantom_names:
            #         code_list[
            #             -1
            #         ] += f"shapes['{phantom_name}'] = _{phantom_name}.shape;"
            return code_list

        def _is_alive(kdn_name, i):
            if kdn_name in op_sched.kdn_names:
                return op_sched.alive_list[i][
                    op_sched.kdn_names.index(kdn_name)
                ]
            elif kdn_name in self.alive_global:
                return self.alive_global[kdn_name]
            else:
                return True

        def _is_proxy_alive(kdn_target, i):
            if f"{kdn_target} data" not in op_sched.kdn_dict:
                # belong to the next block, or del_input
                return False
            includes_phantoms = op_sched.kdn_dict[
                f"{kdn_target} data"
            ].includes_phantoms

            if f"bwd_{kdn_target}" in op_name_list[:i]:
                return False
            if f"bwd_{kdn_target}" in op_name_list[i]:
                return True
            if f"{kdn_target} phantoms" in op_sched.kdn_names:
                return op_sched.alive_list[i][
                    op_sched.kdn_names.index(f"{kdn_target} phantoms")
                ]
            elif (
                includes_phantoms
            ):  # phantom kdn does not exist, need to check data kdn
                #  op_sched.op_list
                return op_sched.alive_list[i][
                    op_sched.kdn_names.index(f"{kdn_target} data")
                ]
            else:
                return f"fwd_{kdn_target}" in op_name_list[:i] or (
                    not op_sched.is_fwd
                )
            #     if op_sched.is_fwd:
            #         if kdn_target == op_sched.input_size[0]:
            #             # input was generated previously
            #             return True
            #         return f"fwd_{kdn_target}" in op_name_list[:i]
            #     else:
            #         return f"bwd_{kdn_target}" in op_name_list[i:]

        def _generate_fake_data(kdn, i, is_self=False):
            # return code for generate the target fake tensor (only for data/grad)
            prep_code = ""
            after_code = ""
            req_shape = kdn.info.tsize
            target_tensor = None
            mt = kdn.main_target
            dict_info = (
                self.info_global if self.aggressive else op_sched.kdn_info
            )

            target_tensor = f"metensor.clone().expand(np.prod(shapes['{mt}']))"
            prep_code += f"{mt}.data = {target_tensor}.view(shapes['{mt}']);"

            for v in kdn.tensor_targets:
                after_code += f"{v}.data = torch.empty(0,device=device); "
            if is_self:
                prep_code += (
                    f"_{mt}.data = {target_tensor}.view(shapes['{mt}']);"
                )
                after_code += f"_{mt}.data = torch.empty(0,device=device);"
            return prep_code, after_code

        def _run_op(op, i):
            # Forward operation
            mt = op.main_target
            if "fwd" in op.name:
                rec = (i > op_sched.op_list.index(op)) or (not op_sched.is_fwd)
                force = rec or not first
                suffix = ""
                if rec and not op.proxy and "loss" not in op.name:
                    suffix = ".data"
                code = (
                    make_str_assign(
                        op.main_code, suffix=suffix, force_special_kwargs=force
                    )
                    + "\n"
                )

                if op.proxy:
                    if (
                        (not during_fwd)
                        and (not op_sched.no_grad)
                        and (mt == op_sched.output_size[0])
                    ):
                        rec = True
                    # code = make_str_assign(op.main_code, prefix="_") + ";"
                    # if not rec:
                    #     code += f"{mt} = _{mt};\n"

                # else:
                #     code = make_str_assign(op.main_code) + "\n"
                code += (
                    make_str_list_assign(
                        op.inplace_code, force_special_kwargs=force
                    )
                    + "\n"
                )
                if op.proxy:
                    for target in op.tensor_targets:
                        code = code.replace(target, "_" + target)
                    if rec:
                        code += f"{mt}.data = _{mt}.data;\n"
                    else:
                        code += (
                            f"{mt} = _{mt}.detach();{mt}.requires_grad_();\n"
                        )
                for bc in op.body_code:
                    suffix = ""
                    if rec and (bc[0] in op.tensor_targets):
                        suffix = ".data"
                    code += (
                        make_str_assign(
                            bc, suffix=suffix, force_special_kwargs=force
                        )
                        + "\n"
                    )
                rec = (i > op_sched.op_list.index(op)) or (not op_sched.is_fwd)
                for user, tensor, phantom_name in op.alias_in_users_phantoms:
                    if rec and _is_proxy_alive(user, i):
                        code += f"_{phantom_name}.data = {tensor}.view(shapes['{phantom_name}']);"

                if not rec:
                    for target in op.tensor_targets:
                        if "loss" not in target:
                            code += f"shapes['{target}'] = {target}.shape;"
                    for phantom_name in op.phantom_names:
                        code += (
                            f"shapes['{phantom_name}'] = _{phantom_name}.shape;"
                        )
                if op.is_rand:
                    code = f"rng_state.get('{op.name}');rng_state.restore('{op.name}')\n{code}"
                return code
            # Backward operation
            elif "bwd" in op.name:
                mt = op.main_target
                rec = op in op_sched.op_list[:i]
                last = not (op in op_sched.op_list[i + 1 :])
                prep_code = ""
                after_code = ""
                for kdn in op.deps_fake:
                    if (
                        not _is_alive(kdn.name, i)
                        # or op_sched.input_size[0] in kdn.name
                    ):
                        fake_code = _generate_fake_data(
                            kdn, i, is_self=(kdn.main_target == op.main_target)
                        )
                        prep_code += fake_code[0]
                        after_code += fake_code[1]
                if rec:
                    prev_i = i - op_sched.op_list[:i][::-1].index(op) - 1
                    rec_list = []
                    for kdn in op.users_global:
                        if DelOp(kdn) in op_sched.op_list[prev_i:i]:
                            rec_list += [kdn.main_target]  # kdn.tensor_targets
                    inputs = ",".join(rec_list)
                    code = f"_{mt}.backward({mt}.grad, inputs=[{inputs}], retain_graph={not last})"
                else:
                    code = f"_{mt}.backward({mt}.grad, retain_graph={not last})"
                bwd_code = f"{prep_code}\n" f"{code}\n" f"{after_code}"
                if op.is_rand:
                    bwd_code = f"rng_state.get('{op.name}');rng_state.restore('{op.name}')\n{bwd_code}"
                return bwd_code

        def _del_op(op, i):
            code = ""
            if op.kdn_type == "data":
                for user, tensor, phantom_name in op.alias_in_users_phantoms:
                    if _is_proxy_alive(user, i):
                        code += f"_{phantom_name}.data = torch.empty(0,device=device); "
                if (
                    op.info is not None
                    and op.info.requires_grad
                    and _is_alive(op.name.replace("data", "phantoms"), i)
                    and op.proxy
                ):
                    code += f"_{op.main_target}.data = torch.empty(0,device=device);"
                    for inp in op.inplace_targets:
                        # code += f"_{inp}.data = torch.empty(0,device=device);"
                        code += f"del _{inp};"

                    if op.includes_phantoms:
                        code += f"del _{op.main_target};"
                    if op.includes_base:
                        if op.proxy:
                            code += f"_{op.main_target}._base.data = torch.empty(0,device=device);"
                        else:
                            code += f"{op.main_target}._base.data = torch.empty(0,device=device);"

                for v in op.tensor_targets:
                    code += f"{v}.data = torch.empty(0,device=device); "

                for v in op.container_targets:
                    code += f"del {v};"

            if op.kdn_type == "grad":
                code += f"{op.main_target}.grad = None;"
            if op.kdn_type == "phantoms":
                code += f"del _{op.main_target};"
                # for inp in op_sched.kdn_dict[f"{op.main_target} phantoms"].inplace_targets:
                #     code += f"del _{inp};"
            return code

        code_list = []
        for i, (op, alive) in enumerate(
            zip(op_sched.op_list, op_sched.alive_list)
        ):
            if op.op_type == "Run":
                code_list.append(_run_op(op, i))
            if op.op_type == "Del":
                code_list.append(_del_op(op, i))
            # if op_sched.del_input_idx == i:
            #     code = "\n"
            #     for target in op_sched.del_input_op.tensor_targets:
            #         code += f"{target}.data = torch.empty(0,device=device);"
            #     code_list[-1] += code
        return code_list
