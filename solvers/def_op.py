# ==========================
# everything about codes, how
# we exec it and how we store things
# -> to replace rotor/Checkpointable.functions
# ==========================
from rkgb.utils.small_fcts import check_attr
import numpy as np
import torch
import warnings


class RunOp:
    def __init__(self, kcn, keep_kcn=False):
        self.name = kcn.name
        self.time = kcn.time
        self.overhead = kcn.overhead
        self.main_target = kcn.main_target
        self.tensor_targets = kcn.tensor_targets
        # self.save_mem = cn.mem
        self.main_code = kcn.main_code
        self.body_code = kcn.body_code
        self.inplace_code = kcn.inplace_code
        self.ff_code = kcn.get_code(force_special_kwargs=True)
        self.deps_fake = [kdn.name for kdn in kcn.deps_fake]
        self.deps_global = [kdn.name for kdn in kcn.deps_global]
        self.users_global = [kdn.name for kdn in kcn.users_global]
        # self.deps_global = kcn.deps_global
        # self.deps_fake = kcn.deps_fake
        # self.users_global = kcn.users_global
        # self.alias_in_users_phantoms = kcn.alias_in_users_phantoms
        # self.phantom_names = kcn.phantom_names
        if keep_kcn:
            self.kcn = kcn
        self.is_fgt = False
        self.op_type = "Run"
        self.proxy = False
        self.no_grad = False
        for kdn in kcn.users:
            if kdn.kdn_type != "data":
                continue
            self.proxy = kdn.info.requires_grad
        self.is_rand = kcn.is_rand

    def __eq__(self, op2):
        return check_attr(self, op2, ["name"])

    def __str__(self):
        return f"Run {self.name}"


class DelOp:
    def __init__(self, kdn, proxy=True):
        self.name = kdn.name
        self.kdn_type = kdn.kdn_type
        self.time = 0
        self.save_mem = kdn.mem
        self.main_target = kdn.main_target
        # self.tensor_targets = kdn.all_targets
        # self.all_targets = list(kdn.deps)[0].all_targets if kdn.deps else []
        self.tensor_targets = kdn.tensor_targets
        self.all_targets = kdn.all_targets
        self.container_targets = kdn.container_targets
        self.inplace_targets = kdn.inplace_targets
        # self.code = kn.get_code()
        # self.requires_grad = kdn.info.requires_grad
        self.info = kdn.info
        self.is_fgt = True
        self.op_type = "Del"
        self.proxy = proxy
        self.includes_phantoms = kdn.includes_phantoms
        self.includes_base = kdn.includes_base
        # self.alias_in_users_phantoms = kdn.alias_in_users_phantoms

    def __eq__(self, op2):
        return check_attr(self, op2, ["name"])

    def __str__(self):
        return f"Del {self.name}"


class OpSchedule:
    def __init__(
        self,
        op_list,
        alive_list,
        input_kdn_data,
        input_kdn_grad,
        output_kdn_data,
        list_kdn,
        no_grad=False,
    ):
        self.op_list = op_list
        self.op_name_list = [op.name for op in self.op_list]
        self.alive_list = alive_list
        L = len(op_list)

        self.no_grad = no_grad

        self.input_size = (
            input_kdn_data.main_target,
            input_kdn_data.mem,
        )
        self.output_size = (
            output_kdn_data.main_target,
            output_kdn_data.mem,
        )
        self.kdn_dict = {kdn.name: kdn for kdn in list_kdn}

        # save the del_input op in case needed
        input_kdn = input_kdn_data
        self.del_input_op = DelOp(input_kdn, proxy=False)
        self.del_input_idx = L

        list_kdn = list_kdn + [input_kdn_grad, input_kdn_data]
        self.mem_sizes = [kdn.mem for kdn in list_kdn]
        self.kdn_names = [kdn.name for kdn in list_kdn]
        self.kdn_info = {
            kdn.name: kdn.info for kdn in list_kdn
        }  # dict: name->info

        self.is_fwd = True
        self.get_mem_time()
        assert self.valid_sched()

    def get_mem_time(self):
        """
        everytime op_list/alive_list are changed, run this to update mem
        """
        L = len(self.op_list)
        self.save = np.zeros(L)
        self.tmp = np.zeros(L)
        input_grad = False
        output_grad = False
        for i, op in enumerate(self.op_list):
            if isinstance(op, RunOp):
                self.tmp[i] = op.overhead
                if "bwd" in op.name:
                    self.is_fwd = False
                    # rotor assumes the space for input data but not input grad
                    for kdn_name in op.users_global:
                        if not input_grad and self.input_size[0] in kdn_name:
                            self.tmp[i:] += self.input_size[1]
                            input_grad = True

            self.save[i] += self.alive_list[i][:-2].dot(
                np.array(self.mem_sizes[:-2])
            )  # input kdn is not included
            # if (
            #     not output_grad
            #     and self.alive_list[i][
            #         self.kdn_names.index(self.output_size[0] + " grad")
            #     ]
            # ):
            #     self.save[i:] -= self.output_size[1]
            #     output_grad = True
        self.overhead = max(self.save + self.tmp) - self.save[-1]
        self.time = sum([op.time for op in self.op_list])

    def get_del_input_idx(self, kg):
        """
        This method is to find the idx where input is no longer needed.
        Should only used for Fn
        """
        input_kdn_name = kg.input_kdn_data.name
        for i, op in enumerate(self.op_list):
            if isinstance(op, RunOp) and input_kdn_name in op.deps_global:
                self.del_input_idx = i + 1

    def del_input(self):
        self.op_list.insert(self.del_input_idx, self.del_input_op)
        alive_status = self.alive_list[self.del_input_idx - 1].copy()
        self.alive_list.insert(self.del_input_idx, alive_status)
        for i in range(self.del_input_idx, len(self.op_list)):
            self.alive_list[i][-1] = False

        self.get_mem_time()
        # self.save = np.append(self.save, self.save[-1])
        # # self.del_input_idx = max(self.op_list.index(kcn)
        # #                     for kcn in input_kdn.users_global
        # #                     if kcn in self.op_list)
        # self.save[self.del_input_idx :] -= input_kdn.mem
        # self.overhead = max(self.save + self.tmp) - self.save[-1]

    def valid_sched(self):
        for i, op in enumerate(self.op_list[1:]):
            if hasattr(op, "deps_global"):
                for kdn_name in op.deps_global:
                    if (
                        not self.alive_list[i][self.kdn_names.index(kdn_name)]
                        and kdn_name not in op.deps_fake
                    ):
                        print(f"{kdn_name} is not alive when {op.name}")
                        return False
        return True


# class RK_Function:
#     def __init__(self, code_fe, code_fn, code_fc, code_bwd):
#         self.code_fe = code_fe
#         self.code_fn = code_fn
#         self.code_fc = code_fc
#         self.code_bwd = code_bwd

#     def exec_fe(self, storage: RK_Storage):
#         self.code_fe.exec(storage)

#     def exec_fn(self, storage: RK_Storage):
#         self.code_fn.exec(storage)

#     def exec_fc(self, storage: RK_Storage):
#         self.code_fc.exec(storage)

#     def exec_bwd(self, storage: RK_Storage):
#         self.code_bwd.exec(storage)
