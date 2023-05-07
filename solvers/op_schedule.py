from rkgb.utils import *
from rkgb.Ptools import P_graph, P_node
from rkgb.Ktools import K_graph, K_C_node, K_D_node
from rkgb.Htools import *
from collections import namedtuple


class Op:
    def __init__(self, kn, fast_forward=False, disabled=False, detach=True):
        self.kn = kn
        self.name = kn.name
        self.fast_forward = fast_forward
        self.disabled = disabled
        self.detach = detach
        self.is_del = isinstance(kn, K_D_node)

    def __repr__(self):
        return "Disabled" * self.disabled + self.name


class OpSchedule:
    def __init__(
        self,
        op_list,
        loss_idx=None,
        cluster=None,
        interfaces=None,
        refine=True,
        correct_overhead=True,
    ):
        # Key role of OpSchedule: taking op_list, analyzing memory stats,
        # keeping info for further solving.
        self.op_list = op_list
        if loss_idx is None:
            # Find the last loss op before the first bwd
            for i, op in enumerate(self.op_list):
                if "loss" in op.name:
                    self.loss_idx = i
                if "bwd" in op.name:
                    break
        else:
            self.loss_idx = loss_idx

        if cluster is not None:
            self.interfaces = cluster.interfaces
            self.list_kdn = cluster.list_kdn
            self.dict_kn = cluster.dict_kn
        else:  # if cluster is not given, get info from op_list
            self.interfaces = interfaces or {
                "inputs_kdn_data": set(),
                "outputs_kdn_data": set(),
                "inputs_kdn_grad": set(),
                "outputs_kdn_grad": set(),
            }
            self.list_kdn = []
            for op in self.op_list:
                if op.is_del:
                    self.list_kdn.append(op.kn)
                else:
                    self.list_kdn.extend(op.kn.users_global)
                    self.list_kdn.extend(op.kn.deps_global)
            self.dict_kn = {
                kdn.name: kdn for kdn in self.list_kdn
            }  # kcn not used
        self.all_interfaces = [
            kdn for inter in self.interfaces.values() for kdn in inter
        ]  # all interface KDN's
        self.interface_names = [kdn.name for kdn in self.all_interfaces]

        if refine:
            self.refine()

        self.op_name_list = [
            (op.name if not op.disabled else "") for op in self.op_list
        ]

        _alive_status = {
            kdn.name: kdn in self.interfaces["inputs_kdn_data"]
            for kdn in self.list_kdn
        }

        self._alive_list = []
        for op in self.op_list:
            if not op.disabled:
                if op.is_del:
                    _alive_status[op.kn.name] = False
                else:
                    for kdn in op.kn.users:
                        if not ("phantoms" in kdn.name and op.fast_forward):
                            _alive_status[kdn.name] = True
            self._alive_list.append(_alive_status.copy())

        L = len(self.op_list)
        self.time = np.zeros(L)
        self.save_mem = np.zeros(L)
        self.overhead = np.zeros(L)

        def _sum_mem(_alive_status_, ignore_list=[]):
            mem = 0
            for k, v in _alive_status_.items():
                if k not in ignore_list and v:
                    d = self.dict_kn[k]
                    mem += d.mem
            return mem

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        for i, (op, _alive_status) in enumerate(
            zip(self.op_list, self._alive_list)
        ):
            self.save_mem[i] = _sum_mem(_alive_status, self.interface_names)
            if (not op.is_del) and (not op.disabled):
                self.time[i] = op.kn.time
                self.overhead[i] = op.kn.overhead

        self.mem = self.save_mem[self.loss_idx]
        self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
        self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])

        self.phantoms = set()
        for kdn in self.list_kdn:
            if (
                self._alive_list[self.loss_idx][kdn.name]
                and not kdn in self.all_interfaces
            ):
                self.phantoms.add(kdn)

        self.fwd_overhead = get_overhead_(
            self.save_mem[: self.loss_idx + 1],
            self.overhead[: self.loss_idx + 1],
        )
        if self.loss_idx < len(self.op_list) - 1:  # bwd is not empty
            self.bwd_overhead = get_overhead_(
                self.save_mem[self.loss_idx + 1 :],
                self.overhead[self.loss_idx + 1 :],
            )

        # names of additional HDNs that are required by BWD
        self.dep_interfaces_data = set()
        for i, op in enumerate(self.op_list[self.loss_idx + 1 :]):
            if (not op.is_del) and (not op.disabled):
                for kdn in op.kn.deps_real:
                    if kdn in self.interfaces["inputs_kdn_data"]:
                        self.dep_interfaces_data.add(self.list_kdn.index(kdn))
                    if kdn in self.interfaces["outputs_kdn_data"]:
                        for kcn in kdn.deps:
                            if (
                                kcn
                                not in self.op_name_list[self.loss_idx + 1 :][
                                    :i
                                ]
                            ):  # if not generated during bwd
                                self.dep_interfaces_data.add(self.list_kdn.index(kdn))

        self.fwd_overhead_correction = []
        self.bwd_overhead_correction = []
        if correct_overhead:
            self.correct_overhead()

    def correct_overhead(self, refine=True):
        # correction terms of overhead, each term represents one step in op_list

        interfaces_status = []
        for kdn in self.interfaces["inputs_kdn_data"]:  # Input of Fwd
            interfaces_status.append((kdn.name, self.loss_idx))  # After fwd
            if self.list_kdn.index(kdn) in self.dep_interfaces_data:
                interfaces_status.append(
                    (kdn.name, len(self.op_list))
                )  # After Bwd
        for kdn in self.interfaces["outputs_kdn_data"]:  # Output of Fwd
            interfaces_status.append((kdn.name, 0))  # Before fwd?
            if self.list_kdn.index(kdn) in self.dep_interfaces_data:
                interfaces_status.append(
                    (kdn.name, len(self.op_list))
                )  # After Bwd
            else:
                interfaces_status.append((kdn.name, -1))  # After Bwd

        for kdn in self.interfaces["outputs_kdn_grad"]:
            interfaces_status.append((kdn.name, len(self.op_list)))  # After Bwd
        for kdn in self.interfaces["inputs_kdn_grad"]:
            interfaces_status.append(
                (kdn.name, self.loss_idx + 1)
            )  # Before Bwd
        self.interfaces_status = interfaces_status
        for i, (op, _alive_status) in enumerate(
            zip(self.op_list, self._alive_list)
        ):
            if i == self.loss_idx:
                continue
            correction_term = {
                "save": self.save_mem[i],
                "overhead": self.overhead[i],
            }
            for kdn_name, index in interfaces_status:
                kdn = self.dict_kn[kdn_name]
                if index == -1:
                    # special case: output_data in BWD without dependency
                    # If outside is _alive, no need to correct;
                    # Otherwise, add kdn to memory
                    if i > self.loss_idx and _alive_status[kdn_name] > 0:
                        correction_term["save"] += kdn.mem
                        correction_term[(self.list_kdn.index(kdn), False)] = -kdn.mem
                    continue

                if (
                    _alive_status[kdn_name] > 0
                    or (index > self.loss_idx) != (i > self.loss_idx)
                    # or not kdn_name
                ):
                    # interfaces_status is useful when:
                    # 1. kdn is not _alive
                    # 2. Fwd to Fwd, Bwd to Bwd
                    continue

                if i >= index:  # if exist before
                    if (  # and not deleted in between
                        kdn_name not in self.op_name_list[index : i + 1]
                    ):
                        correction_term[(self.list_kdn.index(kdn), True)] = -kdn.mem
                    else:
                        correction_term[(self.list_kdn.index(kdn), "always")] = -kdn.mem
                else:  # if exist afterwards
                    if not (kdn in self.interfaces["outputs_kdn_data"]) and (
                        kdn.deps
                        and (
                            list(kdn.deps)[0].name
                            in self.op_name_list[i : index + 1]
                        )
                    ):  # and not generated in between
                        # check if output_data is created after i
                        correction_term[(self.list_kdn.index(kdn), False)] = -kdn.mem
                    elif kdn in self.interfaces["inputs_kdn_data"]:
                        correction_term[(self.list_kdn.index(kdn), False)] = -kdn.mem
                    else:
                        correction_term[(self.list_kdn.index(kdn), "always")] = -kdn.mem

            if (
                i < self.loss_idx
                and correction_term not in self.fwd_overhead_correction
            ):
                self.fwd_overhead_correction.append(correction_term)
            elif (
                i > self.loss_idx
                and correction_term not in self.bwd_overhead_correction
            ):
                self.bwd_overhead_correction.append(correction_term)

        def refine_correction(correction):
            # Some correction terms are not useful because they will not be peak
            min_overhead = max(
                sum(correction_term.values()) for correction_term in correction
            )
            keep_correction = []
            for i, correction_term in enumerate(correction):
                if (
                    correction_term["save"] + correction_term["overhead"]
                    >= min_overhead
                ):
                    # This step will not be the peak even without correction
                    keep_correction.append(correction_term)
            return keep_correction

        if refine:
            self.fwd_overhead_correction = refine_correction(
                self.fwd_overhead_correction
            )
            self.bwd_overhead_correction = refine_correction(
                self.bwd_overhead_correction
            )

    def refine(self):
        for i, op in enumerate(self.op_list):
            if "loss" in op.name:
                op.disabled = True
            if op.is_del:
                # try to delete KDN
                src_i = []  # indices of source KCN's after i
                for kcn in op.kn.deps:
                    if kcn.name in self.op_name_list[i:]:
                        src_i.append(self.op_name_list[i:].index(kcn.name) + i)
                    else:
                        src_i.append(len(self.op_list))
                src_i = src_i or [len(self.op_list)]

                next_used_i = len(self.op_list)  # the next index to use KDN
                for kcn in op.kn.users_real:
                    if kcn.name in self.op_name_list[i:]:
                        next_used_i = min(
                            self.op_name_list[i:].index(kcn.name) + i,
                            next_used_i,
                        )

                if max(src_i) > next_used_i:  # try to use before regenerate
                    op.disabled = True


# def hg_to_cluster(hg: H_graph, kg: K_graph):
#     interfaces = dict()
#     interfaces["inputs_kdn_data"] = set(hdn.kdn for hdn in hg.inputs_hdn_data)
#     interfaces["outputs_kdn_data"] = set(hdn.kdn for hdn in hg.outputs_hdn_data)
#     interfaces["inputs_kdn_grad"] = set(hdn.kdn for hdn in hg.inputs_hdn_grad)
#     interfaces["outputs_kdn_grad"] = set(hdn.kdn for hdn in hg.outputs_hdn_grad)
#     # interfaces["all"] = hg.interfaces
#     list_kcn = []
#     loss_kcn = K_C_node("loss")
#     for kdn in interfaces["outputs_kdn_data"]:
#         loss_kcn.deps_real.add(kdn)
#     for kdn in interfaces["outputs_kdn_grad"]:
#         loss_kcn.users.add(kdn)
#     for kcn in kg.list_kcn:
#         if kcn in hg.all_kcn_inside or kcn.main_target in hg.name:
#             # bottom level hg has no kcn inside
#             list_kcn.append(kcn)
#         if kcn == kg.loss_kcn:
#             loss_idx = len(list_kcn)
#             list_kcn.append(loss_kcn)
#     cluster = Cluster(list_kcn, interfaces, loss_idx)
#     return cluster
