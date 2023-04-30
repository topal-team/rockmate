from rkgb.utils import *
from rkgb.Ptools import P_graph, P_node
from rkgb.Ktools import K_graph, K_C_node, K_D_node
from rkgb.Htools import *
from collections import namedtuple


class Cluster:
    def __init__(self, list_kcn, interfaces, loss_idx):
        self.list_kcn = list_kcn
        self.list_kdn = set()
        for kcn in list_kcn:
            self.list_kdn.update(kcn.users_global)
            self.list_kdn.update(kcn.deps_global)

        self.dict_kn = {
            **{kdn.name: kdn for kdn in self.list_kdn},
            **{kcn.name: kcn for kcn in list_kcn},
        }
        self.interfaces = interfaces
        self.loss_idx = loss_idx

        # self.interfaces = dict()
        # self.interfaces["inputs_kdn_data"] = list_kcn[0].deps_real
        # self.interfaces["outputs_kdn_data"] = ...
        # self.interfaces["inputs_kdn_grad"] = list_kcn[-1].users
        # self.interfaces["outputs_kdn_grad"] = ...


def hg_to_cluster(hg: H_graph, kg: K_graph):
    interfaces = dict()
    interfaces["inputs_kdn_data"] = set(hdn.kdn for hdn in hg.inputs_hdn_data)
    interfaces["outputs_kdn_data"] = set(hdn.kdn for hdn in hg.outputs_hdn_data)
    interfaces["inputs_kdn_grad"] = set(hdn.kdn for hdn in hg.inputs_hdn_grad)
    interfaces["outputs_kdn_grad"] = set(hdn.kdn for hdn in hg.outputs_hdn_grad)
    # interfaces["all"] = hg.interfaces
    list_kcn = []
    loss_kcn = K_C_node("loss")
    for kdn in interfaces["outputs_kdn_data"]:
        loss_kcn.deps_real.add(kdn)
    for kdn in interfaces["outputs_kdn_grad"]:
        loss_kcn.users.add(kdn)
    for kcn in kg.list_kcn:
        if kcn in hg.all_kcn_inside or kcn.main_target in hg.name:
            # bottom level hg has no kcn inside
            list_kcn.append(kcn)
        if kcn == kg.loss_kcn:
            loss_idx = len(list_kcn)
            list_kcn.append(loss_kcn)
    cluster = Cluster(list_kcn, interfaces, loss_idx)
    return cluster


"""
Ptools: clusters, A_clusters which keeps all the schedules
Htools: Hgraph with HCN/HDN, each HCN could lead to one cluster
OpSchedule: always anonymized. 
Init: adding save all schedule to every A_cluster.
HILP: given one Hgraph, solve based on the schedules of clusters of HCN

"""

# class Op:
#     def __init__(self, kn, fast_forward=False, disabled=False, detach=True):
#         self.kn = kn
#         self.name = kn.name
#         self.fast_forward = fast_forward
#         self.disabled = disabled
#         self.detach = detach
#         self.is_del = isinstance(kn, K_D_node)


class OpSchedule:
    def __init__(self, op_list, loss_idx, interfaces, refine=True):
        self.op_list = op_list
        self.loss_idx = loss_idx
        self.interfaces = interfaces
        self.interface_kdns = [
            kdn for inter in interfaces.values() for kdn in inter
        ]
        self.op_name_list = [op.name for op in self.op_list]

        # self.fwd_time = sum(
        #     op.kn.time for op in op_list[: loss_idx + 1] if not op.disabled
        # )
        # self.bwd_time = sum(
        #     op.kn.time for op in op_list[loss_idx + 1 :] if not op.disabled
        # )
        if refine:
            self.refine()

        all_kdns = set()
        for op in self.op_list:
            if op.is_del:
                all_kdns.add(op.kn)
            else:
                all_kdns.update(op.kn.users)
                all_kdns.update(op.kn.deps_global)
        all_kdns.update(
            interfaces["inputs_kdn_grad"]
        )  # inputs grad are not in users

        self.dict_kdn = {kdn.name: kdn for kdn in all_kdns}

        alive_status = {kdn.name: False for kdn in all_kdns}
        for kdn in self.interfaces["inputs_kdn_data"]:
            alive_status[kdn.name] = True

        self.alive_list = []
        # overhead_list = []
        for op in self.op_list:
            if not op.disabled:
                if op.is_del:
                    alive_status[op.kn.name] = False
                else:
                    for kdn in op.kn.users:
                        if not ("phantoms" in kdn.name and op.fast_forward):
                            alive_status[kdn.name] = True
            self.alive_list.append(alive_status.copy())
            # overhead_list.append(
            #     (0 if op.is_del or op.disabled else op.kn.overhead)
            # )

        L = len(self.op_list)
        self.time = np.zeros(L)
        self.save_mem = np.zeros(L)
        self.overhead = np.zeros(L)

        def _sum_mem(alive_status_, ignore_list=[]):
            mem = 0
            for k, v in alive_status_.items():
                if k not in ignore_list and v:
                    d = self.dict_kdn[k]
                    mem += d.mem
            return mem

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        self.interface_names = [kdn.name for kdn in self.interface_kdns]

        for i, (op, alive_status) in enumerate(
            zip(self.op_list, self.alive_list)
        ):
            self.save_mem[i] = _sum_mem(alive_status, self.interface_names)
            if (not op.is_del) and (not op.disabled):
                self.time[i] = op.kn.time
                self.overhead[i] = op.kn.overhead

        self.mem = self.save_mem[self.loss_idx]
        self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
        self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])

        self.phantoms = set()
        for kdn in all_kdns:
            if (
                self.alive_list[self.loss_idx][kdn.name]
                and not kdn in self.interface_kdns
            ):
                self.phantoms.add(kdn)

        self.fwd_overhead = get_overhead_(
            self.save_mem[: self.loss_idx + 1],
            self.overhead[: self.loss_idx + 1],
        )
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
                        self.dep_interfaces_data.add(kdn.name)
                    if kdn in self.interfaces["outputs_kdn_data"]:
                        for kcn in kdn.deps:
                            if (
                                kcn
                                not in self.op_name_list[self.loss_idx + 1 :][
                                    :i
                                ]
                            ):  # if not generated during bwd
                                self.dep_interfaces_data.add(kdn.name)

        self.fwd_overhead_correction = []
        self.bwd_overhead_correction = []
        self.correct_overhead()

    def correct_overhead(self, refine=True):
        # correction terms of overhead, each term represents one step in op_list

        interfaces_status = []
        for kdn in self.interfaces["inputs_kdn_data"]:  # Input of Fwd
            interfaces_status.append((kdn.name, self.loss_idx))  # After fwd
            if kdn.name in self.dep_interfaces_data:
                interfaces_status.append(
                    (kdn.name, len(self.op_list))
                )  # After Bwd
        for kdn in self.interfaces["outputs_kdn_data"]:  # Output of Fwd
            interfaces_status.append((kdn.name, 0))  # Before fwd?
            if kdn.name in self.dep_interfaces_data:
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
        for i, (op, alive_status) in enumerate(
            zip(self.op_list, self.alive_list)
        ):
            if i == self.loss_idx:
                continue
            correction_term = {
                "save": self.save_mem[i],
                "overhead": self.overhead[i],
            }
            for (kdn_name, index) in interfaces_status:
                kdn = self.dict_kdn[kdn_name]
                if index == -1:
                    # special case: output_data in BWD without dependency
                    # If outside is alive, no need to correct;
                    # Otherwise, add kdn to memory
                    if i > self.loss_idx and alive_status[kdn_name] > 0:
                        correction_term["save"] += kdn.mem
                        correction_term[(kdn.name, False)] = -kdn.mem
                    continue

                if (
                    alive_status[kdn_name] > 0
                    or (index > self.loss_idx) != (i > self.loss_idx)
                    # or not kdn_name
                ):
                    # interfaces_status is useful when:
                    # 1. kdn is not alive
                    # 2. Fwd to Fwd, Bwd to Bwd
                    continue

                if i >= index:  # if exist before
                    if (  # and not deleted in between
                        kdn_name not in self.op_name_list[index : i + 1]
                    ):
                        correction_term[(kdn.name, True)] = -kdn.mem
                    else:
                        correction_term[(kdn.name, "always")] = -kdn.mem
                else:  # if exist afterwards
                    if not (kdn in self.interfaces["outputs_kdn_data"]) and (
                        kdn.deps
                        and (
                            list(kdn.deps)[0].name
                            in self.op_name_list[i : index + 1]
                        )
                    ):  # and not generated in between
                        # check if output_data is created after i
                        correction_term[(kdn.name, False)] = -kdn.mem
                    else:
                        correction_term[(kdn.name, "always")] = -kdn.mem

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


def get_autograd_sched_rec(hgraph: H_graph, kg: K_graph):

    for hcn in hgraph.list_hcn:
        if hcn.is_fwd and hcn.sub_graph is not None:
            sub_g = hcn.sub_graph
            if not sub_g.list_sched:
                sub_opt = get_autograd_sched_rec(sub_g, kg)
                sub_g.add_sched(sub_opt)
    cluster = hg_to_cluster(hgraph, kg)
    return get_autograd_sched(cluster)


def get_autograd_sched(
    cluster: Cluster, protect_names=["sources data", "sources grad"]
):
    list_kcn = cluster.list_kcn

    def _can_del(i, kdn):
        if kdn.name in protect_names:
            return False
        for kcn in cluster.list_kcn[i + 1 :]:
            if kdn in kcn.deps_real:
                return False
        # for kcn in kdn.users_real:
        #     if cluster.list_kcn.index(kcn) > i:
        #         return False
        return True

    if len(list_kcn) == 3:  # leaf nodes: fwd+bwd pair & loss
        op_list = [
            Op(list_kcn[0], detach=True),
            Op(list_kcn[1], disabled=True),
            Op(list_kcn[2], detach=True),
        ]
        loss_idx = 1
        # interfaces = dict()
        # interfaces["inputs_kdn_data"] = list_kcn[0].deps_real
        # interfaces["outputs_kdn_data"] = set(
        #     kdn for kdn in list_kcn[0].users if "phantoms" not in kdn.name
        # )
        # interfaces["outputs_kdn_grad"] = set(
        #     kdn for kdn in list_kcn[1].deps_real if "phantoms" not in kdn.name
        # )
        # interfaces["inputs_kdn_grad"] = list_kcn[1].users

    else:
        op_list = []
        alive_list = []
        alive_status = {}
        for kdn in cluster.list_kdn:
            # if hdn not in cluster.interfaces:
            alive_status[kdn.name] = (
                1 if (kdn in cluster.interfaces["inputs_kdn_data"]) else 0
            )

        for i, kcn in enumerate(cluster.list_kcn):
            if i == cluster.loss_idx:
                loss_idx = len(op_list)
            for kdn in kcn.users:
                alive_status[kdn.name] = 1
            op_list.append(Op(kcn, detach=True, disabled=("loss" in kcn.name)))
            alive_list.append(alive_status.copy())

            for kdn_name, alive in alive_status.items():

                kdn = cluster.dict_kn[kdn_name]
                if alive and _can_del(i, kdn):
                    op_list.append(Op(kdn))
                    alive_status[kdn_name] = 0
                    alive_list.append(alive_status.copy())

    return OpSchedule(op_list, loss_idx, cluster.interfaces)
