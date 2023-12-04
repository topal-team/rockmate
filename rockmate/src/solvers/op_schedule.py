from rkgb.utils import *
from rkgb.Ptools import P_graph, P_node
from rkgb.Ktools import K_graph, K_C_node, K_D_node
from rkgb.Htools import *
from collections import namedtuple
from copy import deepcopy
import warnings


class Allocation:
    def __init__(self, name, alloc_type="", mem=0, info=dict(), dtype=torch.float32, size=None):
        """
        Allocation type should be in activation/paramters/buffer
        """
        self.name = name
        self.allo_type = alloc_type
        self.mem = mem
        self.info = info
        self.dtype = dtype
        self.itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 4
        if size:
            self.size = round(size)
            self.mem = size*self.itemsize
        else:
            self.size = round(self.mem/self.itemsize)#element size

    def __repr__(self):
        return self.name


class Activation(Allocation):
    def __init__(self, kdn, dtype=torch.float32):
        super().__init__(
            name=kdn.name,
            alloc_type="Activation",
            mem=kdn.mem,
            info=kdn.info,
            dtype=dtype,
        )
        self.kdn = kdn

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "kdn":  # do not deepcopy kn
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


class Parameter(Allocation):
    def __init__(self, kdn, dtype=torch.float32):
        super().__init__(
            name=kdn.name,
            alloc_type="Parameter",
            mem=kdn.mem,
            info=kdn.info,
            dtype=dtype,
        )
        self.kdn = kdn


class Buffer(Allocation):
    def __init__(self, name, mem=0, info=dict(), dtype=torch.float32, size=None):
        super().__init__(
            name=name, alloc_type="Buffer", mem=mem, info=info, dtype=dtype, size=size
        )
        self.dtype = dtype


class Op:
    def __init__(self, name, disabled=False):
        """
        Op type should be in Compute/Delete/Mapping/Allocate/Offload/Prefetch
        Compute/Delete/Mapping/Allocate happens in the main stream
        """
        self.name = name
        self.disabled = disabled
        self.overhead = 0

    def __repr__(self):
        return self.name


class ComputeOp(Op):
    def __init__(self, kcn, fast_forward=False, disabled=False, detach=True):
        super().__init__(kcn.name, disabled)
        self.kcn = kcn
        self.fast_forward = fast_forward
        self.detach = detach
        self.target = kcn
        self.overhead = kcn.overhead

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "kcn" or k=="target":  # do not deepcopy kn
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result

    def __repr__(self):
        return self.kcn.name


class DeleteOp(Op):
    def __init__(self, alloc: Allocation, disabled=False):
        super().__init__("Delete_" + alloc.name, disabled)
        self.target = alloc

    def __repr__(self):
        return "Delete_" + self.target.name


class MappingOp(Op):
    """
    The memory allocation of sources will be map to targets in buffer.
    the time of running this op is very short, but there is memory overhead.
    """

    def __init__(
        self,
        name: str,
        sources: list,
        targets: list,
        indices: list = None,
        disabled=False,
        copy=False
    ):
        super().__init__("Mapping_" + name, disabled)
        self.sources = sources
        self.targets = targets
        self.indices = indices
        self.copy = copy
        self.overhead = sum(alloc.mem for alloc in sources)  # TODO: update when needed


class AllocateOp(Op):
    def __init__(self, alloc: Allocation, disabled=False):
        super().__init__("Allocate_" + alloc.name, disabled)
        self.target = alloc


class OffloadOp(Op):
    def __init__(
        self,
        alloc: Allocation,
        indices: tuple = (0,-1),
        before: Op = None,
        after: Op = None,
        disabled: bool = False,
    ):
        super().__init__("Offload_" + alloc.name, disabled)
        self.target = alloc
        self.indices = indices
        self.disabled = disabled
        self.before = before
        self.after = after

    def __repr__(self):
        return "Disabled" * self.disabled + f"Offload_{self.target}"


class PrefetchOp(Op):
    def __init__(
        self,
        alloc: Allocation,
        indices: tuple = (0,-1),
        before: Op = None,
        after: Op = None,
        disabled: bool = False,
    ):
        super().__init__("Prefetch_" + alloc.name, disabled)
        self.target = alloc
        self.indices = indices
        self.disabled = disabled
        self.before = before
        self.after = after

    def __repr__(self):
        return "Disabled" * self.disabled + f"Prefetch_{self.target}"


class OpSchedule:
    solver = None

    def __init__(
        self,
        op_list,
        prf_list=[],
        ofl_list=[],
        loss_idx=None,
        cluster=None,
        interfaces=None,
        refine=True,
        correct_overhead=True,
        keep_alive_list=False,
        with_parameters=False,
        init_alive_status: dict = {},
        init_op_list: list = []
    ):
        """
        Key role of OpSchedule: taking op_list, analyzing memory stats,
        keeping info for further solving.
        New role: greedy algorithm to rearrange the prefetch/offload ops
        Parameters are read from the cluster
        """
        self.op_list = op_list
        self.init_op_list = init_op_list
        self.prf_list = prf_list
        self.ofl_list = ofl_list
        self.with_parameters = with_parameters
        if loss_idx is None:
            # Find the last loss op before the first bwd
            for i, op in enumerate(self.op_list):
                if "loss" in op.name:
                    self.loss_idx = i
                if "bwd" in op.name:
                    break
        else:
            self.loss_idx = loss_idx

        if cluster is None:
            warnings.warn("Cluster should be provided to create op_sched")
            self.prepare_allocation_from_op_list(interfaces)
        else:
            self.interfaces = cluster.interfaces
            self.list_alloc = [Activation(kdn) for kdn in cluster.list_kdn]
            self.list_kdn = cluster.list_kdn
            if with_parameters:
                self.list_alloc.extend(
                    [Parameter(kdn) for kdn in cluster.list_kdn_parameters]
                )
                self.list_alloc.extend(self.create_buffer_list())
        self.dict_alloc = {alloc.name: alloc for alloc in self.list_alloc}
        self.all_interfaces = [
            kdn for inter in self.interfaces.values() for kdn in inter
        ]  # all interface KDN's
        self.interface_names = [kdn.name for kdn in self.all_interfaces]

        self.op_name_list = [
            (op.name if not op.disabled else "") for op in self.op_list
        ]

        if refine:
            self.refine()

        alive_list = self.create_alive_list(init_status=init_alive_status)

        L = len(self.op_list)
        self.time = np.zeros(L)
        self.save_mem = np.zeros(L)
        self.overhead = np.zeros(L)

        def _sum_mem(alive_status_, ignore_list=[]):
            mem = 0
            for k, v in alive_status_.items():
                if k not in ignore_list and v:
                    d = self.dict_alloc[k]
                    mem += d.mem
            return mem

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        for i, (op, alive_status) in enumerate(zip(self.op_list, alive_list)):
            self.save_mem[i] = _sum_mem(alive_status, self.interface_names)
            if op.disabled:
                continue
            if isinstance(op, ComputeOp):
                self.time[i] = op.kcn.time
                self.overhead[i] = op.overhead

        self.mem = self.save_mem[self.loss_idx]
        self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
        self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])

        self.phantoms = set()
        for kdn in self.list_kdn:
            if alive_list[self.loss_idx][kdn.name] and not kdn in self.all_interfaces:
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
            if op.disabled:
                continue
            if isinstance(op, ComputeOp):
                for kdn in op.kcn.deps_real:
                    if kdn in self.interfaces["inputs_kdn_data"]:
                        self.dep_interfaces_data.add(self.list_kdn.index(kdn))
                    if kdn in self.interfaces["outputs_kdn_data"]:
                        for kcn in kdn.deps:
                            if (
                                kcn not in self.op_name_list[self.loss_idx + 1 :][:i]
                            ):  # if not generated during bwd
                                self.dep_interfaces_data.add(self.list_kdn.index(kdn))

        self.fwd_overhead_correction = []
        self.bwd_overhead_correction = []
        if correct_overhead:
            self.correct_overhead(alive_list)

        if keep_alive_list:
            self.alive_list = alive_list
        else:
            self.alive_list = []

    def create_buffer_list(self):
        buffer_set = set()
        for op in self.op_list:
            if isinstance(op, MappingOp) and len(op.targets)==1:# merge mapping
                for target in op.targets:
                    buffer_set.add(target)
            elif isinstance(op, AllocateOp):
                buffer_set.add(op.target)

        return list(buffer_set)

    def prepare_allocation_from_op_list(self, interfaces):
        self.interfaces = interfaces or {
            "inputs_kdn_data": set(),
            "outputs_kdn_data": set(),
            "inputs_kdn_grad": set(),
            "outputs_kdn_grad": set(),
        }
        self.list_kdn = []
        for op in self.op_list:
            if isinstance(op, DeleteOp):
                self.list_kdn.append(op.target)
            elif isinstance(op, ComputeOp):
                self.list_kdn.extend([kdn for kdn in op.kcn.users_global])
                self.list_kdn.extend([kdn for kdn in op.kcn.deps_global])
        self.list_alloc = self.list_kdn

    def create_alive_list(self, init_status={}):
        alive_status = {alloc.name: False for alloc in self.list_alloc}
        for k, v in init_status.items():
            alive_status[k] = v
        for kdn in self.interfaces["inputs_kdn_data"]:
            alive_status[kdn.name] = True  # kdn share the name as alloc
        # TODO: add init alive for parameters

        alive_list = []
        for op in self.op_list:
            if op.disabled:
                alive_list.append(alive_status.copy())
                continue
            if isinstance(op, DeleteOp):
                alive_status[op.target.name] = False
            elif isinstance(op, ComputeOp):
                # compute op should not be disabled except loss which is useful for alive status
                for kdn in op.kcn.users:
                    if not ("phantoms" in kdn.name and op.fast_forward):
                        alive_status[kdn.name] = True
            elif isinstance(op, AllocateOp):
                alive_status[op.target.name] = True
            alive_list.append(alive_status.copy())
        return alive_list

    def correct_overhead(self, alive_list, refine=True):
        # correction terms of overhead, each term represents one step in op_list

        interfaces_status = []
        for kdn in self.interfaces["inputs_kdn_data"]:  # Input of Fwd
            interfaces_status.append((kdn.name, self.loss_idx))  # After fwd
            if self.list_kdn.index(kdn) in self.dep_interfaces_data:
                interfaces_status.append((kdn.name, len(self.op_list)))  # After Bwd
        for kdn in self.interfaces["outputs_kdn_data"]:  # Output of Fwd
            interfaces_status.append((kdn.name, 0))  # Before fwd?
            if self.list_kdn.index(kdn) in self.dep_interfaces_data:
                interfaces_status.append((kdn.name, len(self.op_list)))  # After Bwd
            else:
                interfaces_status.append((kdn.name, -1))  # After Bwd

        for kdn in self.interfaces["outputs_kdn_grad"]:
            interfaces_status.append((kdn.name, len(self.op_list)))  # After Bwd
        for kdn in self.interfaces["inputs_kdn_grad"]:
            interfaces_status.append((kdn.name, self.loss_idx + 1))  # Before Bwd
        self.interfaces_status = interfaces_status
        for i, (op, alive_status) in enumerate(zip(self.op_list, alive_list)):
            if i == self.loss_idx:
                continue
            correction_term = {
                "save": self.save_mem[i],
                "overhead": self.overhead[i],
            }
            for kdn_name, index in interfaces_status:
                kdn = self.dict_alloc[kdn_name].kdn
                if index == -1:
                    # special case: output_data in BWD without dependency
                    # If outside is alive, no need to correct;
                    # Otherwise, add kdn to memory
                    if i > self.loss_idx and alive_status[kdn_name] > 0:
                        correction_term["save"] += kdn.mem
                        correction_term[(self.list_kdn.index(kdn), False)] = -kdn.mem
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
                        correction_term[(self.list_kdn.index(kdn), True)] = -kdn.mem
                    else:
                        correction_term[(self.list_kdn.index(kdn), "always")] = -kdn.mem
                else:  # if exist afterwards
                    if not (kdn in self.interfaces["outputs_kdn_data"]) and (
                        kdn.deps
                        and (list(kdn.deps)[0].name in self.op_name_list[i : index + 1])
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
            if self.fwd_overhead_correction:
                self.fwd_overhead_correction = refine_correction(
                    self.fwd_overhead_correction
                )
            if self.bwd_overhead_correction:
                self.bwd_overhead_correction = refine_correction(
                    self.bwd_overhead_correction
                )

    def refine(self):
        for i, op in enumerate(self.op_list):
            if "loss" in op.name:
                op.disabled = True
            if isinstance(op, DeleteOp):
                if isinstance(op.target, Activation):
                    # try to delete KDN
                    src_i = []  # indices of source KCN's after i
                    for kcn in op.target.kdn.deps:
                        if kcn.name in self.op_name_list[i:]:
                            src_i.append(self.op_name_list[i:].index(kcn.name) + i)
                        else:
                            src_i.append(len(self.op_list))
                    src_i = src_i or [len(self.op_list)]

                    next_used_i = len(self.op_list)  # the next index to use KDN
                    for kcn in op.target.kdn.users_real:
                        if kcn.name in self.op_name_list[i:]:
                            next_used_i = min(
                                self.op_name_list[i:].index(kcn.name) + i,
                                next_used_i,
                            )

                    if max(src_i) > next_used_i:  # try to use before regenerate
                        op.disabled = True

                elif isinstance(op.target, Parameter):
                    # TODO: disabled wrong deletion of parameter
                    pass

        self.op_name_list = [
            (op.name if not op.disabled else "") for op in self.op_list
        ]

    def __repr__(self):
        return (
            f"OpSchedule takes {sum(self.time):.2f}ms and cost {self.mem//1024**2} MiB"
        )

    @property
    def peak_mem(self):
        return max(self.save_mem + self.overhead)

    @property
    def simulation_overhead(self):
        all_compute = set(
            op for op in self.op_list if (not op.disabled and isinstance(op, ComputeOp))
        )
        return (
            sum(op.kcn.time for op in all_compute)
            / sum(op.kcn.time for op in all_compute)
            - 1
        )


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
