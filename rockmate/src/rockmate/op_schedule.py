from rkgb.core.base import Node
from rkgb.core.backward import ComputationNode, AllocationNode
from rkgb.core.hierarchical import HierarchicalCluster
from collections import namedtuple
from copy import deepcopy
from typing import List
import warnings
import numpy as np
import torch
from rkgb.lowlevel.ast_add_on import make_str_list_assign


class Allocation:
    def __init__(
        self,
        target_name,
        alloc_type="",
        mem=0,
        info=dict(),
        dtype=torch.float32,
        size=None,
    ):
        """
        Allocation type should be in activation/parameters/buffer
        """
        self.target_name = target_name
        self._alloc_type = alloc_type
        self.mem = mem
        self.info = info
        self.dtype = dtype
        self.itemsize = dtype.itemsize if hasattr(dtype, "itemsize") else 4
        if size:
            self.size = round(size)
            self.mem = size * self.itemsize
        else:
            self.size = round(self.mem / self.itemsize)  # element size

    @property
    def name(self):
        return f"{self.target_name} {self.alloc_type}"

    @property
    def alloc_type(self):
        return self._alloc_type

    def __repr__(self):
        return self.name

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
            if isinstance(v, Node):
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result


class Activation(Allocation):
    def __init__(self, anode: AllocationNode, dtype=torch.float32):
        super().__init__(
            target_name=anode.main_target,
            alloc_type=anode.allocation_type,
            mem=anode.mem,
            info=anode.info,
            dtype=dtype,
        )
        self.kdn = anode
        self.anode = anode


class Parameter(Allocation):
    """
    There is not paramter grad/optim_states nodes from rkgb.
    Need to handle the dependency manually for Allocation.
    """

    def __init__(
        self, pnode, is_grad=False, is_optim_states=False, optimizer_states_factor=2
    ):
        self.pnode = pnode
        self.is_grad = is_grad
        self.is_optim_states = is_optim_states
        self.optimizer_states_factor = optimizer_states_factor
        if not self.is_optim_states:
            mem = pnode.mem
        else:
            mem = pnode.mem * self.optimizer_states_factor

        super().__init__(
            target_name=pnode.param_name,
            alloc_type="param",  # + "_grad"*grad+"_optim_states"*is_optim_states,
            mem=mem,
            info=pnode.info,
            dtype=pnode.info.dtype,
        )

    @property
    def param_name(self):
        return self.pnode.param_name

    @property
    def alloc_type(self):
        return "param" + "_grad" * self.is_grad + "_optim_states" * self.is_optim_states


class Buffer(Allocation):
    def __init__(self, name, mem=0, info=dict(), dtype=torch.float32, size=None):
        super().__init__(
            name=name, alloc_type="Buffer", mem=mem, info=info, dtype=dtype, size=size
        )
        self.dtype = dtype


class Op:
    def __init__(self, target_name, time=0, disabled=False, overhead=0):
        """
        Op type should be in Compute/Delete/Mapping/Allocate/Offload/Prefetch
        Compute/Delete/Mapping/Allocate happens in the main stream
        """
        self._target_name = target_name
        self.disabled = disabled
        self.overhead = overhead
        self._time = time
        self.fct_list = []
        self._op_type = type(self).__name__
        self.record_event = False
        self.wait_events = []

    def add_fct(self, f):
        self.fct_list.append(f)

    def __repr__(self):
        return f"{'Disabled_'*self.disabled}{self.name}"

    @property
    def target_name(self):
        return self._target_name

    @property
    def time(self):
        return self._time

    @property
    def name(self):
        return f"{self.op_type}({self.target_name})"

    @property
    def op_type(self):
        return self._op_type

    @op_type.setter
    def op_type(self, value):
        self._op_type = value

    def __call__(self):
        for f in self.fct_list:
            f()

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
            if isinstance(v, Node):
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))

        return result

    def __eq__(self, op):
        if not type(self) == type(op):
            return False
        return op.name == self.name and self.disabled == op.disabled

    def __hash__(self):
        return id(self)

    def __getstate__(self):
        # fct_list should not be pickled
        state = self.__dict__.copy()
        state["fct_list"] = []
        return state


class SynchronizeOp(Op):
    def __init__(self, name="", disabled=False, stream=None):
        super().__init__(name, disabled=disabled)
        self.op_type = "Synchronize"
        self.stream = stream


class ComputeOp(Op):
    def __init__(
        self, cnode: ComputationNode, fast_forward=False, disabled=False, detach=True
    ):
        super().__init__(cnode.name, disabled=disabled)
        self.fast_forward = fast_forward
        self.detach = detach
        self.target: ComputationNode = cnode
        self.overhead = cnode.mem_overhead
        self.pos_info = {}  # positional information to be filled before compiling
        self.op_type = "Compute"
        self.time_mutiplier = 1# for dynamic batch

    @property
    def time(self):
        # cnode can be replaced during translation
        return self.target.time * self.time_mutiplier if self.target.time is not None else 0

    @property
    def target_name(self):
        return self.target.name


class DeleteOp(Op):
    def __init__(self, alloc: Allocation, disabled=False):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.op_type = f"Delete"

    @property
    def target_name(self):
        return self.target.name


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
        copy=False,
    ):
        super().__init__("Mapping_" + name, disabled=disabled)
        self.sources = sources
        self.targets = targets
        self.indices = indices
        self.copy = copy
        self.overhead = sum(alloc.mem for alloc in sources)  # TODO: update when needed


class AllocateOp(Op):
    def __init__(self, alloc: Allocation, disabled=False, is_optim_states=False):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.op_type = f"Allocate_{alloc.alloc_type}"
        self.is_optim_states = is_optim_states

    @property
    def target_name(self):
        return self.target.name


class OffloadOp(Op):
    def __init__(
        self,
        alloc: Allocation,
        indices: tuple = (0, None),
        disabled: bool = False,
        # grad: bool = False,
        # is_optim_states: bool = False,
        time: float = 0,
    ):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.indices = indices
        self.disabled = disabled
        # self.grad = grad
        # self.is_optim_states = is_optim_states
        self._time = time
        self.op_type = f"Offload"

    @property
    def target_name(self):
        return self.target.name


class PrefetchOp(Op):
    def __init__(
        self,
        alloc: Allocation,
        indices: tuple = (0, None),
        disabled: bool = False,
        # is_optim_states: bool = False,
        time: float = 0,
    ):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.indices = indices
        self.disabled = disabled
        # self.is_optim_states = is_optim_states
        self._time = time
        self.op_type = f"Prefetch"

    @property
    def target_name(self):
        return self.target.name


class OptimizeOp(Op):
    def __init__(
        self,
        list_params,
        cpu=False,
        alloc=None,
        disabled=False,
        time=0,
        overhead=0,
    ):
        self.list_params = list_params
        super().__init__(self.target_name, disabled=disabled, overhead=overhead)
        self.target = alloc or None
        self._time = time
        self.op_type = f"Optimize_{'cpu'*cpu}"
        self.is_cpu = cpu

    @property
    def target_name(self):
        return f"{','.join(self.list_params)}"


class ExecCodeOp(Op):
    def __init__(self, name, code, time=0, disabled=False, overhead=0):
        super().__init__(name, time, disabled, overhead)
        self.code = code


class PrepareOp(Op):
    # Prepare the placeholders for parameters before the iteration
    def __init__(
        self,
        alloc: Allocation,
        device: str = "cpu",
        cpu_placeholder=True,
        cpu_optimize=False,
        pin_memory=True,
        disabled: bool = False,
    ):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.device = device
        self.cpu_placeholder = cpu_placeholder
        self.disabled = disabled
        self.op_type = f"Prepare_{alloc.alloc_type}"
        self.cpu_optimize = cpu_optimize
        self.pin_memory = pin_memory


class OpSchedule:
    solver = None

    def __init__(
        self,
        op_list: List[Op],
        cluster: HierarchicalCluster,
        loss_idx=None,
        with_parameters=False,
        init_alive_status: dict = {},
        init_op_list: list = [],
        restore_op_list: list = [],
        optimizer_states_factor=2,
        correct_overhead=True,
    ):
        """
        OpSchedule contains the operation list and automatically
        compute the alive status given the HierarchicalCluster.
        """
        self.op_list = op_list
        self.cluster = cluster
        self.loss_idx = loss_idx
        self.init_alive_status = init_alive_status
        self.init_op_list = init_op_list  # Place to prepare items in storage
        self.restore_op_list = restore_op_list
        self.with_parameters = with_parameters
        self.optimizer_states_factor = optimizer_states_factor
        self.interfaces = cluster.interfaces
        self.correct_overhead = correct_overhead
        self.with_optimization = any(isinstance(op, OptimizeOp) for op in op_list[::-1])

        self.create_list_alloc(cluster)
        self.get_sched_info()  # get the schedule information for higher level solving

    def __getstate__(self):
        # simulator should not be pickled
        state = self.__dict__.copy()
        state["simulator"] = None
        return state

    def get_occurrences(self):
        self.occurrences = dict()
        for i, op in enumerate(self.op_list):
            if op.name in self.occurrences:
                self.occurrences[op.name].append(i)
            else:
                self.occurrences[op.name] = [i]

    def is_occurred(self, op_name, i, next_i=None):
        if op_name not in self.occurrences:
            return False
        if next_i is None:
            return i in self.occurrences[op_name]
        else:
            return any(i <= i_ <= next_i for i_ in self.occurrences[op_name])

    def simulate_update(self, Simulator, refine_optimize=False):
        """
        Args:
            Simulator: Class defined to simulate op_sched
        """
        self.simulator = Simulator(self)
        self.simulator.refine()  # assume init_op_list remains the same
        if refine_optimize:
            self.simulator.refine_optimize()
        self.op_list = self.simulator.op_list
        self.steps = self.simulator.steps
        self.loss_step = self.simulator.loss_step
        self.loss_idx = self.simulator.loss_idx
        self.get_sched_info()

    def _sum_mem(self, alive_status_, ignore_list=[]):
        return sum(self.dict_alloc[k].mem
                   for k, v in alive_status_.items()
                   if k not in ignore_list and v)

    def get_memory(self, alive_list):
        L = len(self.op_list)
        self.time = np.zeros(L)
        self.save_mem = np.zeros(L)
        self.save_mem_with_interfaces = np.zeros(L)
        self.overhead = np.zeros(L)

        for i, (op, alive_status) in enumerate(zip(self.op_list, alive_list)):
            self.save_mem[i] = self._sum_mem(alive_status, ignore_list=self.interface_names)
            self.save_mem_with_interfaces[i] = self._sum_mem(alive_status)
            if op.disabled:
                continue
            self.time[i] = (
                op.time
                if not (isinstance(op, OffloadOp) or isinstance(op, PrefetchOp))
                else 0
            )
            self.overhead[i] = op.overhead

    def get_sched_info(self):
        """
        To get schedule information for higher level solving:
        - .mem: saved activation memory at the loss step, without interface activations
        - .fwd_time: time of forward operations
        - .bwd_time: time of backward operations
        - .fwd_overhead: during forward, the excess over the ending memory from the peak
        - .bwd_overhead: during backward, the excess over the ending memory from the peak
        - .phantoms: saved anodes at the loss step
        - .deps_interfaces_data: bwd requires interface data nodes which should be kept from fwd
        """
        self.get_occurrences()
        alive_list = self.create_alive_list()
        self.alive_list = alive_list
        self.get_memory(alive_list)

        self.mem = self.save_mem[self.loss_idx]
        self.peak_mem = max(self.save_mem + self.overhead)
        self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
        self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        self.fwd_overhead = get_overhead_(
            self.save_mem[: self.loss_idx + 1],
            self.overhead[: self.loss_idx + 1],
        )
        if self.loss_idx < len(self.op_list) - 1:  # bwd is not empty
            self.bwd_overhead = get_overhead_(
                self.save_mem[self.loss_idx + 1 :],
                self.overhead[self.loss_idx + 1 :],
            ) - self.save_mem[self.loss_idx]
        else:
            self.bwd_overhead = 0

        self.phantoms = set()
        for anode in self.list_anodes:
            if (
                alive_list[self.loss_idx][anode.name]
                and not anode in self.all_interfaces
            ):
                self.phantoms.add(self.cluster.translate_representee_node(anode))

        self.dep_interfaces_data = set()
        for i, op in enumerate(self.op_list[self.loss_idx + 1 :]):
            if op.disabled or not isinstance(op, ComputeOp):
                continue
            for anode in op.target.deps_real:
                if anode in self.interfaces["input_data_anodes"]:
                    # self.dep_interfaces_data.add(self.list_anodes.index(anode))
                    self_anode = self.cluster.translate_representee_node(anode)
                    self.dep_interfaces_data.add(self_anode)
                if anode in self.interfaces["output_data_anodes"]:
                    for cnode in anode.deps:
                        if not self.is_occurred(
                            ComputeOp(cnode).name, self.loss_idx, self.loss_idx + i
                        ):
                            # if not generated during bwd
                            self_anode = self.cluster.translate_representee_node(anode)
                            # self.dep_interfaces_data.add(self.list_anodes.index(self_anode))
                            self.dep_interfaces_data.add(self_anode)

        self.fwd_overhead_correction = []
        self.bwd_overhead_correction = []
        if self.correct_overhead:
            self.add_correction_term(alive_list)

        self.offload_mem = sum(
            op.target.mem for op in self.op_list if isinstance(op, OffloadOp)
        )
        self.prefetch_mem = sum(
            op.target.mem for op in self.op_list if isinstance(op, PrefetchOp)
        )
        self.fwd_wait_time = 0
        self.bwd_wait_time = 0

    def add_correction_term(self, alive_list):
        """
        Correction term works as follows:
        each term corresponds to one step. "save" and "overhead" correspond
        to the inside anodes (non interfaces considered). 
        The other terms are the corrections for the higher level ILP:
        when the mark is "always", the interface will always appear in the overhead
        (still, they are recorded separately with the inside nodes);
        otherwise, the overhead will be written with the higher level ILP variables.
        """

        interfaces_status = []
        for anode in self.interfaces["input_data_anodes"]:  # Input of Fwd
            interfaces_status.append((anode.name, self.loss_idx))  # After fwd
            if anode in self.dep_interfaces_data:
                interfaces_status.append((anode.name, len(self.op_list)))  # After Bwd
        for anode in self.interfaces["output_data_anodes"]:  # Output of Fwd
            interfaces_status.append((anode.name, 0))  # Before fwd?
            if anode in self.dep_interfaces_data:
                interfaces_status.append((anode.name, len(self.op_list)))  # After Bwd
            else:
                interfaces_status.append((anode.name, -1))  # After Bwd

        for anode in self.interfaces["output_grad_anodes"]:
            interfaces_status.append((anode.name, len(self.op_list)))  # After Bwd
        for anode in self.interfaces["input_grad_anodes"]:
            interfaces_status.append((anode.name, self.loss_idx + 1))  # Before Bwd
        self.interfaces_status = interfaces_status
        for i, (op, alive_status) in enumerate(zip(self.op_list, alive_list)):
            if i == self.loss_idx:
                continue
            correction_term = {
                "save": self.save_mem[i], 
                "overhead": self.overhead[i],
            }
            for anode_name, index in interfaces_status:
                anode = self.dict_alloc[anode_name].anode
                if index == -1:
                    # special case: output_data in BWD without dependency
                    # If outside is alive, no need to correct;
                    # Otherwise, add anode to memory
                    if i > self.loss_idx and alive_status[anode_name] > 0:
                        correction_term[(self.list_anodes.index(anode), False)] = (
                            -anode.mem
                        )
                    continue

                if (
                    alive_status[anode_name] > 0
                    or (index > self.loss_idx) != (i > self.loss_idx)
                    # or not anode_name
                ):
                    # interfaces_status is useful when:
                    # 1. anode is not alive
                    # 2. Fwd to Fwd, Bwd to Bwd
                    continue

                if i >= index:  # if exist before
                    if (  # and not deleted in between
                        # anode_name not in self.op_name_list[index : i + 1]
                        self.is_occurred(DeleteOp(anode).name, index, i)
                    ):
                        correction_term[(self.list_anodes.index(anode), True)] = (
                            -anode.mem
                        )
                    else:
                        correction_term[(self.list_anodes.index(anode), "always")] = (
                            -anode.mem
                        )
                else:  # if exist afterwards
                    if not (anode in self.interfaces["output_data_anodes"]) and (
                        anode.deps
                        # and (list(anode.deps)[0].name in self.op_name_list[i : index + 1])
                        and self.is_occurred(
                            ComputeOp(list(anode.deps)[0]).name, i, index
                        )
                    ):  # and not generated in between
                        # check if output_data is created after i
                        correction_term[(self.list_anodes.index(anode), False)] = (
                            -anode.mem
                        )
                    elif anode in self.interfaces["input_data_anodes"]:
                        correction_term[(self.list_anodes.index(anode), False)] = (
                            -anode.mem
                        )
                    else:
                        correction_term[(self.list_anodes.index(anode), "always")] = (
                            -anode.mem
                        )

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

    def create_list_alloc(self, cluster: HierarchicalCluster):
        self.all_interfaces = [
            anode for inter in self.interfaces.values() for anode in inter
        ]  # all interface KDN's
        self.interface_names = [anode.name for anode in self.all_interfaces]

        self.list_alloc = [Activation(anode) for anode in cluster.list_anodes]
        self.list_anodes = cluster.list_anodes
        if self.with_parameters:
            self.list_alloc.extend(
                [Parameter(anode) for anode in cluster.parameter_nodes]
            )
            self.list_alloc.extend(
                [
                    Parameter(anode, is_grad=True)
                    for anode in cluster.parameter_nodes
                    if anode.info.requires_grad
                ]
            )  # add parameter grad allocation
            if self.optimizer_states_factor:
                self.list_alloc.extend(
                    [
                        Parameter(
                            anode,
                            is_optim_states=True,
                            optimizer_states_factor=self.optimizer_states_factor,
                        )
                        for anode in cluster.parameter_nodes
                        if anode.info.requires_grad
                    ]
                )  # add parameter grad allocation
        self.dict_alloc_param = {
            alloc.name: alloc
            for alloc in self.list_alloc
            if isinstance(alloc, Parameter)
        }
        self.dict_alloc = {alloc.name: alloc for alloc in self.list_alloc}

    def create_alive_list(self):
        alive_status = {alloc.name: False for alloc in self.list_alloc}

        for alloc_name, is_alive in self.init_alive_status.items():
            alive_status[alloc_name] = is_alive

        for anode in self.interfaces["input_data_anodes"]:
            alive_status[anode.name] = True

        alive_list = []
        for i, op in enumerate(self.op_list):
            if op.disabled:
                if i == self.loss_idx:
                    for anode in self.interfaces["output_grad_anodes"]:
                        alive_status[anode.name] = True
                alive_list.append(alive_status.copy())
                continue
            if isinstance(op, DeleteOp):
                alive_status[op.target.name] = False
            elif isinstance(op, ComputeOp):
                # compute op should not be disabled except loss which is useful for alive status
                for anode in op.target.users:
                    if not ("phantoms" == anode.allocation_type and op.fast_forward):
                        alive_status[anode.name] = True
                if (
                    self.with_parameters and not op.target.is_fwd
                ):  # assume grad of parameters required by bwd will be generated
                    for pnode in op.target.required_parameter_nodes_real:
                        if pnode.info.requires_grad == True:
                            alive_status[Parameter(pnode, is_grad=True).name] = True
            elif isinstance(op, AllocateOp):
                alive_status[op.target.name] = True
            alive_list.append(alive_status.copy())
        # assert alive_status == self.init_alive_status# cyclic alive status
        # for k,v in self.init_alive_status.items():
        #     assert alive_status[k] == v
        return alive_list

    def add_pos_info(self):
        """
        Prepare positional information of each operation for compiling.
        """
        if not hasattr(self, "alive_list"):
            self.alive_list = self.create_alive_list()
        for index, op in enumerate(self.op_list):
            assert index in self.occurrences[op.name]
            if not isinstance(op, ComputeOp):
                continue
            op.pos_info = {
                "index": index,
                "first_occurrence": index == min(self.occurrences[op.name]),
                "last_occurrence": index == max(self.occurrences[op.name]),
            }
            if op.target.is_fwd:
                # last_before_bwd = True
                bwd_op_name = op.name.replace("FWD", "BWD")
                if bwd_op_name not in self.occurrences:
                    continue
                op.pos_info["next_bwd_idx"] = min(
                    [i for i in self.occurrences[bwd_op_name] if i > index],
                    default=None
                )
                if op.pos_info["next_bwd_idx"] is None:
                    op.pos_info["last_before_bwd"] = False
                else:
                    op.pos_info["last_before_bwd"] = not self.is_occurred(
                        op.name, index + 1, op.pos_info["next_bwd_idx"]
                    )

                # TODO: no_save_list should contain only the one got deleted before bwd
                cnode = op.target
                no_save_nodes = list(cnode.deps_real) + list(cnode.users)
                if self.with_parameters:
                    no_save_nodes += list(cnode.required_parameter_nodes_real)
                    no_save_nodes += list(cnode.required_parameter_nodes_fake)
                op.pos_info["no_save_list"] = [
                    (
                        anode.main_target
                        if hasattr(anode, "main_target")
                        else anode.param_name
                    )
                    for anode in no_save_nodes
                ]
            else:
                op.pos_info["temporary_tensor_names"] = []
                for anode in op.target.deps_fake:
                    if not self.alive_list[index][anode.name]:
                        op.pos_info["temporary_tensor_names"].append(anode.main_target)

                op.pos_info["input_names"] = []
                if not op.pos_info["first_occurrence"]:
                    # prev_i = index - self.op_list[:index][::-1].index(op) - 1
                    prev_i = max(i for i in self.occurrences[op.name] if i < index)
                    for anode in op.target.users:
                        if self.is_occurred(
                            DeleteOp(Activation(anode)).name, prev_i, index
                        ):
                            op.pos_info["input_names"].append(anode.main_target)

                    for pnode in op.target.required_parameter_nodes_real:
                        if self.is_occurred(
                            DeleteOp(Parameter(pnode)).name, prev_i, index
                        ):
                            op.pos_info["input_names"].append(pnode.param_name)
                    if not op.pos_info["input_names"]:
                        op.disabled = True
                        warnings.warn(f"{op.name} is recomputed but no target inputs")

    def __repr__(self) -> str:
        return f"Op_sched takes {sum(self.time):.2f} ms with {self.peak_mem/1024**2} MiB peak mem"
