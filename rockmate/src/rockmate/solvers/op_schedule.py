from rkgb.core.base import Node
from rkgb.core.backward import ComputationNode, AllocationNode
from rkgb.core.hierarchical import HierarchicalCluster
from collections import namedtuple
from copy import deepcopy
from typing import List
import warnings
import numpy as np
import torch
from .simulation import Simulator
from rkgb.lowlevel.ast_add_on import make_str_list_assign

class Allocation:
    def __init__(self, target_name, alloc_type="", mem=0, info=dict(), dtype=torch.float32, size=None):
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
            self.mem = size*self.itemsize
        else:
            self.size = round(self.mem/self.itemsize)#element size

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
    def __init__(self, 
                 anode:AllocationNode,
                 dtype=torch.float32):
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
    def __init__(self, pnode, is_grad=False, is_optim_states=False):
        self.pnode = pnode
        self.is_grad = is_grad
        self.is_optim_states = is_optim_states
        super().__init__(
            target_name=pnode.param_name,
            alloc_type= "param",# + "_grad"*grad+"_optim_states"*is_optim_states,
            mem=pnode.mem + pnode.mem*is_optim_states,
            info=pnode.info,
            dtype=pnode.info.dtype,
        )

    @property
    def param_name(self):
        return self.pnode.param_name
    
    @property
    def alloc_type(self):
        return "param" + "_grad"*self.is_grad+"_optim_states"*self.is_optim_states

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

    # @name.setter
    # def name(self, value):
    #     self._name = value

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
        if not type(self)==type(op):
            return False
        return op.name == self.name and self.disabled==op.disabled

    def __hash__(self):
        return id(self)

class SynchronizeOp(Op):
    def __init__(self, name="", disabled=False):
        super().__init__(name, disabled=disabled)
        self.op_type = "Synchronize"

class ComputeOp(Op):
    def __init__(self, 
                 cnode: ComputationNode,
                 fast_forward=False, disabled=False, detach=True):
        super().__init__(cnode.name, disabled=disabled)
        self.fast_forward = fast_forward
        self.detach = detach
        self.target:ComputationNode = cnode
        self.overhead = cnode.mem_overhead
        self.pos_info = {}#positional information to be filled before compiling

    @property
    def time(self):
        # cnode can be replaced during translation
        return self.target.time if self.target.time is not None else 0
    
    @property
    def kcn(self):
        # TODO: deprecated attribute
        return self.target
    
    @property
    def op_type(self):
        cnode_type = "FWD" if self.target.is_fwd else "BWD"
        return f"Compute"

    @property
    def target_name(self):
        return self.target.name


class DeleteOp(Op):
    def __init__(self, alloc: Allocation, disabled=False):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.op_type = f"Delete"
        # self.grad = grad
        # self.is_optim_states = is_optim_states

    # def __repr__(self):
    #     return "Disabled_"*self.disabled+"Delete_" + self.target.name+"grad"*self.grad+"_optim_stats"*self.is_optim_states
    
    # @property
    # def name(self):
    #     # name changes with target is changed during translation
    #     return "Delete_" + self.target.name


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
        self.is_optim_states=is_optim_states

class OffloadOp(Op):
    def __init__(
        self,
        alloc: Allocation,
        indices: tuple = (0,None),
        disabled: bool = False,
        # grad: bool = False,
        # is_optim_states: bool = False,
        time:float = 0,
    ):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.indices = indices
        self.disabled = disabled
        # self.grad = grad
        # self.is_optim_states = is_optim_states
        self._time = time
        self.op_type = f"Offload_{alloc.alloc_type}"

class PrefetchOp(Op):
    def __init__(
        self,
        alloc: Allocation,
        indices: tuple = (0,None),
        disabled: bool = False,
        # is_optim_states: bool = False,
        time:float = 0,
    ):
        super().__init__(alloc.name, disabled=disabled)
        self.target = alloc
        self.indices = indices
        self.disabled = disabled
        # self.is_optim_states = is_optim_states
        self._time = time
        self.op_type = f"Prefetch_{alloc.alloc_type}"
    
class OptimizeOp(Op):
    def __init__(self, 
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

class ListOp(list):
    def __init__(self, ops: List[Op]):
        super(ListOp, self).__init__(ops)
        self._pop = super(ListOp, self).pop
        self._remove = super(ListOp, self).remove
        self._append = super(ListOp, self).append
        self._insert = super(ListOp, self).insert
        self.time = sum(op.time for op in ops)

    def pop(self,index):
        self.time -= self[index].time
        return self._pop(index)
    def remove(self,op: Op):
        self.time -= op.time
        return self._remove(op)
    def append(self,op: Op):
        self.time += op.time
        return self._append(op)
    def insert(self,i,op: Op):
        self.time += op.time
        return self._insert(i,op)

class Step():
    def __init__(self, op_list:List[Op]) -> None:

        ofl_ops = []
        prf_ops = []
        opt_ops = []
        comp_ops = []
        self.alloc_ops = []
        self.del_ops = []
        for op in op_list:
            if isinstance(op, OffloadOp):
                ofl_ops.append(op)
            elif isinstance(op, PrefetchOp):
                prf_ops.append(op)
            elif isinstance(op, OptimizeOp) and op.is_cpu:
                opt_ops.append(op)
            elif isinstance(op, ComputeOp) or (
                isinstance(op, DeleteOp) and isinstance(op.target, Activation)) or(
                isinstance(op, OptimizeOp) and not op.is_cpu):
                # all happen in the main stream
                comp_ops.append(op)
            elif isinstance(op, DeleteOp) and isinstance(op.target, Parameter):
                self.del_ops.append(op)
            else:#if isinstance(op, AllocateOp):
                self.alloc_ops.append(op)

        self.ofl_ops = ListOp(ofl_ops)
        self.prf_ops = ListOp(prf_ops)
        self.opt_ops = ListOp(opt_ops)
        self.comp_ops = ListOp(comp_ops)

        # if self.prf_ops:
        #     # Assume prefetch ops will not change
        #     code = ""
        #     for op in self.prf_ops:
        #         # if op.is_optim_states:continue
        #         if isinstance(op.target, Parameter) and not op.target.is_optim_states:
        #             code += op.target.pnode.get_code()+"\n"
        #     self.view_param = ExecCodeOp(f"view_{op.name}",
        #                                 code=code)
    
    @property
    def op_list(self):
        # gpu_ops = []
        # cpu_ops = []
        # list_params = []
        # for op in self.opt_ops:
        #     if op.is_cpu:
        #         cpu_ops.append(op)
            # list_params += op.list_params
        #     else:
        #         gpu_ops.append(op)
        # if cpu_ops:
        if not self.opt_ops:
            opt_ops = []
        else:
            list_params = [p for op in self.opt_ops for p in op.list_params]
            # opt_ops = [OptimizeOp(f"cpu_{str(list_params[0])}", 
            #                     list_params=list_params,
            #                     time=sum(op.time for op in self.opt_ops))]
            opt_ops = self.opt_ops
        # cpu_ops = [OptimizeOp(f"cpu_{str(list_params[0])}", list_params=list_params)] if list_params else []
        # opt_ops = cpu_ops+gpu_ops
        
        return (self.alloc_ops
                +self.ofl_ops
                +self.prf_ops
                +self.comp_ops
                +opt_ops
                +self.del_ops
                # +([self.view_param] if self.prf_ops else [])
                )
    
    @property
    def time(self):
        return max(self.ofl_ops.time, 
                   self.prf_ops.time, 
                   self.opt_ops.time, 
                   self.comp_ops.time)

    def all_time(self):
        return (self.ofl_ops.time, 
                   self.prf_ops.time, 
                   self.opt_ops.time, 
                   self.comp_ops.time)

    def max2nd(self):
        t = list(self.all_time())
        t.remove(max(t))
        return max(t)


class OpSchedule:
    solver = None

    def __init__(
        self,
        op_list:List[Op],
        cluster: HierarchicalCluster,
        loss_idx=None,
        with_parameters=False,
        init_alive_status: dict = {},
        init_op_list: list = [],
        restore_op_list: list = []
    ):
        """
        OpSchedule contains the operation list and automatically
        compute the alive status given the HierarchicalCluster.
        """
        self.op_list = op_list
        self.loss_idx = loss_idx
        self.init_alive_status = init_alive_status
        self.init_op_list = init_op_list# Place to prepare items in storage
        self.restore_op_list = restore_op_list
        self.with_parameters = with_parameters
        self.interfaces = cluster.interfaces

        self.get_occurrences()
        self.create_list_alloc(cluster)
        # self.create_alive_list()
        self.get_sched_info()# get the schedule information for higher level solving

    def get_occurrences(self):
        self.occurrences = dict()
        for i, op in enumerate(self.op_list):
            if op.name in self.occurrences:
                self.occurrences.append(i)
            else:
                self.occurrences = [i]

    def simulate_update(self, refine_optimize=False):
        simulator = Simulator(self)
        simulator.refine()# assume init_op_list remains the same
        if refine_optimize:
            simulator.refine_optimize()
        self.op_list = simulator.op_list
        self.loss_idx = simulator.loss_idx
        self.get_sched_info()

    def _sum_mem(self,alive_status_, ignore_list=[]):
        mem = 0
        for k, v in alive_status_.items():
            if k not in ignore_list and v:
                d = self.dict_alloc[k]
                mem += d.mem
        return mem

    def get_sched_info(self):
        """
        To get schedule information for higher level solving:
        - .mem: saved activation memory at the loss step, without interface activations
        - .fwd_time: time of forward operations
        - .bwd_time: time of backward operations
        - .fwd_overhead: during forward, the excess over the ending memory from the peak
        - .bwd_overhead: during backward, the excess over the ending memory from the peak
        - .phantoms: saved anodes at the loss step
        """
        # self.save_mem = self.alive_status.save_mem
        alive_list = self.create_alive_list()

        L = len(self.op_list)
        self.time = np.zeros(L)
        self.save_mem = np.zeros(L)
        self.overhead = np.zeros(L)
        self.interface_mem = np.zeros(L)

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        for i, (op, alive_status) in enumerate(zip(self.op_list, alive_list)):
            self.save_mem[i] = self._sum_mem(alive_status, self.interface_names)
            self.interface_mem[i] = self._sum_mem(alive_status) - self.save_mem[i]
            if op.disabled:
                continue
            self.time[i] = op.time
            self.overhead[i] = op.overhead
            
        self.mem = self.save_mem[self.loss_idx]
        self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
        self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])

        self.phantoms = set()
        for anode in self.list_anodes:
            if alive_list[self.loss_idx][anode.name] and not anode in self.all_interfaces:
                self.phantoms.add(anode)

        self.fwd_overhead = get_overhead_(
            self.save_mem[: self.loss_idx + 1],
            self.overhead[: self.loss_idx + 1],
        )
        if self.loss_idx < len(self.op_list) - 1:  # bwd is not empty
            self.bwd_overhead = get_overhead_(
                self.save_mem[self.loss_idx + 1 :],
                self.overhead[self.loss_idx + 1 :],
            )

        self.dep_interfaces_data = set()
        for i, op in enumerate(self.op_list[self.loss_idx + 1 :]):
            if op.disabled or not isinstance(op, ComputeOp):
                continue
            for anode in op.target.deps_real:
                if anode in self.interfaces["input_data_anodes"]:
                    self.dep_interfaces_data.add(self.list_anodes.index(anode))
                if anode in self.interfaces["output_data_anodes"]:
                    for cnode in anode.deps:
                        if (
                            ComputeOp(cnode) in self.op_list[self.loss_idx + 1 :][:i]
                            # cnode.name not in self.op_name_list[self.loss_idx + 1 :][:i]
                        ):  # if not generated during bwd
                            self.dep_interfaces_data.add(self.list_anodes.index(anode))


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
                [Parameter(anode, is_grad=True) for anode in cluster.parameter_nodes
                    if anode.info.requires_grad]
            )# add parameter grad allocation
            self.list_alloc.extend(
                [Parameter(anode, is_optim_states=True)
                    for anode in cluster.parameter_nodes
                    if anode.info.requires_grad]
            )# add parameter grad allocation
            self.dict_alloc_param = {alloc.name: alloc 
                                 for alloc in self.list_alloc
                                 if isinstance(alloc, Parameter)}
        self.dict_alloc = {alloc.name: alloc for alloc in self.list_alloc}

    def create_alive_list(self):
        alive_status = {alloc.name: False for alloc in self.list_alloc}
        
        for alloc_name, is_alive in self.init_alive_status.items():
            alive_status[alloc_name] = is_alive

        alive_list = []
        for op in self.op_list:
            if op.disabled:
                alive_list.append(alive_status.copy())
                continue
            if isinstance(op, DeleteOp):
                alive_status[op.target.name] = False
            elif isinstance(op, ComputeOp):
                # compute op should not be disabled except loss which is useful for alive status
                for anode in op.target.users:
                    if not ("phantoms" == anode.allocation_type and op.fast_forward):
                        alive_status[anode.name] = True
                if self.with_parameters and not op.target.is_fwd:# assume grad of parameters required by bwd will be generated
                    for pnode in op.target.required_parameter_nodes_real:
                        alive_status[Parameter(pnode).name] = True
            elif isinstance(op, AllocateOp):
                alive_status[op.target.name] = True
            alive_list.append(alive_status.copy())
        # assert alive_status == self.init_alive_status# cyclic alive status
        for k,v in self.init_alive_status.items():
            assert alive_status[k] == v
        return alive_list

    def add_pos_info(self):
        """
        Prepare positional information of each operation for compiling.
        """
        if not hasattr(self, "alive_list"):
            self.alive_list = self.create_alive_list()
        for i, op in enumerate(self.op_list):
            if not isinstance(op, ComputeOp):
                continue
            op.pos_info = {
                "index":i,
                "first_occurrence":not op in self.op_list[:i],
                "last_occurrence":not op in self.op_list[i+1:],
                }
            if op.target.is_fwd:
                last_before_bwd = True
                for j, _op in enumerate(self.op_list[i+1:]):
                    if (isinstance(_op, ComputeOp)
                        and op.target.main_target==_op.target.main_target):
                        if _op.target.is_fwd:
                            last_before_bwd = False
                        else:
                            op.pos_info["next_bwd_idx"] = i+1+j
                            op.pos_info["last_before_bwd"] = last_before_bwd
                        cnode = op.target
                        no_save_nodes = (
                            list(cnode.deps_real)
                            + list(cnode.users)
                            + list(cnode.required_parameter_nodes_real)
                            + list(cnode.required_parameter_nodes_fake)
                        )
                        op.pos_info["no_save_list"] = [anode.main_target
                                                       if hasattr(anode, "main_target")
                                                       else anode.param_name
                                                       for anode in no_save_nodes]
            else:
                op.pos_info["temporary_tensor_names"] = []
                for anode in op.target.deps_fake:#|op.target.deps_real:
                    if not self.alive_list[i][anode.name]:
                        op.pos_info["temporary_tensor_names"].append(anode.main_target)
                        # if anode.main_target == op.target.main_target:
                        #     op.pos_info["temporary_tensor_names"].append(f"_{anode.main_target}")
                op.pos_info["input_names"] = []
                if not op.pos_info["first_occurrence"]:
                    prev_i = i - self.op_list[:i][::-1].index(op) - 1
                    for anode in op.target.users:
                        if DeleteOp(Activation(anode)) in self.op_list[prev_i:i]:
                            op.pos_info["input_names"].append(anode.main_target)

                    for pnode in op.target.required_parameter_nodes_real:
                        if DeleteOp(Parameter(pnode)) in self.op_list[prev_i:i]:
                            op.pos_info["input_names"].append(pnode.param_name)
                    if not op.pos_info["input_names"]:
                        op.disabled = True
                        raise Warning(f"{op.name} is recomputed but no target inputs")

