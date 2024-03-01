# from rkgb.utils import *
# from rkgb.Ptools import P_graph, P_node
# from rkgb.Ktools import K_graph, K_C_node, K_D_node
# from rkgb.Htools import *
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


class AliveSimulator():
    """
    Dynamic alive status of different allocations.
    Functions: 
    1. Easy to check if one allocation is alive for a range of steps.
    2. Easy to update when operations are changed.
    3. Easy to translate allocations for anonymized cluster.
    4. Return peak memory fast.
    Assumption:
    1. Number of operations will not be changed (operations can be disabled).
    2. Number of allocations will not be changed.
    """
    def __init__(self, alive_list, dict_alloc, alloc_categories={}):
        self.update_dict_alloc(dict_alloc)
        self.length = len(alive_list)
        self.alloc_mem_np = np.array([self.dict_alloc[alloc_name].mem 
                                      for alloc_name in self.alloc_names])
        self.alive_np = np.array([[1 if alive_list[i][alloc_name] else 0 
                                   for alloc_name in self.alloc_names]
                                   for i ,_ in enumerate(alive_list)],
                                   dtype=bool)
        
        self.categories = {}
        self.categories["activation"] = [alloc_name for alloc_name in self.alloc_names
                                         if isinstance(self.dict_alloc[alloc_name], Activation)]
        self.categories["parameter"] = [alloc_name for alloc_name in self.alloc_names
                                        if isinstance(self.dict_alloc[alloc_name], Parameter)]
        for k,v in alloc_categories.items():
            self.categories[k] = np.array([self.alloc_names.index(alloc_name) for alloc_name in v])

    def update_dict_alloc(self, dict_alloc):
        """
        For translation: need to ensure the dict_alloc to appear in the same list
        """
        self.dict_alloc = dict_alloc
        self.alloc_names = [alloc_name for alloc_name in sorted(dict_alloc.keys())]
        self.name2key = {
            alloc_name:i for i, alloc_name in enumerate(self.alloc_names)
            }
        
    def _alive(self, key:int, idx_range:range):
        if isinstance(idx_range, int):
            idx_range = range(idx_range, idx_range+1)
        return self.alive_np[idx_range, key]
    
    # def _mem(self, idx:int, category=None):
    #     if category:
    #         return np.sum((self.alive_np[idx]*self.alloc_mem_np)[self.categories[category]])
    #     return np.matmul(self.alive_np[idx], self.alloc_mem_np)
    
    def _mem(self, idx_range:range, category=None):
        """
        Return a (idx_range,)-shape array of memory over the steps range.
        """
        if category:
            if isinstance(idx_range, int):
                idx_range = range(idx_range, idx_range+1)
            return np.sum((self.alive_np[idx_range]*
                           self.alloc_mem_np)[:, self.categories[category]], 
                           axis=1)
        return np.matmul(self.alive_np[idx_range], self.alloc_mem_np)

    def act_mem(self, idx_range, category="activation"):
        return self._mem(idx_range, category=category)

    def param_mem(self, idx_range, category="parameter"):
        return self._mem(idx_range, category=category)

    def save_mem(self, idx_range = None, act_multiplier=1):
        if idx_range is None:
            idx_range = range(0, self.length)
        return act_multiplier*self.act_mem(idx_range)+self.param_mem(idx_range)
    
    def _alive(self, key:int, idx_range:range):
        if isinstance(idx_range, int):
            idx_range = range(idx_range, idx_range+1)
        return self.alive_np[idx_range, key]

    def alive_once(self, key:int, idx_range:range):
        # being alive AT LEAST once
        if isinstance(idx_range, int):
            idx_range = range(idx_range, idx_range+1)
        if isinstance(key, str):
            key = self.name2key[key]
        return bool(max(self._alive[key, idx_range]))

    def alive_always(self, key:int, idx_range:range):
        # being alive AT LEAST once
        if isinstance(idx_range, int):
            idx_range = range(idx_range, idx_range+1)
        if isinstance(key, str):
            key = self.name2key[key]
        return bool(min(self._alive[key, idx_range]))

        
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
        self.create_list_alloc(cluster)
        # self.create_alive_list()
        self.get_sched_info()# get the schedule information for higher level solving

    def simulate_update(self):
        simulator = Simulator(self)
        simulator.refine()# assume init_op_list remains the same
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
            else:
                op.pos_info["temporary_tensor_names"] = []
                for anode in op.target.deps_fake|op.target.deps_real:
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


class Simulator:
    """
    With steps and np-alive status, simulator allows efficient
    simulation of swapping/changing operations. The final output
    of operation list can be used in op_sched updatation.
    """
    def __init__(self, op_sched: OpSchedule):
        self.op_list: List[Op] = op_sched.op_list
        self.loss_idx = op_sched.loss_idx
        alive_list = op_sched.alive_list if hasattr(op_sched, "alive_list") else op_sched.create_alive_list()
        self.alive_list = AliveSimulator(alive_list, dict_alloc=op_sched.dict_alloc)
        self.create_steps()

    def refine_optimize(self):
        steps = self.steps
        for j in range(len(steps)-1, self.loss_step, -1):
            opt_ops: List[OptimizeOp] = list(steps[j].opt_ops)
            opt2user_step = {op.name:None for op in opt_ops}
            steps_avail = {op:steps[j:] for op in opt_ops}

            for i,step in enumerate(steps[:]):
                if None not in opt2user_step.values():break
                for opt_op in opt_ops:
                    if opt2user_step[opt_op.name] is not None:continue
                    # for usr in opt_op.target.pnode.users_real:
                    #     if usr.name in [str(op) for op in step.comp_ops]:
                    for prf_op in step.prf_ops:
                        if prf_op.target.param_name == opt_op.target.pnode.param_name:
                            # print(opt_op.name)
                            opt2user_step[opt_op.name] = i
                            steps_avail[opt_op].extend(steps[:i])
                            # located_ops.append(opt_op)
                            # opt_ops.remove(opt_op)
            for op, avail in steps_avail.items():
                avail_step = max(avail, key=lambda x:x.time-x.opt_ops.time)
                # print(avail_step.time,avail_step.opt_ops.time)
                # print(steps[j].time,steps[j].opt_ops.time)
                if avail_step.time<avail_step.opt_ops.time:continue
                if steps[j].opt_ops.time<steps[j].time:continue
                if (avail_step.opt_ops.time+op.time-avail_step.time>
                    steps[j].opt_ops.time-steps[j].max2nd()):continue
                
                # print(avail_step.time, avail_step.opt_ops.time)
                avail_step.opt_ops.append(op)
                steps[j].opt_ops.remove(op)

        self.refine()
        self.recreate_op_list()

    def create_steps(self):
        self.steps: List[Step] = []
        step_op = []
        for i,op in enumerate(self.op_list):
            if isinstance(op, SynchronizeOp):
                if step_op:self.steps.append(Step(step_op))
                step_op = []
            if i == self.loss_idx:
                self.loss_step = len(self.steps)
            step_op.append(op)
        self.steps.append(Step(step_op))
        self.recreate_op_list()
        
    def recreate_op_list(self):
        self.loss_op = self.op_list[self.loss_idx]
        op_list = []
        for step in self.steps:
            op_list += step.op_list
        self.op_list = op_list
        self.loss_idx = self.op_list.index(self.loss_op)
        
    def refine(self):
        """
        Disable deletion to avoid infeasible operations in the list.
        """
        for i, op in enumerate(self.op_list):
            if op.disabled:continue
            if "loss" in op.name:
                op.disabled = True
            if isinstance(op, DeleteOp):
                if isinstance(op.target, Activation):
                    # try to delete KDN
                    src_i = []  # indices of source KCN's after i
                    for cnode in op.target.anode.deps:
                        c_op = ComputeOp(cnode)
                        if c_op in self.op_list[i:]:
                            src_i.append(self.op_list[i:].index(c_op) + i)
                        else:
                            src_i.append(len(self.op_list))
                    src_i = src_i or [len(self.op_list)]

                    next_used_i = len(self.op_list)  # the next index to use KDN
                    for cnode in op.target.anode.users_real:
                        c_op = ComputeOp(cnode)
                        if c_op in self.op_list[i:]:
                            next_used_i = min(
                                self.op_list[i:].index(c_op) + i,
                                next_used_i,
                            )

                    if max(src_i) > next_used_i:  # try to use before regenerate
                        # print(f"refine {op} for {self.op_name_list[next_used_i]}")
                        op.disabled = True

                elif isinstance(op.target, Parameter):
                    # TODO: disabled wrong deletion of parameter
                    pass

        self.recreate_op_list()



class OpSchedule_old:
    solver = None

    def __init__(
        self,
        op_list:List[Op],
        prf_list=[],
        ofl_list=[],
        loss_idx=None,
        cluster=None,
        interfaces=None,
        refine=True,
        correct_overhead=False,
        keep_alive_list=False,
        with_parameters=False,
        init_alive_status: dict = {},
        init_op_list: list = [],
        restore_op_list: list = []
    ):
        """
        Key role of OpSchedule: taking op_list, analyzing memory stats,
        keeping info for further solving.
        New role: greedy algorithm to rearrange the prefetch/offload ops
        Parameters are read from the cluster
        """
        self._op_list = op_list
        self.init_op_list = init_op_list
        self.restore_op_list = restore_op_list
        self.prf_list = prf_list
        self.ofl_list = ofl_list
        self.with_parameters = with_parameters
        self.from_steps = with_parameters
        
        if loss_idx is None:
            # Find the last loss op before the first bwd
            for i, op in enumerate(self._op_list):
                if "loss" in op.name:
                    self.loss_idx = i
                if isinstance(op, ComputeOp) and not op.target.is_fwd:
                    break
        else:
            self.loss_idx = loss_idx
        if self.from_steps:
            self.create_steps()
        if cluster is None:
            warnings.warn("Cluster should be provided to create op_sched")
            self.prepare_allocation_from_op_list(interfaces)
        else:
            self.interfaces = cluster.interfaces
            self.list_alloc = [Activation(anode) for anode in cluster.list_anodes]
            self.list_anodes = cluster.list_anodes
            if with_parameters:
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
                self.list_alloc.extend(self.create_buffer_list())
        self.dict_alloc = {alloc.name: alloc for alloc in self.list_alloc}
        self.dict_alloc_param = {alloc.name: alloc 
                                 for alloc in self.list_alloc
                                 if isinstance(alloc, Parameter)}
        self.all_interfaces = [
            anode for inter in self.interfaces.values() for anode in inter
        ]  # all interface KDN's
        self.interface_names = [anode.name for anode in self.all_interfaces]

        self._op_name_list = [
            (str(op) if not op.disabled else "") for op in self.op_list
        ]

        if refine:
            print("start refine")
            self.refine()

        alive_list = self.create_alive_list(init_status=init_alive_status)

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
            if isinstance(op, ComputeOp):
                self.time[i] = op.target.time
                self.overhead[i] = op.overhead
            elif isinstance(op, OptimizeOp) and not op.is_cpu:
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

        # names of additional HDNs that are required by BWD
        self.dep_interfaces_data = set()
        for i, op in enumerate(self.op_list[self.loss_idx + 1 :]):
            if op.disabled:
                continue
            if isinstance(op, ComputeOp):
                for anode in op.target.deps_real:
                    if anode in self.interfaces["input_data_anodes"]:
                        self.dep_interfaces_data.add(self.list_anodes.index(anode))
                    if anode in self.interfaces["output_data_anodes"]:
                        for cnode in anode.deps:
                            if (
                                cnode.name not in self.op_name_list[self.loss_idx + 1 :][:i]
                            ):  # if not generated during bwd
                                self.dep_interfaces_data.add(self.list_anodes.index(anode))

        self.fwd_overhead_correction = []
        self.bwd_overhead_correction = []
        if correct_overhead:
            self.correct_overhead(alive_list)

        if keep_alive_list:
            self.alive_list = alive_list
        else:
            self.alive_list = []

    def refine_optimize(self):
        steps = self.steps
        for j in range(len(steps)-1, self.loss_step, -1):
            opt_ops = list(steps[j].opt_ops)
            opt2user_step = {op.name:None for op in opt_ops}
            steps_avail = {op:steps[j:] for op in opt_ops}

            located_ops = []
            for i,step in enumerate(steps[:]):
                if None not in opt2user_step.values():break
                for opt_op in opt_ops:
                    if opt2user_step[opt_op.name] is not None:continue
                    # for usr in opt_op.target.pnode.users_real:
                    #     if usr.name in [str(op) for op in step.comp_ops]:
                    for prf_op in step.prf_ops:
                        if prf_op.target.param_name == opt_op.target.pnode.param_name:
                            # print(opt_op.name)
                            opt2user_step[opt_op.name] = i
                            steps_avail[opt_op].extend(steps[:i])
                            # located_ops.append(opt_op)
                            # opt_ops.remove(opt_op)
            for op, avail in steps_avail.items():
                avail_step = max(avail, key=lambda x:x.time-x.opt_ops.time)
                # print(avail_step.time,avail_step.opt_ops.time)
                # print(steps[j].time,steps[j].opt_ops.time)
                if avail_step.time<avail_step.opt_ops.time:continue
                if steps[j].opt_ops.time<steps[j].time:continue
                if (avail_step.opt_ops.time+op.time-avail_step.time>
                    steps[j].opt_ops.time-steps[j].max2nd()):continue
                
                # print(avail_step.time, avail_step.opt_ops.time)
                avail_step.opt_ops.append(op)
                steps[j].opt_ops.remove(op)

        for i, op in enumerate(self.op_list):
            if "loss" in op.name:
                self.loss_idx = i
            if isinstance(op, ComputeOp) and not op.target.is_fwd:
                break
        self.alive_list = self.create_alive_list()
        self.refine()
        self.recreate_op_list()

    def create_steps(self):
        self.steps = []
        step_op = []
        for i,op in enumerate(self._op_list):
            if isinstance(op, SynchronizeOp):
                if step_op:self.steps.append(Step(step_op))
                step_op = []
            if i == self.loss_idx:
                self.loss_step = len(self.steps)
            step_op.append(op)
        self.steps.append(Step(step_op))
        self.recreate_op_list()
            # print(op, op.time)
        

    def create_alive_np_array(self):
        alive_list = self.alive_list or self.create_alive_list()
        self.np_alloc_mem = np.array([alloc.mem for alloc in self.list_alloc])
        self.np_overhead = np.array([op.overhead for op in self.op_list])
        self.alive_array = np.array([[1 if self.init_alive_status[a.name] else 0 
                                      for a in self.list_alloc]+
                                      [1 if alive_list[i][a.name] else 0 
                                      for a in self.list_alloc] 
                                      for i ,_ in enumerate(self.op_list)])
        self.alive_diff_array = np.diff(self.alive_array, axis=0)

    def create_buffer_list(self):
        buffer_set = set()
        for op in self.op_list:
            if isinstance(op, MappingOp) and len(op.targets)==1:# merge mapping
                for target in op.targets:
                    buffer_set.add(target)
            # elif isinstance(op, AllocateOp):
            #     buffer_set.add(op.target)

        return list(buffer_set)

    def prepare_allocation_from_op_list(self, interfaces):
        self.interfaces = interfaces or {
            "input_data_anodes": set(),
            "output_data_anodes": set(),
            "input_grad_anodes": set(),
            "output_grad_anodes": set(),
        }
        self.list_anodes = []
        for op in self.op_list:
            if isinstance(op, DeleteOp):
                self.list_anodes.append(op.target)
            elif isinstance(op, ComputeOp):
                self.list_anodes.extend([anode for anode in op.target.users_global])
                self.list_anodes.extend([anode for anode in op.target.deps_global])
        self.list_alloc = self.list_anodes

    def create_alive_list(self, init_status={}):
        alive_status = {alloc.name: False for alloc in self.list_alloc}
        for k, v in init_status.items():
            alive_status[k] = v
        for anode in self.interfaces["input_data_anodes"]:
            alive_status[anode.name] = True  # anode share the name as alloc
        for op in self.init_op_list:
            if isinstance(op, AllocateOp):
                alive_status[op.target.name] = True
        self.init_alive_status = alive_status.copy()

        # TODO: add init alive for parameters
        self.bwd2param = {}
        for alloc in self.list_alloc:
            if isinstance(alloc, Parameter) and alloc.is_grad:
                for cnode in alloc.pnode.users_real:
                    if cnode.is_fwd:continue
                    cnode_name = cnode.name
                    # if cnode_name in self.bwd2param:
                    #     self.bwd2param[cnode_name].append(alloc.pnode.param_name+"_grad")
                    #     self.bwd2param[cnode_name].append(alloc.pnode.param_name)
                    # else:
                    #     self.bwd2param[cnode_name] = [alloc.pnode.param_name+"_grad"]
                    #     self.bwd2param[cnode_name].append(alloc.pnode.param_name)
                
        for op in self.op_list:
            if isinstance(op, DeleteOp):
                alive_status[op.target.name] = False
            elif isinstance(op, ComputeOp):
                if op.target.name in self.bwd2param:
                    for anode_name in self.bwd2param[op.target.name]:
                        alive_status[anode_name] = True
            elif isinstance(op, AllocateOp):
                alive_status[op.target.name] = True

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
                if op.target.name in self.bwd2param:
                    for anode_name in self.bwd2param[op.target.name]:
                        alive_status[anode_name] = True
            elif isinstance(op, AllocateOp):
                alive_status[op.target.name] = True
            alive_list.append(alive_status.copy())
        return alive_list

    def correct_overhead(self, alive_list, refine=True):
        # correction terms of overhead, each term represents one step in op_list

        interfaces_status = []
        for anode in self.interfaces["input_data_anodes"]:  # Input of Fwd
            interfaces_status.append((anode.name, self.loss_idx))  # After fwd
            if self.list_anodes.index(anode) in self.dep_interfaces_data:
                interfaces_status.append((anode.name, len(self.op_list)))  # After Bwd
        for anode in self.interfaces["output_data_anodes"]:  # Output of Fwd
            interfaces_status.append((anode.name, 0))  # Before fwd?
            if self.list_anodes.index(anode) in self.dep_interfaces_data:
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
                        correction_term["save"] += anode.mem
                        correction_term[(self.list_anodes.index(anode), False)] = -anode.mem
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
                        anode_name not in self.op_name_list[index : i + 1]
                    ):
                        correction_term[(self.list_anodes.index(anode), True)] = -anode.mem
                    else:
                        correction_term[(self.list_anodes.index(anode), "always")] = -anode.mem
                else:  # if exist afterwards
                    if not (anode in self.interfaces["output_data_anodes"]) and (
                        anode.deps
                        and (list(anode.deps)[0].name in self.op_name_list[i : index + 1])
                    ):  # and not generated in between
                        # check if output_data is created after i
                        correction_term[(self.list_anodes.index(anode), False)] = -anode.mem
                    elif anode in self.interfaces["input_data_anodes"]:
                        correction_term[(self.list_anodes.index(anode), False)] = -anode.mem
                    else:
                        correction_term[(self.list_anodes.index(anode), "always")] = -anode.mem

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
            if op.disabled:continue
            if "loss" in op.name:
                op.disabled = True
            if isinstance(op, DeleteOp):
                if isinstance(op.target, Activation):
                    # try to delete KDN
                    src_i = []  # indices of source KCN's after i
                    for cnode in op.target.anode.deps:
                        if cnode.name in self.op_name_list[i:]:
                            src_i.append(self.op_name_list[i:].index(cnode.name) + i)
                        else:
                            src_i.append(len(self.op_list))
                    src_i = src_i or [len(self.op_list)]

                    next_used_i = len(self.op_list)  # the next index to use KDN
                    for cnode in op.target.anode.users_real:
                        if cnode.name in self.op_name_list[i:]:
                            next_used_i = min(
                                self.op_name_list[i:].index(cnode.name) + i,
                                next_used_i,
                            )

                    if max(src_i) > next_used_i:  # try to use before regenerate
                        # print(f"refine {op} for {self.op_name_list[next_used_i]}")
                        op.disabled = True

                elif isinstance(op.target, Parameter):
                    # TODO: disabled wrong deletion of parameter
                    pass

        self.recreate_op_list()

    def __repr__(self):
        return (
            f"OpSchedule takes {sum(self.time):.2f}ms and cost {self.mem//1024**2} MiB"
        )

    @property
    def peak_mem(self):
        return self.get_peak_mem()
    
    def get_peak_mem(self, with_interface=False, act_multiplier=1):
        if act_multiplier==1:
            save_mem = self.save_mem
            overhead = self.overhead
        else:
            save_mem = []
            for i,alive_status in enumerate(self.alive_list):
                save_mem.append(self.param_mem(i, act_multiplier=act_multiplier)+self.act_mem(i))
            overhead = self.overhead*act_multiplier
        if with_interface:
            return max(np.array(save_mem) + overhead+ self.interface_mem)
        return max(np.array(save_mem) + overhead)
    
    def param_mem(self, i, act_multiplier=1):
        alive_status = self.alive_list[i]
        return sum(self.dict_alloc[a].mem*alive_status[a]
                                    *(act_multiplier if isinstance(self.dict_alloc[a], Parameter) else 0)
                                    for a in alive_status)
    
    def act_mem(self, i):
        alive_status = self.alive_list[i]
        return sum(self.dict_alloc[a].mem*alive_status[a]
                                    *(1 if not isinstance(self.dict_alloc[a], Parameter) else 0)
                                    for a in alive_status)
    
    def _sum_mem(self,alive_status_, ignore_list=[]):
            mem = 0
            for k, v in alive_status_.items():
                if k not in ignore_list and v:
                    d = self.dict_alloc[k]
                    mem += d.mem
            return mem
    
    @property
    def simulation_overhead(self):
        all_compute = set(
            op for op in self.op_list if (not op.disabled and isinstance(op, ComputeOp))
        )
        return (
            sum(op.target.time for op in all_compute)
            / sum(op.target.time for op in all_compute)
            - 1
        )
    
    def optimizer_states_size(self):
        optim_size = 0
        for op in self.op_list:
            if isinstance(op, OptimizeOp) and not op.is_cpu:
                optim_size += sum(self.dict_alloc[anode].mem for anode in op.list_params)
        return optim_size
    
    def cpu_optimize_size(self):
        optim_size = 0
        for op in self.op_list:
            if isinstance(op, OptimizeOp) and op.is_cpu:
                optim_size += sum(self.dict_alloc[anode.strip("cpu_")].mem for anode in op.list_params)
        return optim_size

    @property
    def op_list(self):
        return self._op_list

    @property
    def op_name_list(self):
        return self._op_name_list

    def recreate_op_list(self):
        self.loss_op = self.op_list[self.loss_idx]
        if self.from_steps:
            op_list = []
            for step in self.steps:
                op_list += step.op_list
            self._op_list = op_list
        self._op_name_list = [
            (str(op) if not op.disabled else "") for op in self._op_list
        ]
        self.loss_idx = self._op_list.index(self.loss_op)
