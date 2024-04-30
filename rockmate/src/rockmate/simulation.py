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
from .op_schedule import (
    Allocation,
    Activation,
    Parameter,
    Buffer,
    Op,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    OpSchedule,
    ExecCodeOp,
)


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
            if isinstance(op, SynchronizeOp) or isinstance(op, AllocateOp):
                self.alloc_ops.append(op)
            elif isinstance(op, OffloadOp) and not op.record_event:
                ofl_ops.append(op)
            elif isinstance(op, PrefetchOp) and not op.record_event:
                prf_ops.append(op)
            elif isinstance(op, OptimizeOp) and op.is_cpu:
                opt_ops.append(op)
            # elif isinstance(op, ComputeOp) or (
            #     isinstance(op, DeleteOp) and isinstance(op.target, Activation)) or(
            #     isinstance(op, OptimizeOp) and not op.is_cpu):
            #     # all happen in the main stream
            #     comp_ops.append(op)
            elif isinstance(op, DeleteOp) and isinstance(op.target, Parameter):
                self.del_ops.append(op)
            else:
                comp_ops.append(op)

        self.ofl_ops = ListOp(ofl_ops)
        self.prf_ops = ListOp(prf_ops)
        self.opt_ops = ListOp(opt_ops)
        self.comp_ops = ListOp(comp_ops)

    @property
    def op_list(self):
        
        if not self.opt_ops:
            opt_ops = []
        else:
            list_params = [p for op in self.opt_ops for p in op.list_params]
            opt_ops = self.opt_ops
        return (self.alloc_ops
                +self.ofl_ops
                +self.prf_ops
                +self.comp_ops
                +opt_ops
                +self.del_ops
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
        self.categories["activation"] = np.array([self.alloc_names.index(alloc_name) 
                                                  for alloc_name in self.alloc_names
                                                  if isinstance(self.dict_alloc[alloc_name], Activation)])
        self.categories["parameter"] = np.array([self.alloc_names.index(alloc_name)
                                                 for alloc_name in self.alloc_names
                                                 if isinstance(self.dict_alloc[alloc_name], Parameter)])
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

    def alive_once(self, key:int, idx_range:range):
        # being alive AT LEAST once
        if isinstance(idx_range, int):
            idx_range = range(idx_range, idx_range+1)
        if isinstance(key, str):
            key = self.name2key[key]
        return bool(max(self._alive(key, idx_range)))

    def alive_always(self, key:int, idx_range:range):
        # being alive AT LEAST once
        if isinstance(idx_range, int):
            idx_range = range(idx_range, idx_range+1)
        if isinstance(key, str):
            key = self.name2key[key]
        return bool(min(self._alive(key, idx_range)))

        

class Simulator:
    """
    With steps and np-alive status, simulator allows efficient
    simulation of swapping/changing operations. The final output
    of operation list can be used in op_sched updatation.
    """
    def __init__(self, op_sched: OpSchedule):
        self.op_sched = op_sched
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
        self.get_occurrences()
        
    def get_occurrences(self):
        self.occurrences = dict()
        for i, op in enumerate(self.op_list):
            if op.name in self.occurrences:
                self.occurrences[op.name].append(i)
            else:
                self.occurrences[op.name] = [i]

    def refine(self):
        """
        Disable deletion to avoid infeasible operations in the list.
        """
        for idx, op in enumerate(self.op_list):
            if op.disabled:continue
            if "loss" in op.name:
                op.disabled = True
            if isinstance(op, DeleteOp) and isinstance(op.target, Activation):
                used_idx = [i
                    for cnode in op.target.anode.users_real
                    for i in self.occurrences[ComputeOp(cnode).name]
                    if not "loss" in cnode.name
                    ]
                if not used_idx or max(used_idx) < idx:continue
                next_used_i = min(i for i in used_idx if i >idx)
                
                for cnode in op.target.anode.deps:
                    c_op = ComputeOp(cnode)
                    p_op = PrefetchOp(op.target)
                    if (not self.op_sched.is_occurred(c_op.name, idx, next_used_i)
                    and not self.op_sched.is_occurred(p_op.name, idx, next_used_i)):
                        op.disabled = True
                        break
                        