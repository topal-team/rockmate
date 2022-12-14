# ==========================
# everything about codes, how
# we exec it and how we store things
# -> to replace rotor/Checkpointable.functions
# ==========================
from typing import NamedTuple, Optional
from .utils import *
import numpy as np
# ==========================
# ===== CODE EXECUTOR ======
# ==========================
# -> TODO RK_Storage should help to handle random states
class RK_Storage:
    def __init__(self,device,nn_mod):
        self.gd = {**globals() , "self" : nn_mod , "device" : device}
        self.ld = dict()
    def add_val(self,val,x):
        self.ld[val]=x
    def get_val(self,val):
        try: return self.ld[val]
        except:
            try: return self.gd[val]
            except:
                raise Exception(f"{val} not in the storage")

# CodeAtom :
# -> attributes : .code : str (-> ast ?)
#                 .op_type : Fwd / Bwd / Del / FgtFwd / FgtBwd 
#                 .time ; .mem ; .overhead : int
#                 .main_var / .lvars
# CodeAtom.__init__ is polymorphic, either you
# give it a K_node or all the attributes needed
class CodeAtom: pass

# class Op:
#     def __init__(self,is_fgt,
#             n:pgb.Ktools.K_node):
#         self.is_fgt = is_fgt
#         self.n = n
#         self.overhead = n.overhead.v
#         if self.overhead is None: self.overhead = 0
#         self.main_var = n.main_target
#         self.lvars    = n.all_targets
#         self.run_mem = n.run_mem.v#for debugging
#         if is_fgt: # Fgt
#             self.time = 0#Fgt could happen in ILP solution even Infinity mem
#             self.mem  = - n.run_mem.v
#             if n.is_fwd: self.op_type = "FgtFwd"
#             else: self.op_type = "FgtBwd"
#         else:
#             self.time = n.time
#             self.mem  = n.run_mem.v
#             if n.is_fwd: self.op_type = "Fwd"
#             else: self.op_type = "Bwd"
#         self.name = self.op_type+" "+self.main_var

class RunOp():
    def __init__(self, kcn, keep_kcn=True):
        self.name = kcn.name
        self.time = kcn.time
        self.overhead = kcn.overhead.v
        self.main_target = kcn.main_target
        self.tensor_targets = kcn.tensor_targets
        # self.save_mem = cn.mem.v
        self.main_code = kcn.main_code
        self.body_code = kcn.body_code
        self.code = kcn.get_code()
        # self.deps_fake = [kdn.name for kdn in kcn.deps_fake]
        # self.deps_global = [kdn.name for kdn in kcn.deps_global]
        self.deps_global = kcn.deps_global
        self.deps_fake = kcn.deps_fake
        self.users_global = kcn.users_global
        if keep_kcn: self.kcn = kcn
        self.is_fgt = False
        self.op_type = "Run"
        self.proxy = False
        for kdn in kcn.users:
            if kdn.kdn_type != "data":continue
            self.proxy = kdn.info.requires_grad

class DelOp():
    def __init__(self, kdn):
        self.name = kdn.name
        self.kdn_type = kdn.kdn_type
        self.time = 0
        self.save_mem = kdn.mem.v
        self.main_target = kdn.main_target
        self.all_targets = kdn.all_targets
        # self.code = kn.get_code()
        # self.requires_grad = kdn.info.requires_grad
        self.info = kdn.info
        self.is_fgt = True
        self.op_type = "Del"

class OpSchedule:
    def __init__(self, op_list, alive_list, 
                 list_kdn, output=None, no_grad=False):
        self.no_grad = no_grad
        self.output = output
        self.mem_sizes = [kdn.mem.v for kdn in list_kdn]
        self.kdn_names = [kdn.name for kdn in list_kdn]
        self.op_list = op_list
        self.alive_list = alive_list
        L = len(op_list)
        self.save = np.zeros(L)
        self.tmp = np.zeros(L)
        self.is_fwd = True
        for i,op in enumerate(op_list):
            if isinstance(op, RunOp):
                self.tmp[i] = op.overhead
                if "bwd" in op.name: self.is_fwd=False
            self.save[i] = alive_list[i].dot(np.array(self.mem_sizes))
        self.overhead = max(self.save+self.tmp) - self.save[-1]
        self.time = sum([op.time for op in self.op_list])
        self.del_input_idx = -1

    def del_input(self, kg):
        input_kdn = kg.input_kdn_data
        self.del_input_op = DelOp(input_kdn)
        for i,op in enumerate(self.op_list):
            if isinstance(op, RunOp) and input_kdn in op.deps_global:
                self.del_input_idx = i+1
        # self.del_input_idx = max(self.op_list.index(kcn) 
        #                     for kcn in input_kdn.users_global
        #                     if kcn in self.op_list)
        self.save[self.del_input_idx+1:] -= input_kdn.mem.v
        self.overhead = max(self.save+self.tmp) - self.save[-1]
        

# class OpBlock:
#     def __init__(self, op_sched, alive_list):
#         self.op_sched = op_sched
#         self.alive_list = alive_list
#         save_mem = []
#         tmp_mem = []
#         for op, alive_status in zip(self.op_sched, self.alive_list):
#             if "loss" in op.name:continue
#             if isinstance(op, RunOp): save_mem.append(o.save_mem)
#             tmp_mem.append(o.overhead)
#         self.save_timeline = np.cumsum(np.array([0]+save_mem))
#         #self.overhead_timeline = np.cumsum(np.array(tmp_mem))
#         self.overhead_timeline = np.array(tmp_mem+[0])

#         self.save = self.save_timeline[-1]
#         self.overhead = max(self.save_timeline+self.overhead_timeline) - self.save
#         self.time = sum([op.time for op in self.op_sched])

class RK_Function:
    def __init__(self,code_fe,code_fn,code_fc,code_bwd):
        self.code_fe  = code_fe
        self.code_fn  = code_fn
        self.code_fc  = code_fc
        self.code_bwd = code_bwd
    def exec_fe (self,storage : RK_Storage): self.code_fe.exec(storage)
    def exec_fn (self,storage : RK_Storage): self.code_fn.exec(storage)
    def exec_fc (self,storage : RK_Storage): self.code_fc.exec(storage)
    def exec_bwd(self,storage : RK_Storage): self.code_bwd.exec(storage)
# ==========================

