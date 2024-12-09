# ==========================
# definition file of RK_Sequence
# based on rotor/algorithms/sequence.py
# ==========================

# from .def_op import OpSchedule
from ...op_schedule import Op
from typing import List
from copy import deepcopy

ref_print_atoms = [True]

# ==========================
#  ====== Seq Atom Op =======
# ==========================
class SeqAtomOp:
    def __init__(self, op):
        header = f"{op.op_type} {op.main_target}"
        self.name = header
        self.time = op.time
        self.op = op

    def __str__(self):
        return self.name


# ==========================


# ==========================
# ========= Seq Op =========
# ==========================
class SeqOp:
    pass


class SeqLoss(SeqOp):
    def __init__(self):
        self.time = 0
        self.mem = 0

    def __str__(self):
        return "Loss"


# ** Seq Block Op **
# -> Subclasses : Fn,Fc,Fe,Bwd
class SeqBlockOp(SeqOp):
    def __init__(self, name, index, op_list: List[Op]):
        self.name = name
        self.index = index
        self.op_list = op_list
        # self.time = self.op_sched.time  # sum(o.time for o in body)
        # self.mem = self.op_sched.save_mem[-1]  # sum(o.mem  for o in body)
        # self.overhead = self.op_sched.overhead

    def __str__(self):
        return f"{self.name}_{self.index}"
        # header = f"{self.name} Block {self.index} in {self.time}"
        # if ref_print_atoms[0]:
        #     sb = "\n|  ".join([o.__str__() for o in self.op_sched.op_list])
        #     return f"{header}\n{sb}"
        # else:
        #     return header


class SeqBlockFn(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fn", *args)


class SeqBlockFc(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fc", *args)


class SeqBlockFe(SeqBlockOp):
    def __init__(self, index, option, op_list):
        super().__init__("Fe", index, op_list)
        self.option = option
    def __str__(self):
        return f"{self.name}_{self.index}_{self.option}"

class SeqBlockBwd(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Bwd", *args)


# ==========================


# ==========================
# ======== Sequence ========
# ==========================


class RK_Sequence:
    def __init__(self, l=None):
        if l:
            self.seq = l
        else:
            self.seq = []

    def __str__(self):
        return "\n".join([str(o) for o in self.seq])

    def insert(self, op: SeqOp):
        self.seq.append(op)

    def insert_seq(self, sub_seq):
        self.seq.extend(sub_seq.seq)

    def compute_time(self):
        return sum([o.time for o in self.seq])

    def cut_fwd_bwd(self):
        ln = len(self.seq)
        for i in range(ln):
            if isinstance(self.seq[i], SeqLoss):
                return (
                    RK_Sequence(list(self.seq[:i])),
                    RK_Sequence(list(self.seq[(i + 1) :])),
                )
        raise Exception("Can't cut a Sequence which doesn't have SeqLoss")

    def get_op_list(self):
        return [ deepcopy(op) for rk_op in self.seq
                    for op in rk_op.op_list ]

    def simulate(self, chain, display=True, stopAtLoss=False):
        return simulate(chain, self, display, stopAtLoss)

    def get_peak_memory(self, chain):
        return self.simulate(self, chain, display=False, stopAtLoss=False)
    def get_loss_memory(self, chain):
        return self.simulate(self, chain, display=False, stopAtLoss=True)

        
# ==========================
## Code to estimate the makespan and memory usage of a given sequence

from enum import Enum

Data = Enum("Data", ["X", "Xbar", "Y"])
def make_x(idx):
    return (Data.X, idx)
def make_xb(idx, option):
    return (Data.Xbar, idx, option)
def make_y(idx):
    return (Data.Y, idx)
def make_x_and_xb(index):
    return make_x(index), make_xb(index)

class Storage:
    def __init__(self, chain):
        self.chain = chain
        self.xs = {}
        self.xbs = {}
        self.ys = {}

    def add_x(self, idx):
        self.xs[idx] = True
    def add_xb(self, idx, option):
        assert not self.has_xb(idx)
        self.xbs[idx] = option
    def add_y(self, idx):
        self.ys[idx] = True

    def remove_x(self, idx):
        del self.xs[idx]
    def remove_xb(self, idx):
        del self.xbs[idx]
    def remove_y(self, idx):
        del self.ys[idx]

    def remove_x_or_xb(self, idx):
        if self.has_x(idx):
            self.remove_x(idx)
        else:
            self.remove_xb(idx)

    def has_x(self, idx):
        return idx in self.xs
    def has_y(self, idx):
        return idx in self.ys
    def has_xb(self, idx):
        return idx in self.xbs

    def get_xb_option(self, idx):
        try: 
            return self.xbs[idx]
        except IndexError:
            return None
 
    def usage(self):
        result =  sum(self.chain.cw[i] for i in self.xs)
        result += sum(self.chain.cw[i] for i in self.ys)
        result += sum(self.chain.cbw[i][o] for i, o in self.xbs.items())
        return result

    def __str__(self):
        items = [(i, f"x{i}") for i in self.xs] + [(i, f"y{i}") for i in self.ys] + [(i, f"xb{i}_{o}") for i, o in self.xbs.items()]
        items.sort(key=lambda x: x[0])
        return ", ".join(name for _, name in items)

# def elementUsage(chain, e):
#     t, idx = e[0], e[1]
#     if t is Data.X or Data.Y:
#         return self.cw[idx]
#     else:
#         option = e[2]
#         return self.cbw[idx][option]

# def memUsage(self, storage):
#     return sum(self.elementUsage(e) for e in storage)

# Simulates the execution of the sequence
# Returns the maximum memory usage. If stopAtLoss, returns the memory usage at the time of computing Loss
def simulate(chain, sequence, display=True, stopAtLoss=False):
    if display:
        ref_print_atoms = [False]
    l = chain.ln
    mem = Storage(chain)
    mem.add_x(0)
    maxUsage = mem.usage()
    for op in sequence.seq:
        if display: 
            print("Before", op, f"Usage={mem.usage()}")
            print("Contents:", mem)
        opType = type(op)
        if opType is SeqLoss:
            if not mem.has_x(l) and not mem.has_xb(l):
                raise ValueError(f"Before {op}: no X{l} or Xb{l} in memory")
            mem.add_y(l)
            used = mem.usage()
            opUsage =  used + chain.bwd_tmp[l][0] ##Only one option for Loss
            if stopAtLoss:
                return used
        else:
            index = op.index
            if not mem.has_x(index) and not mem.has_xb(index):
                raise ValueError(f"Before {op}: no X{index} or Xb{index} in memory")
            if opType is SeqBlockFe:
                mem.add_xb(index+1, op.option)
                opUsage = mem.usage() + chain.fwd_tmp[index][op.option]
            if opType is SeqBlockFn:
                mem.add_x(index+1)
                opUsage = mem.usage() + chain.ff_fwd_tmp[index]
                mem.remove_x_or_xb(index)
            if opType is SeqBlockFc:
                mem.add_x(index+1)
                opUsage = mem.usage() + chain.ff_fwd_tmp[index]
            if opType is SeqBlockBwd:
                if not mem.has_y(index+1) or not mem.has_xb(index+1):
                    raise ValueError(f"Before {op}: no Y{index+1} or Xb{index+1} in memory")
                else:
                    mem.add_y(index)
                    option = mem.get_xb_option(index+1)
                    opUsage = mem.usage() + chain.bwd_tmp[index][option]
                    if mem.has_x(index):
                        mem.remove_x(index)
                    # Do not remove xb(index), it will be useful for BWD(index-1)
                    mem.remove_y(index+1)
                    mem.remove_xb(index+1)
        if display: print("During:", opUsage)
        maxUsage = max(maxUsage, opUsage)
    return maxUsage

