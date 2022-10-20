# ==========================
# definition file of RK_Sequence
# based on rotor/algorithms/sequence.py
# ==========================

from .utils import *
from .def_code import RK_Storage, RK_Function
from .def_code import Op, OpBlock

# ==========================
# ====== Seq Atom Op =======
# ==========================
# -> Attributes : .name(str) ; .mem(int) ; .time(int)
# -> Methods    : __str__
class SeqAtomOp:
    def __init__(self,op : Op):
        lvars = list(op.lvars)
        try: lvars.remove(op.main_var)
        except: pass
        header = f"{op.op_type} {op.main_var}"
        if lvars == list():
            self.name = header
        else:
            s = ",".join(lvars)
            self.name = f"{header} ({s})"
        self.time = op.time
        self.mem  = op.mem
        self.op = op
    def __str__(self):
        return self.name
# ==========================



# ==========================
# ========= Seq Op =========
# ==========================
# -> Attributes : .name ; .mem/time
# -> Methods    : __str__
# -> Subclasses : Loss,BlockOp
class SeqOp: pass

class SeqLoss(SeqOp):
    def __init__(self):
        self.time = 0
        self.mem  = 0
    def __str__(self):
        return "Loss"
# ==========================



# ==========================
# ====== Seq Block Op ======
# ==========================
# -> Attributes : .body(SeqAtomOp list) ; .name ; .var ; .mem/time
# -> Methods    : __str__
# -> Subclasses : Fn,Fc,Fe,Bwd
class SeqBlockOp(SeqOp):
    def __init__(self,name,index,op_block : OpBlock):
        self.name=name ; self.index=index
        body = self.body = [SeqAtomOp(o) for o in op_block.body]
        self.time = sum(o.time for o in body)
        self.mem  = sum(o.mem  for o in body)
        self.overhead = op_block.overhead
    def __str__(self):
        header = f"{self.name} Block {self.index} in {self.time}"
        if ref_print_atoms[0]:
            sb = "\n|  ".join([o.__str__() for o in self.body])
            return f"{header}\n{sb}"
        else:
            return header

class SeqBlockFn(SeqBlockOp):
    def __init__(self,*args): super().__init__("Fn",*args)
class SeqBlockFc(SeqBlockOp):
    def __init__(self,*args): super().__init__("Fc",*args)
class SeqBlockFe(SeqBlockOp):
    def __init__(self,*args): super().__init__("Fe",*args)
class SeqBlockBwd(SeqBlockOp):
    def __init__(self,*args): super().__init__("Bwd",*args)
# ==========================



# ==========================
# ======== Sequence ========
# ==========================
# -> Attributes : seq (SeqOp list)
# -> Methods    : insert ; insert_seq ;
#                 __str__ ; plot_mem ; compute_time
class RK_Sequence:
    def __init__(self,l=None):
        if l: self.seq = l
        else: self.seq = []
    def __str__(self):
        return "\n".join([str(o) for o in self.seq])
    def insert(self, op : SeqOp):
        self.seq.append(op)
    def insert_seq(self,sub_seq):
        self.seq.extend(sub_seq.seq)
    def plot_mem(self):
        mem=0 ; l = [mem]
        if ref_print_atoms[0]:
            for blockop in self.seq:
                if not (isinstance(blockop,SeqLoss)):
                    for o in blockop.body:
                        mem+=o.mem
                        l.append(mem)
        else:
            for blockop in self.seq:
                mem+=blockop.mem
                l.append(mem)
        plt.plot(l)
        plt.title("Theoretical : memory used over time")
        plt.show()
    def compute_time(self):
        return sum([o.time for o in self.seq])
    def cut_fwd_bwd(self):
        ln = len(self.seq)
        for i in range(ln):
            if isinstance(self.seq[i],SeqLoss):
                return (RK_Sequence(list(self.seq[:i])),
                        RK_Sequence(list(self.seq[(i+1):])))
        raise Exception(
            f"Can't cut a Sequence which doesn't have SeqLoss")
# ==========================



