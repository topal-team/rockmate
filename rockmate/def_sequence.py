# ==========================
# definition file of RK_Sequence
# based on rotor/algorithms/sequence.py
# ==========================

from .utils import *

# ==========================
# ====== Seq Atom Op =======
# ==========================
# -> Attributes : .name ; .var ; .mem/time_cost 
# -> Methods    : __str__
# -> Subclasses : Fwd/Bwd/Fgt/Del
class SeqAtomOp:
    def __init__(self,name,var,time_cost,mem_cost):
        self.name=name ; self.var=var
        self.time_cost = time_cost
        self.mem_cost  =  mem_cost
    def __str__(self):
        return f"{self.name} {self.var}"

class SeqAtomFwd(SeqAtomOp):
    def __init__(self,*args): super().__init__("Fwd",*args)
class SeqAtomBwd(SeqAtomOp):
    def __init__(self,*args): super().__init__("Bwd",*args)
class SeqAtomFgt(SeqAtomOp):
    def __init__(self,*args): super().__init__("Fgt",*args)
class SeqAtomDel(SeqAtomOp):
    def __init__(self,*args): super().__init__("Del",*args)
# ==========================



# ==========================
# ========= Seq Op =========
# ==========================
# -> Attributes : .name ; .mem/time_cost
# -> Methods    : __str__
# -> Subclasses : Loss,BlockOp
class SeqOp: pass

class SeqLoss(SeqOp):
    def __init__(self):
        self.time_cost = 0
        self.mem_cost  = 0
    def __str__(self):
        return "Loss"
# ==========================



# ==========================
# ====== Seq Block Op ======
# ==========================
# -> Attributes : .body(SeqAtomOp list) ; .name ; .var ; .mem/time_cost
# -> Methods    : __str__
# -> Subclasses : Fn,Fc,Fe,Bwd
class SeqBlockOp(SeqOp):
    def __init__(self,name,index,body):
        self.name=name ; self.index=index ; self.body=body
        self.time_cost = sum(o.time_cost for o in body)
        self.mem_cost  = sum(o.mem_cost  for o in body)
    def __str__(self):
        header = f"{self.name} Block {self.index} in {self.time_cost}"
        if ref_print_atoms[0]:
            sb = "\n|  ".join([o.__str__() for o in body])
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
# -> Methods    : insert ; insert_sequence ;
#                 __str__ ; plot_mem ; compute_time
class RK_Sequence:
    def __init__(self):
        self.seq = []
    def __str__(self):
        return "\n".join(str(self.seq))
    def insert(self, op : SeqOp):
        self.seq.append(op)
    def insert_sequence(self,sub_seq):
        self.extend(sub_seq)
    def plot_mem(self):
        mem=0 ; l = [mem]
        if ref_print_atoms[0]:
            for blockop in self.seq:
                for o in blockop.body:
                    mem+=o.mem_cost
                    l.append(mem)
        else:
            for blockop in self.seq:
                mem+=blockop.mem_cost
                l.append(mem)
        plt.plot(l)
        plt.title("Theoretical : memory used over time")
        plt.show()
    def compute_time(self):
        return sum([o.time_cost for o in self.seq])
# ==========================



