# ==========================
# definition file of RK_Sequence
# based on rotor/algorithms/sequence.py
# ==========================

from rockmate.def_op import OpSchedule

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
    def __init__(self, name, index, op_sched: OpSchedule):
        self.name = name
        self.index = index
        self.op_sched = op_sched
        self.time = self.op_sched.time  # sum(o.time for o in body)
        self.mem = self.op_sched.save[-1]  # sum(o.mem  for o in body)
        self.overhead = self.op_sched.overhead

    def __str__(self):
        header = f"{self.name} Block {self.index} in {self.time}"
        if ref_print_atoms[0]:
            sb = "\n|  ".join([o.__str__() for o in self.op_sched.op_list])
            return f"{header}\n{sb}"
        else:
            return header


class SeqBlockFn(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fn", *args)


class SeqBlockFc(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fc", *args)


class SeqBlockFe(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fe", *args)


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


# ==========================
