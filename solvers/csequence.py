class SeqOp:
    pass


class SeqLoss(SeqOp):
    def __init__(self):
        self.time = 0
        self.mem = 0

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
    def __init__(self, name, index, option):
        self.name = name
        self.index = index
        self.option = option

    def __str__(self):
        if self.option is None:
            return f"{self.name}_{self.index}"
        return f"{self.name}_{self.index}_{self.option}"

class SeqBlockFn(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fn", *args, None)


class SeqBlockFc(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fc", *args, None)


class SeqBlockFe(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Fe", *args)


class SeqBlockBwd(SeqBlockOp):
    def __init__(self, *args):
        super().__init__("Bwd", *args)

class RK_Sequence:
    def __init__(self, l=None):
        if l:
            self.seq = l
        else:
            self.seq = []

    def __str__(self):
        return ", ".join([str(o) for o in self.seq])

    def insert(self, op: SeqOp):
        self.seq.append(op)

    def insert_seq(self, sub_seq):
        self.seq.extend(sub_seq.seq)
