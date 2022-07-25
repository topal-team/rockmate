from .utils import *
from .use_chk import make_sched, Sched_to_ops
import numpy as np
class RK_fwd_option():
    def __init__(self,kg,budget):
        sched_result, chk_g = make_sched(kg, budget)
        Translator = Sched_to_ops(chk_g,kg)
        fwd_ops,bwd_ops = Translator.generate_sched_ops(sched_result)
        _fwd_mem = []
        _bwd_mem = []
        fwd_code = []
        bwd_code = []
        for op in fwd_ops:
            fwd_code.append(op.code)
            if op.is_fgt:
                _fwd_mem.append(-op.node.fgt_mem.v)
            else:
                _fwd_mem.append(op.node.fgt_mem.v)
        for op in bwd_ops:
            bwd_code.append(op.code)
            if op.is_fgt:
                _bwd_mem.append(-op.node.fgt_mem.v)
            else:
                _bwd_mem.append(op.node.fgt_mem.v)
        fwd_mem = np.cumsum(np.array(_fwd_mem))
        bwd_mem = np.cumsum(np.array(_bwd_mem))

        self.code_fwd = "\n".join(fwd_code)
        self.code_bwd = "\n".join(bwd_code)
        self.time_fwd = sum([op.node.time for op in fwd_ops])
        self.time_bwd = sum([op.node.time for op in bwd_ops])
        self.mem_peak_fwd = max(fwd_mem)
        self.mem_peak_bwd = max(bwd_mem)
        self.size_a_bar = fwd_mem[-1]

class RK_block():
    def __init__(self,kg):
        # -- measure max budget --
        # -- try n different relevant budgets --
        self.fwd_opts : RK_fwd_option
        self.code_fast_fwd : str
        self.code_forget_fwd : str
        self.code_forget_inp : str
        self.mem_inp : int
        self.time_fwd : int # for f in self.fwd_opts, assert(f.time_fwd==self.time_fwd)

class RK_chain():
    def __init__(self,list_kg): # K_graph list
        self.blocks : RK_block

