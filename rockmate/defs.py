from .utils import *
from .use_chk import make_sched, Sched_to_ops
import numpy as np
class RK_block_solution():
    def __init__(self,kg,budget_abar,budget_all):
        kg.loss_node.fgt_mem = MemSize(budget_all - budget_abar)
        sched_result, chk_g = make_sched(kg, budget_all)
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
    def __init__(self,kg,nb_budget_abar,nb_budget_all):
        # -- measure max/min budget --
        max_budget  = sum(dict_nodes.values())
        highest_mem = max(dict_nodes.values())

        # -- try n different relevant budgets --
        sols = []
        self.fwd_opts = sols
        l_bd_abar = np.linspace(highest_mem,max_budget,nb_budget_abar)
        l_bd_all  = np.linspace(highest_mem,max_budget,nb_budget_all+2)[2:]
        for bd_abar in l_bd_abar:
            for l_bd_all:
                sol = RK_block_solution(kg,bd_abar,bd_all)
                if sol.is_feasible:
                    sols.append(sol)

        self.fwd_opts : RK_fwd_option
        self.code_fast_fwd : str
        self.code_forget_fwd : str
        self.code_forget_inp : str
        self.mem_inp : int
        self.time_fwd : int # for f in self.fwd_opts, assert(f.time_fwd==self.time_fwd)

class RK_chain():
    def __init__(self,list_kg,nb_budget_abar=10,nb_budget_all=3): # K_graph list
        self.blocks : RK_block

