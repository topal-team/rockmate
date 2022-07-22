from .utils import *
from .use_chk import make_sched, Sched_to_code

class RK_fwd_option():
    def __init__(self,kg,budget):
        sched_result, chk_g = use_chk.make_sched(kg, budget)
        Translator = use_chk.Sched_to_Code(chk_g,kg)
        fwd_code,bwd_code = Translator.generate_sched_code(sched_result)
        self.code_fwd = "\n".join(fwd_code)
        self.code_bwd = "\n".join(bwd_code)
        self.time_fwd = 0
        self.time_bwd = 0
        self.mem_peak_fwd = 0
        self.mem_peak_bwd = 0
        self.size_a_bar = 0

class RK_block():
    def __init__(self,kg):
        # -- measure max budget --
        # -- try n different relevant budgets --
        self.fwd_opts : RK_fwd_option list
        self.code_fast_fwd : str
        self.code_forget_fwd : str
        self.code_forget_inp : str
        self.mem_inp : int
        self.time_fwd : int # for f in self.fwd_opts, assert(f.time_fwd==self.time_fwd)

class RK_chain():
    def __init__(self,list_kg): # K_graph list
        self.blocks : RK_block list

