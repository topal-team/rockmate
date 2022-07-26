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

# I need self.overhead_fwd/bwd in RK_block_solution

class RK_block():
    # self.bloc_name : str
    # self.sols : RK_block_solution list
    # self.code_fast_fwd : str list
    #  -> compute self's output, everything else is deleted (del) -> F_c
    # self.code_fgt_inp : str
    #  -> use it after code_fast_fwd to make F_n
    # self.mem_inp/out : int
    #  -> replace cweight
    # self.time_fwd : int 

    def __init__(self,kg,nb_budget_abar,nb_budget_all):
        self.block_name = (
            f"Block[{kg.hidden_inputs}->{kg.direct_outputs}]")
        # === apply chk nn_budget_abar*nb_budget_all times ===
        size_nodes = [n.fgt_mem.v for n in kg.dict_nodes.values()]
        max_budget  = sum(size_nodes)
        highest_mem = max(size_nodes)

        sols = [] ; self.sols = sols
        l_bd_abar = np.linspace(highest_mem,max_budget,nb_budget_abar)
        l_bd_all  = np.linspace(highest_mem,max_budget,nb_budget_all+2)[2:]
        for bd_abar in l_bd_abar:
            for bd_all in l_bd_all:
                pass
                #sol = RK_block_solution(kg,bd_abar,bd_all)
                #if sol.is_feasible:
                #    sols.append(sol)
        kg.loss_node.fgt_mem = MemSize(0)
        # ====================================================

        # === other things needed for rotor ===
        # = -> code_fast_fwd : compute output but del intermediate var =
        fwd_nodes = sort_based_on_req(kg.loss_node)[:-1] # from pgb/utils
        ff = []
        nodes_done = set()
        def fwd_n(n):
            s = ", ".join(n.all_targets)
            ff.append(piece_of_code(
                n.get_code(),
                f"Fn : {n.main_target} ({s})"))
            nodes_done.add(n)
            for req_n in n.req: try_del(req_n)
        def try_del(n):
            is_fwd = lambda un : un.is_fwd and not un is kg.loss_node
            b = True
            for un in n.used_by:
                if is_fwd(un) and not un in nodes_done:
                    b = False
            if b:
                s = ", ".join(n.all_targets)
                ff.append(piece_of_code(
                    f"del {s}",
                    f"Del : {n.main_target} ({s})"))
        for n in fwd_nodes: fwd_n(n)
        self.code_fast_fwd = bloc_of_code(ff,
            f"Fast forward {self.block_name}")

        # = -> mem_inp =
        memsize = lambda inp : kg.dict_info[inp].memsize
        self.mem_inp = sum([memsize(inp).v for inp in kg.hidden_inputs])
        self.mem_out = memsize(kg.hidden_output)

        # = -> code_fgt_inp =
        s = ", ".join(kg.direct_inputs)
        self.code_fgt_inp = piece_of_code(
            f"del {s}",
            f"Del inputs : {s}")

        # = -> time_fwd =
        self.time_fwd = sum([n.time for n in fwd_nodes])
        for sol in self.sols:
            if sol.time_fwd != self.time_fwd:
                raise Exception(
                  f"One sol for {self.block_name} has a different "\
                  f"time_fwd : {sol.time_fwd} ; compared to "\
                  f"{self.time_fwd} for fast forward")
        # =====================================

    def __repr__(self):
        s = "..."
        # s = f"\n\t{self.code_fast_fwd}\n\t====="
        return (
          f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"\
          f"{self.block_name} :\n"\
          f"\tnb of sol : {len(self.sols)}\n"\
          f"\tmem_inp   : {self.mem_inp}\n"\
          f"\ttime_fwd  : {self.time_fwd}\n"\
          f"\t== FF == : {s}\n"\
          f"\tcode_fgt_inp : {self.code_fgt_inp}"\
          f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


class RK_chain():
    def __init__(self,list_kg,nb_budget_abar=10,nb_budget_all=3):
        l = [] ; self.blocks = l
        for g in list_kg:
            l.append(RK_block(g,nb_budget_abar,nb_budget_all))
            print_debug(l[-1])

    def build_rotor_chain(self):
        # organizes the information for rotor_solver.py as in Rotor
        # -> fw/bw/cw/cbw/fwd_tmp/bwd_tmp
        # -> in those list, one dummy block is added at the end for Loss
        ln = len(self.blocks)
        mkl = lambda n : [[] for _ in range(n)]
        fw = mkl(ln+1)      ; bw  = mkl(ln+1)
        cw = [None]*(ln+2)  ; cbw = mkl(ln+2)
        fwd_tmp = mkl(ln+1) ; bwd_tmp = mkl(ln+1)
        for (i,b) in enumerate(self.blocks):
            for sol in b.sols:
                fw[i].append(sol.time_fwd)
                bw[i].append(sol.time_bwd)
                cbw[i+1].append(sol.size_a_bar)
                fwd_tmp[i].append(sol.overhead_fwd)
                bwd_tmp[i].append(sol.overhead_bwd)
            cw[i] = b.mem_inp
        cw[ln]=self.blocks[-1].mem_out # the final output
        # for the Loss block :
        fw[-1] = [0]    ; bw[-1] = [0]
        cw[-1] = 0      ; cbw[-1] = [0]
        fwd_tmp[-1]=[0] ; bwd_tmp[-1] = [0]
        return (ln,fw,bw,cw,cbw,fwd_tmp,bwd_tmp)




