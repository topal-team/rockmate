from .utils import *
from .use_chk import make_sched, Sched_to_ops
import numpy as np

class RK_block_solution():
    def __init__(self,kg,budget_abar,budget_all):
        kg.loss_node.fgt_mem = MemSize(budget_all - budget_abar)
        sched_result, chk_g = make_sched(kg, budget_all)
        is_f = self.is_feasible = sched_result.feasible
        if is_f:
            Translator = Sched_to_ops(chk_g,kg)
            fwd_ops,bwd_ops = Translator.generate_sched_ops(sched_result)
            def ops_stats(ops):
                N = len(ops)
                overhead = np.zeros(N)
                save = np.zeros(N)
                for i,op in enumerate(ops):
                    if op.is_fgt:
                        save[i:] -= op.node.fgt_mem.v
                    else:
                        save[i:] += op.node.fgt_mem.v
                    overhead[i] = op.node.overhead.v
                return overhead, save
            
            fwd_overhead,fwd_save = ops_stats(fwd_ops)
            bwd_overhead,bwd_save = ops_stats(bwd_ops)

            self.code_fwd = "\n".join(op.code for op in fwd_ops)
            self.code_bwd = "\n".join(op.code for op in bwd_ops)
            self.time_fwd = sum([op.time for op in fwd_ops])
            self.time_bwd = sum([op.time for op in bwd_ops])
            self.size_a_bar = fwd_save[-1]
            self.overhead_fwd = max(fwd_overhead+fwd_save) - fwd_save[-1]
            self.overhead_bwd = max(bwd_overhead+bwd_save) - bwd_save[-1]
# Check if o_b is what you need
# I need self.overhead_fwd/bwd in RK_block_solution

class RK_block():
    # self.bloc_name : str
    # self.sols : RK_block_solution list
    # self.code_fast_fwd : str list
    #  -> compute self's output, everything else is deleted (del) -> F_c
    # self.ff_overhead : int
    #  -> the overhead during fast forward computation
    #  -> TODO : for now, the way I compute it is wrong
    # self.code_fgt_inp : str
    #  -> use it after code_fast_fwd to make F_n
    # self.mem_inp/out : int
    #  -> replace cweight
    # self.time_ff : int 

    def __init__(self,kg,nb_budget_abar,nb_budget_all):
        self.block_name = (
            f"Block[{kg.hidden_inputs}->{kg.direct_outputs}]")
        # === apply chk nn_budget_abar*nb_budget_all times ===
        size_nodes = [n.fgt_mem.v for n in kg.dict_nodes.values()]
        max_budget  = sum(size_nodes)
        highest_mem = max(size_nodes)
        print_debug(
            f"=*=*=*=\nStart {self.block_name}, total cost : "\
            f"{max_budget} and highest_mem : {highest_mem}\n=*=*=*="
            )

        sols = self.sols = []
        l_bd_abar = np.linspace(highest_mem,max_budget,nb_budget_abar)
        l_bd_all  = np.linspace(highest_mem,max_budget,nb_budget_all+2)[2:]
        uniq_sols = set()
        for bd_abar in l_bd_abar:
            for bd_all in l_bd_all:
                if bd_all >= bd_abar:
                    print_debug(
                        f"ask {self.block_name} with : bd_abar = "\
                        f"{bd_abar} and bd_all = {bd_all}")
                    sol = RK_block_solution(kg,bd_abar,bd_all)
                    if sol.is_feasible:
                        t = (sol.size_a_bar,
                            sol.overhead_fwd,
                            sol.overhead_bwd)
                        if not (t in uniq_sols):
                            uniq_sols.add(t)
                            sols.append(sol)
        kg.loss_node.fgt_mem = MemSize(0)
        # ====================================================

        # = -> mem_inp =
        memsize = lambda inp : kg.dict_info[inp].memsize.v
        self.mem_inp = sum([memsize(inp) for inp in kg.hidden_inputs])
        self.mem_out = memsize(kg.hidden_output)

        # === other things needed for rotor ===
        # = -> code_fast_fwd : compute output but del intermediate var =
        # = -> I also compute ff_overhead =
        fwd_nodes = sort_based_on_req(kg.loss_node)[:-1] # from pgb/utils
        ff = []
        nodes_done = set()
        current_mem = 0 ; mem_timeline = []
        def fwd_n(n):
            nonlocal current_mem, mem_timeline
            current_mem += memsize(n.main_target)
            mem_timeline.append(current_mem)
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
                nonlocal current_mem, mem_timeline
                current_mem -= memsize(n.main_target)
                mem_timeline.append(current_mem)
                s = ", ".join(n.all_targets)
                ff.append(piece_of_code(
                    f"del {s}",
                    f"Del : {n.main_target} ({s})"))
        for n in fwd_nodes: fwd_n(n)
        self.code_fast_fwd = bloc_of_code(ff,
            f"Fast forward {self.block_name}")
        self.ff_overhead = max(mem_timeline) - self.mem_out

        # = -> code_fgt_inp =
        s = ", ".join(kg.direct_inputs)
        self.code_fgt_inp = piece_of_code(
            f"del {s}",
            f"Del inputs : {s}")

        # = -> time_ff =
        self.time_ff = sum([n.time for n in fwd_nodes])
        # =====================================

    def __repr__(self):
        s = "..."
        # s = f"\n\t{self.code_fast_fwd}\n\t====="
        return (
          f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"\
          f"{self.block_name} :\n"\
          f"\tnb of sol : {len(self.sols)}\n"\
          f"\tmem_inp   : {self.mem_inp}\n"\
          f"\ttime_ff  : {self.time_ff}\n"\
          f"\t== FF == : {s}\n"\
          f"\tcode_fgt_inp : {self.code_fgt_inp}"\
          f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


class RK_chain():
    def __init__(self,list_kg,nb_budget_abar=10,nb_budget_all=3):
        l = self.blocks = []
        for g in list_kg:
            l.append(RK_block(g,nb_budget_abar,nb_budget_all))
            print_debug(l[-1])

    def build_rotor_chain(self):
        # organizes the information for rotor_solver.py as in Rotor
        # -> fw/bw/cw/cbw/fwd_tmp/bwd_tmp
        # -> in those list, one dummy block is added at the end for Loss

        # -- init variables --
        ln = len(self.blocks)
        mkl = lambda n : [[] for _ in range(n)]
        fw = mkl(ln+1)      ; bw  = mkl(ln+1)
        cw = [None]*(ln+2)  ; cbw = mkl(ln+2)
        fwd_tmp = mkl(ln+1) ; bwd_tmp = mkl(ln+1)
        ff_fwd_tmp = [None]*(ln+1)
        ff_fw      = [None]*(ln+1)
        nb_sol = []

        # -- extract info from each block
        for (i,b) in enumerate(self.blocks):
            nb_sol.append(len(b.sols))
            if nb_sol[-1]==0:
                raise Exception(
                    f"We need at least one solution per block. "\
                    f"Here {b.block_name} has no solution")
            for sol in b.sols:
                fw[i].append(sol.time_fwd)
                bw[i].append(sol.time_bwd)
                cbw[i+1].append(sol.size_a_bar)
                fwd_tmp[i].append(sol.overhead_fwd)
                bwd_tmp[i].append(sol.overhead_bwd)
            cw[i] = b.mem_inp
            ff_fwd_tmp[i] = b.ff_overhead
            ff_fw[i] = b.time_ff
        cw[ln]=self.blocks[-1].mem_out # the final output

        # for the Loss block :
        nb_sol.append(1)
        fw[-1] = [0]    ; bw[-1] = [0]
        cw[-1] = 0      ; cbw[-1] = [0]
        fwd_tmp[-1]=[0] ; bwd_tmp[-1] = [0]
        ff_fwd_tmp[-1] = 0
        ff_fw[-1] = 0

        # return :
        self.ln     = ln
        self.fw     = fw
        self.bw     = bw
        self.cw     = cw
        self.cbw    = cbw
        self.fwd_tmp    = fwd_tmp
        self.bwd_tmp    = bwd_tmp
        self.ff_fwd_tmp = ff_fwd_tmp
        self.ff_fw  = ff_fw
        self.nb_sol = nb_sol




