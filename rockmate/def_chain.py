# ==========================
# definition file of RK_Chain
# also contains RK_Chain builder -> depends on use_chk.py
# based on rotor/algorithms/parameters.py
# ==========================

from .utils import *
from .use_chk import make_sched, Sched_to_ops
from .def_code import CodeAtom, CodeBlock

# ==========================
# ======== RK Block ========
# ==========================
# RK_Block_Solution :
# -> Attributes : .code_fwd ; .code_bwd -> CodeBlock
#                 .time_fwd ; .time_bwd -> int
#                 .overhead_fwd ; .overhead_bwd ; .size_a_bar -> int
# -> Methods    : __init__
class RK_Block_Solution():
    def __init__(self,kg,budget_abar,budget_all):
        kg.loss_node.fgt_mem = MemSize(budget_all - budget_abar)
        sched_result, chk_g = make_sched(kg, budget_all)
        is_f = self.is_feasible = sched_result.feasible
        if is_f:
            Translator = Sched_to_ops(chk_g,kg)
            code_fwd,code_bwd = Translator.generate_sched_ops(sched_result)

            fwd_overhead,fwd_save = code_fwd.mem_timeline()
            bwd_overhead,bwd_save = code_bwd.mem_timeline()

            self.code_fwd = code_fwd
            self.code_bwd = code_bwd
            self.time_fwd = sum([op.time for op in code_fwd.body])
            self.time_bwd = sum([op.time for op in code_bwd.body])
            self.size_a_bar = fwd_save[-1]
            self.overhead_fwd = max(fwd_overhead+fwd_save) - fwd_save[-1]
            self.overhead_bwd = max(bwd_overhead+bwd_save) - bwd_save[-1]


# RK_Block :
# -> Attributes : .block_name               -> str
#                 .sols                     -> RK_Block_Solution list
#                 .code_fn ; .code_fc       -> CodeBlock
#                 .overhead_ff ; .time_ff   -> int
#                 .mem_inp ; .mem_out       -> int
#                 .overhead_fwd ; .overhead_bwd ; .size_a_bar -> int
# -> Methods    : __init__
# -> NB :
#       ff means fast forward ie Fc ~ Fn
#       TODO improve how overhead_ff is computed
class RK_Block():
    def __init__(self,kg,nb_bdg_abar,nb_bdg_all):
        self.block_name = (
            f"Block[{kg.hidden_inputs}->{kg.direct_outputs}]")

        # == budgets to test ==
        nodes_size = [n.fgt_mem.v for n in kg.dict_nodes.values()]
        max_bdg = sum(nodes_size)
        min_bdg = max(nodes_size)
        #l_bd_abar = np.linspace(min_bdg,max_bdg,nb_bdg_abar)
        l_bd_abar = np.linspace(0,max_bdg,nb_bdg_abar)
        l_bd_all  = np.linspace(min_bdg,max_bdg,nb_bdg_all+2)[2:]
        print_debug(
            f"=*=*=*=\nStart {self.block_name}, total cost : "\
            f"{max_bdg} and min_bdg : {min_bdg}\n=*=*=*="
            )

        # == build .sols ==
        sols = self.sols = []
        uniq_sols = set()
        for bd_abar in l_bd_abar:
            for bd_all in l_bd_all:
                if bd_all >= bd_abar:
                    print_debug(
                        f"ask {self.block_name} with : bd_abar = "\
                        f"{bd_abar} and bd_all = {bd_all}")
                    sol = RK_Block_Solution(kg,bd_abar,bd_all)
                    if sol.is_feasible:
                        t = (sol.size_a_bar,
                            sol.overhead_fwd,
                            sol.overhead_bwd)
                        if not (t in uniq_sols):
                            uniq_sols.add(t)
                            sols.append(sol)
        kg.loss_node.fgt_mem = MemSize(0)

        # == build .mem_inp/out ==
        memsize = lambda inp : kg.dict_info[inp].memsize.v
        self.mem_inp = sum([memsize(inp) for inp in kg.hidden_inputs])
        self.mem_out = memsize(kg.hidden_output)

        # == build fast_forward code ==
        fwd_nodes = sort_based_on_req(kg.loss_node)[:-1] # from pgb/utils
        code_ff = []
        nodes_done = set()
        current_mem = 0 ; mem_timeline = []
        def fwd_n(n):
            nonlocal current_mem, mem_timeline
            current_mem += memsize(n.main_target)
            mem_timeline.append(current_mem)
            code_ff.append(CodeAtom(
                code=n.get_code(),
                is_fgt=False,
                n=n))
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
                #s = ", ".join(n.all_targets)
                #code_ff.append(CodeAtom(
                #    code=f"del {s}",
                #    is_fgt=None,
                #    n=n))
                code = ""
                v = n.main_target
                code += f"{v}.data = torch.zeros(0,device=device); "
                #TODO: detect if _{v} exists outside code
                code += f"\nif {v}.requires_grad:\n\t_{v}.data = torch.zeros(0,device=device);"
                code_ff.append(CodeAtom(
                    code=code,
                    is_fgt=True,
                    n=n))
                #for v in n.tensor_targets:
                #    code += (f"{v}.data = torch.zeros(0,device=device); ")
                #    code_ff.append(CodeAtom(
                #        code=code,
                #        is_fgt=None,
                #        n=n))
        for n in fwd_nodes: fwd_n(n)

        # = build .code_fc =
        self.code_fc = CodeBlock(code_ff)
        # = build .code_fn =
        #s = ", ".join(kg.direct_inputs)
        s = "".join(kg.direct_inputs).strip("src")
        code_fgt_inp = CodeAtom(
            code=f"del {s}",
            #n=kg.dict_nodes["fwd_"+s],
            #code=f"{s}.data = torch.zeros(0,device=device);",
            is_fgt=True,
            main_var=kg.direct_inputs[0],
            lvars=kg.direct_inputs,
            is_fwd=True,
            time=0,
            mem=self.mem_inp)
        #self.code_fn = CodeBlock(code_ff+[code_fgt_inp])
        self.code_fn = CodeBlock(code_ff)#TODO: fix Fn by remove the output node of the last block

        # = build .overhead_ff =
        self.overhead_ff = max(mem_timeline) - self.mem_out
        # = build .time_ff ==
        self.time_ff = sum([n.time for n in fwd_nodes])
    # =====================================

    def __str__(self):
        s = "..."
        # s = f"\n\t{self.code_fast_fwd}\n\t====="
        return (
          f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"\
          f"{self.block_name} :\n"\
          f"\tnb of sol : {len(self.sols)}\n"\
          f"\tmem_inp   : {self.mem_inp}\n"\
          f"\ttime_ff  : {self.time_ff}\n"\
          f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# ==========================



# ==========================
# ======== RK CHAIN ========
# ==========================

class RK_Chain():
    def __init__(self,list_kg,nb_budget_abar=10,nb_budget_all=3):
        l = self.body = []
        for g in list_kg:
            l.append(RK_Block(g,nb_budget_abar,nb_budget_all))
            print_debug(l[-1])

    def build_rotor_chain(self):
        # organizes the information for rotor_solver.py as in Rotor
        # -> fw/bw/cw/cbw/fwd_tmp/bwd_tmp
        # -> in those list, one dummy block is added at the end for Loss

        # -- init variables --
        ln = len(self.body)
        mkl = lambda n : [[] for _ in range(n)]
        fw = mkl(ln+1)      ; bw  = mkl(ln+1)
        cw = [None]*(ln+2)  ; cbw = mkl(ln+2)
        fwd_tmp = mkl(ln+1) ; bwd_tmp = mkl(ln+1)
        ff_fwd_tmp = [None]*(ln+1)
        ff_fw      = [None]*(ln+1)
        nb_sol = []

        # -- extract info from each block
        for (i,b) in enumerate(self.body):
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
            ff_fwd_tmp[i] = b.overhead_ff
            ff_fw[i] = b.time_ff
        cw[ln]=self.body[-1].mem_out # the final output

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

# ==========================

