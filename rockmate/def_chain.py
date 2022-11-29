# ==========================
# definition file of RK_Chain
# also contains RK_Chain builder -> depends on use_chk.py
# based on rotor/algorithms/parameters.py
# ==========================

from .utils import *
from .use_chk import make_sched
from .def_code import Op, OpBlock
import math
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
        self.budget_abar = budget_abar
        self.budget_all = budget_all

        kg.loss_node.run_mem = MemSize(budget_all - budget_abar)
        self.sched_result, self.op_list, self.chk_g = make_sched(kg, budget_all)
        is_f = self.is_feasible = self.sched_result.feasible
        if is_f:
            #TODO: find a better way to split
            for i,op in enumerate(self.op_list):
                if "loss" in op.n.name:
                    loss_i = i
                    break
            self.op_block_fwd = OpBlock(self.op_list[:loss_i+1])
            self.op_block_bwd = OpBlock(self.op_list[loss_i+1:])
            self.time_fwd = self.op_block_fwd.time
            self.time_bwd = self.op_block_bwd.time

            #fwd_overhead,fwd_save = op_block_fwd.mem_timeline()
            #bwd_overhead,bwd_save = op_block_bwd.mem_timeline()

            self.size_a_bar = self.op_block_fwd.save
            self.overhead_fwd = self.op_block_fwd.overhead 
            #self.overhead_bwd = self.op_block_bwd.overhead 
            #quick fix:
            self.overhead_bwd = self.op_block_bwd.overhead+self.op_block_bwd.save#-self.size_a_bar


# RK_Block :
# -> Attributes : .block_name               -> str
#                 .sols                     -> RK_Block_Solution list
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
        nodes_size = [n.run_mem.v for n in kg.dict_nodes.values()]
        max_bdg = sum(nodes_size)+max(nodes_size)
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
        kg.loss_node.run_mem = MemSize(0)

        # == build .mem_inp/out ==
        memsize = lambda inp : kg.dict_info[inp].memsize.v
        self.mem_inp = sum([memsize(inp) for inp in kg.hidden_inputs])
        self.mem_out = memsize(kg.hidden_output)

        # == build fast_forward code ==
        fwd_nodes = sort_based_on_req(kg.loss_node)[:-1] # from pgb/utils
        #fwd_nodes should contains only the nodes from the current Kgraph
        code_ff = []
        op_list_fc = []
        op_list_fn = []
        nodes_done = set()
        current_mem = 0 ; mem_timeline = []
        def fwd_n(n):
            nonlocal current_mem, mem_timeline
            current_mem += memsize(n.main_target)
            mem_timeline.append(current_mem)
            #code_ff.append(CodeAtom(
            #    code=n.get_code(),
            #    is_fgt=False,
            #    n=n))
            op_list_fc.append(Op(is_fgt=False,n=n))
            op_list_fn.append(Op(is_fgt=False,n=n))
            nodes_done.add(n)
            for req_n in n.req_global: try_del(req_n)
        def try_del(n):
            is_fwd = lambda un : un.is_fwd and not un is kg.loss_node
            b = True
            for un in n.used_by_global:
                if is_fwd(un) and not un in nodes_done and un in fwd_nodes:
                    b = False
            if b:
                op_list_fn.append(Op(is_fgt=True,n=n))
                if n in fwd_nodes:
                    op_list_fc.append(Op(is_fgt=True,n=n))
        for n in fwd_nodes: fwd_n(n)

        # = build .code_fc =
        #self.code_fc = CodeBlock(code_ff)
        self.op_block_fc = OpBlock(op_list_fc)
        self.op_block_fn = OpBlock(op_list_fn)#TODO:add fgt outputs node from the previous 
        # = build .overhead_ff =
        #self.overhead_ff = max(mem_timeline) - self.mem_out
        self.overhead_ff = self.op_block_fc.overhead 
        # = build .time_ff ==
        #self.time_ff = sum([n.time for n in fwd_nodes])
        self.time_ff = self.op_block_fc.time
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
    def __init__(self,list_kg, nb_budget_abar=10,nb_budget_all=3, mem_unit=None):
        if mem_unit:self.mem_unit = mem_unit
        else: self.mem_unit = 1024**2
        l = self.body = []
        for g in list_kg:
            l.append(RK_Block(g,nb_budget_abar,nb_budget_all))
            print_debug(l[-1])
        # organizes the information for rotor_solver.py as in Rotor
        # -> fw/bw/cw/cbw/fwd_tmp/bwd_tmp
        # -> in those list, one dummy block is added at the end for Loss
        # fw/bw: runtime of fwd/bwd
        # cbw: saved memory in each solution
        # cw: saved memory for each checkpoint solution (only the input)

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
        self.cw     = self.discretize(cw)
        self.cbw    = [self.discretize(x) for x in cbw]
        self.fwd_tmp    = [self.discretize(x) for x in fwd_tmp]
        self.bwd_tmp    = [self.discretize(x) for x in bwd_tmp]
        self.ff_fwd_tmp = self.discretize(ff_fwd_tmp)
        self.ff_fw  = ff_fw
        self.nb_sol = nb_sol

    def discretize(self, values):
        return [math.ceil(value/self.mem_unit) for value in values]

# ==========================

