# ==========================
# rockmate interface with checkmate
# use checkmate to get the schedule
# and translate it from checkmate's R and S
# ==========================

from .utils import *
from .def_code import RawAtomCode,RawBlockCode

# ==========================
# === make the schedule ====
# ==========================
# -> interface with checkmate

def make_sched(kg : pgb.Ktools.K_graph,budget,
        plot_sched=False,solver='SCIPY',
        verbose=None,use_gurobi=True):
    # == verbose ==
    if verbose is None: verbose = ref_verbose[0]
    else: ref_verbose[0]=verbose

    # == check solver ==
    if solver not in cvxpy.installed_solvers():
        raise AttributeError(
            f"{solver} isn't installed ({cvxpy.installed_solvers()})")

    # == try to choose Gurobi ==
    if not gurobi_installed: use_gurobi=False

    # == build Checkmate graph ==
    chk_gb = CHK_GraphBuilder()
    nodes = kg.dict_nodes.values()
    for n in nodes: # -> build nodes
        if n.is_artefact:
            raise Exception("Artefact nodes not supported in CHK yet")
        chk_gb.add_node(name=n.name, cpu_cost=n.time, ram_cost=n.fgt_mem.v)

    for n in nodes: # -> build edges
        for sub_n in n.req:
            chk_gb.add_deps(n.name,sub_n.name)
    chk_g = chk_gb.make_graph()

    # == use Checkmate solver ==
    print_debug(
        f"Ask checkmate to solve a graph with :"\
        f"\n\ttotal cost : {sum(chk_g.cost_ram.values())},"\
        f"\n\tmax   cost : {max(chk_g.cost_ram.values())},"\
        f"\n\tbudget     : {budget}")
    if use_gurobi:
        sched_result = CHK_solve_ilp_gurobi(
                chk_g, budget,
                print_to_console=verbose)
    else:
        sched_result = CHK_solve_checkmate_cvxpy(
                chk_g, budget,
                solver_override=solver,
                verbose=verbose)

    # == result ==
    if sched_result.feasible:
        print("*",end="")
        print_debug('feasible schedule solved')
    if plot_sched: CHK_plot_schedule(sched_result)
    return sched_result, chk_g

# ==========================



# ==========================
# = translate the schedule =
# ==========================
# -> return RawBlockCode

class Sched_to_ops():
    def __init__(self,g,K_graph):
        self.g = g
        self.graph = K_graph
        self.nodes = K_graph.dict_nodes

    def _run_fwd(self, n):
        if n.is_artefact:
            return n.get_code()
        assert(n.name not in self.live)
        # code = (n.get_code())
        # code_list = n.get_code().split('\n')
        # code = code_list[0].replace(n.main_target,"_"+n.main_target)
        # code_list[0] = code
        # code = n.main_code
        code = ast_to_str(make_ast_module([n.main_code]))
        code = code.replace(n.main_target,"_"+n.main_target)

        self.live.append(n.name)
        if n.name not in self.fgt:
            code = (
                f"{code} ; "\
                f"{n.main_target} = _{n.main_target}.detach(); "\
                f"{n.main_target}.requires_grad_()" )
        else: #i.e. recomputation
            #code = (n.code).replace(n.main_target,"_"+n.main_target)
            code = (
                f"{code} ; "\
                f"{n.main_target}.data = _{n.main_target}.data" )
        #if n.main_target == self.graph.output:
        #    fwd_code += f""
        return code+'\n'+ast_to_str(make_ast_module(n.body_code))

    def _run_bwd(self, n):
        assert(n.name not in self.live)
        mt = n.main_target
        code=f"_{mt}.backward({mt}.grad)"
        bwd_code = (
            f"if _{mt}.data.shape == torch.Size([0]):\n"\
            f"\t_{mt}.data = torch.zeros_like({mt}.grad,device=device)\n"\
            f"\t{mt}.data = torch.zeros_like({mt}.grad,device=device)\n"\
            f"\t{code}\n"\
            f"\t_{mt}.data = torch.zeros(0,device=device);"\
            f"\t{mt}.data = torch.zeros(0,device=device)\n"\
            f"else:\n\t{code}\n" )
        self.live.append(n.name)
        return bwd_code

    def _fgt_fwd(self, n):
        assert(n.name in self.live)
        if n.is_artefact: return ""
        code = ""
        v = n.main_target
        code += (f"{v}.data = torch.zeros(0,device=device); "\
                 f"_{v}.data = torch.zeros(0,device=device);")
        for v in n.tensor_targets:
            code += (f"{v}.data = torch.zeros(0,device=device); ")
        self.live.remove(n.name)
        self.fgt.append(n.name)
        return code

    def _fgt_bwd(self, n):
        assert(n.name in self.live)
        code_list = []
        for sub_n in n.used_by:
            for sup_sub_n in sub_n.req:
                if sup_sub_n in self.live:
                    continue
            for t in sub_n.tensor_targets:
                code = f"{t}.grad = None"
                code_list.append(code)
        self.live.remove(n.name)
        self.fgt.append(n.name)
        return ";".join(code_list)

    def generate_sched_ops(self, sched_result):
        self.schedule = sched_result.schedule
        self.sched_ops = []
        self.live = []#record which grad is live
        self.fgt = []#record what is forgotten
        for op in self.schedule:
            if isinstance(op, CHK_OperatorEvaluation):
                node_name = self.g.node_names[op.id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    code = self._run_fwd(node)
                else:
                    code = self._run_bwd(node)
                res = RawAtomCode(node,code,is_fgt=False,is_fwd=is_fwd)
                self.sched_ops.append(res)

            elif isinstance(op, CHK_DeallocateRegister):
                node_name = self.g.node_names[op.op_id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    code = self._fgt_fwd(node)
                else:
                    code = self._fgt_bwd(node)
                res = RawAtomCode(node,code,is_fgt=True,is_fwd=is_fwd)
                self.sched_ops.append(res)

            elif isinstance(op, CHK_AllocateRegister):
                pass

            else:
                raise TypeError(f"Operation not recognize:{type(op)}")

        fwd_ops = []
        bwd_ops = []
        fwd = True
        for i,op in enumerate(self.sched_ops):
            if "loss" in op.code:
                fwd=False
            if fwd:
                fwd_ops.append(op)
            else:
                bwd_ops.append(op)
        return RawBlockCode(fwd_ops),RawBlockCode(bwd_ops)



# ==========================

