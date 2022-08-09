# ==========================
# rockmate interface with checkmate
# use checkmate to get the schedule
# and translate it from checkmate's R and S
# ==========================

from .utils import *
from .def_code import CodeAtom, CodeBlock

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
# -> return CodeBlock

class Sched_to_ops():
    def __init__(self,g,K_graph):
        self.g = g
        self.graph = K_graph
        self.nodes = K_graph.dict_nodes

    def _run_fwd(self, n, non_grad=False, rec=False):
        assert(n.name not in self.live)
        self.live.append(n.name)
        if n.is_artefact or non_grad:
            return n.get_code()
        if "LOSS" in n.get_code():
            return n.get_code()
        # code = (n.get_code())
        # code_list = n.get_code().split('\n')
        # code = code_list[0].replace(n.main_target,"_"+n.main_target)
        # code_list[0] = code
        # code = n.main_code
        code = ast_to_str(make_ast_module([n.main_code]))
        code = code.replace(n.main_target,"_"+n.main_target)
        body_code = ""
        if rec: #i.e. recomputation
            #code = (n.code).replace(n.main_target,"_"+n.main_target)
            code = (
                f"{code} ; "\
                f"{n.main_target}.data = _{n.main_target}.data" )
            for c in n.body_code:
                if "view" in ast_to_str(c.value):
                    body_code += ast_to_str(c.targets) + ".data = " + ast_to_str(c.value)+";"
                else:
                    body_code += ast_to_str(c)+";"
        else:
            code = (
                f"{code} ; "\
                f"{n.main_target} = _{n.main_target}.detach(); "\
                f"{n.main_target}.requires_grad_()" )
            body_code = ast_to_str(make_ast_module(n.body_code))
        #if n.main_target == self.graph.output:
        #    fwd_code += f""
        return code+'\n'+body_code

    def _run_bwd(self, n, rec=False):
        assert(n.name not in self.live)
        mt = n.main_target
        if rec:#TODO: check if retain_graph=True changes memory need
            rec_list = []
            if sub_list is None:
                for sub_n in n.used_by:
                    if sub_n.name in self.fgt:
                        rec_list += sub_n.tensor_targets
            code=f"_{mt}.backward({mt}.grad, inputs={rec_list}, retain_graph=True)"
        else:
            code=f"_{mt}.backward({mt}.grad, retain_graph=True)"
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

    def _fgt_fwd(self, n, non_grad=False):
        assert(n.name in self.live)
        if n.is_artefact: return ""
        code = ""
        v = n.main_target
        code += f"{v}.data = torch.zeros(0,device=device); "
        if not non_grad:
            code += f"_{v}.data = torch.zeros(0,device=device);"
        for v in n.tensor_targets:
            code += (f"{v}.data = torch.zeros(0,device=device); ")
        self.live.remove(n.name)
        self.fgt.append(n.name)
        return code

    def _fgt_bwd(self, n):
        assert(n.name in self.live)
        code_list = []
        for sub_n in n.used_by:
            to_fgt = True
            for sup_sub_n in sub_n.req:
                if sup_sub_n in self.live:
                    #TODO: mark the output grad that can be fgt
                    to_fgt = False
                    continue
            if to_fgt:
                self.fgt.append(sub_n.name)
                for t in sub_n.tensor_targets:
                    code = f"{t}.grad = None"
                    code_list.append(code)
        self.live.remove(n.name)
        #self.fgt.append(n.name)
        return ";".join(code_list)

    def generate_sched_ops(self, sched_result):
        self.schedule = sched_result.schedule
        self.sched_ops = []
        self.ops = []
        self.live = []#record which grad is live
        self.fgt = []#record what is forgotten
        for op in self.schedule:
            if isinstance(op, CHK_OperatorEvaluation):
                node_name = self.g.node_names[op.id]#"fwd_"+main_target
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                rec = True if node_name in self.ops else False
                if is_fwd:
                    # TODO: this should be a attribute of node
                    if 'loss' in node_name or self.graph.dict_info[node_name[4:]].requires_grad:
                        code = self._run_fwd(node, rec=rec)
                    else:
                        code = self._run_fwd(node, non_grad=True, rec=rec)
                else:
                    code = self._run_bwd(node, rec=rec)
                self.ops.append(node_name)
                res = CodeAtom(code,is_fgt=False,n=node)
                self.sched_ops.append(res)

            elif isinstance(op, CHK_DeallocateRegister):
                node_name = self.g.node_names[op.op_id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    if 'loss' in node_name or self.graph.dict_info[node_name[4:]].requires_grad:
                        code = self._fgt_fwd(node)
                    else:
                        code = self._fgt_fwd(node, non_grad=True)
                else:
                    code = self._fgt_bwd(node)
                res = CodeAtom(code,is_fgt=True,n=node)
                self.sched_ops.append(res)

            elif isinstance(op, CHK_AllocateRegister):
                pass

            else:
                raise TypeError(f"Operation not recognize:{type(op)}")

        fwd_ops = []
        bwd_ops = []
        fwd = True
        for i,op in enumerate(self.sched_ops):
            if "LOSS" in op.code:
                fwd=False
            if fwd:
                fwd_ops.append(op)
            else:
                bwd_ops.append(op)
        return CodeBlock(fwd_ops),CodeBlock(bwd_ops)



# ==========================

