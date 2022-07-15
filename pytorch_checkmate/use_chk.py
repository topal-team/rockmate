from .Ktools import K_graph
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# ==========================
# === make the schedule ====
# ==========================

from checkmate.core.graph_builder import GraphBuilder
from checkmate.core.schedule import ScheduledResult, ILPAuxData
try:
    from checkmate.core.solvers.gurobi_solver import solve_ilp_gurobi
except:
    gurobi_installed = False
from checkmate.core.solvers.cvxpy_solver import solve_checkmate_cvxpy
from checkmate.plot.graph_plotting import plot_schedule
import cvxpy

def make_sched(kg : K_graph,budget,plot_sched=False,solver='SCIPY',verbose=False,show_debug=False,use_gurobi=True):
    if solver not in cvxpy.installed_solvers():
        raise AttributeError("please choose from the installed solvers:"+ str(cvxpy.installed_solvers()))
    
    chk_gb = GraphBuilder()
    nodes = list(kg.dict_nodes.values())
    for kn in nodes:
        chk_gb.add_node(name=kn.name, cpu_cost=kn.time, ram_cost=-kn.fgt_mem.v)
    # TODO: add constraint to not forget artifact nodes
    for kn in nodes:
        for sub_n in kn.req:
            if sub_n.fgt_mem:
                chk_gb.add_deps(kn.name,sub_n.name)
            else:
                print("yep, sometimes sub_n.fgt_mem is None")
    chk_g = chk_gb.make_graph()
    max_cost = max(chk_g.cost_ram.values())
    if show_debug:
        for kn in nodes:
            if -kn.fgt_mem.v == max_cost:
                print(kn.get_code())
    print('total cost:', sum(chk_g.cost_ram.values()),
          'max cost:', max_cost,
          'budget:', budget)
    if not gurobi_installed: use_gurobi=False
    if use_gurobi:
        sched_result = solve_ilp_gurobi(
                chk_g, budget, print_to_console=not verbose)
    else:
        sched_result = solve_checkmate_cvxpy(
                chk_g, budget,
                solver_override=solver,
                verbose =verbose)
    if sched_result.feasible: print('feasible schedule solved')

    if plot_sched: plot_schedule(sched_result)
    return sched_result, chk_g

# ==========================



# ==========================
# = translate the schedule =
# ==========================

from checkmate.core.schedule import (
    OperatorEvaluation,
    DeallocateRegister,
    AllocateRegister)

class Sched_to_Code():
    def __init__(self,g,K_graph):
        self.g = g
        self.graph = K_graph
        self.nodes = K_graph.dict_nodes

    def _run_fwd(self, n):
        if n.is_artefact:
            return n.get_code()
        assert(n.name not in self.live)
        # code = (n.get_code())
        code_list = n.get_code().split('\n')
        code = code_list[0].replace(n.main_target,"_"+n.main_target)
        # code_list[0] = code
        
        self.live.append(n.name)
        if n.name not in self.fgt:
            code_list[0] = (
                f"{code} ; "\
                f"{n.main_target} = _{n.main_target}.detach(); "\
                f"{n.main_target}.requires_grad_()" )
        else: #i.e. recomputation
            #code = (n.code).replace(n.main_target,"_"+n.main_target)
            code_list[0] = (
                f"{code} ; "\
                f"{n.main_target}.data = _{n.main_target}.data" )
        #if n.main_target == self.graph.output:
        #    fwd_code += f""
        return '\n'.join(code_list)

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

    def generate_sched_code(self, sched_result):
        self.schedule = sched_result.schedule
        self.sched_code = []
        self.live = []#record which grad is live
        self.fgt = []#record what is forgotten
        for op in self.schedule:
            if isinstance(op, OperatorEvaluation):
                node_name = self.g.node_names[op.id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    code = self._run_fwd(node)
                else:
                    code = self._run_bwd(node)
                self.sched_code.append(code)

            elif isinstance(op, DeallocateRegister):
                node_name = self.g.node_names[op.op_id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    code = self._fgt_fwd(node)
                else:
                    code = self._fgt_bwd(node)

                self.sched_code.append(code)
            elif isinstance(op, AllocateRegister):
                self.sched_code.append("")

            else:
                raise TypeError(f"Operation not recognize:{type(op)}")

        return self.sched_code

# ==========================

