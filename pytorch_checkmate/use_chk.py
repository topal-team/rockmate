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
from checkmate.core.solvers.cvxpy_solver import solve_checkmate_cvxpy
from checkmate.plot.graph_plotting import plot_schedule

def make_sched(kg : K_graph,budget,plot_sched=False,solver='SCIPY',verbose=False):
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
    print('total cost:', sum(chk_g.cost_ram.values()),
          'max cost:', max(chk_g.cost_ram.values()),
          'budget:', budget)

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
        code = (n.get_code()).replace(n.main_target,"_"+n.main_target)
        self.live.append(n.name)
        if n.name not in self.fgt:
            fwd_code = (
                f"{code} ; "\
                f"{n.main_target} = _{n.main_target}.detach(); "\
                f"{n.main_target}.requires_grad_()" )
        else: #i.e. recomputation
            code = (n.code).replace(n.main_target,"_"+n.main_target)
            fwd_code = (
                f"{code} ; "\
                f"{n.main_target}.data = _{n.main_target}.data" )
        #if n.main_target == self.graph.output:
        #    fwd_code += f""
        return fwd_code

    def _run_bwd(self, n):
        assert(n.name not in self.live)
        mt = n.main_target
        code=f"_{mt}.backward({mt}.grad)"
        bwd_code = (
            f"if _{mt}.data.shape == torch.Size([0]):\n"\
            f"\t_{mt}.data = torch.zeros_like({targ}.grad,device=device)\n"\
            f"\t{targ}.data = torch.zeros_like({targ}.grad,device=device)\n"\
            f"\t{code}\n"\
            f"\t_{targ}.data = torch.zeros(0,device=device)"\
            f"\t{targ}.data = torch.zeros(0,device=device)\n"\
            f"else:\n\t{code}\n" )
        self.live.append(n.name)
        return bwd_code

    def _fgt_fwd(self, n):
        assert(n.name in self.live)
        if n.is_artefact: return ""
        code = ""
        for v in n.tensor_targets:
            code += f"{v}.data = torch.zeros(0,device=device);_{v}.data = torch.zeros(0,device=device);"
        self.live.remove(n.name)
        self.fgt.append(n.name)
        return code
    
    def _fgt_bwd(self, n):
        assert n.name in self.live
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

