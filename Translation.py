import torch
import torch.nn as nn
import read_trace_code
from Dtools import *
from Stools import *
from Ktools import *

from checkmate.core.schedule import OperatorEvaluation, DeallocateRegister, AllocateRegister


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Sched_to_Code():
    def __init__(self,g,K_graph):
        self.g = g
        self.graph = K_graph
        self.nodes = K_graph.dict_nodes
        
    def _run_fwd(self, n):
        if n.is_artefact:
            return n.get_code()
        assert n.name not in self.live
        code = (n.get_code()).replace(n.main_target,"_"+n.main_target)
        self.live.append(n.name)
        if n.name not in self.fgt: 
            fwd_code = f"{code} ; {n.main_target} = _{n.main_target}.detach(); {n.main_target}.requires_grad_()"
        else:
            code = (n.code).replace(n.main_target,"_"+n.main_target)
            fwd_code = f"{code} ; {n.main_target}.data = _{n.main_target}.data"
        #if n.main_target == self.graph.output:
        #    fwd_code += f""
        return fwd_code

    def _run_bwd(self, n):
        assert n.name not in self.live
        code='_{o}.backward({o}.grad)'.format(o=n.main_target)
        targ = n.main_target
        bwd_code = f"if _{targ}.data.shape == torch.Size([0]):\n"
        bwd_code += f"\t_{targ}.data = torch.zeros_like({targ}.grad,device=device);{targ}.data = torch.zeros_like({targ}.grad,device=device)\n"
        bwd_code += f"\t{code}\n"
        bwd_code += f"\t_{targ}.data = torch.zeros(0,device=device);{targ}.data = torch.zeros(0,device=device)\n"
        bwd_code += f"else:\n\t{code}\n"
        self.live.append(n.name)
        return bwd_code

    def _fgt_fwd(self, n):
        assert n.name in self.live
        if n.is_artefact:
            return ""
        code = ""
        for v in n.all_targets: 
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
            for t in sub_n.all_targets:
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



