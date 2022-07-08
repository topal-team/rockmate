import torch
import torch.nn as nn
from Btools import *
import read_trace_code
import Btools as Btools
from Btools import *
from D_gr_to_DF_gr import *

from checkmate.core.schedule import OperatorEvaluation, DeallocateRegister, AllocateRegister


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Sched_to_Code():
    def __init__(self,g,sched_result,K_nodes):
        self.g = g
        self.schedule = sched_result.schedule
        self.nodes = K_nodes
        
    def _run_fwd(self, n):
        assert n.name not in self.live
        code = (n.code).replace(n.main_target,"_"+n.main_target)
        self.live.append(n.name)
        if n.name not in self.fgt: 
            return f"{code} ; {n.main_target} = _{n.main_target}.detach(); {n.main_target}.requires_grad_()"
        else:
            code = (n.code).replace(n.main_target,"_"+n.main_target)
            return f"{code} ; {n.main_target}.data = _{n.main_target}.data"
        #return n.code

    def _run_bwd(self, n):
        assert n.name not in self.live
        code='_{o}.backward({o}.grad)'.format(o=n.main_target)
        targ = n.main_target
        bwd_code = f"if _{targ}.data.shape == torch.Size([0]):\n"
        bwd_code += f"\t_{targ}.data = torch.zeros_like({targ}.grad,device=device);{targ}.data = torch.zeros_like({targ}.grad,device=device)\n"
        bwd_code += f"\t{code}\n"
        bwd_code += f"\t_{targ}.data = torch.zeros(0,device=device)\n"
        bwd_code += f"else:\n\t{code}\n"
        self.live.append(n.name)
        return bwd_code

    def _fgt_fwd(self, n):
        assert n.name in self.live
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
            code = f"{sub_n.main_target}.grad = None"
            code_list.append(code)
        self.live.remove(n.name)
        self.fgt.append(n.name)
        return ";".join(code_list)
        
    def run_sched(self):
        self.sched_code = []
        self.live = []#record which grad is live
        self.fgt = []#record what is forgotten
        for op in self.schedule:
            if isinstance(op, OperatorEvaluation):
                #print('operation', op.id)
                node_name = self.g.node_names[op.id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    code = self._run_fwd(node)
                else:
                    code = self._run_bwd(node) 
                #assert op.operator_cost==node.time
                self.sched_code.append(code)
                
            if isinstance(op, DeallocateRegister):
                #print('deallocate', op.op_id)
                node_name = self.g.node_names[op.op_id]
                node = self.nodes[node_name]
                is_fwd = node.is_fwd
                if is_fwd:
                    code = self._fgt_fwd(node)
                else:
                    code = self._fgt_bwd(node) 
                
                self.sched_code.append(code)

            
        return self.sched_code



