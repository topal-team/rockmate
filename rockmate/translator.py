from .utils import *
from .def_sequence import *
from .def_code import *

class Translator():#to execute Op 
    def __init__(self,storage, fwd_seq, bwd_seq):
        self.storage = storage
        self.live = {}#variables -> CodeAtom
        self.fgt = []#variables been fgt
        self.code = []
        self.grad = {}
        self.fwd_op_sched = []
        self.bwd_op_sched = []
        self.op_info = []
        self.fwd_code = []
        # for sb in fwd_seq.seq:
        #     self.fwd
        #     for sa in sb.body:
        #         self.op_info.append((sa.op.n.main_target, 
        #                                  sa.op.is_fgt, sa.op.n.is_fwd))
        #         self.fwd_op_sched.append(sa.op)
        # for sb in bwd_seq.seq:
        #     for sa in sb.body:
        #         self.op_info.append((sa.op.n.main_target, 
        #                                  sa.op.is_fgt, sa.op.n.is_fwd))
        #         self.bwd_op_sched.append(sa.op)
        # self.op_sched_origin = self.fwd_op_sched+self.bwd_op_sched
        # self.output = list(self.fwd_op_sched[-1].n.req_real)[0].main_target#TODO:check correct
        # self.op_sched = self.fwd_op_sched+self.bwd_op_sched
        # self.mt2op = {}
        # for op in self.op_sched.op_list:
        #     if not op.is_fgt: self.mt2op[op.n.main_target] = op
        # self.mem_timeline = []
        # self.overhead_timeline = []

    def _estimate_memory(self):
        mem = 0
        for k,v in self.live.items():
            mt, data = k.split(".")
            if v: mem += self.mt2op[mt].mem
        return mem


    def translate(self, op_sched, during_fwd=True):
        # Fc/Fn cases
        if op_sched.no_grad:
            code_list = []#["with torch.no_grad():"]
            for i,op in enumerate(op_sched.op_list):
                if op.op_type == "Run": 
                    if "loss" in op.main_target: code_list.append("")
                    else:
                        # code = ast_to_str(make_ast_module([op.main_code]))
                        # code += "\n"+ast_to_str(make_ast_module(op.body_code))
                        # code = op.code
                        # code = "\t".join(code.splitlines(True))
                        code_list.append(f"{op.code}")
                elif op.kdn_type == "data":
                    code = "" 
                    for target in op.all_targets:
                        code += f"del {target};"
                    code_list.append(code)
                else:code_list.append("")
                if op_sched.del_input_idx == i: 
                    code = "\n"
                    for target in op_sched.del_input_op.tensor_targets:
                        code += f"{target}.data = torch.zeros(0,device=device);"
                    code_list[-1] += code
            code_list[-1] += f"\n{op_sched.output.main_target}.requires_grad_()"
            return code_list#["\n".join(code_list)]#Fc/Fn needs indent so run as one command

        def _is_alive(kdn_name, i):
            if kdn_name in op_sched.kdn_names:
                return op_sched.alive_list[i][op_sched.kdn_names.index(kdn_name)]
            else:
                return True
        
        def _generate_fake_data(kdn, i, is_self=False):
            # return code for generate the target fake tensor (only for data/grad)
            prep_code = ""
            after_code = ""
            if True: # aggressive way to save memory
                req_shape = kdn.info.tsize
                target_tensor = None
                mt = kdn.main_target
                if is_self:target_tensor = f"{kdn.main_target}.grad"
                # TODO: go through all the live tensors
                for name,info in op_sched.kdn_info.items():
                    if "data" not in name:continue
                    if (np.prod(info.tsize)==np.prod(req_shape) 
                        and _is_alive(name, i)):
                        target_tensor = name.split(" ")[0]#main_target
                # for k,v in self.live.items():
                #     if not v: continue
                #     if (np.prod(self.op.main_target2op[k[:-5]].n.info.tsize) ==
                #        np.prod(req_shape)):
                #        target_tensor = k
                if not target_tensor:# No available live tensor to use
                    target_tensor = f"torch.zeros({req_shape},device=device)"
                prep_code += f"{mt}.data = {target_tensor}.reshape({req_shape});"
                prep_code += ";".join([make_str_assign(bc) for bc in 
                                        list(kdn.deps)[0].body_code])+"\n"
                after_code += f"{mt}.data = torch.zeros(0,device=device);"
                for v in kdn.all_targets:
                    after_code += (f"{v}.data = torch.zeros(0,device=device); ")
                if is_self:
                    prep_code += f"_{mt}.data = {target_tensor}.reshape({req_shape});"
                    after_code += f"_{mt}.data = torch.zeros(0,device=device);"
            return prep_code, after_code
        
        def _run_op(op, i):
            # code = ""
            if "fwd" in op.name:
                rec = ((i>op_sched.op_list.index(op)) or 
                        (not op_sched.is_fwd)) 
                if op.proxy:
                    if ((not during_fwd) and (not op_sched.no_grad) and#Fe in bwd
                        (op.main_target == op_sched.output.main_target)):
                        rec = True
                    proxy_code = make_str_assign(op.main_code, prefix="_")
                    code = (
                        f"{proxy_code};"\
                        f"{op.main_target} = _{op.main_target}.detach();"\
                        f"{op.main_target}.requires_grad_()")
                    if rec:
                        code = (
                            f"{proxy_code};"\
                            f"{op.main_target}.data = _{op.main_target}.data;")
                else:
                    code = make_str_assign(op.main_code)
                for bc in op.body_code:
                    suffix = ""
                    if (rec and (bc[0] in op.tensor_targets)):
                        suffix = ".data"
                    code += "\n" + make_str_assign(bc, suffix=suffix)
                return code
                # code = ast_to_str(make_ast_module([op.main_code]))
                # if op.proxy:
                #     code = code.replace(op.main_target,f"_{op.main_target}")
                #     code = (
                #         f"{code}; "\
                #         f"{op.main_target} = _{op.main_target}.detach(); "\
                #         f"{op.main_target}.requires_grad_()")
                # if i>op_sched.op_list.index(op) or (not op_sched.is_fwd):
                #     code = code.replace(op.main_target,f"_{op.main_target}")
                #     code = (
                #         f"{code}; "\
                #         f"{op.main_target}.data = _{op.main_target}.data; "\
                #         f"{op.main_target}.requires_grad_()")
                #     body_code = ast_to_str(make_ast_module(op.body_code))
                #     for target in op.all_targets:
                #         body_code.replace(target, f"{target}.data")

                # return code + "\n"+ast_to_str(make_ast_module(op.body_code))
            elif "bwd" in op.name:
                mt = op.main_target
                rec = op in op_sched.op_list[:i]
                last = not (op in op_sched.op_list[i+1:])
                prep_code = ""
                after_code = ""
                for kdn in op.deps_fake:
                    if not _is_alive(kdn.name, i):
                        fake_code = _generate_fake_data(kdn, i, 
                            is_self=(kdn.main_target==op.main_target))
                        prep_code += fake_code[0]
                        after_code += fake_code[1]
                if rec:
                    prev_i = i - op_sched.op_list[:i][::-1].index(op) - 1
                    rec_list = []
                    for kdn in op.users_global:
                        # if not _is_alive(kdn.name, i):
                        if DelOp(kdn) in op_sched.op_list[prev_i:i]:
                            rec_list += kdn.all_targets
                    inputs = ",".join(rec_list)
                    code = f"_{mt}.backward({mt}.grad, inputs=[{inputs}], retain_graph={not last})"
                else:
                    code=f"_{mt}.backward({mt}.grad, retain_graph={not last})"
                bwd_code = (
                    f"{prep_code}\n"\
                    f"{code}\n"\
                    f"{after_code}")
                return bwd_code


        def _del_op(op, i):
            code = ""
            if op.kdn_type == "data":
                if (op.info.requires_grad and 
                    _is_alive(op.name.replace("data", "phantoms"), i)):
                    code += f"_{op.main_target}.data = torch.zeros(0,device=device);"
                for v in op.tensor_targets:
                    code += (f"{v}.data = torch.zeros(0,device=device); ")
            if op.kdn_type == "grad":
                code += f"{op.main_target}.grad = None"
            if op.kdn_type == "phantoms":
                code += f"del _{op.main_target}"
            return code
        
        code_list = []
        for i, (op, alive) in enumerate(zip(op_sched.op_list, op_sched.alive_list)):
            if op.op_type == "Run": code_list.append(_run_op(op, i))
            if op.op_type == "Del": code_list.append(_del_op(op, i))
        return code_list
