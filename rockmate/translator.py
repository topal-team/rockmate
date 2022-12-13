from .utils import *
from .def_sequence import *

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


    def translate(self, op_sched):
        # Fc/Fn cases
        if op_sched.no_grad:
            code_list = ["with torch.no_grad():"]
            for i,op in enumerate(op_sched.op_list):
                if op.op_type == "Run": code_list.append(f"\t{op.code}")
                elif op.kdn_type == "data": 
                    for target in op.all_targets:
                        code_list.append(f"\tdel {target}")
                if op_sched.del_input_idx == i: 
                    for target in op_sched.del_input_op.all_targets:
                        code_list.append(f"\tdel {target}") 
            return "\n".join(code_list)

        def _is_alive(kdn, i):
            if kdn.name in op_sched.kdn_names:
                return op_sched.alive_list[i][op_sched.kdn_names.index(kdn.name)]
            else:
                return True
        
        def _generate_fake_tensor(kdn, proxy=False):
            # return code for generate the target fake tensor (only for data/grad)
            prep_code = ""
            after_code = ""
            if True: # aggressive way to save memory
                req_shape = kdn.info.tsize
                target_tensor = None
                # TODO: go through all the live tensors
                # for k,v in self.live.items():
                #     if not v: continue
                #     if (np.prod(self.op.main_target2op[k[:-5]].n.info.tsize) ==
                #        np.prod(req_shape)):
                #        target_tensor = k
                if not target_tensor:# No available live tensor to use
                    target_tensor = f"torch.zeros({req_shape},device=device)"
                prep_code += f"{kdn.main_target}.data = {target_tensor}.reshape({req_shape});"
                after_code += f"{kdn.main_target}.data = torch.zeros(0,device=device);"
                if proxy:
                    prep_code += f"_{kdn.main_target}.data = {target_tensor}.reshape({req_shape});"
                    after_code += f"_{kdn.main_target}.data = torch.zeros(0,device=device);"
            return prep_code, after_code
        
        def _run_op(op, i):
            code = ""
            if "fwd" in op.name:
                code = ast_to_str(make_ast_module([op.main_code]))
                if op.proxy:
                    code = code.replace(op.main_target,f"_{op.main_target}")
                    code = (
                        f"{code} ; "\
                        f"{op.main_target} = _{op.main_target}.detach(); "\
                        f"{op.main_target}.requires_grad_()")
                code += "\n"+ast_to_str(make_ast_module(op.body_code))
                return code
            elif "bwd" in op.name:
                prep_code = ""
                after_code = ""
                for kdn in op.deps_fake:
                    if not _is_alive(kdn, i):
                        fake_code = _generate_fake_tensor(kdn, kdn.info.requires_grad)
                        prep_code += fake_code[0]
                        after_code += fake_code[1]
                code = f"_{op.main_target}.backward({op.main_target}.grad, retain_graph={False})"
                bwd_code = (
                    f"{prep_code}\n"\
                    f"{code}\n"\
                    f"{after_code}")
                # TODO: recompute bwd is not supported yet
                return bwd_code


        def _del_op(op, i):
            code = ""
            if op.kdn_type == "data":
                if op.info.requires_grad:
                    code += f"_{op.main_target}.data = torch.zeros(0,device=device);"
                for v in op.all_targets:
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
        return "\n".join(code_list)

    # def translate(self,bwd=True):
    #     # self.cached_set = set([op for op in self.fwd_op_sched 
    #     #                        if (op.n.cached and not op.is_fgt and op.n.main_target != self.output)])
    #     # self.insert_pos = {}
    #     # for op in self.cached_set:
    #     #     is_reqed_list = [(op.n in every_op.n.req_real and not every_op.is_fgt and every_op.n.is_fwd)
    #     #                      for every_op in self.op_sched] 
    #     #     self.insert_pos[len(is_reqed_list) - is_reqed_list[::-1].index(True)-1] = op

    #     for i,op in enumerate(self.fwd_op_sched):
    #         rec = self.op_info[i] in self.op_info[:i]
    #         last = self.op_info[i] not in self.op_info[i+1:]

    #         if op.is_fgt==None or op.is_fgt:
    #             self._fgt_fwd(op,rec=rec,last=last)
    #         else:
    #             self._run_fwd(op,rec=rec,last=last)
    #         self.mem_timeline.append(self._estimate_memory())
    #         self.overhead_timeline.append(self._estimate_memory()+op.overhead)
    #         # if i in self.insert_pos.keys():
    #         #     #print("cached mem will be removed",insert_pos[i].name)
    #         #     self.code[i] += "\n"+self._fgt_fwd(self.insert_pos[i],return_code=True,only_data=True)


    #     self.fwd_code = self.code
    #     self.code = []
    #     if bwd:
    #         for i,op in enumerate(self.bwd_op_sched):
    #             j = i+ len(self.fwd_op_sched)
    #             rec = self.op_info[j] in self.op_info[:j]
    #             last = self.op_info[j] not in self.op_info[j+1:]
    #             try:
    #                 if op.n.is_fwd:
    #                     if op.is_fgt==None or op.is_fgt:
    #                         self._fgt_fwd(op,rec=rec,last=last)
    #                     else:
    #                         self._run_fwd(op,rec=rec,last=last)
    #                 else:
    #                     if op.is_fgt:
    #                         self._fgt_bwd(op,rec=rec,last=last)
    #                     else:
    #                         self._run_bwd(op,rec=rec,last=last)
    #             except Exception as e:
    #                 self.code.append("")
    #                 #print(e)
    #                 break
    #             self.mem_timeline.append(self._estimate_memory())
    #             self.overhead_timeline.append(self._estimate_memory()+op.overhead)
    #             # if j in self.insert_pos.keys():
    #             #     #print("cached mem will be removed",insert_pos[i].name)
    #             #     self.code[-1] += "\n"+self._fgt_fwd(self.insert_pos[j],return_code=True,only_data=True)
    #     self.bwd_code = self.code
    #     # del output.grad
    #     for i, op in enumerate(self.bwd_op_sched[::-1]):
    #         n = op.n
    #         if "loss" in op.name and op.is_fgt:
    #             isOutput = False
    #             for sub_n in n.req_global:
    #                 if self.output in sub_n.name:
    #                     isOutput = True
    #                     break
    #             if isOutput:
    #                 self.bwd_code[-i-1] += f"{self.output}.grad = None"
    #                 break

    # # def execute(self):
    # #     for code in self.code:
    # #         exec(code, self.storage.gd, self.storage.ld)

    # def _run_fwd(self, op, rec=False, last=False, return_code=False):
    #     #rec = op.name in self.done
    #     n = op.n
    #     if "loss" in n.name:
    #         self.code.append("")
    #         return None
    #     mt = n.main_target
    #     #assert(f"{mt}.data" not in self.live)
    #     #self.live.append(n.name)
    #     if n.is_artefact or "LOSS" in n.get_code() or not op.n.info.requires_grad: 
    #         if rec:
    #             code = ""
    #             mc = [n.main_code] if n.main_code else []
    #             for c in mc+n.body_code:
    #                 try:
    #                     if ast_to_str(c.targets) in n.tensor_targets:
    #                         code += ast_to_str(c.targets) + ".data = " + ast_to_str(c.value)+";"
    #                     else:
    #                         code += ast_to_str(c)+";"
    #                 except: code += ast_to_str(c)+";"
    #         else:
    #             code = n.get_code()
    #         self.code.append(code)
    #         #exec(code, self.storage.gd, self.storage.ld)
    #         self.live[f"{mt}.data"] = [op.name]#we assume .data can only from one op
    #         return None 
    #     code = ast_to_str(make_ast_module([n.main_code]))
    #     body_code = ""
    #     if rec:# and not n.abar:#i.e. recomputation
    #         if not n.abar: code = code.replace(mt,f"_{mt}.data")
    #         else: code = code.replace(mt,f"_{mt}")
    #         code = (
    #             f"{code} ; "\
    #             f"{mt}.data = _{mt}.data;"\
    #             f"{mt}.requires_grad_()")
    #         for c in n.body_code:
    #             if "view" in ast_to_str(c.value):
    #                 body_code += ast_to_str(c.targets) + ".data = " + ast_to_str(c.value)+";"
    #             else:
    #                 body_code += ast_to_str(c)+";"
    #     else:
    #         code = code.replace(mt,f"_{mt}")
    #         code = (
    #             f"{code} ; "\
    #             f"{mt} = _{mt}.detach(); "\
    #             f"{mt}.requires_grad_()")
    #         body_code = ast_to_str(make_ast_module(n.body_code))
    #         self.grad[f"{mt}"] = {}
    #         self.live[f"{mt}.grad"] = []
    #     #if True:
    #     #    code += f"{mt}.requires_grad_()"
    #     self.live[f"{mt}.data"] = [op.name]#we assume .data can only from one op
    #     if return_code:return code
    #     self.code.append(code+'\n'+body_code)

    # def _run_bwd(self, op, rec=False, last=False, sub_list=None, return_code=False):
    #     n = op.n
    #     if "loss" in n.name:
    #         self.code.append("")
    #         return None
    #     mt = n.main_target 
    #     #rec = op.name in self.done
    #     #assert(f"{mt}.data" not in self.live)
    #     if rec:
    #         rec_list = []
    #         if sub_list is None:#TODO: future work to allow recompute grad separately
    #             for sub_n in n.used_by_global:
    #                 smt = sub_n.main_target
    #                 if "Bwd "+op.main_var not in self.live[f"{smt}.grad"]:
    #                     #self.live[f"{smt}.grad"].append("Bwd "+op.main_var)
    #                     rec_list += sub_n.tensor_targets
    #         inputs = ",".join(rec_list)
    #         code=f"_{mt}.backward({mt}.grad, inputs=[{inputs}], retain_graph={not last})"
    #     else:
    #         code=f"_{mt}.backward({mt}.grad, retain_graph={not last})"

    #     pre_code = ""
    #     after_code = ""
    #     for deps_kdn in list(n.req_fake)+[n]:
    #         req_shape = deps_kdn.info.tsize
    #         target_tensor = None
    #         for k,v in self.live.items():
    #             if not v: continue
    #             if (np.prod(self.mt2op[k[:-5]].n.info.tsize) ==
    #                np.prod(req_shape)):
    #                target_tensor = k
    #         #target_tensor = None
    #         if not target_tensor:
    #             #print("Need to create something", op.name)
    #             target_tensor = f"torch.zeros({req_shape},device=device)"

    #         if len(self.live[f"{deps_kdn.main_target}.data"])==0 and deps_kdn.info.requires_grad:
    #             pre_code += f"{deps_kdn.main_target}.data = {target_tensor}.reshape({req_shape});"
    #             after_code += f"{deps_kdn.main_target}.data = torch.zeros(0,device=device);"
    #             if deps_kdn.main_target == n.main_target:
    #                 pre_code += f"_{deps_kdn.main_target}.data = {target_tensor}.reshape({req_shape});"
    #                 after_code += f"_{deps_kdn.main_target}.data = torch.zeros(0,device=device);"

    #     bwd_code = (
    #         f"{pre_code}"\
    #         f"{code}\n"\
    #         f"{after_code}")
    #     for sub_n in n.used_by_global:
    #         # if sub_n.main_target == n.main_target:
    #         #     continue
    #         if sub_n.info.requires_grad:
    #             smt = sub_n.main_target
    #             self.live[f"{smt}.grad"].append("Bwd "+op.main_var)
    #     if return_code:return code
    #     self.code.append(bwd_code)


    # def _fgt_fwd(self, op, rec=False, last=False, return_code=False, only_data=False):
        
    #     n = op.n
    #     if "loss" in n.name:
    #         self.code.append("")
    #         return None
    #     #assert(f"{mt}.data" in self.live)
    #     if n.is_artefact: code = ""
    #     elif n.abar:
    #         mt = n.main_target
    #         #code = f"{mt}.data = torch.zeros(0,device=device); "
    #         code =""
    #         if op.n.info and op.n.info.requires_grad:
    #             if not only_data:code += f"del _{mt};"
    #             else: code += f"_{mt}.data = torch.zeros(0,device=device); "
    #         for v in n.tensor_targets:
    #             code += (f"{v}.data = torch.zeros(0,device=device); ")
    #         try:
    #             self.live[f"{mt}.data"].remove("Fwd "+op.main_var)
    #         except: pass
    #     #if n.abar:code = f"del _{mt};" 
                
    #     else:
    #         mt = n.main_target
    #         code =""
    #         if op.n.info and op.n.info.requires_grad:
    #             code += f"_{mt}.data = torch.zeros(0,device=device);"
    #         #code = f"{mt}.data = torch.zeros(0,device=device); "
    #         for v in n.tensor_targets:
    #             code += (f"{v}.data = torch.zeros(0,device=device); ")
    #         try:
    #             self.live[f"{mt}.data"].remove("Fwd "+op.main_var)
    #         except: pass
    #     if return_code:return code
    #     self.code.append(code)

    # def _fgt_bwd(self, op, rec=False, last=False, sp=False, return_code=False):
    #     n = op.n
    #     #assert(n.name in self.live)
    #     if "loss" in n.name:
    #         self.code.append("")
    #         return None
    #     code_list = []
    #     for sub_n in n.used_by_global:
    #         if "loss" in sub_n.name:continue
    #         if sub_n.info.requires_grad:
    #             smt = sub_n.main_target

    #             self.live[f"{smt}.grad"].remove("Bwd "+op.main_var)
    #             if len(self.live[f"{smt}.grad"])==0:
    #                 if hasattr(op, "sp"):
    #                     delattr(op, "sp")
    #                 else:
    #                     if (last and sub_n not in n.used_by_real):
    #                         continue
    #                 code_list.append(f"{smt}.grad = None")
    #                 for t in sub_n.tensor_targets:
    #                     code = f"{t}.grad = None"
    #                     code_list.append(code)
    #     if return_code:return code
    #     self.code.append(";".join(code_list))
