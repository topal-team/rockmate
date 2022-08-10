from .utils import *
from .def_code import RK_Storage
from .def_chain import RK_Chain
from .rotor_solver import seq_builder

def print_memsizes(list_kg):
    di = list_kg[-1].dict_info
    for kg in list_kg:
        for n in kg.dict_nodes.values():
            mt = n.main_target 
            try: print_debug(
                f"{mt} : memsize {di[mt].memsize} ; "\
                f"fm {n.fgt_mem.v}",end="")
            except: print_debug("\nloss")
    print_debug("\n")

class Executor():#to execute CodeAtom
    def __init__(self,storage):
        self.storage = storage
        self.live = {}#variables -> CodeAtom
        self.fgt = []#variables been fgt
        self.done = []#CodeAtom already did
        self.code = []
        self.grad = {}

    def translate(self,code_atom):
        if code_atom.n.is_fwd:
            if code_atom.is_fgt==None or code_atom.is_fgt:
                self._fgt_fwd(code_atom)
            else:
                self._run_fwd(code_atom)
        else:
            if code_atom.is_fgt:
                self._fgt_bwd(code_atom)
            else:
                self._run_bwd(code_atom)

    def exec(self):
        for code in self.code:
            exec(code, self.storage.gd, self.storage.ld)

    def _run_fwd(self, code_atom):
        rec = code_atom.name in self.done
        n = code_atom.n
        mt = n.main_target
        #assert(f"{mt}.data" not in self.live)
        #self.live.append(n.name)
        if n.is_artefact or "LOSS" in n.get_code() or not code_atom.n.info.requires_grad: 
            code = n.get_code()
            self.code.append(code)
            #exec(code, self.storage.gd, self.storage.ld)
            self.done.append(code_atom.name) 
            self.live[f"{mt}.data"] = [code_atom.name]#we assume .data can only from one code_atom
            return None 
        code = ast_to_str(make_ast_module([n.main_code]))
        code = code.replace(mt,"_"+mt)
        body_code = ""
        if rec:#code_atom.name not in self.live[f"{mt}.data"]:#i.e. recomputation
            code = (
                f"{code} ; "\
                f"{mt}.data = _{mt}.data" )
            for c in n.body_code:
                if "view" in ast_to_str(c.value):
                    body_code += ast_to_str(c.targets) + ".data = " + ast_to_str(c.value)+";"
                else:
                    body_code += ast_to_str(c)+";"
        else:
            code = (
                f"{code} ; "\
                f"{mt} = _{mt}.detach(); "\
                f"{mt}.requires_grad_()" )
            body_code = ast_to_str(make_ast_module(n.body_code))
            self.grad[f"{mt}"] = {}
            self.live[f"{mt}.grad"] = []
        self.live[f"{mt}.data"] = [code_atom.name]#we assume .data can only from one code_atom
        self.code.append(code+'\n'+body_code)
        #exec(code+'\n'+body_code, self.storage.gd, self.storage.ld)
        self.done.append(code_atom.name) 

    def _run_bwd(self, code_atom, sub_list=None):
        n = code_atom.n
        mt = n.main_target 
        rec = code_atom.name in self.done
        #assert(f"{mt}.data" not in self.live)
        if rec:#TODO: check if retain_graph=True changes memory need
            rec_list = []
            if sub_list is None:#TODO: future work to allow recompute grad separately
                for sub_n in n.used_by:
                    smt = sub_n.main_target
                    if code_atom.name not in self.live[f"{smt}.grad"]:
                        rec_list += sub_n.tensor_targets
            inputs = ",".join(rec_list)
            code=f"_{mt}.backward({mt}.grad, inputs=[{inputs}], retain_graph=True)"
        else:
            code=f"_{mt}.backward({mt}.grad, retain_graph=True)"
        if len(self.live[f"{mt}.data"])==0:
            bwd_code = (
                f"_{mt}.data = torch.zeros_like({mt}.grad,device=device)\n"\
                f"{mt}.data = torch.zeros_like({mt}.grad,device=device)\n"\
                f"{code}\n"\
                f"_{mt}.data = torch.zeros(0,device=device);"\
                f"{mt}.data = torch.zeros(0,device=device)\n")
        else:
            bwd_code = code
        for sub_n in n.used_by:
            if True:#TODO:sub_n.requires_grad
                smt = sub_n.main_target
                self.live[f"{smt}.grad"].append(code_atom.name)
        self.code.append(bwd_code)
        #exec(bwd_code, self.storage.gd, self.storage.ld)
        self.done.append(code_atom.name) 

    def _fgt_fwd(self, code_atom):
        n = code_atom.n
        #assert(f"{mt}.data" in self.live)
        if n.is_artefact: code = ""
        else:
            mt = n.main_target
            code = f"{mt}.data = torch.zeros(0,device=device); "
            if code_atom.n.info and code_atom.n.info.requires_grad:
                code += f"_{mt}.data = torch.zeros(0,device=device);"
            for v in n.tensor_targets:
                code += (f"{v}.data = torch.zeros(0,device=device); ")
            self.live[f"{mt}.data"].remove("Fwd "+code_atom.main_var)
        self.code.append(code)
        #exec(code, self.storage.gd, self.storage.ld)
        self.done.append(code_atom.name) 

    def _fgt_bwd(self, code_atom):
        n = code_atom.n
        #assert(n.name in self.live)
        code_list = []
        for sub_n in n.used_by:
            smt = sub_n.main_target
            self.live[f"{smt}.grad"].remove("Bwd "+code_atom.main_var)
            if len(self.live[f"{smt}.grad"])==0:
                code_list.append(f"{smt}.grad = None")
                for t in sub_n.tensor_targets:
                    code = f"{t}.grad = None"
                    code_list.append(code)
        self.code.append(";".join(code_list))
        #exec(";".join(code_list), self.storage.gd, self.storage.ld)
        self.done.append(code_atom.name) 


class CheckpointedModule(): #torch.nn.Module):
    def __init__(self,original_mod,dict_inputs,verbose=False, mem_limit=None):
        if not mem_limit:
            mem_limit = torch.cuda.get_device_properties(0).total_memory*0.9
        ref_verbose[0] = verbose
        self.device = get_device()
        # -- use pytorch graph builder to get the list of K_graphs --
        pgb_res = pgb.make_all_graphs(
           original_mod,dict_inputs,
           verbose=verbose,
           bool_kg = False) # we don't need the whole K_graph
        self.list_kg = pgb_res.K_graph_list

        self.init_code = ast_to_str(self.list_kg[0].init_code)

        self.output = self.list_kg[-1].direct_outputs[-1]

        print_memsizes(self.list_kg) # to debug

        # -- use checkmate to solve all the blocks --
        rk_chain = RK_Chain(self.list_kg,2,2)

        # -- solve the chain like rotor --
        seq,functions = seq_builder(rk_chain, mem_limit)

        self.functions = functions
        self.fwd_seq,self.bwd_seq = seq.cut_fwd_bwd()
        self.original_mod = original_mod
        self.storage =  RK_Storage(self.device,self.original_mod)
        self.executor = Executor(self.storage)
        for sb in self.fwd_seq.seq:
            for sa in sb.body:
                self.executor.translate(sa.code)
        self.fwd_code = self.executor.code[:]
        self.executor.code = []
        for sb in self.bwd_seq.seq:
            for sa in sb.body:
                self.executor.translate(sa.code)
        self.bwd_code = self.executor.code

    def forward(self,input):
        self.storage.add_val("src",input) # hardcoded
        #self.storage.gd["executor"] = self.executor 
        exec(self.init_code,self.storage.gd,self.storage.ld)
        #self.fwd_seq.exec(self.storage,self.functions)
        for code in self.fwd_code:
            exec(code,self.storage.gd,self.storage.ld)

        return self.storage.get_val(self.output)

    def backward(self):
        for code in self.bwd_code:
            exec(code,self.storage.gd,self.storage.ld)
        #self.bwd_seq.exec(self.storage,self.functions)




