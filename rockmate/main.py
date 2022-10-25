from .utils import *
from .def_code import RK_Storage
from .def_chain import RK_Chain
from .rotor_solver import seq_builder, Executor

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

class CheckpointedModule(torch.nn.Module):
    def __init__(self,original_mod,dict_inputs,verbose=False, 
                 mem_limit=None, mem_slots=5000):
        super(CheckpointedModule,self).__init__()
        if mem_limit:
            self.mem_limit = mem_limit
        else:
            self.mem_limit = torch.cuda.get_device_properties(0).total_memory*0.9
        self.mem_slots = mem_slots
        ref_verbose[0] = verbose
        self.device = get_device()
        # -- use pytorch graph builder to get the list of K_graphs --
        self.pgb_res = pgb.make_all_graphs(
           original_mod,dict_inputs,
           verbose=verbose,
           bool_kg = False) # we don't need the whole K_graph
        self.list_kg = self.pgb_res.K_graph_list
        #print("kgraph",torch.cuda.memory_allocated())
        self.init_code = ast_to_str(self.list_kg[0].init_code)

        self.output = self.list_kg[-1].direct_outputs[-1]

        print_memsizes(self.list_kg) # to debug

        # -- use checkmate to solve all the blocks --
        mem_unit = self.mem_limit//self.mem_slots
        print_debug("mem_unit", mem_unit)
        print_debug("mem_limit", self.mem_limit)
        self.rk_chain = RK_Chain(self.list_kg,10,3, mem_unit=mem_unit)
        #print("rkchain",torch.cuda.memory_allocated())

        # -- solve the chain like rotor --
        seq = seq_builder(self.rk_chain, self.mem_limit//mem_unit)
        #print("seqbuilder",torch.cuda.memory_allocated())
        
        self.fwd_seq,self.bwd_seq = seq.cut_fwd_bwd()
        self.original_mod = original_mod
        self.storage =  RK_Storage(self.device,self.original_mod)
        self.executor = Executor(self.storage,self.fwd_seq,self.bwd_seq)
        #print("executor",torch.cuda.memory_allocated())
        self.executor.translate(bwd=True)
        self.fwd_code = self.executor.fwd_code
        self.bwd_code = self.executor.bwd_code

        """
        for sb in self.fwd_seq.seq:
            for sa in sb.body:
                try:
                    self.executor.translate(sa.op)
                except:
                    print(f"Failed to translate {sa.op.name}")
        self.fwd_code = self.executor.code[:]
        self.executor.code = []
        for sb in self.bwd_seq.seq:
            for sa in sb.body:
                try:
                    self.executor.translate(sa.op)
                except:
                    print(f"Failed to translate {sa.op.name}")
        self.bwd_code = self.executor.code
        """

    def forward(self,input, record_mem = False):
        self.storage.add_val("src",input) # hardcoded
        exec(self.init_code,self.storage.gd,self.storage.ld)
        torch.cuda.reset_peak_memory_stats()
        self.max_mem = [torch.cuda.max_memory_allocated()]
        self.allo_mem = [torch.cuda.memory_allocated()]
        for code in self.fwd_code:
            try:
                exec(code,self.storage.gd,self.storage.ld)
            except Exception as e:
                print(f"Failed to execute code:\n {code}")
                print(e)
                break
            if record_mem:
                self.max_mem.append(torch.cuda.max_memory_allocated())
                self.allo_mem.append(torch.cuda.memory_allocated())
        return self.storage.get_val(self.output)

    def backward(self,record_mem=False):
        for code in self.bwd_code:
            try:
                exec(code,self.storage.gd,self.storage.ld)
            except:
                print(f"Failed to execute code:\n {code}")
                break
            if record_mem:
                self.max_mem.append(torch.cuda.max_memory_allocated())
                self.allo_mem.append(torch.cuda.memory_allocated())

    def expect_time(self):
        # Sum of the measured time of each operation for one batch
        return self.fwd_seq.compute_time()+self.bwd_seq.compute_time() 

    def expect_mem(self, save=False):
        # Peak mem based on the measured memory/overhead of each operation
        mem = 0;l=[mem]
        for op in self.executor.op_list:
            if "loss" in op.name: l.append(mem);continue
            mem += op.mem
            l.append(mem)
            if not save: l[-1]+= op.overhead
        return l
    def reinit(self):
        self.original_mod.zero_grad()
        self.storage.ld = {}

        #k_l = list(self.storage.ld.keys())
        #for k in k_l: del self.storage.ld[k]
