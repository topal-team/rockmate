from .utils import *
from .def_code import RK_Storage
from .def_chain import RK_Chain
from .rotor_solver import seq_builder, Executor
from .translator import Translator

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
    def __init__(self,original_mod, dict_inputs,
                    mem_limit=None, mem_unit=None,
                    verbose=False, get_chain=False,
                    get_sequence=False, get_code=False): 
        super(CheckpointedModule,self).__init__()
        ref_verbose[0] = verbose
        self.device = get_device()
        self.original_mod = original_mod
        # for k,v in original_mod.state_dict().items():
        #     self.register_parameter(k,v)
        self.mem_unit = mem_unit if mem_unit else 1024**2
        # -- use pytorch graph builder to get the list of K_graphs --
        self.pgb_res = pgb.make_all_graphs(
           original_mod,dict_inputs,
           verbose=verbose,
           bool_kg = False) # we don't need the whole K_graph
        self.list_kg = self.pgb_res.K_graph_list
        self.init_code = ast_to_str(self.list_kg[0].init_code)
        self.output = self.list_kg[-1].output_kdn_data
        if get_chain:self.get_chain()
        if get_sequence:self.get_sequence(mem_limit)
        if get_code:self.get_code()

    def get_chain(self,nb_budget_abar=10,nb_budget_all=5):
        # -- use checkmate to solve all the blocks --
        self.rk_chain = RK_Chain(self.list_kg,nb_budget_abar,nb_budget_all,
                                 mem_unit=self.mem_unit)

    def get_sequence(self,mem_limit):
        if mem_limit:
            self.mem_limit = mem_limit
        else:
            self.mem_limit = torch.cuda.get_device_properties(0).total_memory*0.9
        print_debug("mem_limit", self.mem_limit)
        # -- solve the chain like rotor --
        self.seq = seq_builder(self.rk_chain, self.mem_limit//self.mem_unit)
        self.fwd_seq,self.bwd_seq = self.seq.cut_fwd_bwd()
        self.fwd_op_list = [op for seq in self.fwd_seq.seq 
                            for op in seq.op_sched.op_list]
        self.bwd_op_list = [op for seq in self.bwd_seq.seq 
                            for op in seq.op_sched.op_list]
        
    def get_code(self, aggressive=True):
        self.storage =  RK_Storage(self.device,self.original_mod)
        self.translator = Translator(self.storage, aggressive=aggressive)
        fwd_code = []
        for seq_block in self.fwd_seq.seq:
            fwd_code.append(self.translator.translate(seq_block.op_sched,True))
        bwd_code = []
        for seq_block in self.bwd_seq.seq:
            bwd_code.append(self.translator.translate(seq_block.op_sched,False))
        self.fwd_code = fwd_code
        self.bwd_code = bwd_code
        self.full_code = []
        for code_list in fwd_code + bwd_code:
            self.full_code += code_list

    def _exec(self, code_list, record_mem=False):
        for code in code_list:
            if record_mem:
                torch.cuda.reset_peak_memory_stats()
                self.mem_before = torch.cuda.memory_allocated()
                self.max_before = torch.cuda.max_memory_allocated()
            try:
                exec(code,self.storage.gd,self.storage.ld)
            except Exception as e:
                print(f"Failed to execute code:\n {code}")
                print(e)
                break
            if record_mem:
                allo_mem = torch.cuda.memory_allocated()-self.mem_before
                peak_mem = torch.cuda.max_memory_allocated()-self.max_before
                self.max_mem.append(peak_mem-allo_mem)
                self.allo_mem.append(allo_mem)

    def forward(self,input, record_mem=False):
        self.storage.add_val("src",input) # hardcoded
        exec(self.init_code,self.storage.gd,self.storage.ld)
        
        self.max_mem = []
        self.allo_mem = []
        for code_list, seq in zip(self.fwd_code, self.fwd_seq.seq):
            if seq.op_sched.no_grad:
                with torch.no_grad():self._exec(code_list, record_mem)
            else:
                with torch.enable_grad():self._exec(code_list, record_mem)
                
        return self.storage.get_val(self.output.main_target)

    def backward(self,record_mem=False):
        if record_mem:self.allo_mem[-1] += self.output.info.memsize.v
        #output grad is generated outside
        for code_list, seq in zip(self.bwd_code, self.bwd_seq.seq):
            if seq.op_sched.no_grad:
                with torch.no_grad():self._exec(code_list, record_mem)
            else:
                with torch.enable_grad():self._exec(code_list, record_mem)
        
    def expect_time(self):
        # Sum of the measured time of each operation for one batch
        return self.fwd_seq.compute_time()+self.bwd_seq.compute_time() 

    def expect_mem(self, overhead=False):
        # Peak mem based on the measured memory/overhead of each operation
        pred_mem = []; acc_mem = np.zeros(len(self.fwd_seq.seq))
        for i, seq in enumerate(self.fwd_seq.seq+self.bwd_seq.seq):
            op_sched = seq.op_sched
            mem_sizes = np.array([op_sched.kdn_info[k].memsize.v 
                                    for k in op_sched.kdn_names])
            for s, op in zip(op_sched.save, op_sched.op_list):
                acc_mem[seq.index] = s
                pred_mem.append(sum(acc_mem))
                if overhead and op.op_type == "Run":pred_mem[-1] += op.overhead
        return pred_mem
    def reinit(self):
        self.original_mod.zero_grad()
        self.storage.ld = {}
