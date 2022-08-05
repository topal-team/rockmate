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

    def forward(self,input):
        storage = self.storage = RK_Storage(self.device,self.original_mod)
        storage.add_val("src",input) # hardcoded
        exec(self.init_code,storage.gd,storage.ld)
        self.fwd_seq.exec(storage,self.functions)
        return storage.get_val(self.output)

    def backward(self):
        self.bwd_seq.exec(self.storage,self.functions)




