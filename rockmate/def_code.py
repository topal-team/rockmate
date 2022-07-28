# ==========================
# everything about codes, how
# we exec it and how we store things
# -> to replace rotor/Checkpointable.functions
# ==========================

from .utils import *

# ==========================
# ===== CODE EXECUTOR ======
# ==========================
# -> TODO RK_Storage should help to handle random states
class RK_Storage:
    def __init__(self,device,nn_mod):
        self.gd = {**globals() , "self" : nn_mod , "device" : device}
        self.ld = dict()
    def add_val(self,val,x):
        self.ld[val]=x
    def get_val(self,val):
        try: return self.ld[val]
        except:
            try: return self.gd[val]
            except:
                raise Exception(f"{val} not in the storage")

# CodeAtom :
# -> attributes : .code : str (-> ast ?)
#                 .op_type : Fwd / Bwd / Del / FgtFwd / FgtBwd 
#                 .time ; .mem ; .overhead : int
#                 .main_var / .lvars
# CodeAtom.__init__ is polymorphic, either you
# give it a K_node or all the attributes needed
class CodeAtom:
    def __init__(self,code,is_fgt,
            n : pgb.Ktools.K_node=None,
            main_var=None,lvars=None,is_fwd=None,time=None,mem=None):
        # is_fgt : ternaire boolean (is_fgt = None means Del)
        self.code = code
        if n:
            self.overhead = n.overhead.v
            self.main_var = n.main_target
            self.lvars    = n.all_targets
            is_fwd   = n.is_fwd
            if is_fgt is None: # Del
                self.time = 0
                self.mem  = n.run_mem.v
            elif is_fgt: # Fgt
                self.time = 0
                self.mem  = - n.fgt_mem.v
            else:
                self.time = n.time
                self.mem  = n.fgt_mem.v
        else:
            self.overhead = 0
            self.main_var = main_var
            self.lvars    = lvars
            self.time     = time
            self.mem      = mem
        # = op_type : =
        if is_fgt is None:
            self.op_type = "Del"
        elif is_fgt:
            if is_fwd: self.op_type = "FgtFwd"
            else: self.op_type = "FgtBwd"
        else:
            if is_fwd: self.op_type = "Fwd"
            else: self.op_type = "Bwd"


class CodeBlock:
    def __init__(self,body):
        self.body = body

    def mem_timeline(self):
        overhead = [o.overhead for o in self.body]
        current_mem = 0 ; mem_timeline = []
        for o in self.body:
            current_mem += o.mem
            mem_timeline.append(current_mem)
        return np.array(overhead),np.array(mem_timeline)

    def exec(self,storage : RK_Storage):
        for c in self.body:
            exec(c.code,storage.gd,storage.ld)

class RK_Function:
    def __init__(self,code_fe,code_fn,code_fc,code_bwd):
        self.code_fe  = code_fe
        self.code_fn  = code_fn
        self.code_fc  = code_fc
        self.code_bwd = code_bwd
    def exec_fe (self,storage : RK_Storage): self.code_fe.exec(storage)
    def exec_fn (self,storage : RK_Storage): self.code_fn.exec(storage)
    def exec_fc (self,storage : RK_Storage): self.code_fc.exec(storage)
    def exec_bwd(self,storage : RK_Storage): self.code_bwd.exec(storage)
# ==========================

