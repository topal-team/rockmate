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
class CodeAtom: pass

class Op:
    def __init__(self,is_fgt,
            n:pgb.Ktools.K_node):
        self.is_fgt = is_fgt
        self.n = n
        self.overhead = n.overhead.v
        if self.overhead is None: self.overhead = 0
        self.main_var = n.main_target
        self.lvars    = n.all_targets
        self.run_mem = n.run_mem.v#for debugging
        if is_fgt: # Fgt
            self.time = 0#Fgt could happen in ILP solution even Infinity mem
            self.mem  = - n.run_mem.v
            if n.is_fwd: self.op_type = "FgtFwd"
            else: self.op_type = "FgtBwd"
        else:
            self.time = n.time
            self.mem  = n.run_mem.v
            if n.is_fwd: self.op_type = "Fwd"
            else: self.op_type = "Bwd"
        self.name = self.op_type+" "+self.main_var
        

class OpBlock:
    def __init__(self, op_list):
        self.body = op_list
        save_mem = []
        tmp_mem = []
        for o in self.body:
            if "loss" in o.name:continue
            save_mem.append(o.mem)
            tmp_mem.append(o.overhead)
        self.save_timeline = np.cumsum(np.array([0]+save_mem))
        #self.overhead_timeline = np.cumsum(np.array(tmp_mem))
        self.overhead_timeline = np.array(tmp_mem+[0])

        self.save = self.save_timeline[-1]
        self.overhead = max(self.save_timeline+self.overhead_timeline) - self.save
        self.time = sum([op.time for op in self.body])



class CodeBlock: pass

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

