# =====================
# = INSPECTION OF KCN =
# =====================

from pgb.utils.imports import *
from pgb.utils import def_info,ast_add_on,small_fcts
from pgb.utils.global_vars import print_debug
import gc

# ======================
# = CREATE A FRESH ENV =
# ======================

def generate_our_global(sg,model,device):
    our_global = globals().copy()
    our_global["self"] = model
    our_global["device"] = device
    for inp in sg.direct_inputs:
        info = sg.dict_info[inp]
        x = def_info.generate_val(info,device)
        our_global[inp]=x
    return our_global

def generate_tmp_local(sn,sg,our_global,device,tmp_local=None):
    if tmp_local is None:
        tmp_local = dict()
        exec(sg.init_node.get_code(),our_global,tmp_local)
    for req_sn in sn.deps.keys():
        if (not (req_sn is sg.init_node)
        and req_sn.main_target not in tmp_local):
            # we create the main_target value, and we run the body_code
            # but the body_code may requires some artefacts
            req_sn_mt = req_sn.main_target
            main_info = sg.dict_info[req_sn_mt]
            req_sn_mt_value = def_info.generate_val(main_info,device)
            if isinstance(req_sn_mt_value,torch.Tensor):
                req_sn_mt_value = req_sn_mt_value.clone()
            tmp_local[req_sn_mt] = req_sn_mt_value
            for req_req_sn in req_sn.deps.keys():
                if not (req_req_sn is sg.init_node):
                    for req_req_tar in req_req_sn.all_targets:
                        req_req_info = sg.dict_info[req_req_tar]
                        tmp_local[req_req_tar] = (
                            def_info.generate_val(req_req_info,device))
            exec(ast_add_on.make_str_list_assign(req_sn.body_code),
                our_global,tmp_local)
    return tmp_local

def generate_deep_tmp_local(sn,sg,our_global,device):
    tmp_local = dict()
    for req_sn in sn.deps.keys():
        generate_tmp_local(
            req_sn,sg,our_global,device,tmp_local=tmp_local)
        exec(req_sn.get_code(), our_global, tmp_local)
    return tmp_local

# ======================



# ======================
# TRACE GRAD_FN TO KNOW
# WHAT IS NEEDED TO BWD
# -> REAL_DEPS
# -> PHANTOMS
# ======================

def get_useful_vars(sn,sg,our_global,device):
    params = dict(our_global['self'].named_parameters())
    print_debug(f"Try to open {sn.main_target}'s grad_fn")
    # == INIT ==
    tmp_local = generate_deep_tmp_local(sn,sg,our_global,device)
    exec(sn.get_code(), our_global, tmp_local)
    mt = sn.main_target
    fn = tmp_local[mt].grad_fn
    explicit_vars = set() # set of Tensors
    phantom_vars  = set() # set of Tensors

    # == SEARCH THROUGH GRAD_FN == 
    def trace_gradfn(f,path): # path useless, just testing TO REMOVE
        if hasattr(f,"variable"):
            explicit_vars.add(f.variable)
        for attr in dir(f):
            x = getattr(f,attr)
            if (attr != "variable" and isinstance(x,torch.Tensor)):
                is_para = False ; is_input = False
                for p in params.values():
                    if p is x: is_para  = True
                for t in our_global.values():
                    if t is x: is_input = True
                if not is_para and not is_input:
                    phantom_vars.add(x)
        if hasattr(f,"next_functions"):
            for k,t in enumerate(f.next_functions):
                trace_gradfn(t[0],path+[k])
    trace_gradfn(fn,[])

    # == recognize which var are concerned ==
    used_vars = explicit_vars.union(phantom_vars)
    used_ptrs = [v.data_ptr() for v in used_vars]

    req_real = []
    req_ptrs = []
    print_debug(f"SEE WHICH VARS ARE USEFUL FOR {sn.main_target}")
    for name,val in tmp_local.items():
        if (name not in sg.direct_inputs
        and isinstance(val,torch.Tensor)
        and val.data_ptr() in used_ptrs):
            req_real.append(name)
            req_ptrs.append(val.data_ptr())
            print_debug(f"usefull var : {name}")

    # == check for the presence of phantoms ==
    exist_phantoms = False
    for v in phantom_vars:
        p = v.data_ptr()
        if p not in req_ptrs:
            exist_phantoms = True
            print_debug(f"yes {mt} have phantoms")

    return req_real,exist_phantoms

# ======================



# ==========================
# ======= INSPECTION =======
# ==========================

class Inspection_result():
    def __init__(self):
        self.relevant     = False # -> turn True if result of inspection
        self.mem_del_fwd  = rotor_MemSize(0)
        self.overhead_fwd = rotor_MemSize(0)
        self.overhead_bwd = rotor_MemSize(0)
        self.mem_run_fwd  = rotor_MemSize(0)
        self.mem_run_bwd  = rotor_MemSize(0)
        self.mem_fgt_fwd  = rotor_MemSize(0)
        self.mem_fgt_bwd  = rotor_MemSize(0)
        self.time_run_fwd = 0
        self.time_run_bwd = 0

# TODO : clean the following 100 lines
class inspector():
    def __init__(self,sn,sg,our_global,device):
        self.sn = sn
        self.sg = sg
        self.mt = sn.main_target
        self.info = sg.dict_info[self.mt]
        self.timer = rotor.timing.make_timer(device)
        self.memUsage = rotor.memory.MeasureMemory(device)
        self.our_global = our_global
        self.tmp_local = generate_tmp_local(sn,sg,our_global,device)
        self.ret = Inspection_result()
        self.ret.relevant = True

    # -- aux --
    def measure_time(self, fct, inter_fct=None):
        t = self.timer.measure_median(fct,samples=1)
        nb_repeat = 1
        measures = [t] ; tot = t
        while (tot < time_min_duration or nb_repeat < time_min_repeat):
            if inter_fct: inter_fct()
            t = self.timer.measure_median(fct,samples=1)
            measures.append(t)
            tot += t ; nb_repeat += 1
        if len(measures)>2:
            return (
                (sum(measures)-max(measures)-min(measures))
                /(len(measures)-2))
        else:np.median(measures)
    # ---------

    # === FORWARD ===
    # -- measure forward --
    def measure_fwd(self,only_run=False):
        def fct_run_fwd():
            self.code_run_fwd = self.sn.get_code()
            exec(self.code_run_fwd, self.our_global, self.tmp_local)

        def fct_fgt_fwd():
            for tar in self.sn.tensor_targets:
                self.tmp_local[tar].data = torch.zeros(0,device=device)

        def fct_del_fwd():
            code = ""
            for tar in self.sn.tensor_targets:
                code += f"del {tar};"
            self.code_del_fwd = code#Only include the phantom part 
            exec(self.code_del_fwd, self.our_global, self.tmp_local)
        gc.disable()
        _ , mem_run_fwd , peak_fwd = self.memUsage.measure(fct_run_fwd)
        overhead_fwd = peak_fwd - mem_run_fwd
        self.ret.overhead_fwd = overhead_fwd
        self.ret.mem_run_fwd = mem_run_fwd
        if not only_run:
            _ , mem_del_fwd , _ = self.memUsage.measure(fct_del_fwd)
            self.ret.mem_del_fwd = small_fcts.minus_mem(mem_del_fwd)
            _ , _ , _ = self.memUsage.measure(fct_run_fwd)

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret.mem_fgt_fwd = small_fcts.minus_mem(mem_fgt_fwd)
            self.ret.time_run_fwd = time_run_fwd
        gc.enable()
    # ===============

    # === BACKWARD ===

    def fct_run_bwd(self):
        self.code_run_bwd = f"{self.mt}.backward({self.mt}.grad)"
        exec(self.code_run_bwd, self.our_global, self.tmp_local)

    def fct_fgt_bwd(self):
        for req_sn in self.sn.deps.keys():
            if not req_sn.is_artefact:
                for tar in req_sn.tensor_targets:
                    self.tmp_local[tar].grad = None
    def fct_prepare_bwd(self):
        self.tmp_local = generate_tmp_local(
            self.sn,self.sg,self.our_global,device)
        self.code_run_fwd = self.sn.get_code()
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.sn.main_target].grad = (
            def_info.generate_val(self.info,device))

    # measure
    def measure_bwd(self):
        if self.info.requires_grad:
            gc.disable()
            self.fct_prepare_bwd()
            _ , mem_run_bwd , peak_bwd = (
                self.memUsage.measure(self.fct_run_bwd))
            overhead_bwd = peak_bwd - mem_run_bwd
            _ , mem_fgt_bwd , _ = self.memUsage.measure(self.fct_fgt_bwd)

            self.fct_prepare_bwd()
            time_run_bwd = self.measure_time(
                self.fct_run_bwd, self.fct_prepare_bwd)
            # overhead_bwd contains n.target.data now /!\
            gc.enable()
            self.ret.overhead_bwd = overhead_bwd
            self.ret.mem_run_bwd  = mem_run_bwd
            self.ret.mem_fgt_bwd  = small_fcts.minus_mem(mem_fgt_bwd)
            self.ret.time_run_bwd = time_run_bwd

# ==========================

