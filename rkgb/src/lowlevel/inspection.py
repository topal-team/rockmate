# =====================
# = INSPECTION OF KCN =
# =====================

import sys
import gc
from src.lowlevel import ast_add_on
from src.lowlevel import constants
from src.lowlevel.variable_info import VariableInfo
from src.core import base

# ========================================
# = CREATE A FRESH ENVIRONNEMENT TO EXEC =
# ========================================

def generate_our_global(sg,model,device):
    our_global = globals().copy()
    our_global.update(sg.dict_constants)
    our_global["self"] = model
    our_global["device"] = device
    for inp in sg.whole_model_inputs.union(set(sg.inputs)):
        info = sg.dict_info[inp]
        x = info.generate_value(device)
        our_global[inp]=x
    return our_global


def generate_tmp_local(sn,sg,our_global,device):
    tmp_local = dict()
    exec(
        sg.init_node.get_code(force_special_kwargs=True),
        our_global,tmp_local)
    req_sn_todo = list(sn.deps.keys())
    set_req_sn_todo = set(req_sn_todo)
    while req_sn_todo != []:
        req_sn = req_sn_todo.pop(0)
        if set(req_sn.deps).intersection(set_req_sn_todo) != set():
            req_sn_todo.append(req_sn) # not his turn yet
            continue
        else:
            set_req_sn_todo.remove(req_sn)
        if not (req_sn is sg.init_node):
            # we create the main_target value, and we run the body_code
            # but the body_code may requires some artifacts
            # thus we need req of req
            req_sn_mt = req_sn.main_target
            main_info = sg.dict_info[req_sn_mt]
            req_sn_mt_value = main_info.generate_value(device)
            if isinstance(req_sn_mt_value,torch.Tensor):
                req_sn_mt_value = req_sn_mt_value.clone()
            tmp_local[req_sn_mt] = req_sn_mt_value
            body_code = ast_add_on.make_str_list_assign(
                req_sn.body_code,
                force_special_kwargs=True)
            for req_req_sn in req_sn.deps.keys():
                if not (req_req_sn is sg.init_node):
                    for req_req_tar in req_req_sn.all_targets:
                        if req_req_tar in body_code and req_req_tar not in tmp_local:
                            req_req_info = sg.dict_info[req_req_tar]
                            tmp_local[req_req_tar] = (
                                req_req_info.generate_value(device))
            exec(body_code,our_global,tmp_local)
    return tmp_local

# ======================



# ======================
# TRACE GRAD_FN TO KNOW
# WHAT IS NEEDED TO BWD
# -> REAL_DEPS
# -> PHANTOMS
# ======================

# -> auxiliary function for "get_useful_vars" below
def trace_grad_fn(grad_fn,main_target="var",params=None,our_global=None):
    if params is None: params = dict()
    if our_global is None: our_global = dict()
    explicit_vars  = set() # set of Tensors
    phs_found = set() # set of (ph_val * ph_name)
    def aux(f,path):
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
                    path_str = [f".next_functions[{k}][0]" for k in path]
                    ph_name = (
                        f"{main_target}.grad_fn" 
                        + "".join(path_str)
                        + "." + attr)
                    phs_found.add((x,ph_name))
        if hasattr(f,"next_functions"):
            for k,t in enumerate(f.next_functions):
                aux(t[0],path+[k])
    aux(grad_fn,[])
    return explicit_vars,phs_found



def get_useful_vars(sn,sg,our_global,device):
    params = dict(our_global['self'].named_parameters())
    print_debug(f"Try to open {sn.main_target}'s grad_fn")
    # == INIT ==
    dict_info = sg.dict_info
    mt = sn.main_target
    tmp_local = generate_tmp_local(sn,sg,our_global,device)
    exec(
        sn.get_code(force_special_kwargs=True), 
        our_global, tmp_local)
    sn_val = tmp_local[mt]
    hasattr_base = not (sn_val._base is None)

    # == SEARCH THROUGH GRAD_FN == 
    grad_fn = sn_val.grad_fn
    (explicit_vars,phs_found) = trace_grad_fn(
        grad_fn,sn.main_target,params,our_global
    )

    # == recognize which var are concerned ==
    explicit_deps = []
    data_ptr_ph_deps = dict() 
    # ph_name -> data_owner_name with equal data_ptr
    valid_view_ph_deps = dict() 
    # ph_name -> (var_name,data_owner_name) st .view == ph

    for name,val in tmp_local.items():
        if (name not in sg.inputs
        and isinstance(val,torch.Tensor)):
            if name not in dict_info: 
                print(
                    f"Warning: {name} is a Tensor in tmp_local"\
                    f" which isn't in dict_info ? How is it ?",
                    file = sys.stderr)
                data_owner_name = name
            else:
                data_owner_name = dict_info[name].data_owner_name
            if data_owner_name == base.Graph.default_param_target_string:
                continue
            for explicit_var in explicit_vars:
                if val is explicit_var:
                    explicit_deps.append(data_owner_name)
                    break
            for ph_val,ph_name in phs_found:
                if val is ph_val:
                    explicit_deps.append(data_owner_name)
                if val.data_ptr() == ph_val.data_ptr():
                    data_ptr_ph_deps[ph_name] = data_owner_name
                    if torch.numel(val) == torch.numel(ph_val):
                        try:
                            if torch.equal(val.view(ph_val.shape),ph_val):
                                valid_view_ph_deps[ph_name] = (
                                    name,data_owner_name)
                        except: pass
                        # -> applying val.view raise an error if 
                        # -> val stride and size isn't compatible with
                        # -> the original data_owner
    
    # == check for the presence of original phantoms ==
    exist_phs = False
    original_phs = []
    for ph_val,ph_name in phs_found:
        if ph_name not in data_ptr_ph_deps:
            exist_phs = True
            original_phs.append(ph_name)
            print_debug(f"{ph_name} is an original ph of {mt}")

    # == clean data_ptr_ph_deps ==
    for ph_name in valid_view_ph_deps:
        del data_ptr_ph_deps[ph_name]

    return (
        explicit_deps,
        data_ptr_ph_deps,
        valid_view_ph_deps,
        exist_phs,
        original_phs,
        hasattr_base,
    )

# ======================



# ==========================
# ======= INSPECTION =======
# ==========================

class Inspection_result():
    def __init__(self):
        self.relevant     = False # -> turn True if result of inspection
        self.mem_del_fwd  = 0
        self.overhead_fwd = 0
        self.overhead_bwd = 0
        self.mem_run_fwd  = 0
        self.mem_run_bwd  = 0
        self.mem_fgt_fwd  = 0
        self.mem_fgt_bwd  = 0
        self.time_run_fwd = 0
        self.time_run_bwd = 0

class inspector():
    # -> We define an inspector class to save every intermediate 
    # -> information used during inspection, very helpful to debug.
    def __init__(self,sn,sg,our_global,device):
        self.sn = sn
        self.sg = sg
        self.mt = sn.main_target
        self.info = sg.dict_info[self.mt]
        self.timer = irotor.make_timer(device)
        self.memUsage = irotor.MeasureMemory(device)
        self.our_global = our_global
        self.tmp_local = generate_tmp_local(sn,sg,our_global,device)
        self.ret = Inspection_result()
        self.ret.relevant = True
        self.device = device

    # -- aux --
    def measure_time(self, fct, inter_fct=None):
        t = self.timer.measure(fct)
        nb_repeat = 1
        measures = [t] ; tot = t
        while (tot < constants.time_min_duration
        or nb_repeat < constants.time_min_repeat):
            if inter_fct: inter_fct()
            t = self.timer.measure(fct)
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
            self.code_run_fwd = self.sn.get_code(force_special_kwargs=True)
            exec(self.code_run_fwd, self.our_global, self.tmp_local)

        def fct_fgt_fwd():
            for tar in self.sn.tensor_targets:
                val = self.tmp_local[tar]
                val.data = torch.zeros(0,device=self.device)
                if val._base is not None:
                    val._base.data = torch.empty(0,device=self.device)
                
        def fct_del_fwd():
            code = ""
            for tar in self.sn.tensor_targets:
                code += f"del {tar};"
            self.code_del_fwd = code
            exec(self.code_del_fwd, self.our_global, self.tmp_local)

        gc.disable()
        # -> We don't want the gc to disturb the memory measurement
        _ , mem_run_fwd , peak_fwd = self.memUsage.measure(fct_run_fwd)
        overhead_fwd = peak_fwd - mem_run_fwd
        self.ret.overhead_fwd = overhead_fwd
        self.ret.mem_run_fwd = mem_run_fwd
        if not only_run:
            _ , mem_del_fwd , _ = self.memUsage.measure(fct_del_fwd)
            self.ret.mem_del_fwd = - mem_del_fwd
            fct_run_fwd()

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret.mem_fgt_fwd = - mem_fgt_fwd
            self.ret.time_run_fwd = time_run_fwd
        gc.enable()
    # ===============

    # === BACKWARD ===

    def fct_run_bwd(self):
        self.code_run_bwd = f"{self.mt}.backward({self.mt}.grad)"
        exec(self.code_run_bwd, self.our_global, self.tmp_local)

    def fct_fgt_bwd(self):
        for req_sn in self.sn.deps.keys():
            if not req_sn.is_artifact:
                for tar in req_sn.tensor_targets:
                    self.tmp_local[tar].grad = None
    def fct_prepare_bwd(self):
        self.tmp_local = generate_tmp_local(
            self.sn,self.sg,self.our_global,self.device)
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.sn.main_target].grad = (
            self.info.generate_value(self.device))

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
            self.ret.mem_fgt_bwd  = - mem_fgt_bwd
            self.ret.time_run_bwd = time_run_bwd
    
            self.tmp_local.clear()

# ==========================

