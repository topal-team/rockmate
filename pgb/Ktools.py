from .utils import *
from . import def_info
from .Stools import S_node,S_graph
import copy
import gc
# ==========================
# ====== K structure =======
# ==========================

# ************
# * K_C_node *
# ************

class K_C_node():
    def __init__(self,
            target="/!\\ No target /!\\",
            is_fwd=True,
            is_rand=False,
            main_code=None,body_code=None,
            deps_real=None,deps_fake=None,
            deps_through_size_artefacts=None,
            unique_id_generator = None):
        # ** informative **
        self.main_target = mt = target
        self.all_targets = []
        self.tensor_targets = []
        self.name        = f"fwd_{mt}" if is_fwd else f"bwd_{mt}"
        self.is_fwd      = is_fwd
        self.is_rand     = is_rand
        self.main_code   = main_code # target * AST
        self.body_code   = body_code if body_code else [] # (str*AST) list
        if unique_id_generator is None: self.unique_id = id(self)
        else:
            u = unique_id_generator[0]
            self.unique_id = u
            unique_id_generator[0] = u+1

        # ** deps/used_by **
        self.deps_real    = deps_real if deps_real else set() # KDN set
        self.deps_fake    = deps_fake if deps_fake else set() # KDN set
        self.deps_global  = set() # KDN set
        self.users_global = set() # KDN set
        self.users        = set() # KDN set
        da = deps_through_size_artefacts
        self.deps_through_size_artefacts = da if da else set() # KCN set
        # -> just for the toposort, we don't need the reciprocal users_..

        # ** inspection **
        self.time         = None
        self.overhead     = None
        self.has_phantoms = None # TO REMOVE ?
    def deps_only_global(self):
        return self.deps_global - self.deps_real.union(self.deps_fake)
    def users_only_global(self):
        return self.users_global - self.users

    def __eq__(self,kcn2,force_order=False,raise_exception=False):
        kcn1 = self
        b = (
        check_attr(kcn1,kcn2,
            ["name","main_target","is_fwd","all_targets","is_rand",
             "tensor_targets","overhead","has_phantoms"],
            raise_exception=raise_exception)
        and kcn1.get_code() == kcn2.get_code())

        # ** deps/users **
        mt = lambda nl : [rn.main_target for rn in nl]
        s = sort_nodes if force_order else (lambda s : s)
        for attr in ["deps_real","deps_fake","deps_global",
            "users","users_global","deps_through_size_artefacts"]:
            c = mt(s(getattr(kcn1,attr))) == mt(s(getattr(kcn2,attr)))
            b *= c
            if not c and raise_exception:
                raise Exception(f"kcns differ on attr {attr}")

        # ** time **
        t1 = kcn1.time
        t2 = kcn2.time
        r = ref_reasonable_rate[0]
        if not (((t1 == t2)
            or (isinstance(t1,float) and isinstance(t2,float)
            and (abs(t1 - t2) < (r * max(t1,t2)))))):return False
        if not b and raise_exception:
            raise Exception("kcns differ on attr .time")
        return bool(b)
    def __hash__(self):
        return self.unique_id

    def get_main_code(self):
        return make_str_assign(self.main_code)
    def get_code(self):
        mc = make_str_assign(self.main_code)
        mc = "" if mc == "" else mc+"\n"
        bc = make_str_list_assign(self.body_code)
        return mc + bc


# ************
# * K_D_node *
# ************

class K_D_node():
    def __init__(self,
            kdn_type = "/!\\ No kdn_type/!\\",
            target   = "/!\\ No target /!\\",
            all_targets = None,
            info        = None,
            deps        = None,
            unique_id_generator = None):
        # ** informative **
        self.kdn_type    = kdn_type # data, grad or phantoms
        self.main_target = mt = target
        self.all_targets = all_targets if all_targets else [target]
        self.name        = f"{mt} {self.kdn_type}"
        self.mem         = MemSize(0)
        self.info        = info
        if unique_id_generator is None: self.unique_id = id(self)
        else:
            u = unique_id_generator[0]
            self.unique_id = u

        # ** deps/used_by **
        self.users_real   = set() # KCN set
        self.users_fake   = set() # KCN set
        self.users_global = set() # KCN set
        self.deps_global  = set() # KCN set
        self.deps         = deps if deps else set() # KCN set
    def deps_only_global(self):
        return self.deps_global - self.deps
    def users_only_global(self):
        return self.users_global - self.users_real.union(self.users_fake)

    def __eq__(self,kdn2,force_order=False,raise_exception=False):
        kdn1 = self
        b = check_attr(kdn1,kdn2,
            ["name","mem","kdn_type",
             "main_target","all_targets"],
            raise_exception=raise_exception)
        # ** deps/users **
        mt = lambda nl : [rn.main_target for rn in nl]
        s = sort_nodes if force_order else (lambda s : s)
        for attr in ["users_real","users_fake",
            "deps","users_global","deps_global"]:
            c = mt(s(getattr(kdn1,attr))) == mt(s(getattr(kdn2,attr)))
            b *= c
            if not c and raise_exception:
                raise Exception(f"kdns differ on attr {attr}")
        return bool(b)
    def __hash__(self):
        return self.unique_id


# ***********
# * K_graph *
# ***********

class K_graph():
    def __init__(self,sg : S_graph = None,unique_id_generator=None):
        self.dict_kn  = dict() # KDN/KCN.name -> KDN/KCN
        self.list_kcn = []     # KCN list : Toposorted
        self.list_kdn = []     # KDN list : Arbitrary order

        self.input_kdn_data  = None # e.g. KDN _13.data
        self.output_kdn_data = None # e.g. KDN _116.data
        self.loss_kcn        = None
        self.output_kdn_grad = None # e.g. KDN _116.grad
        self.input_kdn_grad  = None # e.g. KDN _13.grad
        # -> for a standalone K_graph, input_kdn_data/grad are fresh nodes
        # -> otherwise they are shared with the previous k_graph
        # -> output_kdn_data/grad are shared with the next one

        self.init_code = make_ast_list_assign(sg.init_node.body_code)
        self.dict_info = sg.dict_info
        # -> no more .dict_rand
        # -> random op have been inserted at the end of Simplification
        self.unique_id_generator = unique_id_generator
        # -> to generate K_node.__hash__
        self.sg = sg # TO REMOVE

    def make_users(self):
        for kcn in self.list_kcn:
            for req_kdn in kcn.deps_real: req_kdn.users_real.add(kcn)
            for req_kdn in kcn.deps_fake: req_kdn.users_fake.add(kcn)
        for kdn in self.list_kdn:
            for req_kcn in kdn.deps: req_kcn.users.add(kdn)
    def init_deps_and_users_global(self):
        for kcn in self.list_kcn:
            kcn.deps_global = kcn.deps_real.union(kcn.deps_fake)
            kcn.users_global = set(kcn.users)
        for kdn in self.list_kdn:
            kdn.deps_global = set(kdn.deps)
            kdn.users_global = kdn.users_real.union(kdn.users_fake)

    def sort_list_kcn(self):
        # we want to use sort_based_on_deps over list_kcn
        # but to do so we need an origin_node, ie a root of
        # the "deps" relation between KCN.
        leaves_kcn = set()
        for kcn in self.list_kcn:
            if not kcn.is_fwd and len(kcn.users) == 0:
                leaves_kcn.add(kcn)
        root_kdn = K_D_node(deps = leaves_kcn,
            unique_id_generator = self.unique_id_generator)
        root_kcn = K_C_node(deps_real=set([root_kdn]),
            unique_id_generator = self.unique_id_generator)
        self.list_kcn = l = sort_based_on_deps(root_kcn)
        l.remove(root_kcn)

    def __eq__(self,g2,force_order=False,raise_exception=False):
        g1 = self
        eq_node= lambda n1,n2 : n1.__eq__(n2,force_order,raise_exception)
        b = (ast_to_str(self.init_code) == ast_to_str(g2.init_code))
        if not b and raise_exception:
            raise Exception("K_graphs diff init_code")
        for attr in ["input_kdn_grad","output_kdn_grad",
            "loss_kcn","input_kdn_data","output_kdn_data"]:
            b *= eq_node(getattr(g1,attr),getattr(g2,attr))
        for attr in ["list_kcn","list_kdn"]:
            for kn1,kn2 in zip(getattr(g1,attr),getattr(g2,attr)):
                b *= eq_node(kn1,kn2)
        keys1 = list(g1.dict_kn)
        keys2 = list(g2.dict_kn)
        #if force_order:
        #    keys1 = sort_names(keys1)
        #    keys2 = sort_names(keys2)
        #b *= (keys1 == keys2)
        if not b and raise_exception:
            raise Exception("Kgraphs differ on dict_kn's keys (order?)")
        for k in keys1:
            b *= eq_node(g1.dict_kn[k],g2.dict_kn[k])
        b *= check_attr(g1,g2,["dict_info"],raise_exception)
        return bool(b)
    def __hash__(self):
        return id(self)

# ==========================



# ==========================
# === generate tmp local ===
# fresh env to exec one node 
# ==========================

def generate_our_global(sg,model,device):
    our_global = globals().copy()
    our_global["self"] = model
    our_global["device"] = device
    for inp in sg.direct_inputs:
        info = sg.dict_info[inp]
        x = def_info.generate_val(info,device)
        our_global[inp]=x
        """ TO REMOVE
        if inp in sg.hidden_inputs:
            info.memsize = MemSize(int(tensorMsize(x)))
            # need to be done at least once """
    return our_global

def generate_tmp_local(sn,sg : S_graph,our_global,tmp_local=None):
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
            # if isinstance(req_sn_mt_value,torch.Tensor):
            #   req_sn_mt_value = req_sn_mt_value.clone()
            tmp_local[req_sn_mt] = req_sn_mt_value
            for req_req_sn in req_sn.deps.keys():
                if not (req_req_sn is sg.init_node):
                    for req_req_tar in req_req_sn.all_targets:
                        req_req_info = sg.dict_info[req_req_tar]
                        tmp_local[req_req_tar] = (
                            def_info.generate_val(req_req_info,device))
            exec(make_str_list_assign(req_sn.body_code),
                our_global,tmp_local)
    return tmp_local

def generate_deep_tmp_local(sn,sg,our_global):
    tmp_local = dict()
    for req_sn in sn.deps.keys():
        generate_tmp_local(req_sn,sg,our_global,tmp_local=tmp_local)
        exec(req_sn.get_code(), our_global, tmp_local)
    return tmp_local

# ==========================



# ==========================
# == get dep and phantoms ==
# trace grad_fn to know what
#   is needed to backward
# ==========================

def get_useful_vars(sn : S_node,sg : S_graph,our_global):
    params = dict(our_global['self'].named_parameters())
    print_debug(f"Try to open {sn.main_target}'s grad_fn")
    # == INIT ==
    tmp_local = generate_deep_tmp_local(sn,sg,our_global)
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


# ==========================



# ==========================
# ======= INSPECTION =======
# ==========================

class Inspection_result():
    def __init__(self):
        self.mem_del_fwd  = MemSize(0)
        self.overhead_fwd = MemSize(0)
        self.overhead_bwd = MemSize(0)
        self.mem_run_fwd  = MemSize(0)
        self.mem_run_bwd  = MemSize(0)
        self.mem_fgt_fwd  = MemSize(0)
        self.mem_fgt_bwd  = MemSize(0)
        self.time_run_fwd = 0
        self.time_run_bwd = 0

class inspector():
    def __init__(self, sn : S_node, sg : S_graph,our_global):
        self.sn = sn
        self.sg = sg
        self.mt = sn.main_target
        self.info = sg.dict_info[self.mt]
        self.timer = rotor.timing.make_timer(device)
        self.memUsage = rotor.memory.MeasureMemory(device)
        self.our_global = our_global
        self.tmp_local = generate_tmp_local(sn,sg,our_global)
        self.ret = Inspection_result()
    
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
            return (sum(measures)-max(measures)-min(measures))/(len(measures)-2)
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
            self.ret.mem_del_fwd = minus_mem(mem_del_fwd)
            _ , _ , _ = self.memUsage.measure(fct_run_fwd)

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret.mem_fgt_fwd = minus_mem(mem_fgt_fwd)
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
        self.tmp_local = generate_tmp_local(self.sn,self.sg,self.our_global)
        self.code_run_fwd = self.sn.get_code()
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.sn.main_target].grad = (
            def_info.generate_val(self.info,device))

    # measure
    def measure_bwd(self):
        #def fct_run_fwd():
        #    self.code_run_fwd = self.n.get_code() 
        #    exec(self.code_run_fwd, self.our_global, self.tmp_local)
        if self.info.requires_grad:
            #self.tmp_local[self.mt].data = generate_val(self.info,device)
            #self.tmp_local[self.mt].grad = generate_val(self.info,device)
            gc.disable()
            self.fct_prepare_bwd()
            _ , mem_run_bwd , peak_bwd = self.memUsage.measure(self.fct_run_bwd)
            overhead_bwd = peak_bwd - mem_run_bwd
            _ , mem_fgt_bwd , _ = self.memUsage.measure(self.fct_fgt_bwd)
            #fct_run_fwd()
            #self.timer.measure_median(fct_run_fwd)

            #self.tmp_local[self.n.main_target].grad = generate_val(self.info,device)
            self.fct_prepare_bwd()
            time_run_bwd = self.measure_time(self.fct_run_bwd, self.fct_prepare_bwd)
            # overhead_bwd contains n.target.data now /!\
            gc.enable()
            self.ret.overhead_bwd = overhead_bwd
            self.ret.mem_run_bwd  = mem_run_bwd
            self.ret.mem_fgt_bwd  = minus_mem(mem_fgt_bwd)
            self.ret.time_run_bwd = time_run_bwd
    # # ===============
# ==========================



# ==========================
# = Move from S to K graph =
# ==========================

# aux function to handle verbose and device
def aux_init_S_to_K(model,verbose,d):
    global device
    device = d if d else (
        get_device_and_check_all_same_device(model,dict(),True))
    if not (verbose is None): ref_verbose[0] = verbose

# the function that does it all
def aux_build_S_to_K(sg : S_graph,model,prev_kg=None):
    # -- init --
    dict_KCN_fwd  = dict() # mt -> KCN(fwd)
    dict_KCN_bwd  = dict() # mt -> KCN(bwd)
    dict_KDN_data = dict() # mt -> KDN(data)
    dict_KDN_grad = dict() # ...
    dict_KDN_phantoms = dict()
    unique_id_generator = [0]
    kg = K_graph(sg,unique_id_generator)

    # ============  
    def handle_node(sn : S_node):
        mt = sn.main_target
        print_debug(f"start to handle {mt}'s S_node in S_to_K")
        our_global = generate_our_global(sg,model,device)
        info = sg.dict_info[mt]

        # For artefact nodes :
        #   -> if KCN2 only need KCN1.size, it means in sg there is
        #   -> an artefact node for KCN1.size to avoid useless dep
        #   -> between KCN2 and KCN1. We decided to do NOT have KDN(size)
        #   -> in fact we just need KCN1 to be ordered before KCN2 in
        #   -> the toposort. To do so we create a tmp special dep:
        #   -> "deps_through_size_artefacts" when we find artft in sn.deps
        if sn.is_artefact: return ()

        # *** build the fwd part ***
        sn_deps = set(sn.deps.keys())
        sn_deps.discard(sg.init_node)

        # -> handle artefact deps :
        kcn_deps_art_kcn = set()
        for req_sn in sn_deps:
            if req_sn.is_artefact:
                sn_deps.discard(req_sn)
                req_real_sn = list(req_sn.deps.keys())[0] # art's parent
                kcn_deps_art_kcn.add(dict_KCN_fwd[req_real_sn.main_target])

        # -> get kdn_data deps for fwd
        sn_deps_mt = [req_sn.main_target for req_sn in sn_deps]
        kcn_fwd_deps = set(
            dict_KDN_data[mt] for mt in sn_deps_mt)

        # -> KCN(fwd)
        kcn_fwd = K_C_node(
            target    = mt,
            is_fwd    = True,
            main_code = sn.main_code,
            body_code = sn.body_code,
            deps_real = kcn_fwd_deps,
            deps_through_size_artefacts = kcn_deps_art_kcn,
            unique_id_generator = unique_id_generator)
        kcn_fwd.all_targets = sn.all_targets
        kcn_fwd.tensor_targets = sn.tensor_targets
        dict_KCN_fwd[mt] = kcn_fwd

        # -> KDN(data)
        kdn_data = K_D_node(
            kdn_type    = "data",
            target      = mt,
            all_targets = sn.tensor_targets,
            info        = info,
            deps        = set([kcn_fwd]),
            unique_id_generator = unique_id_generator)
        dict_KDN_data[mt] = kdn_data


        # *** build the bwd part ***
        if info.requires_grad:
            # -> get kdn_data and phantoms deps for bwd
            deps_real_name,exist_phantoms = (
                get_useful_vars(sn,sg,our_global))
            bwd_deps_real_mt = (
                set(deps_real_name).intersection(set(sn_deps_mt)))
            kcn_bwd_deps_real = set(
                dict_KDN_data[mt] for mt in bwd_deps_real_mt)
            kcn_bwd_deps_fake = (
                kcn_fwd_deps - kcn_bwd_deps_real)
            kcn_bwd_deps_fake.add(kdn_data)

            # -> KCN(bwd)
            kcn_bwd = K_C_node(
                target    = mt,
                is_fwd    = False,
                deps_real = kcn_bwd_deps_real,
                deps_fake = kcn_bwd_deps_fake,
                unique_id_generator = unique_id_generator)
            dict_KCN_bwd[mt] = kcn_bwd

            # -> KDN(phantoms)
            if exist_phantoms:
                kdn_phantoms = K_D_node(
                    kdn_type    = "phantoms",
                    target      = mt,
                    info        = info,
                    deps        = set([kcn_fwd]),
                    unique_id_generator = unique_id_generator)
                dict_KDN_phantoms[mt] = kdn_phantoms
                kcn_bwd.deps_real.add(kdn_phantoms)
                kcn_fwd.has_phantoms = True
            else: kcn_fwd.has_phantoms = False

            # -> KDN(grad)
            kdn_grad = K_D_node(
                kdn_type    = "grad",
                info        = info,
                target      = mt,
                unique_id_generator = unique_id_generator)
            dict_KDN_grad[mt] = kdn_grad
            kcn_bwd.deps_real.add(kdn_grad)

            # -> KDN(grad).deps of fwd_deps
            for req_sn_mt in sn_deps_mt:
                if req_sn_mt in dict_KDN_grad: #i.e. requires_grad
                    dict_KDN_grad[req_sn_mt].deps.add(kcn_bwd)


        # *** inspection ***
        if device == torch.device("cpu"):
            res = Inspection_result()
        else:
            ins = inspector(sn,sg,our_global)
            ins.measure_fwd()
            ins.measure_bwd()
            res = ins.ret

        # -> fwd ins
        kcn_fwd.overhead = res.overhead_fwd
        kcn_fwd.time     = res.time_run_fwd
        kdn_data.mem     = res.mem_fgt_fwd
        info.memsize     = res.mem_fgt_fwd # -> may correct

        # -> bwd ins
        if info.requires_grad:
            kcn_bwd.overhead = res.overhead_bwd
            kcn_bwd.time     = res.time_run_bwd
            kdn_grad.mem     = kdn_data.mem

            # -> phantoms ins
            if ref_test_phantoms_detection[0]:
                exist_diff=res.mem_run_fwd.v - res.mem_fgt_fwd.v > 0
                if exist_diff or exist_phantoms:
                    print(f"For node {mt}: mem_diff : {exist_diff} "\
                          f"and detection {exist_phantoms}")

            if exist_phantoms:
                kdn_phantoms.mem = MemSize(
                    res.mem_run_fwd.v - res.mem_fgt_fwd.v)

        # ins.tmp_local.clear()
    # ============ 


    for sn in sg.nodes:
        handle_node(sn)

    # -> loss_node
    kg.output_kdn_data=output_kdn_data = dict_KDN_data[sg.hidden_output]
    kg.output_kdn_grad=output_kdn_grad = dict_KDN_grad[sg.hidden_output]
    kg.loss_kcn=loss_kcn = K_C_node(
        target    = "loss",
        is_fwd    = True,
        main_code = ("loss",make_ast_constant("LOSS")),
        deps_real = set([output_kdn_data]),
        unique_id_generator = unique_id_generator)
    loss_kcn.time     = 0
    loss_kcn.overhead = MemSize(0)
    dict_KCN_fwd[loss_kcn.main_target] = loss_kcn
    output_kdn_grad.deps.add(loss_kcn)

    # -> store the nodes
    kg.list_kcn = (
        list(dict_KCN_fwd.values()) +
        list(dict_KCN_bwd.values()))
    kg.list_kdn = (
        list(dict_KDN_data.values()) +
        list(dict_KDN_grad.values()) +
        list(dict_KDN_phantoms.values()))
    for kn in kg.list_kcn+kg.list_kdn: kg.dict_kn[kn.name]=kn

    # -> build "users" attributes as reciprocal of "deps"
    kg.make_users()


    # *** global relations ***
    kg.init_deps_and_users_global()
    # -> get input_kdn_data/grad from prev_kg
    if prev_kg:
        kg.input_kdn_data=input_kdn_data = prev_kg.output_kdn_data
        kg.input_kdn_grad=input_kdn_grad = prev_kg.output_kdn_grad
    # -> or create fresh vars in case kg is a standalone graph
    else:
        if len(sg.hidden_inputs) != 1: inp_mt = "sources"
        else: inp_mt = sg.hidden_inputs[0]
        kg.input_kdn_data=input_kdn_data = K_D_node(
            kdn_type = "data", target = inp_mt,
            all_targets = sg.direct_inputs,
            unique_id_generator = unique_id_generator)
        kg.input_kdn_grad=input_kdn_grad = K_D_node(
            kdn_type = "grad", target = inp_mt,
            all_targets = sg.direct_inputs,
            unique_id_generator = unique_id_generator)

    # -> users of inp_data and deps of inp_grad
    input_sn_users_mt = [
        sn.main_target for sn in sg.init_node.users.keys()]
    input_kdn_data_users = set(
        dict_KCN_fwd[mt] for mt in input_sn_users_mt)
    input_kdn_grad_deps  = set(
        dict_KCN_bwd[mt] for mt in input_sn_users_mt
        if mt in dict_KCN_bwd)

    # -> make deps/users_global
    input_kdn_data.users_global.update(input_kdn_data_users)
    for user_kcn in input_kdn_data_users:
        user_kcn.deps_global.add(input_kdn_data)
    input_kdn_grad.deps_global.update(input_kdn_grad_deps)
    for user_kcn in input_kdn_grad_deps:
        user_kcn.users_global.add(input_kdn_grad)

    # -> TOPOSORT list_kcn
    kg.sort_list_kcn()

    return kg


def S_to_K(sg : S_graph,model,verbose=None,device=None):
    aux_init_S_to_K(model,verbose,device)
    return aux_build_S_to_K(sg,model,prev_kg = None)


def S_list_to_K_list(list_sg,model,verbose=None,device=None):
    aux_init_S_to_K(model,verbose,device)
    list_kg = []
    prev_kg = None
    for sg in list_sg:
        prev_kg = kg = aux_build_S_to_K(sg,model,prev_kg)
        list_kg.append(kg)
    return list_kg


# ==========================



# ==========================
# === Copying functions ====
# ==========================

def copy_K_C_node(kcn : K_C_node):
    new_kcn = K_C_node()
    new_kcn.main_target  = kcn.main_target
    new_kcn.all_targets  = list(kcn.all_targets)
    new_kcn.tensor_targets = list(kcn.tensor_targets)
    new_kcn.name         = kcn.name
    new_kcn.is_fwd       = kcn.is_fwd
    new_kcn.main_code    = kcn.main_code
    new_kcn.body_code    = [tuple(c) for c in kcn.body_code]
    new_kcn.time         = kcn.time
    new_kcn.overhead     = kcn.overhead
    new_kcn.has_phantoms = kcn.has_phantoms
    new_kcn.unique_id    = kcn.unique_id
    for attr in ["deps_real","deps_fake","deps_global",
        "users","users_global","deps_through_size_artefacts"]:
        setattr(new_kcn,attr,set()) # /!\
    return new_kcn

def copy_K_D_node(kdn : K_D_node):
    new_kdn = K_D_node()
    new_kdn.kdn_type    = kdn.kdn_type
    new_kdn.main_target = kdn.main_target
    new_kdn.all_targets = list(kdn.all_targets)
    new_kdn.name        = kdn.name
    new_kdn.mem         = kdn.mem
    new_kdn.info        = kdn.info
    new_kdn.unique_id   = kdn.unique_id
    for attr in ["users_real","users_fake",
        "deps","users_global","deps_global"]:
        setattr(new_kdn,attr,set()) # /!\
    return new_kdn


def copy_K_graph(kg : K_graph):
    new_kg = K_graph(kg.sg) # TO CHANGE
    new_kg.dict_info = dict(kg.dict_info)
    new_kg.init_code = kg.init_code
    new_kg.unique_id_generator = copy_generator(kg.unique_id_generator)

    # == NODES ==
    new_dict_kn = new_kg.dict_kn
    new_kg.list_kcn = new_list_kcn = (
        [copy_K_C_node(kcn) for kcn in kg.list_kcn])
    new_kg.list_kdn = new_list_kdn = (
        [copy_K_D_node(kdn) for kdn in kg.list_kdn])
    for kn in new_list_kcn+new_list_kdn: new_dict_kn[kn.name]=kn

    # -- edges --
    for new_kcn,old_kcn in zip(new_list_kcn,kg.list_kcn):
        for attr in ["deps_real","deps_fake","users",
            "deps_through_size_artefacts"]:
            old_edges = getattr(old_kcn,attr)
            for old_aux_kn in old_edges:
                getattr(new_kcn,attr).add(new_dict_kn[old_aux_kn.name])
    for new_kdn,old_kdn in zip(new_list_kdn,kg.list_kdn):
        for attr in ["users_real","users_fake","deps"]:
            old_edges = getattr(old_kdn,attr)
            for old_aux_kn in old_edges:
                getattr(new_kdn,attr).add(new_dict_kn[old_aux_kn.name])

    # -- global edges --
    # /!\ new_kg is a standalone graph /!\
    new_kg.init_deps_and_users_global()
    old_inp_data = kg.input_kdn_data
    old_inp_grad = kg.input_kdn_grad
    new_kg.input_kdn_data=new_inp_data = copy_K_D_node(old_inp_data)
    new_kg.input_kdn_grad=new_inp_grad = copy_K_D_node(old_inp_grad)
    for old_fst_kcn in old_inp_data.users_only_global():
        new_fst_kcn = new_dict_kn[old_fst_kcn.name]
        new_fst_kcn.deps_global.add(new_inp_data)
        new_inp_data.users_global.add(new_fst_kcn)
    for old_lst_kcn in old_inp_grad.deps_only_global():
        new_lst_kcn = new_dict_kn[old_lst_kcn.name]
        new_lst_kcn.users_global.add(new_inp_grad)
        new_inp_grad.deps_global.add(new_lst_kcn)

    new_kg.output_kdn_data = new_dict_kn[kg.output_kdn_data.name]
    new_kg.output_kdn_grad = new_dict_kn[kg.output_kdn_grad.name]
    new_kg.loss_kcn = new_dict_kn[kg.loss_kcn.name]
    return new_kg

# ==========================



# ==========================
# === printing functions ===
# ==========================

color_kcn_fwd  = "blue"
color_kcn_bwd  = "blueviolet"
color_special  = "green"
color_kdn      = "olive"

def get_color(kn):
    if isinstance(kn,K_D_node): return color_kdn
    if kn.is_fwd: return color_kcn_fwd
    return color_kcn_bwd

def aux_print_graph(dot,kg,uniq_num):
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,**kwargs): dot.edge(uni(i1),uni(i2),**kwargs)

    # *** nodes ***
    def print_kcn(kcn):
        mt = kcn.main_target
        if mt == "loss":
            node(kcn.name,"LOSS KCN",color=color_special)
        else:
            lbl = kcn.get_code() if kcn.is_fwd else f"backward of {mt}"
            node(kcn.name,lbl,color=get_color(kcn),tooltip = (
                f"Time : {kcn.time}\n"\
                f"Mem overhead : {kcn.overhead}"))
    def print_kdn(kdn):
        node(kdn.name,kdn.name,color=get_color(kdn),
            tooltip = f"Mem {kdn.mem}")

    for kcn in kg.list_kcn: print_kcn(kcn)
    for kdn in kg.list_kdn: print_kdn(kdn)

    # *** edges ***
    for kcn in kg.list_kcn:
        for req_kdn in kcn.deps_real:
            c = get_color(req_kdn)
            edge(req_kdn.name,kcn.name,color=c)
        for req_kdn in kcn.deps_fake:
            c = get_color(req_kdn)
            edge(req_kdn.name,kcn.name,color=c,style="dashed")
    for kdn in kg.list_kdn:
        for req_kcn in kdn.deps:
            edge(req_kcn.name,kdn.name,color=get_color(req_kcn))

    # *** io - global relations ***
    inp_data = kg.input_kdn_data
    inp_grad = kg.input_kdn_grad
    kwargs = {"color":color_special , "style":"dashed"}
    node(inp_data.name,inp_data.name,**kwargs)
    node(inp_grad.name,inp_grad.name,**kwargs)
    for user_inp_data in inp_data.users_only_global():
        edge(inp_data.name,user_inp_data.name,**kwargs)
    for req_inp_grad in inp_grad.deps_only_global():
        edge(req_inp_grad.name,inp_grad.name,**kwargs)


def print_K_graph(kg : K_graph,name=None,open=True):
    if name is None: name = "complet K-graph"
    print(
        f"Forward + Backward graph with Computation and "\
        f"Data nodes: {len(kg.list_kcn)} + {len(kg.list_kdn)}")
    dot = graphviz.Digraph(name,
        comment="K_graph = Forward + Backward with Comp and Data nodes")
    aux_print_graph(dot,kg,0)
    graph_render(dot,open,"K") # from utils.py


def print_K_graph_list(list_kg,name=None,open=True):
    if name is None: name = "seq K-graphs"
    list_nb_kcn = [len(kg.list_kcn) for kg in list_kg]
    list_nb_kdn = [len(kg.list_kdn) for kg in list_kg]
    tot_nb_kcn = sum(list_nb_kcn)
    tot_nb_kdn = sum(list_nb_kdn)
    str_list_nb_kcn = "+".join(str(i) for i in list_nb_kcn)
    str_list_nb_kdn = "+".join(str(i) for i in list_nb_kdn)
    print(
        f"{len(list_kg)} K_graphs in seq, with :\n"\
        f"{str_list_nb_kcn} = {tot_nb_kcn} Comp nodes\n"\
        f"{str_list_nb_kdn} = {tot_nb_kdn} Data nodes\n"\
        f"=> total of {tot_nb_kcn + tot_nb_kdn} nodes")
    dot = graphviz.Digraph(
        name,
        comment="K_graph list : cut fwd+bwd with Comp and Data nodes")
    for i in range(len(list_kg)):
        aux_print_graph(dot,list_kg[i],i)
    graph_render(dot,open,"K") # from utils.py

# ==========================

