from .utils import *
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
            main_code=None,body_code=None,
            deps_real=None,deps_fake=None,
            deps_through_size_artefacts=None):
        # ** informative **
        self.main_target = mt = target
        self.name        = f"fwd_{mt}" if is_fwd else f"bwd_{mt}"
        self.is_fwd      = is_fwd
        self.main_code   = main_code # AST
        self.body_code   = body_code if body_code else [] # AST list

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
        self.inspector    = None # TO REMOVE
        self.has_phantoms = None # TO REMOVE ?

    def __eq__(self,kcn2):
        kcn1 = self
        b = (
        check_attr(kcn1,kcn2,
            ["name","main_target","is_fwd",
             "overhead","has_phantoms"],
            raise_exception=False)
        and kcn1.get_code() == kcn2.get_code())

        # ** deps/users **
        get_mt = lambda nl : [rn.main_target for rn in nl]
        for attr in [
            "deps_real","deps_fake","deps_global",
            "users","users_global",
            "deps_through_size_artefacts"]:
            b *= get_mt(getattr(kcn1,attr)) == get_mt(getattr(kcn2,attr))

        # ** time **
        t1 = kcn1.time
        t2 = kcn2.time
        r = ref_reasonable_rate[0]
        if not (((t1 == t2)
            or (isinstance(t1,float) and isinstance(t2,float)
            and (abs(t1 - t2) < (r * max(t1,t2)))))):return False
        return bool(b)
    def __hash__(self):
        return self.name.__hash__()

    def full_code(self):
        if self.main_code is None: mc = []
        else: mc = [self.main_code]
        return make_ast_module(mc + self.body_code)
    def get_main_code(self):
        if self.main_code is None: mc = []
        else: mc = [self.main_code]
        return ast_to_str(make_ast_module(mc))
    def get_code(self):
        return ast_to_str(self.full_code())


# ************
# * K_D_node *
# ************

class K_D_node():
    def __init__(self,
            kdn_type = "/!\\ No kdn_type/!\\",
            target   = "/!\\ No target /!\\",
            all_targets = None,
            deps        = None):
        # ** informative **
        self.kdn_type    = kdn_type # data, grad or phantoms
        self.main_target = mt = target
        self.name        = f"{mt} {self.kdn_type}"
        self.all_targets = all_targets if all_targets else [target]
        self.mem         = MemSize(0)

        # ** deps/used_by **
        self.users_real   = set() # KCN set
        self.users_fake   = set() # KCN set
        self.users_global = set() # KCN set
        self.deps_global  = set() # KCN set
        self.deps         = deps if deps else set() # KCN set

    def __eq__(self,kdn2):
        kdn1 = self
        b = check_attr(kdn1,kdn2,
            ["name","mem","kdn_type",
             "main_target","all_targets"],
            raise_exception=False)
        # ** deps/users **
        get_mt = lambda nl : [rn.main_target for rn in nl]
        for attr in [
            "users_real","users_fake","deps",
            "users_global","deps_global"]:
            b *= get_mt(getattr(kdn1,attr)) == get_mt(getattr(kdn2,attr))
        return bool(b)
    def __hash__(self):
        return self.name.__hash__()


# ***********
# * K_graph *
# ***********

class K_graph():
    def __init__(self,sg : S_graph):
        self.dict_kn  = dict() # KDN/KCN.name -> KDN/KCN
        self.list_kcn = []     # KCN list : Toposorted
        self.list_kdn = []     # KDN list : Arbitrary order

        self.input_kdn_data  = None # e.g. KDN _13.data
        self.output_kdn_data = None # e.g. KDN _116.data
        self.loss_kcn        = None
        self.output_kdn_grad = None # e.g. KDN _116.grad
        self.input_kdn_grad  = None # e.g. KDN _13.grad
        # -> for the first K_graph, input_kdn_data/grad are fresh nodes
        # -> otherwise they are shared with the previous k_graph
        # -> output_kdn_data/grad are shared with the next one

        self.init_code = make_ast_module(sg.init_node.body_code)
        self.dict_info = sg.dict_info
        self.dict_rand = sg.dict_rand
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
        root_kdn = K_D_node(deps = leaves_kcn)
        root_kcn = K_C_node(deps_real=set([root_kdn]))
        self.list_kcn = l = sort_based_on_deps(root_kcn)
        l.remove(root_kcn)

    def __eq__(self,g2): # aggressive
        if ast_to_str(self.init_code) != ast_to_str(g2.init_code):
            raise Exception("diff init_code")
        return check_attr(self,g2,["sg",
            "dict_kn","list_kcn","list_kdn",
            "input_kdn_data","output_kdn_data","loss_kcn",
            "input_kdn_grad","output_kdn_grad",
            "dict_info","dict_rand"],raise_exception=True)
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
        x = generate_val(info,device)
        our_global[inp]=x
        if inp in sg.hidden_inputs:
            info.memsize = MemSize(int(tensorMsize(x)))
            # need to be done at least once
    return our_global

def generate_tmp_local(sn,sg : S_graph,our_global):
    tmp_local = {}
    exec(sg.init_node.get_code(),our_global,tmp_local)
    for req_sn in sn.deps.keys():
        if not (req_sn is sg.init_node):
            # we create the main_target value, and we run the body_code
            # but the body_code may requires some artefacts
            req_sn_mt = req_sn.main_target
            main_info = sg.dict_info[req_sn_mt]
            tmp_local[req_sn_mt] = generate_val(main_info,device)
            for req_req_sn in req_sn.deps.keys():
                if not (req_req_sn is sg.init_node):
                    for req_req_tar in req_req_sn.all_targets:
                        req_req_info = sg.dict_info[req_req_tar]
                        tmp_local[req_req_tar] = (
                            generate_val(req_req_info,device))
            for c in req_sn.body_code:
                exec(ast_to_str(c),our_global,tmp_local)
    """ TODO
    if sn.is_rand:
        for req_rd in sn.deps_rand:
            exec(sg.dict_rand[req_rd],our_global,tmp_local)
    """
    return tmp_local

def generate_deep_tmp_local(sn,sg,our_global):
    tmp_local = dict()
    for req_sn in sn.deps.keys():
        tmp_local.update(generate_tmp_local(req_sn,sg,our_global))
        exec(req_sn.get_code(), our_global, tmp_local)
    return tmp_local

# ==========================



# ==========================
# == get dep and phantoms ==
# trace grad_fn to know what
#   is needed to backward
# ==========================

# TODO : TO TEST "exist_phantoms"
# -> I don't want to use inspection to detect phantoms
# -> Ktools shouldn't requires a GPU to produce a correct graph

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

# TODO : rewrite + clean this section

class inspector():
    def __init__(self, sn : S_node, sg : S_graph,our_global):
        self.sn = sn
        self.mt = sn.main_target
        self.info = sg.dict_info[self.mt]
        self.timer = rotor.timing.make_timer(device)
        self.memUsage = rotor.memory.MeasureMemory(device)
        self.our_global = our_global
        self.tmp_local = generate_tmp_local(sn,sg,our_global)
        self.ret = {}
    
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
        self.ret["overhead_fwd"] = overhead_fwd
        self.ret["mem_run_fwd"] = mem_run_fwd
        if not only_run:
            _ , mem_del_fwd , _ = self.memUsage.measure(fct_del_fwd)
            self.ret["mem_del_fwd"] = minus_mem(mem_del_fwd)
            _ , _ , _ = self.memUsage.measure(fct_run_fwd)

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret["mem_fgt_fwd"] = minus_mem(mem_fgt_fwd)
            self.ret["time_run_fwd"] = time_run_fwd
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
        self.code_run_fwd = self.sn.get_code()
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.sn.main_target].grad = generate_val(self.info,device)

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
            self.ret["overhead_bwd"] = overhead_bwd
            self.ret["mem_run_bwd"]  = mem_run_bwd
            self.ret["mem_fgt_bwd"]  = minus_mem(mem_fgt_bwd)
            self.ret["time_run_bwd"] = time_run_bwd
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
    kg = K_graph(sg)

    # ============  
    def handle_node(sn : S_node):
        mt = sn.main_target
        print_debug(f"start to handle {mt}'s S_node in S_to_K")
        our_global = generate_our_global(sg,model,device)

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
            deps_through_size_artefacts = kcn_deps_art_kcn)
        dict_KCN_fwd[mt] = kcn_fwd

        # -> KDN(data)
        kdn_data = K_D_node(
            kdn_type    = "data",
            target      = mt,
            all_targets = sn.tensor_targets,
            deps        = set([kcn_fwd]))
        dict_KDN_data[mt] = kdn_data


        # *** build the bwd part ***
        info = sg.dict_info[mt]
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
                deps_fake = kcn_bwd_deps_fake)
            dict_KCN_bwd[mt] = kcn_bwd

            # -> KDN(phantoms)
            if exist_phantoms:
                kdn_phantoms = K_D_node(
                    kdn_type    = "phantoms",
                    target      = mt,
                    deps        = set([kcn_fwd]))
                dict_KDN_phantoms[mt] = kdn_phantoms
                kcn_bwd.deps_real.add(kdn_phantoms)
                kcn_fwd.has_phantoms = True
            else: kcn_fwd.has_phantoms = False

            # -> KDN(grad)
            kdn_grad = K_D_node(
                kdn_type    = "grad",
                target      = mt)
            dict_KDN_grad[mt] = kdn_grad
            kcn_bwd.deps_real.add(kdn_grad)

            # -> KDN(grad).deps of fwd_deps
            for req_sn_mt in sn_deps_mt:
                if req_sn_mt in dict_KDN_grad: #i.e. requires_grad
                    dict_KDN_grad[req_sn_mt].deps.add(kcn_bwd)


        # *** inspection ***
        ins = inspector(sn,sg,our_global)
        ins.measure_fwd()
        ins.measure_bwd()
        res = ins.ret

        # -> fwd ins
        kcn_fwd.ins = ins
        kcn_fwd.overhead = res["overhead_fwd"]
        kcn_fwd.time     = res["time_run_fwd"]
        kdn_data.mem     = res["mem_fgt_fwd"]
        info.memsize     = res["mem_fgt_fwd"]

        # -> bwd ins
        if info.requires_grad:
            kcn_bwd.ins = ins
            kcn_bwd.overhead = res["overhead_bwd"]
            kcn_bwd.time     = res["time_run_bwd"]
            kdn_grad.mem     = res["mem_fgt_bwd"] # assert : = mem_fgt_fwd

            # -> phantoms ins
            if exist_phantoms:
                kdn_phantoms.mem = MemSize(
                    res["mem_run_fwd"].v - res["mem_fgt_fwd"].v)

        ins.tmp_local.clear()
    # ============ 


    for sn in sg.nodes:
        handle_node(sn)

    # -> loss_node
    kg.output_kdn_data=output_kdn_data = dict_KDN_data[sg.hidden_output]
    kg.output_kdn_grad=output_kdn_grad = dict_KDN_grad[sg.hidden_output]
    kg.loss_kcn=loss_kcn = K_C_node(
        target    = "loss",
        is_fwd    = True,
        main_code = make_ast_constant("LOSS"),
        deps_real = set([output_kdn_data]))
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
    # -> or create fresh vars in case kg is the first K_graph
    else:
        kg.input_kdn_data=input_kdn_data = K_D_node(
            kdn_type = "data", target = "source",
            all_targets = sg.direct_inputs)
        kg.input_kdn_grad=input_kdn_grad = K_D_node(
            kdn_type = "grad", target = "source",
            all_targets = sg.direct_inputs)

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
    inp_data_users_only_global = (
        inp_data.users_global -
        inp_data.users_real.union(inp_data.users_fake))
    inp_grad_deps_only_global = (
        inp_grad.deps_global - inp_grad.deps)
    for user_inp_data in inp_data_users_only_global:
        edge(inp_data.name,user_inp_data.name,**kwargs)
    for req_inp_grad in inp_grad_deps_only_global:
        edge(req_inp_grad.name,inp_grad.name,**kwargs)


def print_K_graph(kg : K_graph,name="complete K-graph",open=True):
    print(
        f"Forward + Backward graph with Computation and "\
        f"Data nodes: {len(kg.list_kcn)} + {len(kg.list_kdn)}")
    dot = graphviz.Digraph(name,
        comment="K_graph = Forward + Backward with Comp and Data nodes")
    aux_print_graph(dot,kg,0)
    graph_render(dot,open,"K") # from utils.py


def print_K_graph_list(list_kg,name="seq K-graphs",open=True):
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

