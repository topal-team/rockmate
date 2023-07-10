# ==========================
# ====== K structure =======
# ==========================

from rkgb.utils import *
from rkgb.Stools import S_node,S_graph

# ************
# * K_C_node *
# ************

class K_C_node():
    def __init__(self,
            target="/!\\ No target /!\\",
            all_targets       = None,
            tensor_targets    = None,
            inplace_targets   = None,
            container_targets = None,
            is_fwd  = True,
            is_rand = False,
            main_code    = None,
            inplace_code = None,
            body_code    = None,
            deps_real = None,
            deps_fake = None,
            deps_through_size_artefacts=None,
            unique_id_generator = None):
        # ** informative **
        self.main_target = mt = target
        atars = all_targets
        ttars = tensor_targets
        itars = inplace_targets
        ctars = container_targets
        self.all_targets       = atars if atars else [mt]
        self.tensor_targets    = ttars if ttars else [mt]
        self.inplace_targets   = itars if itars else []
        self.container_targets = ctars if ctars else []
        self.name        = f"fwd_{mt}" if is_fwd else f"bwd_{mt}"
        self.is_fwd      = is_fwd
        self.is_rand     = is_rand
        self.main_code   = main_code # target * AST
        self.inplace_code= inplace_code if inplace_code else []
        self.body_code   = body_code if body_code else [] # (str*AST) list
        self.unique_id   = small_fcts.use_generator(unique_id_generator,self)

        # ** deps/used_by **
        self.deps_real    = deps_real if deps_real else set() # KDN set
        self.deps_fake    = deps_fake if deps_fake else set() # KDN set
        self.deps_global  = set() # KDN set
        self.users_global = set() # KDN set
        self.users        = set() # KDN set
        self.deps_impossible_to_restore = set() # (KDN * str) set
        da = deps_through_size_artefacts
        self.deps_through_size_artefacts = da if da else set() # KCN set
        # -> just for the toposort, we don't need the reciprocal users_..

        # ** inspection **
        self.time         = None
        self.overhead     = None
        self.phantom_names = []
        self.alias_in_users_phantoms = []

    def deps_only_global(self):
        return self.deps_global - self.deps_real.union(self.deps_fake)
    def users_only_global(self):
        return self.users_global - self.users

    def __eq__(self,kcn2,force_order=False,raise_exception=False):
        kcn1 = self
        try:
            b = (
            small_fcts.check_attr(kcn1,kcn2,
                ["name","main_target","is_fwd",
                "all_targets","container_targets",
                "tensor_targets","inplace_targets",
                "is_rand","overhead",],#"phantom_names",
                #"alias_in_users_phantoms"],
                raise_exception=raise_exception)
            and kcn1.full_code() == kcn2.full_code())
            if not b and raise_exception: raise Exception(
                f"{kcn1.main_target} and {kcn2.main_target} KCN differ on "\
                f"code : {kcn1.full_code()}\n===\n{kcn2.full_code()}")

            # ** deps/users **
            mmt = lambda nl : [rn.main_target for rn in nl]
            s = shared_methods.sort_nodes if force_order else (lambda s : s)
            for attr in ["deps_real","deps_fake","deps_global",
                "users","users_global","deps_through_size_artefacts"]:
                c = mmt(s(getattr(kcn1,attr))) == mmt(s(getattr(kcn2,attr)))
                b *= c
                if not c and raise_exception:
                    raise Exception(f"kcns differ on attr {attr}")
            mmt2 = lambda nl : [(r[0].main_target,r[1]) for r in nl]
            b *= small_fcts.clean__eq__(
                mmt2(kcn1.deps_impossible_to_restore),
                mmt2(kcn2.deps_impossible_to_restore),
                raise_exception=raise_exception)

            # ** time **
            t1 = kcn1.time
            t2 = kcn2.time
            r = global_vars.ref_reasonable_rate[0]
            if not (((t1 == t2)
                or (isinstance(t1,float) and isinstance(t2,float)
                and (abs(t1 - t2) < (r * max(t1,t2)))))):return False
            if not b and raise_exception:
                raise Exception("kcns differ on attr .time")
            return bool(b)
        except AttributeError as a: return kcn1.__hash__() == kcn2.__hash__()
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self)
    def clean_hash_in_sets(self):
        for attr in ["deps_real","deps_fake","deps_global",
            "users","users_global","deps_through_size_artefacts"]:
            s1 = getattr(self,attr)
            s2 = set()
            for x in s1:
                s2.add(x)
            setattr(self,attr,s2)


    def get_main_code(self,force_special_kwargs=False):
        return ast_add_on.make_str_assign(
            self.main_code,force_special_kwargs)
    def get_code(self,*args, **kwargs):
        return shared_methods.get_code(self,*args, **kwargs)
    def full_code(self,*args, **kwargs):
        return shared_methods.full_code(self,*args, **kwargs)


# ************
# * K_D_node *
# ************

class K_D_node():
    def __init__(self,
            kdn_type = "/!\\ No kdn_type/!\\",
            target   = "/!\\ No target /!\\",
            all_targets       = None,
            tensor_targets    = None,
            inplace_targets   = None,
            container_targets = None,
            info        = None,
            deps        = None,
            unique_id_generator = None):
        # ** informative **
        self.kdn_type    = kdn_type # data, grad or phantoms
        self.main_target = mt = target
        atars = all_targets
        ttars = tensor_targets
        itars = inplace_targets
        ctars = container_targets
        self.all_targets       = atars if atars else [mt]
        self.tensor_targets    = ttars if ttars else [mt]
        self.inplace_targets   = itars if itars else []
        self.container_targets = ctars if ctars else []
        self.name        = f"{mt} {self.kdn_type}"
        self.mem         = 0
        self.info        = info
        self.includes_base = False
        self.includes_phantoms = False
        self.unique_id   = small_fcts.use_generator(unique_id_generator,self)
        # ** deps/used_by **
        self.users_real   = set() # KCN set
        self.users_fake   = set() # KCN set
        self.users_global = set() # KCN set
        self.deps_global  = set() # KCN set
        self.deps         = deps if deps else set() # KCN set
        self.users_impossible_to_restore = set() # (KCN * str) set
        self.alias_in_users_phantoms = []
    def deps_only_global(self):
        return self.deps_global - self.deps
    def users_only_global(self):
        return self.users_global - self.users_real.union(self.users_fake)

    def __eq__(self,kdn2,force_order=False,raise_exception=False):
        kdn1 = self
        try:
            b = small_fcts.check_attr(kdn1,kdn2,
                ["name","mem","kdn_type","main_target",
                "all_targets","container_targets",
                "tensor_targets","inplace_targets",
                "includes_phantoms",
                "includes_base"],
                #"alias_in_users_phantoms"],
                raise_exception=raise_exception)
            # ** deps/users **
            mt = lambda nl : [rn.main_target for rn in nl]
            s = shared_methods.sort_nodes if force_order else (lambda s : s)
            for attr in ["users_real","users_fake",
                "deps","users_global","deps_global"]:
                c = mt(s(getattr(kdn1,attr))) == mt(s(getattr(kdn2,attr)))
                b *= c
                if not c and raise_exception:
                    raise Exception(f"kdns differ on attr {attr}")
            mmt2 = lambda nl : [(r[0].main_target,r[1]) for r in nl]
            b *= small_fcts.clean__eq__(
                mmt2(kdn1.users_impossible_to_restore),
                mmt2(kdn2.users_impossible_to_restore),
                raise_exception=raise_exception)
            return bool(b)
        except AttributeError as a: return kdn1.__hash__() == kdn2.__hash__()
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self)
    """ USELESS
    def clean_hash_in_sets(self):
        for attr in ["users_real","users_fake",
            "deps","users_global","deps_global"]:
            s1 = getattr(self,attr)
            s2 = set()
            for x in s1:
                s2.add(x)
            setattr(self,attr,s2)
    """


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

        self.init_code = shared_methods.get_code_ast(sg.init_node)
        self.dict_info = sg.dict_info
        self.dict_constants = sg.dict_constants
        # -> no more .dict_rand
        # -> random op have been inserted at the end of Simplification
        self.unique_id_generator = unique_id_generator
        # -> to generate K_node.__hash__
        self.sg = sg

    def make_users(self):
        for kcn in self.list_kcn:
            for req_kdn in kcn.deps_real: req_kdn.users_real.add(kcn)
            for req_kdn in kcn.deps_fake: req_kdn.users_fake.add(kcn)
            for req_kdn,ph_name in kcn.deps_impossible_to_restore:
                req_kdn.users_impossible_to_restore.add((kcn,ph_name))
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
        self.list_kcn = l = shared_methods.sort_based_on_deps(root_kcn)
        l.remove(root_kcn)

    def __eq__(self,g2,force_order=False,raise_exception=False):
        g1 = self
        eq_node= lambda n1,n2 : n1.__eq__(n2,force_order,raise_exception)
        b = (ast_add_on.ast_to_str(self.init_code) 
            == ast_add_on.ast_to_str(g2.init_code))
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
        if force_order:
            keys1 = shared_methods.sort_names(keys1)
            keys2 = shared_methods.sort_names(keys2)
        b *= (keys1 == keys2)
        if not b and raise_exception:
            raise Exception("Kgraphs differ on dict_kn's keys (order?)")
        #for k in keys1:
        #    b *= eq_node(g1.dict_kn[k],g2.dict_kn[k])
        b *= small_fcts.check_attr(g1,g2,
            ["dict_info","dict_constants"],raise_exception)
        return bool(b)
    def __hash__(self):
        return id(self)
    def clean_hash_in_sets(self):
        for attr in ["input_kdn_grad","output_kdn_grad",
            "loss_kcn","input_kdn_data","output_kdn_data"]:
            getattr(self,attr).clean_hash_in_sets()
        for kn in self.list_kcn + self.list_kdn:
            kn.clean_hash_in_sets()

# ==========================



# ==========================
# = Move from S to K graph =
# ==========================

# aux function to handle verbose and device
def aux_init_S_to_K(model,verbose,d):
    global device
    device = d if d else (
        small_fcts.get_device_and_check_all_same_device(model,dict(),True))
    if not (verbose is None): global_vars.ref_verbose[0] = verbose
    for n,p in model.named_parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

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
        our_global = def_inspection.generate_our_global(sg,model,device)
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
        sn_deps_copy = set(sn_deps)
        for req_sn in sn_deps_copy:
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
            target       = mt,
            all_targets       = sn.all_targets,
            tensor_targets    = sn.tensor_targets,
            inplace_targets   = sn.inplace_targets,
            container_targets = sn.container_targets,
            is_fwd       = True,
            is_rand      = sn.is_rand,
            main_code    = sn.main_code,
            inplace_code = sn.inplace_code,
            body_code    = sn.body_code,
            deps_real    = kcn_fwd_deps,
            deps_through_size_artefacts = kcn_deps_art_kcn,
            unique_id_generator = unique_id_generator)
        dict_KCN_fwd[mt] = kcn_fwd

        # -> KDN(data)
        kdn_data = K_D_node(
            kdn_type    = "data",
            target      = mt,
            all_targets       = sn.all_targets,
            tensor_targets    = sn.tensor_targets,
            inplace_targets   = sn.inplace_targets,
            container_targets = sn.container_targets,
            info        = info,
            deps        = set([kcn_fwd]),
            unique_id_generator = unique_id_generator)
        dict_KDN_data[mt] = kdn_data


        # *** build the bwd part ***
        if info.requires_grad:
            # -> get kdn_data and phantoms deps for bwd
            (explicit_deps,
            data_ptr_only_ph_deps,
            valid_view_ph_deps,
            exist_phs,
            original_phs,
            hasattr_base) = (
                def_inspection.get_useful_vars(sn,sg,our_global,device))
            all_deps_mt = set(explicit_deps).union(
                set(data_ptr_only_ph_deps.values()).union(
                set([t[1] for t in valid_view_ph_deps.values()])))
            bwd_deps_real_mt = (
                all_deps_mt.intersection(set(sn_deps_mt)))
            kcn_bwd_deps_real = set(
                dict_KDN_data[mt] for mt in bwd_deps_real_mt)
            kcn_bwd_deps_fake = (
                kcn_fwd_deps - kcn_bwd_deps_real)
            kdn_data.includes_base = hasattr_base
            if mt in all_deps_mt:
                kcn_bwd_deps_real.add(kdn_data)
                data_includes_phantoms = kdn_data.includes_phantoms = True
            else:
                kcn_bwd_deps_fake.add(kdn_data)
                data_includes_phantoms = False

            # -> KCN(bwd)
            kcn_bwd = K_C_node(
                target    = mt,
                is_fwd    = False,
                deps_real = kcn_bwd_deps_real,
                deps_fake = kcn_bwd_deps_fake,
                all_targets       = sn.all_targets,
                tensor_targets    = sn.tensor_targets,
                inplace_targets   = sn.inplace_targets,
                container_targets = sn.container_targets,
                unique_id_generator = unique_id_generator)
            dict_KCN_bwd[mt] = kcn_bwd

            # -> phantom deps
            for ph_name,(used_name,owner_name) in valid_view_ph_deps.items():
                if owner_name not in dict_KDN_data: raise Exception(
                    f"Warning : {ph_name}'s owner is {owner_name} "\
                    f"but we cannot find it's KDN_data node ??"\
                    f"its used name is {used_name}")
                used_kdn = dict_KDN_data[owner_name]
                used_kcn = dict_KCN_fwd[owner_name]
                used_kdn.alias_in_users_phantoms.append(
                    (mt,used_name,ph_name))
                used_kcn.alias_in_users_phantoms.append(
                    (mt,used_name,ph_name))
            for ph_name,owner_name in data_ptr_only_ph_deps.items():
                if owner_name not in dict_KDN_data: raise Exception(
                    f"Warning : {ph_name}'s owner is {owner_name}"\
                    f"but we cannot find it's KDN_data node ??")
                used_kdn = dict_KDN_data[owner_name]
                kcn_bwd.deps_impossible_to_restore.add((used_kdn,ph_name))
            kcn_fwd.phantom_names = (
                list(valid_view_ph_deps.keys())
                + list(data_ptr_only_ph_deps.keys())
                + original_phs)


            # -> KDN(phantoms)
            if exist_phs and not data_includes_phantoms:
                kdn_phantoms = K_D_node(
                    kdn_type    = "phantoms",
                    target      = mt,
                    info        = info,
                    deps        = set([kcn_fwd]),
                    all_targets       = sn.all_targets,
                    tensor_targets    = sn.tensor_targets,
                    inplace_targets   = sn.inplace_targets,
                    container_targets = sn.container_targets,
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
                all_targets       = sn.all_targets,
                tensor_targets    = sn.tensor_targets,
                inplace_targets   = sn.inplace_targets,
                container_targets = sn.container_targets,
                unique_id_generator = unique_id_generator)
            dict_KDN_grad[mt] = kdn_grad
            kcn_bwd.deps_real.add(kdn_grad)

            # -> KDN(grad).deps of fwd_deps
            for req_sn_mt in sn_deps_mt:
                if req_sn_mt in dict_KDN_grad: #i.e. requires_grad
                    dict_KDN_grad[req_sn_mt].deps.add(kcn_bwd)
        else:
            data_includes_phantoms = False


        # *** inspection ***
        if (device == torch.device("cpu")
        or "torch.split_with_sizes" in sn.get_code()):
            res = def_inspection.Inspection_result()
        else:
            ins = def_inspection.inspector(sn,sg,our_global,device)
            ins.measure_fwd()
            ins.measure_bwd()
            res = ins.ret

        # -> fwd ins
        kcn_fwd.overhead = res.overhead_fwd
        kcn_fwd.time     = res.time_run_fwd
        # kdn_data.mem     = info.memsize
        if data_includes_phantoms:
            kdn_data.mem = res.mem_run_fwd
        else:
            kdn_data.mem = res.mem_fgt_fwd

        # -> bwd ins
        if info.requires_grad:
            kcn_bwd.overhead = res.overhead_bwd
            kcn_bwd.time     = res.time_run_bwd
            kdn_grad.mem     = kdn_data.mem

            # -> phantoms ins
            if global_vars.ref_test_phantoms_detection[0]:
                exist_diff=res.mem_run_fwd - res.mem_fgt_fwd > 0
                if exist_diff or exist_phs:
                    print(f"For node {mt}: mem_diff : {exist_diff} "\
                          f"and detection {exist_phs}")

            if exist_phs and not data_includes_phantoms:
                kdn_phantoms.mem = (
                    res.mem_run_fwd - res.mem_fgt_fwd)

    # ============ 


    for sn in sg.nodes:
        handle_node(sn)

    # -> loss_node
    kg.output_kdn_data=output_kdn_data = dict_KDN_data[sg.hidden_output]
    kg.output_kdn_grad=output_kdn_grad = dict_KDN_grad[sg.hidden_output]
    kg.loss_kcn=loss_kcn = K_C_node(
        target    = "loss",
        is_fwd    = True,
        main_code = ("loss",ast_add_on.make_ast_constant("LOSS")),
        deps_real = set([output_kdn_data]),
        unique_id_generator = unique_id_generator)
    loss_kcn.time     = 0
    loss_kcn.overhead = 0
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
        inp_mt = "sources"
        # TO REMOVE
        # if len(sg.hidden_inputs) != 1: inp_mt = "sources"
        # else: inp_mt = sg.hidden_inputs[0]
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
    new_kcn.all_targets       = list(kcn.all_targets)
    new_kcn.tensor_targets    = list(kcn.tensor_targets)
    new_kcn.inplace_targets   = list(kcn.inplace_targets)
    new_kcn.container_targets = list(kcn.container_targets)
    new_kcn.name         = kcn.name
    new_kcn.is_fwd       = kcn.is_fwd
    new_kcn.is_rand      = kcn.is_rand
    new_kcn.main_code    = kcn.main_code
    new_kcn.inplace_code = [tuple(c) for c in kcn.inplace_code]
    new_kcn.body_code    = [tuple(c) for c in kcn.body_code]
    new_kcn.time         = kcn.time
    new_kcn.overhead     = kcn.overhead
    new_kcn.alias_in_users_phantoms = list(kcn.alias_in_users_phantoms)
    new_kcn.phantom_names = list(kcn.phantom_names)
    new_kcn.unique_id    = kcn.unique_id
    for attr in ["deps_real","deps_fake","deps_global",
        "users","users_global","deps_through_size_artefacts"]:
        setattr(new_kcn,attr,set()) # /!\
    return new_kcn

def copy_K_D_node(kdn : K_D_node):
    new_kdn = K_D_node()
    new_kdn.kdn_type    = kdn.kdn_type
    new_kdn.main_target = kdn.main_target
    new_kdn.all_targets       = list(kdn.all_targets)
    new_kdn.tensor_targets    = list(kdn.tensor_targets)
    new_kdn.inplace_targets   = list(kdn.inplace_targets)
    new_kdn.container_targets = list(kdn.container_targets)
    new_kdn.name        = kdn.name
    new_kdn.mem         = kdn.mem
    new_kdn.info        = kdn.info
    new_kdn.includes_base = kdn.includes_base
    new_kdn.includes_phantoms = kdn.includes_phantoms
    new_kdn.alias_in_users_phantoms = list(kdn.alias_in_users_phantoms)
    new_kdn.unique_id   = kdn.unique_id
    for attr in ["users_real","users_fake",
        "deps","users_global","deps_global"]:
        setattr(new_kdn,attr,set()) # /!\
    return new_kdn


def copy_K_graph(kg : K_graph):
    new_kg = K_graph(kg.sg)
    new_kg.dict_info = dict(kg.dict_info)
    new_kg.dict_constants = dict(kg.dict_constants)
    new_kg.init_code = kg.init_code
    new_kg.unique_id_generator = small_fcts.copy_generator(
            kg.unique_id_generator)

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
        for old_req_kdn,ph_name in old_kcn.deps_impossible_to_restore:
            new_kcn.deps_impossible_to_restore.add(
                (new_dict_kn[old_req_kdn.name],str(ph_name)))
    for new_kdn,old_kdn in zip(new_list_kdn,kg.list_kdn):
        for attr in ["users_real","users_fake","deps"]:
            old_edges = getattr(old_kdn,attr)
            for old_aux_kn in old_edges:
                getattr(new_kdn,attr).add(new_dict_kn[old_aux_kn.name])
        for old_user_kcn,ph_name in old_kdn.users_impossible_to_restore:
            new_kdn.users_impossible_to_restore.add(
                (new_dict_kn[old_user_kcn.name],str(ph_name)))


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
                f"Mem overhead : {irotor.MemSize(kcn.overhead)}"))
    def print_kdn(kdn):
        node(kdn.name,kdn.name,color=get_color(kdn),
            tooltip = f"Mem {irotor.MemSize(kdn.mem)}")

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


def print_K_graph(kg : K_graph,name=None,open=True,render_format="svg"):
    if name is None: name = "Fwd_and_bwd_graph"
    print(
        f"Forward + Backward graph with Computation and "\
        f"Data nodes: {len(kg.list_kcn)} + {len(kg.list_kdn)}")
    dot = graphviz.Digraph(name,
        comment="K_graph = Forward + Backward with Comp and Data nodes")
    aux_print_graph(dot,kg,0)
    small_fcts.graph_render(dot,open,"K",render_format)


def print_K_graph_list(list_kg,name=None,open=True,render_format="svg"):
    if name is None: name = "Sequentialized_Fwd_Bwd_graph"
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
        comment="K_graph list : sequentialized fwd+bwd with Comp and Data nodes")
    for i in range(len(list_kg)):
        aux_print_graph(dot,list_kg[i],i)
    small_fcts.graph_render(dot,open,"K",render_format)

# ==========================

