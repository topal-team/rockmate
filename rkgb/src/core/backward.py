# ==========================
# ====== K structure =======
# ==========================

from .utils import *
from .Stools import S_node,S_graph

# ************
# * K_C_node *
# ************

class K_C_node(base.Node):
    def __init__(self,
            main_target="/!\\ No target /!\\",
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
            deps_through_artifacts=None,
            other_obj=None):
        # ** informative **
        super().__init__("KC",other_obj,main_target=main_target)
        mt = main_target
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

        # ** deps/used_by **
        self.deps_real    = deps_real if deps_real else set() # KDN set
        self.deps_fake    = deps_fake if deps_fake else set() # KDN set
        self.deps_global  = set() # KDN set
        self.users_global = set() # KDN set
        self.users        = set() # KDN set
        self.deps_impossible_to_restore = set() # (KDN * str) set
        da = deps_through_artifacts
        self.deps_through_artifacts = da if da else set() # KCN set
        # -> just for the toposort, we don't need the reciprocal users_..

        # ** inspection **
        self.time         = None
        self.overhead     = None
        self.phantom_names = []
        self.alias_in_users_phantoms = []

    def get_all_standard_deps(self):
        return set().union(
            *[bdn.deps for bdn in self.deps_real],
            self.deps_through_artifacts)
    def get_all_standard_users(self):
        return set().union(
            *[bdn.users_real for bdn in self.users])

    @property
    def deps_only_global(self):
        return self.deps_global - self.deps_real.union(self.deps_fake)
    @property
    def users_only_global(self):
        return self.users_global - self.users


# ************
# * K_D_node *
# ************

class K_D_node(base.Node):
    def __init__(self,
            kdn_type = "/!\\ No kdn_type/!\\",
            main_target = "/!\\ No target /!\\",
            all_targets       = None,
            tensor_targets    = None,
            inplace_targets   = None,
            container_targets = None,
            info      = None,
            deps      = None,
            other_obj = None):
        # ** informative **
        super().__init__("KD",other_obj,main_target=main_target)
        self.kdn_type = kdn_type # data, grad or phantoms
        mt = main_target
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
        # ** deps/used_by **
        self.users_real   = set() # KCN set
        self.users_fake   = set() # KCN set
        self.users_global = set() # KCN set
        self.deps_global  = set() # KCN set
        self.deps         = deps if deps else set() # KCN set
        self.users_impossible_to_restore = set() # (KCN * str) set
        self.alias_in_users_phantoms = []
    
    @property
    def deps_only_global(self):
        return self.deps_global - self.deps
    @property
    def users_only_global(self):
        return self.users_global - self.users_real.union(self.users_fake)

    def get_all_standard_deps(self):
        return set().union(
            *[bcn.deps_real for bcn in self.deps])
    def get_all_standard_users(self):
        return set().union(
            *[bcn.users for bcn in self.users_real])



# ***********
# * K_graph *
# ***********

class K_graph(base.Graph):
    has_fake_input_kdn_grad = False
    def __init__(self,sg : S_graph):
        super().__init__("K")
        if not (sg is None): self.inherit_base_attributes(sg)
        self.dict_rand = dict() # random operations have been inserted at the end of simplification
        self.sg = sg

        self.dict_kn  = dict() # KDN/KCN.name -> KDN/KCN
        self.list_kcn = []     # KCN list : Toposorted
        self.list_kdn = []     # KDN list : Arbitrary order

        self.input_kdn_data        = None # e.g. KDN _13.data
        self.list_outputs_kdn_data = None # e.g. KDN _116.data
        self.loss_kcn              = None
        self.list_outputs_kdn_grad = None # e.g. KDN _116.grad
        self.input_kdn_grad        = None # e.g. KDN _13.grad
        # /!\ A K_graph always has a single input_node
        # /!\ BUT can have several outputs
        # -> for a standalone K_graph, input_kdn_data/grad are fresh nodes
        # -> otherwise they are shared with the previous k_graph
        # -> output_kdn_data/grad are shared with the next one

        # ** useful dicts **
        self.dict_KCN_fwd  = dict() # mt -> KCN(fwd)
        self.dict_KCN_bwd  = dict() # mt -> KCN(bwd)
        self.dict_KDN_data = dict() # mt -> KDN(data)
        self.dict_KDN_grad = dict() # ...
        self.dict_KDN_phantoms = dict()

        # ** init and final codes **
        self.init_code = sg.init_node.get_code_ast()
        self.dict_output_viewing_code = sg.dict_output_viewing_code 
        if not (sg.special_output_node is None):
            self.outputs_wrapping_code = sg.special_output_node.get_code_ast()
        else:
            self.outputs_wrapping_code = ast.parse("")

    @property # FOR ORIGINAL ROCKMATE COMPATIBILITY
    def output_kdn_data(self):
        if len(self.list_outputs_kdn_data) != 1:
            warnings.warn(
                "Several output nodes, you shouldn't use "\
                "`output_kdn_data` but `list_outputs_kdn_data")
        return self.list_outputs_kdn_data[0]
    @property # FOR ORIGINAL ROCKMATE COMPATIBILITY
    def output_kdn_grad(self):
        if len(self.list_outputs_kdn_grad) != 1:
            warnings.warn(
                "Several output nodes, you shouldn't use "\
                "`output_kdn_grad` but `list_outputs_kdn_grad")
        return self.list_outputs_kdn_grad[0]
    

    # FOR ORIGINAL ROCKMATE COMPATIBILITY 
    def fake_input_kdn_grad(self):
        if self.input_kdn_grad is not None:
            self.has_fake_input_kdn_grad = False
        else:
            self.has_fake_input_kdn_grad = True
            self.input_kdn_grad=input_kdn_grad = K_D_node(
                kdn_type = "grad", main_target = "sources",
                all_targets = self.sg.inputs,
                other_obj = self)
            firsts_mt = [sn.mt for sn in self.sg.init_node.users]
            self.dict_KDN_grad[input_kdn_grad.mt] = input_kdn_grad
            self.dict_kn[input_kdn_grad.name] = input_kdn_grad
            input_kdn_grad_deps = set(
                self.dict_KCN_bwd[mt] for mt in firsts_mt
                if mt in self.dict_KCN_bwd)
            input_kdn_grad.deps_global.update(input_kdn_grad_deps)
            for user_kcn in input_kdn_grad_deps:
                user_kcn.users_global.add(input_kdn_grad)
        assert self.input_kdn_grad is not None

    def release_fake_input_kdn_grad(self):
        if self.has_fake_input_kdn_grad:
            self.has_fake_input_kdn_grad = False
            input_kdn_grad = self.input_kdn_grad
            self.input_kdn_grad = None
            del self.dict_KDN_grad[input_kdn_grad.mt]
            del self.dict_kn[input_kdn_grad.name]
            for first_kcn in input_kdn_grad.deps_global:
                first_kcn.users_global.remove(input_kdn_grad)



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
        root_kdn = K_D_node(deps = leaves_kcn,other_obj=self)
        root_kcn = K_C_node(deps_real=set([root_kdn]),other_obj=self)
        self.list_kcn = l = RK_sort_based_on_deps(root_kcn)
        l.remove(root_kcn)

    def make_kcns_number(self):
        for i,kcn in enumerate(self.list_kcn):
            setattr(kcn,"_number",i)

    """ # NOT MAINTAINED 
    def __eq__(self,g2,force_order=False,raise_exception=False):
        g1 = self
        eq_node= lambda n1,n2 : n1.__eq__(n2,force_order,raise_exception)
        b = (ast_add_on.ast_to_str(self.init_code) 
            == ast_add_on.ast_to_str(g2.init_code))
        if not b and raise_exception:
            raise Exception("K_graphs diff init_code")
        for attr in ["input_kdn_grad","list_outputs_kdn_grad",
            "loss_kcn","input_kdn_data","list_outputs_kdn_data"]:
            n1 = getattr(g1,attr)
            n2 = getattr(g2,attr)
            if (n1 is None) != (n2 is None):
                b = False
            else:
                b *= eq_node(getattr(g1,attr),getattr(g2,attr))
        for attr in ["list_kcn","list_kdn"]:
            for kn1,kn2 in zip(getattr(g1,attr),getattr(g2,attr)):
                b *= eq_node(kn1,kn2)
        keys1 = list(g1.dict_kn)
        keys2 = list(g2.dict_kn)
        if force_order:
            keys1 = base.Node.sort_names(keys1)
            keys2 = base.Node.sort_names(keys2)
        b *= (keys1 == keys2)
        if not b and raise_exception:
            raise Exception("K_graphs differ on dict_kn's keys (order?)")
        #for k in keys1:
        #    b *= eq_node(g1.dict_kn[k],g2.dict_kn[k])
        b *= small_fcts.check_attr(g1,g2,
            ["dict_info","dict_constants"],raise_exception)
        return bool(b)
    """
# ==========================



# ==========================
# = Move from S to K graph =
# ==========================

# aux function to handle verbose and device
def aux_init_S_to_K(model,verbose,d):
    global device
    device = d if d else (
        small_fcts.get_device_and_check_all_same_device(model,dict(),True))
    if not (verbose is None): constants.ref_verbose[0] = verbose
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

# the function that does it all
def aux_build_S_to_K(sg : S_graph,
        model,
        prev_kg : K_graph=None,
        is_really_first_graph=False,
        do_inspection=True):
    kg = K_graph(sg)
    dict_KCN_fwd = kg.dict_KCN_fwd
    dict_KCN_bwd = kg.dict_KCN_bwd
    dict_KDN_data = kg.dict_KDN_data
    dict_KDN_grad = kg.dict_KDN_grad
    dict_KDN_phantoms = kg.dict_KDN_phantoms

    # ============  
    def handle_node(sn : S_node):
        mt = sn.main_target
        print_debug(f"start to handle {mt}'s S_node in S_to_K")
        our_global = def_inspection.generate_our_global(sg,model,device)
        info = sg.dict_info[mt]

        # For artifact nodes :
        #   -> if KCN2 only need KCN1.size, it means in sg there is
        #   -> an artifact node for KCN1.size to avoid useless dep
        #   -> between KCN2 and KCN1. We decided to do NOT have KDN(size)
        #   -> in fact we just need KCN1 to be ordered before KCN2 in
        #   -> the toposort. To do so we create a tmp special dep:
        #   -> "deps_through_artifacts" when we find artifact in sn.deps
        if sn.is_artifact: return ()

        # *** build the fwd part ***
        sn_deps = set(sn.deps.keys())
        if sg.init_node in sn_deps:
            raise Exception("sg.init_node has been unhooked ?!?")

        # -> handle artifact deps :
        kcn_deps_art_kcn = set()
        sn_deps_copy = set(sn_deps)
        for req_sn in sn_deps_copy:
            if req_sn.is_artifact:
                sn_deps.discard(req_sn)
                req_real_sn = list(req_sn.deps.keys())[0] # art's parent
                kcn_deps_art_kcn.add(dict_KCN_fwd[req_real_sn.main_target])

        # -> get kdn_data deps for fwd
        sn_deps_mt = [req_sn.main_target for req_sn in sn_deps]
        kcn_fwd_deps = set(
            dict_KDN_data[mt] for mt in sn_deps_mt)

        # -> KCN(fwd)
        kcn_fwd = K_C_node(
            main_target       = mt,
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
            deps_through_artifacts = kcn_deps_art_kcn,
            other_obj = kg)
        dict_KCN_fwd[mt] = kcn_fwd

        # -> KDN(data)
        kdn_data = K_D_node(
            kdn_type    = "data",
            main_target       = mt,
            all_targets       = sn.all_targets,
            tensor_targets    = sn.tensor_targets,
            inplace_targets   = sn.inplace_targets,
            container_targets = sn.container_targets,
            info        = info,
            deps        = set([kcn_fwd]),
            other_obj = kg)
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
                main_target       = mt,
                all_targets       = sn.all_targets,
                tensor_targets    = sn.tensor_targets,
                inplace_targets   = sn.inplace_targets,
                container_targets = sn.container_targets,
                is_fwd    = False,
                deps_real = kcn_bwd_deps_real,
                deps_fake = kcn_bwd_deps_fake,
                other_obj = kg)
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
                    main_target       = mt,
                    all_targets       = sn.all_targets,
                    tensor_targets    = sn.tensor_targets,
                    inplace_targets   = sn.inplace_targets,
                    container_targets = sn.container_targets,
                    info        = info,
                    deps        = set([kcn_fwd]),
                    other_obj = kg)
                dict_KDN_phantoms[mt] = kdn_phantoms
                kcn_bwd.deps_real.add(kdn_phantoms)
                kcn_fwd.has_phantoms = True
            else: kcn_fwd.has_phantoms = False

            # -> KDN(grad)
            kdn_grad = K_D_node(
                kdn_type    = "grad",
                info        = info,
                main_target       = mt,
                all_targets       = sn.all_targets,
                tensor_targets    = sn.tensor_targets,
                inplace_targets   = sn.inplace_targets,
                container_targets = sn.container_targets,
                other_obj = kg)
            dict_KDN_grad[mt] = kdn_grad
            kcn_bwd.deps_real.add(kdn_grad)

            # -> KDN(grad).deps of fwd_deps
            for req_sn_mt in sn_deps_mt:
                if req_sn_mt in dict_KDN_grad: #i.e. requires_grad
                    dict_KDN_grad[req_sn_mt].deps.add(kcn_bwd)
        else:
            data_includes_phantoms = False


        # *** inspection ***
        if (not do_inspection
        or device == torch.device("cpu")):
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
            if constants.ref_test_phantoms_detection[0]:
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
    kg.list_outputs_kdn_data = list_outputs_kdn_data \
        = [dict_KDN_data[out.mt] for out in sg.output_nodes]
    kg.list_outputs_kdn_grad = list_outputs_kdn_grad \
        = [dict_KDN_grad[out.mt] for out in sg.output_nodes]
    kg.loss_kcn=loss_kcn = K_C_node(
        main_target = "loss",
        is_fwd    = True,
        main_code = ("loss",ast_add_on.make_ast_constant("LOSS")),
        deps_real = set(list_outputs_kdn_data),
        other_obj = kg)
    loss_kcn.time     = 0
    loss_kcn.overhead = 0
    dict_KCN_fwd[loss_kcn.main_target] = loss_kcn
    for kdn in list_outputs_kdn_grad:
        kdn.deps.add(loss_kcn)

    # -> list of nodes
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

    # ** input nodes **
    # -> get input_kdn_data/grad from prev_kg
    sources_mt = "sources"
    if prev_kg:
        is_sources = False
        nb_input_kdn = len(prev_kg.list_outputs_kdn_data)
        if nb_input_kdn != 1:
            raise Exception(
                f"Except the last one, K_graph always has "\
                f"exactly one output. Error here, prev_kg "\
                f"has {nb_input_kdn} outputs"
            )
        kg.input_kdn_data=input_kdn_data = prev_kg.list_outputs_kdn_data[0]
        kg.input_kdn_grad=input_kdn_grad = prev_kg.list_outputs_kdn_grad[0]
    # -> or create fresh vars in case kg is a standalone graph
    else:
        is_sources = True
        kg.input_kdn_data=input_kdn_data = K_D_node(
            kdn_type = "data", main_target = sources_mt,
            all_targets = sg.inputs,
            other_obj = kg)
        if sg.sources_req_grad or not is_really_first_graph:
            kg.input_kdn_grad=input_kdn_grad = K_D_node(
                kdn_type = "grad", main_target = sources_mt,
                all_targets = sg.inputs,
                other_obj = kg)
        else:
            kg.input_kdn_grad = None

    # ** make deps/users_global with inputs **
    # -> users of inp_data
    kg.dict_KDN_data[input_kdn_data.mt] = input_kdn_data
    kg.dict_kn[input_kdn_data.name] = input_kdn_data
    firsts_mt = [sn.mt for sn in sg.init_node.users]
    input_kdn_data_users = set(dict_KCN_fwd[mt] for mt in firsts_mt)
    input_kdn_data.users_global.update(input_kdn_data_users)
    for user_kcn in input_kdn_data_users:
        user_kcn.deps_global.add(input_kdn_data)

    # -> deps of inp_grad
    if not is_sources or sg.sources_req_grad or not is_really_first_graph:
        kg.dict_KDN_grad[input_kdn_grad.mt] = input_kdn_grad
        kg.dict_kn[input_kdn_grad.name] = input_kdn_grad
        input_kdn_grad_deps = set(
            dict_KCN_bwd[mt] for mt in firsts_mt
            if mt in dict_KCN_bwd)
        input_kdn_grad.deps_global.update(input_kdn_grad_deps)
        for user_kcn in input_kdn_grad_deps:
            user_kcn.users_global.add(input_kdn_grad)

    # -> TOPOSORT list_kcn
    kg.sort_list_kcn()

    return kg


def S_to_K(sg : S_graph,model,verbose=None,device=None):
    aux_init_S_to_K(model,verbose,device)
    return aux_build_S_to_K(sg,model,prev_kg = None,is_really_first_graph=True)

class K_graph_list(list):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

def S_list_to_K_list(list_sg,model,verbose=None,device=None):
    aux_init_S_to_K(model,verbose,device)
    list_kg = []
    prev_kg = None
    for sg in list_sg:
        prev_kg = kg = aux_build_S_to_K(sg,model,prev_kg,
            is_really_first_graph=(prev_kg is None))
        list_kg.append(kg)
    return K_graph_list(list_kg)


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
        "users","users_global","deps_through_artifacts"]:
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
    new_kg.inherit_base_attributes(kg)
    new_kg.has_fake_input_kdn_grad = kg.has_fake_input_kdn_grad
    new_kg.init_code = kg.init_code
    new_kg.outputs_wrapping_code = kg.outputs_wrapping_code
    new_kg.dict_output_viewing_code = kg.dict_output_viewing_code  

    # == NODES ==
    new_dict_kn = new_kg.dict_kn
    new_kg.list_kcn = new_list_kcn = (
        [copy_K_C_node(kcn) for kcn in kg.list_kcn])
    new_kg.list_kdn = new_list_kdn = (
        [copy_K_D_node(kdn) for kdn in kg.list_kdn])
    for kn in new_list_kcn+new_list_kdn:
        new_dict_kn[kn.name]=kn

    # Some inputs may be missing (sources...)
    for kdn in (
            list(kg.dict_KDN_data.values())
        +   list(kg.dict_KDN_grad.values())):
        if kdn.name not in new_dict_kn:
            new_dict_kn[kdn.name] = copy_K_D_node(kdn)

    new_kg.dict_KCN_fwd = dict(
        (kn.mt,new_dict_kn[kn.name]) for kn in kg.dict_KCN_fwd.values())
    new_kg.dict_KCN_bwd = dict(
        (kn.mt,new_dict_kn[kn.name]) for kn in kg.dict_KCN_bwd.values())
    new_kg.dict_KDN_data = dict(
        (kn.mt,new_dict_kn[kn.name]) for kn in kg.dict_KDN_data.values())
    new_kg.dict_KDN_grad = dict(
        (kn.mt,new_dict_kn[kn.name]) for kn in kg.dict_KDN_grad.values())
    new_kg.dict_KDN_phantoms = dict(
        (kn.mt,new_dict_kn[kn.name]) for kn in kg.dict_KDN_phantoms.values())

    # -- edges --
    for new_kcn,old_kcn in zip(new_list_kcn,kg.list_kcn):
        for attr in ["deps_real","deps_fake","users",
            "deps_through_artifacts"]:
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
    new_kg.dict_KDN_data[new_inp_data.mt] = new_inp_data
    new_kg.dict_kn[new_inp_data.name] = new_inp_data
    for old_fst_kcn in old_inp_data.users_only_global:
        new_fst_kcn = new_dict_kn[old_fst_kcn.name]
        new_fst_kcn.deps_global.add(new_inp_data)
        new_inp_data.users_global.add(new_fst_kcn)
    if old_inp_grad is None:
        new_kg.input_kdn_grad = None
    else:
        new_kg.input_kdn_grad=new_inp_grad = copy_K_D_node(old_inp_grad)
        new_kg.dict_KDN_grad[new_inp_grad.mt] = new_inp_grad
        new_kg.dict_kn[new_inp_grad.name] = new_inp_grad
        for old_lst_kcn in old_inp_grad.deps_only_global:
            new_lst_kcn = new_dict_kn[old_lst_kcn.name]
            new_lst_kcn.users_global.add(new_inp_grad)
            new_inp_grad.deps_global.add(new_lst_kcn)

    new_kg.list_outputs_kdn_data \
        = [new_dict_kn[out.name] for out in kg.list_outputs_kdn_data]
    new_kg.list_outputs_kdn_grad \
        = [new_dict_kn[out.name] for out in kg.list_outputs_kdn_grad]
    new_kg.loss_kcn = new_dict_kn[kg.loss_kcn.name]
    return new_kg

# ==========================

# ====================
"""
# /!\ NOT WORKING /!\ 
# TODO :
# -> Need better edges for input_kdn_data/grad
# -> 1) include kcn bwd nodes
# -> 2) deps_real or fake
# -> Recognize inputs in def_inspection when opening grad_fn
def K_list_to_K(kl : K_graph_list,sg : S_graph) -> K_graph:
    kl = [copy_K_graph(block_kg) for block_kg in kl]
    nb_block = len(kl)
    whole_kg = K_graph(sg)
    whole_list_kdn \
        = whole_kg.list_kdn \
        = sum([kg.list_kdn for kg in kl],[])
    whole_list_kcn \
        = whole_kg.list_kcn \
        = sum([kg.list_kcn for kg in kl],[])
    whole_kg.loss_kcn = kl[-1].loss_kcn
    whole_kg.input_kdn_data = kl[0].input_kdn_data
    whole_kg.list_outputs_kdn_data = kl[-1].list_outputs_kdn_data
    whole_kg.list_outputs_kdn_grad = kl[-1].list_outputs_kdn_grad
    whole_kg.input_kdn_grad = kl[0].input_kdn_grad

    # == Merge output of one block with input of the next one ==
    for index in range(nb_block-1):
        block_kg : K_graph = kl[index]
        next_kg  : K_graph = kl[index+1]
        # only one output since not last block
        output_kdn_data : K_D_node = block_kg.output_kdn_data
        output_kdn_grad : K_D_node = block_kg.output_kdn_grad

        # -> Unplug the loss_kcn
        loss_kcn = block_kg.loss_kcn
        whole_list_kcn.remove(loss_kcn)
        output_kdn_data.users_real.discard(loss_kcn)
        output_kdn_data.users_fake.discard(loss_kcn)
        output_kdn_data.users_global.discard(loss_kcn)
        output_kdn_grad.deps.discard(loss_kcn)
        output_kdn_grad.deps_global.discard(loss_kcn)

        # Unplug next_input_kdns
        next_input_data : K_D_node = next_kg.input_kdn_data
        next_input_grad : K_D_node = next_kg.input_kdn_grad

        # merge output with next_input
        for next_user_of_inp_data in next_input_data.users_global:
            next_user_of_inp_data.deps_global.add(output_kdn_data)
            next_user_of_inp_data.deps_real.add(output_kdn_data)
            next_user_of_inp_data.deps_global.remove(next_input_data)
            output_kdn_data.users_global.add(next_user_of_inp_data)
            output_kdn_data.users_real.add(next_user_of_inp_data)
        for next_req_of_inp_grad in next_input_grad.deps_global:
            next_req_of_inp_grad.users_global.add(output_kdn_grad)
            next_req_of_inp_grad.users.add(output_kdn_grad)
            next_req_of_inp_grad.users_global.remove(next_input_grad)
            output_kdn_grad.deps_global.add(next_req_of_inp_grad)
            output_kdn_grad.deps.add(next_req_of_inp_grad)

    # make dicts of K_nodes
    inputs_kdn = [whole_kg.input_kdn_data]
    if whole_kg.input_kdn_grad is not None:
        inputs_kdn.append(whole_kg.input_kdn_grad)

    for kdn in whole_list_kdn + inputs_kdn:
        whole_kg.dict_kn[kdn.name] = kdn
        if kdn.kdn_type == "data":
            whole_kg.dict_KDN_data[kdn.mt] = kdn
        elif kdn.kdn_type == "grad":
            whole_kg.dict_KDN_grad[kdn.mt] = kdn
        else:
            whole_kg.dict_KDN_phantoms[kdn.mt] = kdn
    for kcn in whole_list_kcn:
        whole_kg.dict_kn[kcn.name] = kcn
        if kcn.is_fwd:
            whole_kg.dict_KCN_fwd[kcn.mt] = kcn
        else:
            whole_kg.dict_KCN_bwd[kcn.mt] = kcn

    # toposort the whole graph
    whole_kg.sort_list_kcn()

    return whole_kg
"""

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

def aux_print_K_graph_message(kg : K_graph):
    return (
        f"K_graph - Forward + Backward graph, "\
        f"{len(kg.list_kcn)} K_C_nodes; {len(kg.list_kdn)} K_D_nodes"
    )

def aux_print_K_graph_list_message(lkg : K_graph_list):
    list_nb_kcn = [len(kg.list_kcn) for kg in lkg]
    list_nb_kdn = [len(kg.list_kdn) for kg in lkg]
    tot_nb_kcn = sum(list_nb_kcn)
    tot_nb_kdn = sum(list_nb_kdn)
    str_list_nb_kcn = "+".join(str(i) for i in list_nb_kcn)
    str_list_nb_kdn = "+".join(str(i) for i in list_nb_kdn)
    return (
        f"K_graph_list - Sequentialized Forward + Backward graphs, "\
        f"{len(lkg)} blocks, with :\n"\
        f"     -> {str_list_nb_kcn} = {tot_nb_kcn} Comp nodes\n"\
        f"     -> {str_list_nb_kdn} = {tot_nb_kdn} Data nodes\n"\
        f"     => total of {tot_nb_kcn + tot_nb_kdn} nodes"
    )

def aux_print_K_graph_name(kg : K_graph,name=None):
    if name is not None: return name
    else: return "Forward_and_Backward_K_graph"

def aux_print_K_graph_list_name(lkg : K_graph_list,name=None):
    if name is not None: return name
    else: return "Sequentialized_Forward_and_Backward_K_graph_list"

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
    kwargs = {"color":color_special , "style":"dashed"}
    inp_data = kg.input_kdn_data
    inp_users = list(inp_data.users_only_global)
    if len(inp_users)!=0:
        node(inp_data.name,inp_data.name,**kwargs)
        for user_inp_data in inp_users:
            edge(inp_data.name,user_inp_data.name,**kwargs)
    inp_grad = kg.input_kdn_grad
    if inp_grad is not None:
        node(inp_grad.name,inp_grad.name,**kwargs)
        for req_inp_grad in inp_grad.deps_only_global:
            edge(req_inp_grad.name,inp_grad.name,**kwargs)


def print_K_graph(kg : K_graph,name=None,open=True,render_format="svg",dot=None,uniq_num=0):
    if dot is None:
        render = True
        name = aux_print_K_graph_name(kg,name)
        dot = graphviz.Digraph(name,comment=name)
    else:
        render = False
    aux_print_graph(dot,kg,uniq_num)
    if render:
        small_fcts.graph_render(dot,open,"K",render_format)


def print_K_graph_list(lkg : K_graph_list,name=None,open=True,render_format="svg"):
    name = aux_print_K_graph_list_name(lkg,name)
    dot = graphviz.Digraph(name,comment=name)
    for i in range(len(lkg)):
        aux_print_graph(dot,lkg[i],i)
    small_fcts.graph_render(dot,open,"K",render_format)

# ==========================

