# ==========================
# ====== K structure =======
# ==========================

import warnings
import ast
import torch
from src.lowlevel import ast_add_on
from src.lowlevel import constants
from src.lowlevel import measure
from src.lowlevel import inspection
from src.core import base
from src.core.simplified import SimplifiedNode,SimplifiedGraph


class ForwardBackwardComputationNode(base.Node):
    def __init__(self,
            main_target=base.Node.no_target_string,
            simplified_node : SimplifiedNode = None,
            is_fwd  = True,
            is_rand = False,
            main_code    = None,
            inplace_code = None,
            body_code    = None,
            deps_real = None,
            deps_fake = None,
            deps_through_artifacts=None,
            backward_graph=None):
        # ** informative **
        super().__init__(main_target,
            parent_structure_with_id_generator=backward_graph)
        # - inherits from simplified_node:
        if simplified_node:
            for attr in [
                    "all_targets","tensor_targets",
                    "inplace_targets","container_targets"]:
                setattr(self,attr,getattr(simplified_node,attr))
        # - basic attributes:
        self.name = f"FWD[{main_target}]" if is_fwd else f"BWD[{main_target}]"
        self.is_fwd = is_fwd
        self.is_rand = is_rand
        self.main_code = main_code # tuple (target * AST)
        self.inplace_code = inplace_code if inplace_code else []
        self.body_code = body_code if body_code else [] # (str*AST) list
        # - deps and users:
        self.deps_real    = deps_real if deps_real else set() # AllocationNode set
        self.deps_fake    = deps_fake if deps_fake else set() # AllocationNode set
        self.users        = set()
        self.deps_impossible_to_restore = set() # TODO : REMOVE deps_impossible_to_restore
        if deps_through_artifacts: # ComputationNode set
            self.deps_through_artifacts = deps_through_artifacts
        else:
            self.deps_through_artifacts = set()
            # -> just for the toposort, we don't need the reciprocal users_..
        # - inspection:
        self.time = None
        self.overhead = None
        self.phantom_names = []
        self.alias_in_users_phantoms = []

    def get_all_standard_deps(self):
        return set().union(
            *[req_allocation_node.deps 
              for req_allocation_node in self.deps_real],
            self.deps_through_artifacts)
    def get_all_standard_users(self):
        return set().union(
            *[req_allocation_node.users_real 
              for req_allocation_node in self.users])

# ************
# * ForwardBackwardAllocationNode *
# ************

class ForwardBackwardAllocationNode(base.Node):
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
        super().__init__(other_obj,main_target=main_target)
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
    
    def get_all_standard_deps(self):
        return set().union(
            *[bcn.deps_real for bcn in self.deps])
    def get_all_standard_users(self):
        return set().union(
            *[bcn.users for bcn in self.users_real])



# ***********
# * ForwardBackwardGraph *
# ***********

class ForwardBackwardGraph(base.Graph):
    has_fake_input_kdn_grad = False
    def __init__(self,sg : SimplifiedGraph):
        super().__init__()
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
        # /!\ A ForwardBackwardGraph always has a single input_node
        # /!\ BUT can have several outputs
        # -> for a standalone ForwardBackwardGraph, input_kdn_data/grad are fresh nodes
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
        if not (sg.wrapper_output_node is None):
            self.outputs_wrapping_code = sg.wrapper_output_node.get_code_ast()
        else:
            self.outputs_wrapping_code = ast.parse("")

    def __iter__(self):
        return iter(self.computation_nodes)

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
            self.input_kdn_grad=input_kdn_grad = ForwardBackwardAllocationNode(
                kdn_type = "grad", main_target = constants.init_target_string,
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
        root_kdn = ForwardBackwardAllocationNode(deps = leaves_kcn,other_obj=self)
        root_kcn = ForwardBackwardComputationNode(deps_real=set([root_kdn]),other_obj=self)
        self.list_kcn = l = self.get_sorted_nodes_by_following_deps_relation()
        l.remove(root_kcn)

    def make_kcns_number(self):
        for i,kcn in enumerate(self.list_kcn):
            setattr(kcn,"_number",i)

# ==========================



# ==========================
# = Move from S to K graph =
# ==========================

# the function that does it all
def aux_build_S_to_K(sg : SimplifiedGraph,
        model,
        device,
        do_inspection=True):
    kg = ForwardBackwardGraph(sg)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    dict_KCN_fwd = kg.dict_KCN_fwd
    dict_KCN_bwd = kg.dict_KCN_bwd
    dict_KDN_data = kg.dict_KDN_data
    dict_KDN_grad = kg.dict_KDN_grad
    dict_KDN_phantoms = kg.dict_KDN_phantoms

    # ============  
    def handle_node(sn : SimplifiedNode):
        mt = sn.main_target
        our_global = inspection.generate_our_global(sg,model,device)
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
        kcn_fwd = ForwardBackwardComputationNode(
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
        kdn_data = ForwardBackwardAllocationNode(
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
                inspection.get_useful_vars(sn,sg,our_global,device))
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
            kcn_bwd = ForwardBackwardComputationNode(
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
                kdn_phantoms = ForwardBackwardAllocationNode(
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
            kdn_grad = ForwardBackwardAllocationNode(
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
            res = inspection.Inspection_result()
        else:
            ins = inspection.inspector(sn,sg,our_global,device)
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
    kg.loss_kcn=loss_kcn = ForwardBackwardComputationNode(
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
    if prev_kg:
        is_sources = False
        nb_input_kdn = len(prev_kg.list_outputs_kdn_data)
        if nb_input_kdn != 1:
            raise Exception(
                f"Except the last one, ForwardBackwardGraph always has "\
                f"exactly one output. Error here, prev_kg "\
                f"has {nb_input_kdn} outputs"
            )
        kg.input_kdn_data=input_kdn_data = prev_kg.list_outputs_kdn_data[0]
        kg.input_kdn_grad=input_kdn_grad = prev_kg.list_outputs_kdn_grad[0]
    # -> or create fresh vars in case kg is a standalone graph
    else:
        is_sources = True
        kg.input_kdn_data=input_kdn_data = ForwardBackwardAllocationNode(
            kdn_type = "data", main_target = constants.init_target_string,
            all_targets = sg.inputs,
            other_obj = kg)
        if sg.sources_req_grad or not is_really_first_graph:
            kg.input_kdn_grad=input_kdn_grad = ForwardBackwardAllocationNode(
                kdn_type = "grad", main_target = constants.init_target_string,
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




# ==========================

# ==========================
# === printing functions ===
# ==========================

color_kcn_fwd  = "blue"
color_kcn_bwd  = "blueviolet"
color_special  = "green"
color_kdn      = "olive"

def get_color(kn):
    if isinstance(kn,ForwardBackwardAllocationNode): return color_kdn
    if kn.is_fwd: return color_kcn_fwd
    return color_kcn_bwd

def aux_print_ForwardBackwardGraph_message(kg : ForwardBackwardGraph):
    return (
        f"ForwardBackwardGraph - Forward + Backward graph, "\
        f"{len(kg.list_kcn)} ForwardBackwardComputationNodes; {len(kg.list_kdn)} ForwardBackwardAllocationNodes"
    )

def aux_print_ForwardBackwardGraph_list_message(lkg : ForwardBackwardGraph_list):
    list_nb_kcn = [len(kg.list_kcn) for kg in lkg]
    list_nb_kdn = [len(kg.list_kdn) for kg in lkg]
    tot_nb_kcn = sum(list_nb_kcn)
    tot_nb_kdn = sum(list_nb_kdn)
    str_list_nb_kcn = "+".join(str(i) for i in list_nb_kcn)
    str_list_nb_kdn = "+".join(str(i) for i in list_nb_kdn)
    return (
        f"ForwardBackwardGraph_list - Sequentialized Forward + Backward graphs, "\
        f"{len(lkg)} blocks, with :\n"\
        f"     -> {str_list_nb_kcn} = {tot_nb_kcn} Comp nodes\n"\
        f"     -> {str_list_nb_kdn} = {tot_nb_kdn} Data nodes\n"\
        f"     => total of {tot_nb_kcn + tot_nb_kdn} nodes"
    )

def aux_print_ForwardBackwardGraph_name(kg : ForwardBackwardGraph,name=None):
    if name is not None: return name
    else: return "Forward_and_Backward_ForwardBackwardGraph"

def aux_print_ForwardBackwardGraph_list_name(lkg : ForwardBackwardGraph_list,name=None):
    if name is not None: return name
    else: return "Sequentialized_Forward_and_Backward_ForwardBackwardGraph_list"

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
                f"Mem overhead : {measure.MemSize(kcn.overhead)}"))
    def print_kdn(kdn):
        node(kdn.name,kdn.name,color=get_color(kdn),
            tooltip = f"Mem {measure.MemSize(kdn.mem)}")

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

    # *** io - global relations *** # TO REMOVE = il n'y a plus de list de backward
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


def print_ForwardBackwardGraph(kg : ForwardBackwardGraph,name=None,open=True,render_format="svg",dot=None,uniq_num=0):
    if dot is None:
        render = True
        name = aux_print_ForwardBackwardGraph_name(kg,name)
        dot = graphviz.Digraph(name,comment=name)
    else:
        render = False
    aux_print_graph(dot,kg,uniq_num)
    if render:
        small_fcts.graph_render(dot,open,"K",render_format)


def print_ForwardBackwardGraph_list(lkg : ForwardBackwardGraph_list,name=None,open=True,render_format="svg"):
    name = aux_print_ForwardBackwardGraph_list_name(lkg,name)
    dot = graphviz.Digraph(name,comment=name)
    for i in range(len(lkg)):
        aux_print_graph(dot,lkg[i],i)
    small_fcts.graph_render(dot,open,"K",render_format)

# ==========================

