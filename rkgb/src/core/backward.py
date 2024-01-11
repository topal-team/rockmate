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
        super().__init__(main_target,
            parent_structure_with_id_generator=backward_graph)
        # - basic attributes:
        self.name = f"FWD[{main_target}]" if is_fwd else f"BWD[{main_target}]"
        self.is_fwd = is_fwd
        self.is_rand = is_rand
        self.main_code = main_code # tuple (target * AST)
        self.inplace_code = inplace_code if inplace_code else []
        self.body_code = body_code if body_code else [] # (str*AST) list
        # - inherits target attributes from simplified_node:
        # Not specially useful, but can help debugging
        if simplified_node:
            for attr in [
                    "all_targets","tensor_targets",
                    "inplace_targets","container_targets"]:
                setattr(self,attr,getattr(simplified_node,attr))
        # - deps and users:
        self.deps_real    = deps_real or set()
        self.deps_fake    = deps_fake or set()
        self.deps_global  = set()
        self.users_global = set()
        self.users        = set()
        # => all: AllocationNode sets
        if deps_through_artifacts: # ComputationNode set
            self.deps_through_artifacts = deps_through_artifacts
        else:
            self.deps_through_artifacts = set()
            # -> just for the toposort, we don't need the reciprocal attribute (users_..)
        # - inspection:
        self.time = None
        self.mem_overhead = None
        self.has_phantoms = None

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
            main_target = base.Node.no_target_string,
            allocation_type = "/!\\ No allocation_type/!\\",
            all_targets       = None,
            tensor_targets    = None,
            inplace_targets   = None,
            container_targets = None,
            info      = None,
            deps      = None,
            backward_graph = None):
        # ** informative **
        super().__init__(main_target,
            parent_structure_with_id_generator=backward_graph)
        self.allocation_type = allocation_type # data, grad or phantoms
        self.all_targets = all_targets or [main_target]
        self.tensor_targets = tensor_targets or [main_target]
        self.inplace_targets = inplace_targets or []
        self.container_targets = container_targets or []
        self.name = f"{main_target} {self.allocation_type}"
        self.mem  = 0
        self.info = info
        self.has_attribute__base = False
        self.includes_phantoms = False
        # ** deps/users **
        self.deps         = deps or set()
        self.deps_global  = set()
        self.users_global = set()
        self.users_real   = set()
        self.users_fake   = set()
        # => all: ComputationNode sets
    
    def get_all_standard_deps(self):
        return set().union(
            *[computation_node.deps_real
              for computation_node in self.deps])
    def get_all_standard_users(self):
        return set().union(
            *[computation_node.users
              for computation_node in self.users_real])



# ***********
# * ForwardBackwardGraph *
# ***********

class ForwardBackwardGraph(base.Graph):
    input_data_node = None
    list_output_data_nodes = None
    loss_computation_node = None
    list_output_grad_nodes = None
    input_grad_node = None
    # Note: We no longer have chain/list of K_graph,
    # as we fully moved to hierarchical structures,
    # hence the input_data/grad is simply the ad hoc
    # source node, so it could be removed, but it would
    # require to adapt quite a lot of lines in the compiler.
    # So make it easier for the moment I keep them.
    # !Warning!: input_grad_node is None if 
    # none of the inputs requires a gradient.

    def __init__(self,
            simplified_graph : SimplifiedGraph = None,
            original_mod : torch.nn.Module = None,
            inspection_device = None,
            do_inspection = True):
        # 2 constructors: if given a simplified_graph, 
        # then move from S to FB => run inspection,
        # build the backward part and allocation nodes.
        # otherwise return an empty graph
        super().__init__()
        self.dict_nodes = dict() # node name -> node
        self.computation_nodes = [] # Toposorted
        self.allocation_nodes = [] # Arbitrary order
        self.dict_fwd_cnodes = dict()
        self.dict_bwd_cnodes = dict()
        self.dict_data_anodes = dict()
        self.dict_grad_anodes = dict()
        self.dict_phantom_anodes = dict()
        # => all: dict: main_targets => Node
        if simplified_graph is not None:
            if original_mod is None or inspection_device is None: 
                raise Exception(
                    "You need to pass original_mod and inspection_device"\
                    "to ForwardBackwardGraph.__init__ (or let "\
                    "`simplified_graph` to None to get an empty graph")
            self.inherit_base_attributes(simplified_graph)
            # init and final codes:
            self.init_code = simplified_graph.init_node.get_code_ast()
            self.dict_output_viewing_code = dict(
                simplified_graph.dict_output_viewing_code)
            


            self.set_computation_node_numbers()



    def make_users(self):
        for cnode in self.computation_nodes:
            for req_anode in cnode.deps_real: req_anode.users_real.add(cnode)
            for req_anode in cnode.deps_fake: req_anode.users_fake.add(cnode)
        for anode in self.allocation_nodes:
            for req_cnode in anode.deps: req_cnode.users.add(anode)
    def init_deps_and_users_global(self):
        for cnode in self.computation_nodes:
            cnode.deps_global = cnode.deps_real.union(cnode.deps_fake)
            cnode.users_global = set(cnode.users)
        for anode in self.allocation_nodes:
            anode.deps_global = set(anode.deps)
            anode.users_global = anode.users_real.union(anode.users_fake)

    def set_computation_node_numbers(self):
        for i,cnode in enumerate(self.computation_nodes):
            setattr(cnode,"_number",i)

    # ****************
    def __iter__(self):
        return iter(self.computation_nodes)

    def make_temporary_global_root_node_to_deps_relation(self):
        # OVERWRITE base.Graph METHOD
        leaves_cnodes = []
        for cnode in self.computation_nodes:
            if not cnode.is_fwd and len(cnode.users) == 0:
                leaves_cnodes.append(cnode)
        if len(leaves_cnodes):
            return False,leaves_cnodes[0]
        else:
            root_allonode = ForwardBackwardAllocationNode(
                deps=leaves_cnodes,backward_graph=self)
            fresh_cnode_root = ForwardBackwardComputationNode(
                deps_real=set([root_allonode]),backward_graph=self)
            return True,fresh_cnode_root
    def remove_temporary_global_root_node(self,fresh_root):
        # We don't need the user relation, as we only use this
        # root_node to toposort; hence nothing to unplug
        pass

    



# ==========================



# ==========================
# = Move from S to K graph =
# ==========================

# the function that does it all
def aux_build_S_to_K(sg : SimplifiedGraph,
        original_mod,
        device,
        do_inspection=True):
    kg = ForwardBackwardGraph(sg)
    dict_compnode_fwd = kg.dict_compnode_fwd
    dict_compnode_bwd = kg.dict_compnode_bwd
    dict_KDN_data = kg.dict_KDN_data
    dict_KDN_grad = kg.dict_KDN_grad
    dict_KDN_phantoms = kg.dict_KDN_phantoms

    # ============  
    def handle_node(sn : SimplifiedNode):
        mt = sn.main_target
        our_global = inspection.generate_our_global(sg,original_mod,device)
        info = sg.dict_info[mt]

        # For artifact nodes :
        #   -> if compnode2 only need compnode1.size, it means in sg there is
        #   -> an artifact node for compnode1.size to avoid useless dep
        #   -> between compnode2 and compnode1. We decided to do NOT have KDN(size)
        #   -> in fact we just need compnode1 to be ordered before compnode2 in
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
            allocation_type    = "data",
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
            # Open grad_fn and collect backward dependencies:
            (   real_dependencies_of_bwd,
                exist_phantoms,
                has_attribute__base ) \
                = inspection.get_relevant_dependencies_via_grad_fn(
                    sn,our_global,tmp_local
                )
            fake_dependencies_of_bwd = kcn_fwd_deps - real_dependencies_of_bwd
            if sn.main_target in real_dependencies_of_bwd:
                attach_phantoms_to_data_node = True
                kdn_data.includes_phantoms = True

            else:
                attach_phantoms_to_data_node = False

            bwd_deps_real_mt = (
                all_deps_mt.intersection(set(sn_deps_mt)))
            kcn_bwd_deps_real = set(
                dict_KDN_data[mt] for mt in bwd_deps_real_mt)
            kcn_bwd_deps_fake = (
                kcn_fwd_deps - kcn_bwd_deps_real)
            kdn_data.has_attribute__base = has_attribute__base
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

            # -> KDN(phantoms)
            if exist_phs and not data_includes_phantoms:
                kdn_phantoms = ForwardBackwardAllocationNode(
                    main_target       = mt,
                    allocation_type    = "phantoms",
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
                main_target       = mt,
                allocation_type    = "grad",
                info        = info,
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
        kcn_fwd.mem_overhead = res.mem_overhead_fwd
        kcn_fwd.time     = res.time_run_fwd
        # kdn_data.mem     = info.memsize
        if data_includes_phantoms:
            kdn_data.mem = res.mem_run_fwd
        else:
            kdn_data.mem = res.mem_fgt_fwd

        # -> bwd ins
        if info.requires_grad:
            kcn_bwd.mem_overhead = res.mem_overhead_bwd
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
    loss_kcn.mem_overhead = 0
    dict_KCN_fwd[loss_kcn.main_target] = loss_kcn
    for kdn in list_outputs_kdn_grad:
        kdn.deps.add(loss_kcn)

    # -> list of nodes
    kg.computation_nodes = (
        list(dict_KCN_fwd.values()) +
        list(dict_KCN_bwd.values()))
    kg.allocation_nodes = (
        list(dict_KDN_data.values()) +
        list(dict_KDN_grad.values()) +
        list(dict_KDN_phantoms.values()))
    for kn in kg.computation_nodes+kg.allocation_nodes: kg.dict_kn[kn.name]=kn

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
            main_target = constants.init_target_string,
            allocation_type = "data",
            all_targets = sg.inputs,
            other_obj = kg)
        if sg.sources_req_grad or not is_really_first_graph:
            kg.input_kdn_grad=input_kdn_grad = ForwardBackwardAllocationNode(
                main_target = constants.init_target_string,
                allocation_type = "grad",
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

    # -> TOPOSORT computation_nodes
    kg.sort_computation_nodes()

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
        f"{len(kg.computation_nodes)} ForwardBackwardComputationNodes; {len(kg.allocation_nodes)} ForwardBackwardAllocationNodes"
    )

def aux_print_ForwardBackwardGraph_list_message(lkg : ForwardBackwardGraph_list):
    list_nb_kcn = [len(kg.computation_nodes) for kg in lkg]
    list_nb_kdn = [len(kg.allocation_nodes) for kg in lkg]
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
                f"Mem overhead : {measure.MemSize(kcn.mem_overhead)}"))
    def print_kdn(kdn):
        node(kdn.name,kdn.name,color=get_color(kdn),
            tooltip = f"Mem {measure.MemSize(kdn.mem)}")

    for kcn in kg.computation_nodes: print_kcn(kcn)
    for kdn in kg.allocation_nodes: print_kdn(kdn)

    # *** edges ***
    for kcn in kg.computation_nodes:
        for req_kdn in kcn.deps_real:
            c = get_color(req_kdn)
            edge(req_kdn.name,kcn.name,color=c)
        for req_kdn in kcn.deps_fake:
            c = get_color(req_kdn)
            edge(req_kdn.name,kcn.name,color=c,style="dashed")
    for kdn in kg.allocation_nodes:
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

