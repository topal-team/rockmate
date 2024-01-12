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
            info = None,
            main_code    = None,
            inplace_code = None,
            body_code    = None,
            deps_real = None,
            deps_fake = None,
            deps_through_artifacts=None,
            forwardbackward_graph=None):
        super().__init__(main_target,
            parent_structure_with_id_generator=forwardbackward_graph)
        # - basic attributes:
        self.name = f"FWD[{main_target}]" if is_fwd else f"BWD[{main_target}]"
        self.is_fwd = is_fwd
        self.is_rand = is_rand
        self.info = info
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
        self.deps_real = deps_real or set()
        self.deps_fake = deps_fake or set()
        self.users     = set()
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
            simplified_node = None,
            info      = None,
            deps      = None,
            forwardbackward_graph = None):
        # ** informative **
        super().__init__(main_target,
            parent_structure_with_id_generator=forwardbackward_graph)
        self.allocation_type = allocation_type # data, grad or phantoms
        # inherits target attributes from simplified_node:
        if simplified_node is not None:
            for attr in [
                    "all_targets","tensor_targets",
                    "inplace_targets","container_targets"]:
                setattr(self,attr,getattr(simplified_node,attr))
        else:
            self.all_targets = [main_target]
            self.tensor_targets = [main_target]
            self.inplace_targets = []
            self.container_targets = []
        self.name = f"{main_target} {self.allocation_type}"
        self.mem  = 0
        self.info = info
        self.has_attribute__base = False
        self.includes_phantoms = False
        # ** deps/users **
        self.deps       = deps or set()
        self.users_real = set()
        self.users_fake = set()
        # => all: ComputationNode sets
    
    def get_all_standard_deps(self):
        return set().union(
            *[computation_node.deps_real
              for computation_node in self.deps])
    def get_all_standard_users(self):
        return set().union(
            *[computation_node.users
              for computation_node in self.users_real])


class ParameterNode(base.ParameterNode):
    """
    backward.ParameterNode sub class base.ParameterNode
    only to change `.users` attribute by `.users_real/fake` 
    """
    def __init__(self,*args):
        super().__init__(*args)
        del self.users
        self.users_real = set()
        self.users_fake = set()

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
            do_inspection = True,
            inspection_device = None):
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
        self.dict_phantoms_anodes = dict()
        # => all: dict: main_targets => Node
        if simplified_graph is not None:
            if original_mod is None or inspection_device is None: 
                raise Exception(
                    "You need to pass original_mod and inspection_device"\
                    "to ForwardBackwardGraph.__init__ (or let "\
                    "`simplified_graph` to None to get an empty graph")
            self.inherit_base_attributes(simplified_graph)
            self.init_code = simplified_graph.init_node.get_code_ast()
            self.dict_output_viewing_code = dict(simplified_graph.dict_output_viewing_code)

            for sn_to_proceed in simplified_graph.nodes:
                self.process_and_inspect_node(
                    sn_to_proceed,
                    simplified_graph,
                    original_mod,
                    do_inspection,
                    inspection_device)
            self.make_special_loss_and_io_nodes()
            self.store_all_nodes()
            self.make_reciprocal_users_attributes()
            self.computation_nodes = self.get_sorted_nodes_by_following_deps_relation()
            self.set_computation_node_numbers()

    # ======= MAIN LOOP ========
    def process_and_inspect_node(self,
            sn_to_proceed : SimplifiedNode,
            simplified_graph : SimplifiedGraph,
            original_mod : torch.nn.Module,
            do_inspection,
            inspection_device):
        # 0) Create the execution environment
        our_global = inspection.Inspector.generate_global_env(
            self,inspection_device)
        tmp_local = inspection.Inspector.generate_local_env(
            sn_to_proceed,simplified_graph,our_global,
            original_mod,inspection_device
        )

        # =====================================
        # == Part 1 : BUILD THE FORWARD PART ==
        # =====================================
        # 1) Forward Computation Node
        sn_deps_targets = [
            req_sn.main_target 
            for req_sn in sn_to_proceed.deps]
        fwd_cnode_deps = set(
            self.dict_data_anodes[req_target]
            for req_target in sn_deps_targets
        )
        fwd_cnode = ForwardBackwardComputationNode(
            main_target           = sn_to_proceed.main_target,
            simplified_node       = sn_to_proceed,
            forwardbackward_graph = self,
            is_fwd       = True,
            is_rand      = sn_to_proceed.is_rand,
            info         = sn_to_proceed.info,
            main_code    = sn_to_proceed.main_code,
            inplace_code = sn_to_proceed.inplace_code,
            body_code    = sn_to_proceed.body_code,
            deps_real    = fwd_cnode_deps,
            deps_through_artifacts = set(
                self.dict_fwd_cnodes[req_sn.main_target]
                for req_sn in sn_to_proceed.deps_through_artifacts
            )
        )
        self.dict_fwd_cnodes[sn_to_proceed.main_target] = fwd_cnode

        # 2) Data Allocation Node
        data_anode = ForwardBackwardAllocationNode(
            main_target     = sn_to_proceed.main_target,
            allocation_type = "data",
            simplified_node = sn_to_proceed,
            info = sn_to_proceed.info,
            deps = set([fwd_cnode]),
            forwardbackward_graph = self)
        self.dict_data_anodes[sn_to_proceed.main_target] = data_anode

        # ======================================
        # == Part 2 : BUILD THE BACKWARD PART ==
        # ======================================
        if sn_to_proceed.info.requires_grad:
            # 1) Open grad_fn and collect backward dependencies:
            (   bwd_real_dependencies,
                bool_bwd_requires_fwd_data,
                bool_exist_phantoms,
                has_attribute__base ) \
                = inspection.get_relevant_dependencies_via_grad_fn(
                    sn_to_proceed,our_global,tmp_local
                )
            bwd_cnode_deps_real = set(
                self.dict_data_anodes[req_target]
                for req_target in bwd_real_dependencies
            )
            bwd_cnode_deps_fake = fwd_cnode_deps
            data_anode.has_attribute__base = has_attribute__base
            if bool_bwd_requires_fwd_data:
                bwd_cnode_deps_real.add(data_anode)
                data_anode.includes_phantoms = True
            else:
                bwd_cnode_deps_fake.add(data_anode)
            
            # 2) Backward Computation Node
            bwd_cnode = ForwardBackwardComputationNode(
                main_target = sn_to_proceed.main_target,
                simplified_node = sn_to_proceed,
                is_fwd = False,
                info = sn_to_proceed.info,
                deps_real = bwd_cnode_deps_real, # we add grad_anode latter on
                deps_fake = bwd_cnode_deps_fake,
                forwardbackward_graph = self
            )
            self.dict_bwd_cnodes[sn_to_proceed.main_target] = bwd_cnode

            # 3) Phantom Allocation Node
            if bool_exist_phantoms and not data_anode.includes_phantoms:
                phantoms_anode = ForwardBackwardAllocationNode(
                    main_target = sn_to_proceed.main_target,
                    allocation_type = "phantoms",
                    simplified_node = sn_to_proceed,
                    info = sn_to_proceed.info,
                    deps = set([fwd_cnode]),
                    forwardbackward_graph = self
                )
                self.dict_phantoms_anodes[sn_to_proceed.main_target] = phantoms_anode
                fwd_cnode.has_phantoms = True
            else:
                fwd_cnode.has_phantoms = False

            # 4) Grad Allocation Node
            grad_anode = ForwardBackwardAllocationNode(
                main_target = sn_to_proceed.main_target,
                allocation_type = "grad",
                simplified_node = sn_to_proceed,
                info = sn_to_proceed.info,
                forwardbackward_graph = self
            )
            self.dict_grad_anodes[sn_to_proceed.main_target] = grad_anode
            bwd_cnode.deps_real.add(grad_anode)
            # grad_anode depends on the bwd node of the users of the fwd_cnode
            # which aren't proceed it (as we process them in the topo order)
            # But we can plug bwd_cnode to its users, which are previously
            # created grad_anodes.
            for req_target in sn_deps_targets:
                if req_target in self.dict_grad_anodes: # ie requires_grad
                    self.dict_grad_anodes[req_target].deps.add(bwd_cnode)
        else:
            bool_bwd_requires_fwd_data = False
            bool_exist_phantoms = False
            
        # =========================
        # == Part 3 : INSPECTION ==
        # =========================
        # 1) Run the inspection
        if not do_inspection:
            inspection_result = inspection.InspectionResult()
        else:
            if inspection_device.type == "cpu":
                timer = measure.TimerCPU()
                memory_tracker = measure.MemoryTrackerCPU()
            elif inspection_device.type == "cuda":
                timer = measure.TimerCUDA(inspection_device)
                memory_tracker = measure.MemoryTrackerCUDA(inspection_device)
            else:
                raise Exception(
                    f"Unrecognized device type: neither 'cpu' nor "\
                    f"'cuda' but {inspection_device.type}")
            inspector = inspection.InspectorDefault(
                sn_to_proceed,
                inspection_device,
                our_global, tmp_local,
                timer, memory_tracker,
                simplified_graph,
                original_mod
            )
            inspection_result = inspector.inspect()
        
        # 2) Fill corresponding node attributes
        # - Forward Computation Node:
        fwd_cnode.mem_overhead = inspection_result.mem_overhead_fwd
        fwd_cnode.time = inspection_result.time_fwd
        # - Data Allocation Node:
        if bool_bwd_requires_fwd_data: # ie data_anode includes phantoms
            data_anode.mem = inspection_result.mem_run_fwd
        else:
            data_anode.mem = inspection_result.mem_fgt_fwd

        if sn_to_proceed.info.requires_grad:
            # - Backward Computation Node:
            bwd_cnode.mem_overhead = inspection_result.mem_overhead_bwd
            bwd_cnode.time = inspection_result.time_bwd
            # - Grad Allocation Node:
            grad_anode.mem = inspection_result.mem_fgt_fwd
            # Note: It's mem_fgt_FWD not bwd;
            # Moreover it used to be: grad_anode.mem := data_anode.mem
            # But I don't understand why: TO TEST / TO CHANGE

            # - Phantoms Allocation Node:
            if bool_exist_phantoms and not bool_bwd_requires_fwd_data:
                phantoms_anode.mem = (
                    inspection_result.mem_run_fwd 
                    - inspection_result.mem_fgt_fwd
                )
            # If you want to test an other way to detect phantoms:
            # exist_diff=res.mem_run_fwd - res.mem_fgt_fwd > 0
            # if exist_diff or exist_phs:
            #   print(f"For node {mt}: mem_diff : {exist_diff} "\
            #         f"and detection {exist_phs}")
    # ======= END OF MAIN LOOP =======

    # ===================================================
    # == Small methods to generate the last attributes ==
    def make_special_loss_and_io_nodes(self,
            simplified_graph : SimplifiedGraph):
        # Outputs:
        self.list_output_data_nodes = [
            self.dict_data_anodes[output_sn.main_target]
            for output_sn in simplified_graph.output_nodes
        ]
        self.list_output_grad_nodes = [
            self.dict_grad_anodes[output_sn.main_target]
            for output_sn in simplified_graph.output_nodes
        ]
        # Loss:
        loss_cnode = ForwardBackwardComputationNode(
            main_target = "loss",
            is_fwd    = True,
            main_code = ("loss",ast_add_on.make_ast_constant("LOSS")),
            deps_real = set(self.list_output_data_nodes),
            forwardbackward_graph = self
        )
        self.loss_computation_node = loss_cnode
        loss_cnode.time = 0
        loss_cnode.mem_overhead = 0
        self.dict_fwd_cnodes[loss_cnode.main_target] = loss_cnode
        for output_grad_anode in self.list_output_grad_nodes:
            output_grad_anode.deps.add(loss_cnode)
        # Inputs:
        self.input_data_node = ForwardBackwardAllocationNode(
            main_target = constants.init_target_string,
            allocation_type = "data",
            all_targets = simplified_graph.input_targets,
            forwardbackward_graph=self)
        self.input_data_node.all_targets = simplified_graph.input_targets
        if simplified_graph.sources_req_grad:
            self.input_grad_node = ForwardBackwardAllocationNode(
                main_target = constants.init_target_string,
                allocation_type = "grad",
                all_targets = simplified_graph.input_targets,
                forwardbackward_graph=self)

    def store_all_nodes(self):
        cnodes = self.computation_nodes = (
            list(self.dict_fwd_cnodes.values()) +
            list(self.dict_bwd_cnodes.values()))
        anodes = self.allocation_nodes = (
            list(self.dict_data_anodes.values()) +
            list(self.dict_grad_anodes.values()) +
            list(self.dict_phantoms_anodes.values()))
        for node in cnodes + anodes:
            self.dict_nodes[node.name] = node

    def make_reciprocal_users_attributes(self):
        for cnode in self.computation_nodes:
            for req_anode in cnode.deps_real: req_anode.users_real.add(cnode)
            for req_anode in cnode.deps_fake: req_anode.users_fake.add(cnode)
        for anode in self.allocation_nodes:
            for req_cnode in anode.deps: req_cnode.users.add(anode)

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
                deps=leaves_cnodes,
                forwardbackward_graph=self)
            fresh_cnode_root = ForwardBackwardComputationNode(
                deps_real=set([root_allonode]),
                forwardbackward_graph=self)
            return True,fresh_cnode_root
    def remove_temporary_global_root_node(self,fresh_root):
        # We don't need the users relation, as we only use this
        # root_node to toposort; hence nothing to unplug
        pass

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

