# ==========================
# ====== K structure =======
# ==========================

import torch
from rkgb.lowlevel import ast_add_on
from rkgb.lowlevel import constants
from rkgb.lowlevel import measure
from rkgb.lowlevel.anonymize import SimplifiedNodeAnonymizationMaterial
from rkgb.lowlevel import inspection
from rkgb.core import base
from rkgb.core.simplified import SimplifiedNode,SimplifiedGraph

class ComputationNode(base.Node):
    _topological_number = None
    def __init__(self,
            main_target=base.Node.no_target_string,
            simplified_node : SimplifiedNode = None,
            is_fwd  = True,
            is_rand = False,
            info = None,
            main_code    = None,
            inplace_code = None,
            body_code    = None,
            main_fct     = None,
            deps_real = None,
            deps_fake = None,
            deps_through_artifacts=None,
            required_parameter_nodes_real=None,
            required_parameter_nodes_fake=None,
            graph=None):
        super().__init__(main_target,parent_structure_with_id_generator=graph)
        # - basic attributes:
        self.name = f"FWD[{main_target}]" if is_fwd else f"BWD[{main_target}]"
        self.is_fwd = is_fwd
        self.is_rand = is_rand
        self.info = info
        self.main_code = main_code # tuple (target * AST)
        self.inplace_code = inplace_code or []
        self.body_code = body_code or [] # (str*AST) list
        self.main_fct = main_fct or ""
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
        self.deps_through_artifacts = deps_through_artifacts or set()
        # -> just for the toposort, we don't need the reciprocal attribute (users_..)
        self.required_parameter_nodes_real = required_parameter_nodes_real or set()
        self.required_parameter_nodes_fake = required_parameter_nodes_fake or set()
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
# * AllocationNode *
# ************

class AllocationNode(base.Node):
    def __init__(self,
            main_target = base.Node.no_target_string,
            allocation_type = "/!\\ No allocation_type/!\\",
            simplified_node = None,
            info      = None,
            deps      = None,
            graph = None):
        # ** informative **
        super().__init__(main_target,parent_structure_with_id_generator=graph)
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
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        del self.users
        self.users_real = set()
        self.users_fake = set()

# ***********
# * Graph *
# ***********

class ForwardAndBackwardGraph(base.Graph):
    source_data_anode : AllocationNode = None
    list_output_data_anodes : list[AllocationNode] = None
    loss_cnode : ComputationNode = None
    list_output_grad_anodes : list[AllocationNode] = None
    source_grad_anode : AllocationNode = None
    # Note: We no longer have chain/list of K_graph,
    # as we fully moved to hierarchical structures,
    # hence the source_data/grad are trivial ad hoc
    # source nodes, so they could be removed, but it would
    # require to adapt quite a lot of lines in the compiler.
    # So make it easier for the moment I keep them.
    # !Warning!: source_grad_anode is None if 
    # none of the inputs requires a gradient.

    def __init__(self,
            simplified_graph : SimplifiedGraph = None,
            original_mod : torch.nn.Module = None,
            current_device = None,
            inspection_device = None,
            do_inspection = True,
            dict_mt_to_sn_ano_material = None):
        # 2 constructors: if given a simplified_graph, 
        # then move from S to FB => run inspection,
        # build the backward part and allocation nodes.
        # otherwise return an empty graph
        super().__init__()
        self.dict_nodes = dict() # node name -> node
        self.computation_nodes = [] # Toposorted
        self.allocation_nodes = [] # Arbitrary order
        self.parameter_nodes = []
        self.dict_fwd_cnodes = dict()
        self.dict_bwd_cnodes = dict()
        self.dict_data_anodes = dict()
        self.dict_grad_anodes = dict()
        self.dict_phantoms_anodes = dict()
        # => all: dict: main_targets => Node
        if simplified_graph is not None:
            if any(x is None for x in [
                    original_mod,
                    current_device,inspection_device,
                    dict_mt_to_sn_ano_material]):
                raise Exception(
                    "You need to pass all required args"\
                    "to ForwardAndBackwardGraph.__init__ (or let "\
                    "`simplified_graph` to None to get an empty graph")
            self.inherit_base_attributes(simplified_graph)
            self.init_code = simplified_graph.init_code

            dict_old_param_node_to_new = dict(
                (param_node,ParameterNode(node_to_clone=param_node))
                for param_node in simplified_graph.parameter_nodes)
            self.parameter_nodes = [
                dict_old_param_node_to_new[param_node]
                for param_node in simplified_graph.parameter_nodes]
            self.parameter_nodes_required_for_init_code = [
                dict_old_param_node_to_new[param_node]
                for param_node in simplified_graph.init_node.required_parameter_nodes
            ]
            # Note: these are backward.ParameterNodes not base.ParameterNodes
            dict_old_param_node_to_new_param_node = dict(
                zip(simplified_graph.parameter_nodes,self.parameter_nodes))
            
            # To avoid running twice equivalent codes
            dict_sn_ano_id_to_ano_tracing_result = dict()
            dict_sn_ano_id_to_inspection_result = dict()

            for sn_to_proceed in simplified_graph.nodes:
                self.process_and_inspect_node(
                    sn_to_proceed,
                    simplified_graph,
                    original_mod,
                    do_inspection,
                    current_device,
                    inspection_device,
                    dict_old_param_node_to_new_param_node,
                    dict_mt_to_sn_ano_material,
                    dict_sn_ano_id_to_ano_tracing_result,
                    dict_sn_ano_id_to_inspection_result)
            self.make_special_loss_and_output_nodes(simplified_graph)
            self.store_all_nodes()
            self.make_reciprocal_users_attributes()
            self.computation_nodes = self.get_sorted_nodes_by_following_deps_relation()
            self.make_special_input_nodes(simplified_graph)
            self.set_computation_node_topological_numbers()

    # ======= MAIN LOOP ========
    def process_and_inspect_node(self,
            sn_to_proceed : SimplifiedNode,
            simplified_graph : SimplifiedGraph,
            original_mod : torch.nn.Module,
            do_inspection,
            current_device,
            inspection_device,
            dict_old_param_node_to_new_param_node,
            dict_mt_to_sn_ano_material,
            dict_sn_ano_id_to_ano_tracing_result,
            dict_sn_ano_id_to_inspection_result):
        # 0) Create the execution environment
        our_global = inspection.Inspector.generate_global_env(
            self,current_device,inspection_device,original_mod)
        tmp_local = inspection.Inspector.generate_local_env(
            sn_to_proceed,simplified_graph,our_global,
            original_mod,
            current_device,inspection_device
        )

        sn_ano_material : SimplifiedNodeAnonymizationMaterial = dict_mt_to_sn_ano_material[sn_to_proceed.mt]

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
        fwd_cnode_required_param_nodes = set(
            dict_old_param_node_to_new_param_node[param_node]
            for param_node in sn_to_proceed.required_parameter_nodes
        )
        fwd_cnode = ComputationNode(
            main_target     = sn_to_proceed.main_target,
            simplified_node = sn_to_proceed,
            graph   = self,
            is_fwd  = True,
            is_rand = sn_to_proceed.is_rand,
            info    = sn_to_proceed.info,
            main_code    = sn_to_proceed.main_code,
            inplace_code = sn_to_proceed.inplace_code,
            body_code    = sn_to_proceed.body_code,
            main_fct     = sn_to_proceed.main_fct,
            deps_real    = fwd_cnode_deps,
            deps_through_artifacts = set(
                self.dict_fwd_cnodes[req_sn.main_target]
                for req_sn in sn_to_proceed.deps_through_artifacts
            ),
            required_parameter_nodes_real=fwd_cnode_required_param_nodes
        )
        self.dict_fwd_cnodes[sn_to_proceed.main_target] = fwd_cnode

        # 2) Data Allocation Node
        data_anode = AllocationNode(
            main_target     = sn_to_proceed.main_target,
            allocation_type = "data",
            simplified_node = sn_to_proceed,
            info = sn_to_proceed.info,
            deps = set([fwd_cnode]),
            graph = self)
        self.dict_data_anodes[sn_to_proceed.main_target] = data_anode

        # ======================================
        # == Part 2 : BUILD THE BACKWARD PART ==
        # ======================================
        if sn_to_proceed.info.requires_grad:
            # 1) Open grad_fn and collect backward dependencies:
            # + anonymization to avoid running twice on equivalent nodes
            if sn_ano_material.anonymous_id in dict_sn_ano_id_to_ano_tracing_result:
                ano_tracing_result = dict_sn_ano_id_to_ano_tracing_result[sn_ano_material.anonymous_id]
                try:
                    tracing_result = inspection.reverse_translate_tracing_result(sn_ano_material,ano_tracing_result)
                except: # not exactly exactly equivalent nodes => some weird unused names
                    tracing_result = inspection.get_relevant_dependencies_via_grad_fn(
                    sn_to_proceed,our_global,tmp_local)
            else:
                tracing_result = inspection.get_relevant_dependencies_via_grad_fn(
                    sn_to_proceed,our_global,tmp_local)
                try:
                    ano_tracing_result = inspection.anonymize_tracing_result(sn_ano_material,tracing_result)
                    dict_sn_ano_id_to_ano_tracing_result[sn_ano_material.anonymous_id] = ano_tracing_result
                except:
                    pass
            # Results:
            (   bwd_real_dependencies,
                bool_bwd_requires_fwd_data,
                bool_exist_phantoms,
                parameter_strs_found,
                has_attribute__base ) = tracing_result
            data_anode.has_attribute__base = has_attribute__base
            # - real deps
            bwd_cnode_deps_real = set(
                self.dict_data_anodes[req_target]
                for req_target in bwd_real_dependencies
            )
            for anode in fwd_cnode_deps:
                if anode.mem < 1024**2 and anode not in bwd_cnode_deps_real:
                    bwd_cnode_deps_real.add(anode)
            # - fake deps
            bwd_cnode_deps_fake = fwd_cnode_deps - bwd_cnode_deps_real
            # - phantoms
            if bool_bwd_requires_fwd_data:
                bwd_cnode_deps_real.add(data_anode)
                data_anode.includes_phantoms = True
            else:
                bwd_cnode_deps_fake.add(data_anode)
            # - real param deps
            bwd_cnode_required_param_nodes_real = set(
                param_node
                for param_node in fwd_cnode_required_param_nodes
                if param_node.param_str in parameter_strs_found
            )
            # - fake param deps
            bwd_cnode_required_param_nodes_fake = (
                fwd_cnode_required_param_nodes
                - bwd_cnode_required_param_nodes_real
            )
            
            # 2) Backward Computation Node
            bwd_cnode = ComputationNode(
                main_target = sn_to_proceed.main_target,
                simplified_node = sn_to_proceed,
                is_fwd = False,
                main_fct = sn_to_proceed.main_target+".backward",
                info = sn_to_proceed.info,
                deps_real = bwd_cnode_deps_real, # we add grad_anode latter on
                deps_fake = bwd_cnode_deps_fake,
                required_parameter_nodes_real=bwd_cnode_required_param_nodes_real,
                required_parameter_nodes_fake=bwd_cnode_required_param_nodes_fake,
                graph = self
            )
            self.dict_bwd_cnodes[sn_to_proceed.main_target] = bwd_cnode

            # 3) Phantom Allocation Node
            if bool_exist_phantoms and not data_anode.includes_phantoms:
                phantoms_anode = AllocationNode(
                    main_target = sn_to_proceed.main_target,
                    allocation_type = "phantoms",
                    simplified_node = sn_to_proceed,
                    info = sn_to_proceed.info,
                    deps = set([fwd_cnode]),
                    graph = self
                )
                self.dict_phantoms_anodes[sn_to_proceed.main_target] = phantoms_anode
                bwd_cnode.deps_real.add(phantoms_anode)
                fwd_cnode.has_phantoms = True
            else:
                fwd_cnode.has_phantoms = False

            # 4) Grad Allocation Node
            grad_anode = AllocationNode(
                main_target = sn_to_proceed.main_target,
                allocation_type = "grad",
                simplified_node = sn_to_proceed,
                info = sn_to_proceed.info,
                graph = self
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
        elif sn_ano_material.anonymous_id in dict_sn_ano_id_to_inspection_result:
            inspection_result = dict_sn_ano_id_to_inspection_result[sn_ano_material.anonymous_id]
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
                current_device,
                inspection_device,
                our_global, tmp_local,
                timer, memory_tracker,
                simplified_graph,
                original_mod
            )
            inspection_result = inspector.inspect()
        dict_sn_ano_id_to_inspection_result[sn_ano_material.anonymous_id] = inspection_result
        
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
    def make_special_loss_and_output_nodes(self,
            simplified_graph : SimplifiedGraph):
        # Outputs:
        self.list_output_data_anodes = [
            self.dict_data_anodes[output_sn.main_target]
            for output_sn in simplified_graph.output_nodes
        ]
        self.list_output_grad_anodes = [
            self.dict_grad_anodes[output_sn.main_target]
            for output_sn in simplified_graph.output_nodes
            if output_sn.info.requires_grad
        ]
        # Loss:
        loss_cnode = ComputationNode(
            main_target = "loss",
            is_fwd    = True,
            main_code = ("loss",ast_add_on.make_ast_constant("LOSS")),
            deps_real = set(self.list_output_data_anodes),
            graph = self
        )
        self.loss_cnode = loss_cnode
        loss_cnode.time = 0
        loss_cnode.mem_overhead = 0
        self.dict_fwd_cnodes[loss_cnode.main_target] = loss_cnode
        for output_grad_anode in self.list_output_grad_anodes:
            output_grad_anode.deps.add(loss_cnode)


    def make_special_input_nodes(self,
            simplified_graph : SimplifiedGraph):
        # 1) Input Data Allocation Node
        source_data_anode = AllocationNode(
            main_target = constants.init_target_string,
            allocation_type = "data",
            graph = self)
        source_data_anode.all_targets = simplified_graph.input_targets
        self.source_data_anode = source_data_anode
        self.dict_data_anodes[source_data_anode.main_target] = source_data_anode
        self.dict_nodes[source_data_anode.name] = source_data_anode

        # 2) Users of source_data_anode
        source_data_anode.users_real = set(
            self.dict_fwd_cnodes[sn.main_target]
            for sn in simplified_graph.init_node.users
        ) # Not reciprocal !

        if simplified_graph.sources_req_grad:
            # 3) Input Grad Allocation Node
            source_grad_anode = AllocationNode(
                main_target = constants.init_target_string,
                allocation_type = "grad",
                graph = self)
            source_grad_anode.all_targets = simplified_graph.input_targets,
            self.source_grad_anode = source_grad_anode
            self.dict_grad_anodes[source_grad_anode.main_target] = source_grad_anode
            self.dict_nodes[source_grad_anode.name] = source_grad_anode

            # 4) Deps of source_grad_cnode
            source_grad_anode.deps = set(
                self.dict_bwd_cnodes[sn.main_target]
                for sn in simplified_graph.init_node.users
            ) # Not reciprocal !

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
        cnode : ComputationNode
        anode : AllocationNode
        for cnode in self.computation_nodes:
            for req_anode in cnode.deps_real: req_anode.users_real.add(cnode)
            for req_anode in cnode.deps_fake: req_anode.users_fake.add(cnode)
            for param_node in cnode.required_parameter_nodes_real:
                param_node.users_real.add(cnode)
            for param_node in cnode.required_parameter_nodes_fake:
                param_node.users_fake.add(cnode)
        for anode in self.allocation_nodes:
            for req_cnode in anode.deps: req_cnode.users.add(anode)

    def set_computation_node_topological_numbers(self):
        for i,cnode in enumerate(self.computation_nodes):
            cnode._topological_number = i

    # **********************************
    # == OVERWRITE base.Graph METHODS ==
    @property
    def _lists_of_nodes(self):
        return [self.computation_nodes,self.allocation_nodes]

    def make_temporary_global_root_node_to_deps_relation(self):
        # OVERWRITE base.Graph METHOD
        leaf_cnodes = []
        for cnode in self.computation_nodes:
            if not cnode.is_fwd and len(cnode.users) == 0:
                leaf_cnodes.append(cnode)
        root_anode = AllocationNode(
            deps=leaf_cnodes,
            graph=self)
        fresh_root_cnode = ComputationNode(
            deps_real=set([root_anode]),
            graph=self)
        return fresh_root_cnode
    def remove_temporary_global_root_node(self,fresh_root):
        # We don't need the users relation, as we only use this
        # root_node to toposort; hence nothing to unplug
        pass

    # = print and render =
    def __str__(self):
        return f"Forward+Backward and Computation+Allocation Graph:"\
               f"{len(self.computation_nodes)} Computation Nodes, "\
               f"{len(self.allocation_nodes)} Allocation Nodes."

    @staticmethod
    def get_render_color(node):
        color_fwd_cnode = "blue"
        color_bwd_cnode = "blueviolet"
        color_anode = "olive"
        color_parameter_node = "black"
        if isinstance(node,AllocationNode):
            return color_anode
        elif isinstance(node,ParameterNode):
            return color_parameter_node
        else:
            assert isinstance(node,ComputationNode)
            if node.is_fwd: return color_fwd_cnode
            else: return color_bwd_cnode

    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            include_parameter_nodes=False,
            include_artifact_edges=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        color_special = "green"

        # 1) Parameter nodes
        if include_parameter_nodes:
            for param_node in self.parameter_nodes:
                param_node : ParameterNode
                if param_node.view_targets == []:
                    render_label = param_node.param_str
                elif only_function_name:
                    render_label = "\n".join(
                        [param_node.param_str]+param_node.view_targets)
                else:
                    render_label = f"{param_node.param_str}\n{param_node.get_code()}"
                dot.node(
                    param_node.param_str,
                    render_label,
                    color = self.__class__.get_render_color(param_node),
                    style = "dashed")
                
        # 2) Nodes
        for cnode in self.computation_nodes:
            cnode : ComputationNode
            if cnode.main_target == "loss":
                dot.node(cnode.name,"LOSS computation",color=color_special)
            else:
                if cnode.is_fwd:
                    if only_function_name:
                        render_label = f"{cnode.mt} : {cnode.main_fct}"
                    else:
                        render_label = cnode.get_code()
                else:
                    render_label = cnode.main_fct
                dot.node(
                    cnode.name,
                    render_label,
                    color = self.__class__.get_render_color(cnode),
                    tooltip = (
                        f"Time : {cnode.time}\n Memory Overhead : "\
                        f"{measure.pretty_format_memory(cnode.mem_overhead)}")
                )

        for anode in self.allocation_nodes:
            anode : AllocationNode
            dot.node(
                anode.name,
                anode.name,
                color = self.__class__.get_render_color(anode),
                tooltip = "Memory : "+measure.pretty_format_memory(anode.mem)
            )

        # 3) edges
        for cnode in self.computation_nodes:
            for req_anode in cnode.deps_real:
                dot.edge(req_anode.name,cnode.name,
                    color=self.__class__.get_render_color(cnode))
            for req_anode in cnode.deps_fake:
                dot.edge(req_anode.name,cnode.name,
                    color=self.__class__.get_render_color(cnode),
                    style="dashed")

            if include_artifact_edges:
                for req_cnode in cnode.deps_through_artifacts:
                    dot.edge(req_cnode.name,cnode.name,style="dotted")
            if include_parameter_nodes:
                for req_param in cnode.required_parameter_nodes_real:
                    dot.edge(req_param.param_str,cnode.name)
                for req_param in cnode.required_parameter_nodes_fake:
                    dot.edge(req_param.param_str,cnode.name,style="dashed")

        for anode in self.allocation_nodes:
            for req_cnode in anode.deps:
                dot.edge(req_cnode.name,anode.name,
                    color=self.__class__.get_render_color(cnode))
        
        # 4) source_data/grad_cnode
        kwargs = {"color":color_special , "style":"dashed"}
        source_data = self.source_data_anode
        if source_data.users_real != set():
            dot.node(source_data.name,source_data.name,**kwargs)
            for user_anode in source_data.users_real:
                dot.edge(source_data.name,user_anode.name,**kwargs)
        source_grad = self.source_grad_anode
        if source_grad is not None:
            dot.node(source_grad.name,source_grad.name,**kwargs)
            for req_anode in source_grad.deps:
                dot.edge(req_anode.name,source_grad.name,**kwargs)

        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )


# ==========================