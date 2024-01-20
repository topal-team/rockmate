# ==========================
# ====== H structure =======
# ==========================
pip_editable_broken_imports = False
if pip_editable_broken_imports:
    from lowlevel.measure import pretty_format_memory
    from lowlevel import anonymize
    from core import base
    from core.backward import ForwardAndBackwardGraph, ComputationNode, AllocationNode
    from core.backward import ParameterNode as bwdParameterNode
    from core.partitioned import PartitionedStructure, PartitionedCluster, PartitionedGraph, PartitionedNode
else:
    from rkgb.lowlevel.measure import pretty_format_memory
    from rkgb.lowlevel.variable_info import VariableInfo
    from rkgb.lowlevel import anonymize
    from rkgb.core import base
    from rkgb.core.backward import ForwardAndBackwardGraph, ComputationNode, AllocationNode
    from rkgb.core.backward import ParameterNode as bwdParameterNode
    from rkgb.core.partitioned import PartitionedStructure, PartitionedCluster, PartitionedGraph, PartitionedNode


# from rkgb.lowlevel.measure import pretty_format_memory
# from rkgb.lowlevel.variable_info import VariableInfo
# from rkgb.lowlevel import anonymize
# from rkgb.core import base
# from rkgb.core.backward import ForwardAndBackwardGraph, ComputationNode, AllocationNode
# from rkgb.core.backward import ParameterNode as bwdParameterNode
# from rkgb.core.partitioned import PartitionedStructure, PartitionedCluster, PartitionedGraph, PartitionedNode

# ************
# * HierarchicalComputationNode *
# ************
class HierarchicalComputationNode(base.Node):
    def __init__(self,
            name,
            main_target = None,
            cnode : ComputationNode = None,
            sub_cluster = None,
            is_fwd = True,
            _topological_number = -1,
            hierarchical_graph = None):
        super().__init__(main_target,
            parent_structure_with_id_generator=hierarchical_graph)
        self.name = name  # e.g. Fwd_1
        self.cnode = cnode
        self.sub_cluster : HierarchicalCluster = sub_cluster
        self.is_fwd = is_fwd
        self.is_leaf = bool(main_target is None)
        self.deps = set()  # HAN set
        self.users = set()  # HAN set
        self.deps_through_artifacts = set()  # HCN set
        self.required_parameter_nodes_real = set()
        self.required_parameter_nodes_fake = set()
        self._topological_number = _topological_number
        # About self._topological_number :
        # When toposorting list_hcn we want to respect as much as possible
        # the original order, so list_hcn toposort consists of:
        # -> inverse Breath First Search (as any toposort)
        # -> if equal, then choose the one which comes first in list_cnode.
        # And that's what self._topological_number is used for: base.Node.get_num
        # (for D, S and K nodes we can get the number from main_target,
        # but some H_nodes don't have a main_target).

    def get_all_standard_deps(self):
        return set().union(
            *[han.deps for han in self.deps],
            self.deps_through_artifacts)
    def get_all_standard_users(self):
        return set().union(
            *[han.users for han in self.users])
    # def does_require_grad(self): # TO REMOVE USELESS ???
        # if not self.is_leaf:
            # return True
        # else:
            # assert self.info is not None
            # return self.info.requires_grad


# ************
# * HierarchicalAllocationNode *
# ************
class HierarchicalAllocationNode(base.Node):
    anode : AllocationNode = None
    main_target : str = None
    name : str = None
    is_data : bool = None
    deps : set[HierarchicalComputationNode] = None
    users : set[HierarchicalComputationNode] = None
    def __init__(self,anode : AllocationNode = None):
        self.deps = set()
        self.users = set()
        if anode is None:
            super().__init__()
        else:
            super().__init__(anode.mt)
            self.anode = anode
            self.name = HierarchicalAllocationNode.make_name_from_anode(anode)
            self.mem = anode.mem
            if anode.allocation_type == "data":
                self.is_data = True
            elif anode.allocation_type == "grad":
                self.is_data = False
            else:
                assert(anode.allocation_type == "phantoms")
                raise Exception(
                    "A Hierarchical Allocation Node cannot represent "\
                    "a backward.anode of type 'phantoms'")

    def get_all_standard_deps(self):
        return set().union(
            *[hcn.deps for hcn in self.deps])
    def get_all_standard_users(self):
        return set().union(
            *[hcn.users for hcn in self.users])
        
    @staticmethod
    def make_name_from_anode(anode):
        return f"Hierarchical_{anode.allocation_type}_{anode.mt}"



class HierarchicalParameterNode(base.ParameterNode):
    """
    Sub class base.ParameterNode, to add .users_real/fake 
    just as backward.ParameterNode does; but it's clearer
    to have two different classes, 
    """
    def __init__(self,original_param_node : bwdParameterNode):
        super().__init__(node_to_clone=original_param_node)
        del self.users
        self.users_real = set()
        self.users_fake = set()
        self.original_param_node = original_param_node




# ***********
# * HierarchicalGraph *
# ***********
class HierarchicalGraph(base.Graph):
    def __init__(self,
            h_cluster,
            p_graph : PartitionedGraph,
            fb_graph : ForwardAndBackwardGraph):
        super().__init__()
        # /!\ All the HCN's and HAN's should be in the same level. /!\ 
        self.graph_nb = p_graph.graph_nb
        self.name = f"HierarchicalGraph({self.graph_nb})"
        self.cluster : HierarchicalCluster = h_cluster
        self.dict_HN = dict()  #  name -> HN
        self.list_HCNs = []  #  toposorted
        self.list_HANs = []  #  including interface HANs
        self.input_data_HANs = set()  # HAN set -> inputs' data
        self.output_data_HANs = set()  # HAN set -> outputs' data
        self.output_grad_HANs = set()  # HAN set -> outputs' grad
        self.input_grad_HANs = set()  # HAN set -> inputs' grad
        self.loss_hcn = None

        # == Useful dictionaries to build interfaces and edges ==
        dict_mt_to_data_han = dict()
        dict_mt_to_grad_han = dict()
        dict_han_to_anode : dict[HierarchicalAllocationNode, AllocationNode] = dict()
        # A han represents exactly one anode
        dict_cnode_to_hcn : dict[ComputationNode, HierarchicalComputationNode] = dict()
        # At a fixed level of depth, a cnode can be found in only one hcn
        # Warning: the opposite isn't true: one HCN may contain multiple cnodes
        #  -> These two dict make it super easy to build edges

        # ======== PROCEED EACH NODE ONE BY ONE ========
        # For a P_node: build the corresponding HCNs/HANs (fwd/bwd + data/grad)
        for pn_to_proceed in p_graph.nodes:
            # Generate pn's sub structure if needed
            if (pn_to_proceed.is_leaf 
            and pn_to_proceed.main_target not in fb_graph.dict_bwd_cnodes):
                pn_sub_cluster = None
            else:
                pn_sub_cluster = HierarchicalCluster(fb_graph,p_node = pn_to_proceed)
            
            # There are two cases:
            # If pn_to_proceed is a leaf node, then we simply generate corresponding
            # fwd_hcn(mt), data_han(mt), bwd_hcn(mt) and grad_han(mt) (if requires_grad)
            # otherwise we have a non trivial sub cluster representing multiple nodes
            # CASE 1: bottom
            if pn_to_proceed.is_leaf:
                target_to_proceed = pn_to_proceed.main_target
                # 1) Forward Hierarchical Computation Node
                fwd_cnode = fb_graph.dict_fwd_cnodes[target_to_proceed]
                fwd_hcn = HierarchicalComputationNode(
                    fwd_cnode.name,
                    main_target = target_to_proceed,
                    cnode = fwd_cnode,
                    sub_cluster = pn_sub_cluster,
                    is_fwd = True,
                    _topological_number = fwd_cnode._topological_number)
                dict_cnode_to_hcn[fwd_cnode] = fwd_hcn

                # 2) Data Hierarchical Allocation Node
                if target_to_proceed not in dict_mt_to_data_han: # interfaces of sub clusters overlap 
                    data_anode = fb_graph.dict_data_anodes[target_to_proceed]
                    data_han = HierarchicalAllocationNode(data_anode)
                    dict_han_to_anode[data_han] = data_anode
                    dict_mt_to_data_han[target_to_proceed] = data_han

                if target_to_proceed in fb_graph.dict_bwd_cnodes:
                    # 3) Backward Hierarchical Computation Node
                    bwd_cnode = fb_graph.dict_bwd_cnodes[target_to_proceed]
                    bwd_hcn = HierarchicalComputationNode(
                        bwd_cnode.name,
                        main_target = target_to_proceed,
                        cnode = bwd_cnode,
                        sub_cluster = pn_sub_cluster,
                        is_fwd = False,
                        _topological_number = bwd_cnode._topological_number)
                    dict_cnode_to_hcn[bwd_cnode] = bwd_hcn

                    # 4) Grad Hierarchical Allocation Node
                    if target_to_proceed not in dict_mt_to_grad_han: # interfaces of sub clusters overlap
                        grad_anode = fb_graph.dict_grad_anodes[target_to_proceed]
                        grad_han = HierarchicalAllocationNode(grad_anode)
                        dict_han_to_anode[grad_han] = grad_anode
                        dict_mt_to_grad_han[target_to_proceed] = grad_han

            # CASE 2: NOT bottom
            else:
                # 1) topological_number
                # => fwd_hcn.topological_number = min cnode.topological_number for cnode inside HCN
                # => and bwd_hcn's topological_number is the max
                # Reminder: topological_number guide the toposort, so 
                # we want fwd_hcn to be place at the position of the first cnode it includes
                # and bwd_hcn to come on the last position
                fwd_hcn_num = 999999
                bwd_hcn_num = -1
                for cnode in pn_sub_cluster.list_cnodes:
                    if cnode._topological_number is None: continue
                    if cnode.is_fwd:
                        fwd_hcn_num = min(fwd_hcn_num,cnode._topological_number)
                    else:
                        bwd_hcn_num = max(bwd_hcn_num,cnode._topological_number)

                # 2) Hierarchical Computation Nodes
                fwd_hcn = HierarchicalComputationNode(
                    f"FWD[{pn_to_proceed.name}]",
                    sub_cluster = pn_sub_cluster,
                    is_fwd = True,
                    _topological_number = fwd_hcn_num)
                bwd_hcn = HierarchicalComputationNode(
                    f"BWD[{pn_to_proceed.name}]",
                    sub_cluster = pn_sub_cluster,
                    is_fwd = False,
                    _topological_number = bwd_hcn_num)
                for cnode in pn_sub_cluster.list_cnodes:
                    dict_cnode_to_hcn[cnode] = fwd_hcn if cnode.is_fwd else bwd_hcn
                    # fwd_hcn represents all the fwd cnodes inside of
                    #  the sub graph and bwd_hcn all the bwd cnodes

                # 3) Hierarchical Allocation Nodes
                # -> The interfaces, ie inputs/outputs, of the sub cluster
                for anode in pn_sub_cluster.all_interfaces:
                    han = HierarchicalAllocationNode(anode)
                    if han.is_data:
                        if anode.mt not in dict_mt_to_data_han:
                            dict_han_to_anode[han] = anode
                            dict_mt_to_data_han[han.mt] = han
                    else:
                        if anode.mt not in dict_mt_to_grad_han:
                            dict_han_to_anode[han] = anode
                            dict_mt_to_grad_han[han.mt] = han
        # ===== END OF PROCESSING EACH PN =====
        
        # 1) Loss HCN
        self.loss_hcn = HierarchicalComputationNode(
            f"Loss_HCN[Graph_{self.graph_nb}]",
            main_target = "loss"
        )

        # 2) Missing HANs
        # When we processed each pn, for non bottom ones we created
        # all the interfaces (good), but for bottom ones, we only
        # have the targets created, not the deps.
        # Hence we might miss some inputs:
        for inp_mt in h_cluster.p_cluster.inputs_mt:
            if inp_mt not in dict_mt_to_data_han:
                data_anode = fb_graph.dict_data_anodes[inp_mt]
                han_data = HierarchicalAllocationNode(data_anode)
                dict_han_to_anode[han_data] = data_anode
                dict_mt_to_data_han[han_data.mt] = data_anode
                if inp_mt in fb_graph.dict_grad_anodes:
                    grad_anode = fb_graph.dict_grad_anodes[inp_mt]
                    if grad_anode in h_cluster.all_interfaces:
                        han_grad = HierarchicalAllocationNode(grad_anode)
                        dict_han_to_anode[han_grad] = grad_anode
                        dict_mt_to_grad_han[han_grad.mt] = grad_anode

        # 3) Register the nodes and interfaces
        self.list_HANs = list(dict_han_to_anode.keys())
        self.list_HCNs = list(dict_cnode_to_hcn.values())
        self.list_HCNs.append(self.loss_hcn)
        self.dict_nodes = dict(
            [(han.name,han) for han in self.list_HANs]
          + [(hcn.name,hcn) for hcn in self.list_HCNs]
        )
        dict_anode_to_han = dict((anode,han) for (han,anode) in dict_han_to_anode.items())
        self.input_data_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["input_data_anodes"])
        self.output_data_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["output_data_anodes"])
        self.input_grad_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["input_grad_anodes"])
        self.output_grad_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["output_grad_anodes"])

        # ====== EDGES ======
        # 4) Classic edges
        # The idea is, for each 'HAN', we have the corresponding 'anode'
        # and for 'cnode' in 'anode'.deps/users, we have the 'HCN' which
        # contains 'cnode' => we transpose the edge to 'HAN'.deps/users.add('HCN)
        for han in self.list_HANs:
            anode = dict_han_to_anode[han]
            for cnode in anode.deps: # TO REMOVE:  if anode.mt != constants.init_target_string else anode.deps_global:
                if cnode is not fb_graph.loss_cnode:
                    if cnode in dict_cnode_to_hcn:
                        hcn = dict_cnode_to_hcn[cnode]
                        han.deps.add(hcn)
                        hcn.users.add(han)
            for cnode in anode.users_real: # TO REMOVE: if anode.mt != constants.init_target_string else anode.users_global:
                if cnode is not fb_graph.loss_cnode:
                    if cnode in dict_cnode_to_hcn:
                        hcn = dict_cnode_to_hcn[cnode]
                        if not (hcn in han.deps):
                            han.users.add(hcn)
                            hcn.deps.add(han)

        # 5) Loss edges
        for han in self.output_data_HANs:
            han.users.add(self.loss_hcn)
            self.loss_hcn.deps.add(han)
        for han in self.output_grad_HANs:
            han.deps.add(self.loss_hcn)
            self.loss_hcn.users.add(han)

        # 6) Artifact edges
        for cnode,hcn in dict_cnode_to_hcn.items():
            for req_via_art_cnode in cnode.deps_through_artifacts:
                if req_via_art_cnode in dict_cnode_to_hcn:
                    req_via_art_hcn = dict_cnode_to_hcn[req_via_art_cnode]
                    if req_via_art_hcn is not hcn:
                        hcn.deps_through_artifacts.add(
                            req_via_art_hcn
                        )

        self.list_HCNs = self.get_sorted_nodes_by_following_deps_relation()

        # 7) Parameter nodes
        self.hierarchical_parameter_nodes : list[HierarchicalParameterNode] = []
        for param_node in self.cluster.parameter_nodes:
            h_param_node = HierarchicalParameterNode(param_node)
            self.hierarchical_parameter_nodes.append(h_param_node)
            for cnode in param_node.users_real:
                if cnode in dict_cnode_to_hcn:
                    hcn = dict_cnode_to_hcn[cnode]
                    hcn.required_parameter_nodes_real.add(h_param_node)
                    h_param_node.users_real.add(hcn)
            for cnode in param_node.users_fake:
                if cnode in dict_cnode_to_hcn:
                    hcn = dict_cnode_to_hcn[cnode]
                    hcn.required_parameter_nodes_fake.add(h_param_node)
                    h_param_node.users_fake.add(hcn)

    # **********************************
    # == OVERWRITE base.Graph METHODS ==
    def __iter__(self):
        return iter(self.list_HCNs)
    
    def make_temporary_global_root_node_to_deps_relation(self):
        # OVERWRITE base.Graph METHOD
        fresh_root_hcn = HierarchicalComputationNode("")
        fresh_root_hcn.deps = set(self.input_grad_HANs)
        # -> Not enough, some han_grad are missing
        # -> We need all the roots of the 'deps' relation
        leaf_hcns = set()
        for hcn in self.list_HCNs:
            if not hcn.is_fwd:
                if len(hcn.get_all_standard_users())==0:
                    leaf_hcns.add(hcn)
        root_han = HierarchicalAllocationNode()
        root_han.deps = leaf_hcns
        fresh_root_hcn.deps.add(root_han)
        return fresh_root_hcn
    
    def remove_temporary_global_root_node(self, fresh_root):
        # We don't need the users relation, as we only use this
        # root_node to toposort; hence nothing to unplug
        pass

    # = print and render =
    def __str__(self):
        return self.name # TO IMPROVE
    
    @staticmethod
    def get_render_color(node):
        color_hcn_fwd = "blue"
        color_hcn_bwd = "blueviolet"
        color_special = "green"
        color_han  = "olive"
        color_parameter_node = "black"
        if isinstance(node,HierarchicalParameterNode):
            return color_parameter_node
        if node.main_target == "loss":
            return color_special
        elif isinstance(node, HierarchicalAllocationNode):
            return color_han
        elif node.is_fwd:
            return color_hcn_fwd
        else:
            return color_hcn_bwd
    
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            include_parameter_nodes=True,
            include_artifact_edges=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        color_edge = "black"

        # 1) Computation Nodes
        for hcn in self.list_HCNs:
            if hcn.main_target is not None:
                if hcn.cnode is None:
                    label = hcn.name
                else:
                    if only_function_name:
                        code = hcn.cnode.main_fct
                    else:
                        code = hcn.cnode.get_code()
                    label = f"{hcn.name}\n{code}"
            else:
                label = f"{hcn.name}\nCluster size: {hcn.sub_cluster.p_cluster.size}"
            dot.node(hcn.name,label,color=self.__class__.get_render_color(hcn))
        # 2) Allocation Nodes
        for han in self.list_HANs:
            dot.node(han.name,han.name,
                color=self.__class__.get_render_color(han),
                tooltip=f"Mem {pretty_format_memory(han.mem)}")

        # 3) Edges
        for hcn in self.list_HCNs:
            for req_han in hcn.deps:
                dot.edge(req_han.name, hcn.name, color=color_edge)
            for user_han in hcn.users:
                dot.edge(hcn.name, user_han.name, color=color_edge)
            if include_artifact_edges:
                for req_han in hcn.deps_through_artifacts:
                    dot.edge(req_han.name,hcn.name, color=color_edge, style ="dotted")

        # 4) Parameter nodes
        if include_parameter_nodes:
            for h_param_node in self.hierarchical_parameter_nodes:
                if h_param_node.view_targets == []:
                    label = h_param_node.param_str
                elif only_function_name:
                    label = "\n".join(
                        [h_param_node.param_str]+h_param_node.view_targets)
                else:
                    label = f"{h_param_node.param_str}\n{h_param_node.get_code()}"
                dot.node(
                    h_param_node.param_str,
                    label,
                    color = self.__class__.get_render_color(h_param_node),
                    style = "dashed")
                for user_hcn in h_param_node.users_real:
                    dot.edge(h_param_node.param_str,user_hcn.name)
                for user_hcn in h_param_node.users_fake:
                    dot.edge(h_param_node.param_str,user_hcn.name,style="dashed")

        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )



# ***********************
# * HierarchicalCluster *
# ***********************
class HierarchicalCluster():
    is_bottom : bool = None
    # Warning: A bottom cluster is composed of ComputationNodes
    # and AllocationNodes from backward.py; not Hierarchical ones
    list_cnodes : list[ComputationNode] = None
    list_anodes : list[AllocationNode] = None
    loss_cnode : ComputationNode = None
    loss_idx : int = None
    dict_nodes : dict = None
    interfaces : dict[str, set[AllocationNode]]
    parameter_nodes : list[bwdParameterNode] = None
    translator : anonymize.ClusterTranslator = None
    p_cluster : PartitionedCluster = None
    p_node : PartitionedNode = None
    representee_cluster = None # : HierarchicalCluster
    partitionings : list[HierarchicalGraph] = None
    list_schedules = None #: list[Op_sched] LATER 

    def __init__(self,
            fb_graph : ForwardAndBackwardGraph,
            p_cluster : PartitionedCluster = None,
            p_node : PartitionedNode = None):
        """ Two constructors: one from PartitionedNode the other from PartitionedCluster
        """
        if (p_cluster is None) == (p_node is None):
            raise Exception(
                "HierarchicalCluster.__init__ takes either a PartitionedCluster "\
                "or a PartitionedNode (one or the other, not both)")
        if p_node is not None and p_node.sub_cluster is not None:
            p_cluster = p_node.sub_cluster
        # ======================================================
        # ====== FIRST H_CLUSTER CONSTRUCTOR: FROM P_NODE ======
        if p_cluster is None:
            target_to_proceed = p_node.main_target
            assert(target_to_proceed in fb_graph.dict_bwd_cnodes)
            self.is_bottom = True
            self.p_node = p_node
            self.main_target = target_to_proceed
            self.representee_cluster = self
            self.name = f"BottomHCluster({target_to_proceed})"

            # Warning: A bottom cluster is composed of ComputationNodes
            # and AllocationNodes from backward.py; not Hierarchical ones
            # 1) target_to_proceed's nodes
            data_anode = fb_graph.dict_data_anodes[target_to_proceed]
            grad_anode = fb_graph.dict_grad_anodes[target_to_proceed]
            fwd_cnode = fb_graph.dict_fwd_cnodes[target_to_proceed]
            bwd_cnode = fb_graph.dict_bwd_cnodes[target_to_proceed]

            # 2) Loss Computation Node
            self.loss_cnode = ComputationNode("loss")
            self.loss_cnode.deps_real = {data_anode}
            self.loss_cnode.users = {grad_anode}
            self.list_cnodes = [fwd_cnode,self.loss_cnode,bwd_cnode]
            self.loss_idx = 1

            # 3) Additional AllocationNodes: deps/users and phantoms
            all_anodes = set([data_anode,grad_anode])
            if target_to_proceed in fb_graph.dict_phantoms_anodes:
                all_anodes.add(fb_graph.dict_phantoms_anodes[target_to_proceed])
            input_data_anodes = set(fwd_cnode.deps_real)
            input_grad_anodes = set(bwd_cnode.users)

            # 4) In case user of global source node
            if fwd_cnode in fb_graph.source_data_anode.users_real:
                input_data_anodes.add(fb_graph.source_data_anode)
            if fb_graph.source_grad_anode in bwd_cnode.users:
                input_grad_anodes.add(fb_graph.source_grad_anode)
            all_anodes.update(input_data_anodes)
            all_anodes.update(input_grad_anodes)
            self.list_anodes = list(all_anodes)

            # 5) Parameter nodes
            self.parameter_nodes = list(fwd_cnode.required_parameter_nodes_real) # fake is empty, and bwd_cnode.params \included in fwd.params

            # 6) useful attributes
            self.make_dict_nodes()
            self.interfaces = {
                "input_data_anodes" : input_data_anodes,
                "output_data_anodes" : {data_anode},
                "output_grad_anodes" : {grad_anode},
                "input_grad_anodes"  : input_grad_anodes
            }
        # ================ END FIRST CONSTRUCTOR ===================


        # ==========================================================
        # ====== SECOND H_CLUSTER CONSTRUCTOR: FROM P_CLUSTER ======
        else:
            self.is_bottom = False
            self.p_cluster = p_cluster
            p_cluster.h_cluster = self
            self.cluster_nb = p_cluster.cluster_nb
            self.ano_cluster_id = p_cluster.ano_cluster_id
            self.name = f"H_Cluster_{self.cluster_nb}_Ano_id_{self.ano_cluster_id}"

            all_main_targets_contained = set(sn.main_target for sn in p_cluster.s_nodes)
            # 1) Collect all the ComputationNodes
            # Reminder we are talking about backward.py's nodes
            # not HierarchicalComputationNodes
            self.list_cnodes = []
            self.loss_cnode = ComputationNode("loss")
            # Note: we collect them in the topological order
            for cnode in fb_graph.computation_nodes:
                if cnode.main_target in all_main_targets_contained:
                    self.list_cnodes.append(cnode)
                elif cnode == fb_graph.loss_cnode:
                    self.loss_idx = len(self.list_cnodes)
                    self.list_cnodes.append(self.loss_cnode)
                    # => We substitute the original Loss Node by the new one

            # 2) AllocationNodes
            all_anodes = set(
                anode for anode in fb_graph.allocation_nodes
                if anode.main_target in all_main_targets_contained)
            self.interfaces = dict()
            self.interfaces["input_data_anodes" ] = input_data_anodes  = set()
            self.interfaces["output_data_anodes"] = output_data_anodes = set()
            self.interfaces["output_grad_anodes"] = output_grad_anodes = set()
            self.interfaces["input_grad_anodes" ] = input_grad_anodes  = set()

            # 3) Inputs
            for input_mt in p_cluster.inputs_mt:
                # Input data
                input_data = fb_graph.dict_data_anodes[input_mt]
                input_data_anodes.add(input_data)
                all_anodes.add(input_data)
                # Input grad
                if input_mt in fb_graph.dict_grad_anodes:
                    input_grad = fb_graph.dict_grad_anodes[input_mt]
                    input_grad_anodes.add(input_grad)
                    all_anodes.add(input_grad)

            # 4) Outputs
            # => We plug the new fresh Loss_cnode to outputs'data/grad
            # but we don't pollute output nodes (as they belong to fb_graph)
            # => not reciprocal edges
            for output_mt in p_cluster.outputs_mt:
                output_data = fb_graph.dict_data_anodes[output_mt]
                output_data_anodes.add(output_data)
                self.loss_cnode.deps_real.add(output_data)
                # output_data/grad_anode is already in all_anodes
                # as output_mt is in all_main_targets_contained
                if output_mt in fb_graph.dict_grad_anodes:
                    output_grad = fb_graph.dict_grad_anodes[output_mt]
                    output_grad_anodes.add(output_grad)
                    self.loss_cnode.users.add(output_grad)

            self.list_anodes = list(all_anodes)
            self.make_dict_nodes()

            # 5) Parameter Nodes
            self.parameter_nodes = list(set().union(
                *[cnode.required_parameter_nodes_real # fake is included in the rest
                  for cnode in self.list_cnodes]))

            # 6) Translator
            self.translator = p_cluster.translator
            self.translator.enrich_with_cnodes_and_anodes(self)

            # 7) Partitionings
            if p_cluster is p_cluster.representee_cluster:
                self.representee_cluster = self
                self.partitionings = []
                for pg in p_cluster.partitionings:
                    self.partitionings.append(
                        HierarchicalGraph(self,pg,fb_graph)
                    )
            else:
                # Not representee of its equivalent class
                p_representee = p_cluster.representee_cluster
                if p_representee.h_cluster is not None:
                    self.representee_cluster = p_representee.h_cluster
                else:
                    h_representee = HierarchicalCluster(fb_graph,p_cluster = p_representee)
                    self.representee_cluster = h_representee
                    # 7) It's important to have equivalent list_anodes order
                    assert(len(h_representee.list_anodes) == len(self.list_anodes))
                    old_self_list_anodes = self.list_anodes
                    representee_anonymized_list_anodes = [
                        h_representee.translator.to_ano(anode) # translate
                        for anode in h_representee.list_anodes
                    ]
                    self.list_anodes = [
                        self.translator.from_ano(ano) # reverse translate
                        for ano in representee_anonymized_list_anodes
                    ]
                    assert(set(old_self_list_anodes) == set(self.list_anodes))
        # ================ END SECOND CONSTRUCTOR ==================


    def make_dict_nodes(self):
        self.dict_nodes = dict(
            [(anode.name,anode) for anode in self.list_anodes]
          + [(cnode.name,cnode) for cnode in self.list_cnodes]
        )

    @property
    def all_interfaces(self):
        return set().union(*self.interfaces.values())
    
    def __str__(self):
        return self.name # TO IMPROVE
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            include_parameter_nodes=True,
            include_artifact_edges=True,
            no_message = False,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        cluster = self.representee_cluster
        if not no_message:
            if len(cluster.partitionings) == 0:
                print("Sorry your cluster doesn't have any partitioning, "\
                    "ie corresponding Hierarchical Graph.")
            if cluster is not self:
                print(
                    f"Warning : your cluster has some equivalent ones, "\
                    f"{cluster.name} is the representee of the equivalent class, "\
                    f"so we render its graphs")
        for i,hg in enumerate(cluster.partitionings):
            if name is not None:
                graph_name = f"{i}-th partitioning of {name}"
            else:
                graph_name = None
            hg.render(graph_name,
                view,
                only_function_name,
                include_parameter_nodes,
                include_artifact_edges,
                directory,
                render_format,
                render,
                dot)





class HierarchicalStructure():
    def __init__(self,
            partitioned_structure : PartitionedStructure,
            forward_and_backward_graph : ForwardAndBackwardGraph):
        self.dict_info = partitioned_structure.dict_info
        # Build all the clusters:
        self.main_cluster = HierarchicalCluster(
            p_cluster = partitioned_structure.main_cluster,
            fb_graph = forward_and_backward_graph)
        # Secondary attributes
        self.make_all_targets_like_attributes(partitioned_structure)
        for cluster in self.all_unique_clusters: cluster.list_schedules = []
        for cluster in self.all_bottom_clusters: cluster.list_schedules = []


    def make_all_targets_like_attributes(self,partitioned_structure):
        # Collect all clusters and graphs
        self.all_clusters = set(
            p_cluster.h_cluster 
            for p_cluster in partitioned_structure.all_clusters)
        self.all_unique_clusters = set(
            p_cluster.h_cluster 
            for p_cluster in partitioned_structure.all_unique_clusters)
        self.all_unique_graphs = set().union(
            *[set(h_cluster.partitionings) 
            for h_cluster in self.all_unique_clusters])
        # Dict to easily find them based on cluster_nb
        self.dict_cluster_nb_to_cluster = dict(
            (h_cluster.cluster_nb,h_cluster)
            for h_cluster in self.all_clusters
        )
        # Collect bottom ones
        self.all_bottom_clusters = set()
        for hg in self.all_unique_graphs:
            for hcn in hg.list_HCNs:
                if hcn.sub_cluster is not None and hcn.sub_cluster.is_bottom:
                    self.all_bottom_clusters.add(hcn.sub_cluster)
        
        self.dict_mt_to_bottom_cluster = dict(
            (h_cluster.main_target,h_cluster)
            for h_cluster in self.all_bottom_clusters
        )
        self.dict_nb_to_bottom_cluster = dict(
            (base.Node.get_num_tar(h_cluster.main_target),h_cluster)
            for h_cluster in self.all_bottom_clusters
        )
        

    # ***************************************************
    # == Methods to get some insights of the structure ==
    def biggest_sub_graph(self):
        return max(
            self.all_unique_graphs,
            key = lambda hg : len(hg.list_HCNs)
        )
    
    def max_depth(self):
        def explore(cluster : HierarchicalCluster,depth):
            max_depth = depth
            if cluster.representee_cluster is cluster:
                for hg in cluster.partitionings:
                    for hn in hg.nodes:
                        if hn.sub_cluster is not None:
                            max_depth = max(max_depth,
                                explore(hn.sub_cluster,depth+1))
            return max_depth
        return explore(self.main_cluster,0)
    
    def get_insights_of_the_structure(self):
        print("Insights of the Hierarchical Structure :")
        print(f"- A total of {len(self.all_clusters)} clusters")
        print(f"- with {len(self.all_unique_clusters)} unique clusters")
        print(f"- A maximum depth of {self.max_depth()} "\
              "(0 means there is no partitioning, i.e. a flat structure)")
        biggest_hg = self.biggest_sub_graph()
        print("The biggest sub graph has:")
        print(f"- A total of {len(biggest_hg.list_HCNs)} computations")
        print(f"- {len([hcn for hcn in biggest_hg.list_HCNs if hcn.is_fwd])} of which are forward computations")
        print(f"- A total of {len(biggest_hg.list_HCNs)} allocations")

    def __str__(self): # TO IMPROVE
        return "HierarchicalStructure whose main cluster is:\n"+str(self.main_cluster)
    
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            include_parameter_nodes=True,
            include_artifact_edges=True,
            no_message=False,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        self.main_cluster.render(
            name,view,only_function_name,
            include_parameter_nodes,
            include_artifact_edges,
            no_message,
            directory,render_format,
            render,dot
        )

