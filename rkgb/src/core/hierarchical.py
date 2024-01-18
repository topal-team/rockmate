# ==========================
# ====== H structure =======
# ==========================

from src.lowlevel import constants
from src.lowlevel.variable_info import VariableInfo
from src.lowlevel import anonymize
from src.core import base
from src.core.backward import ForwardAndBackwardGraph, ComputationNode, AllocationNode
from src.core.backward import ParameterNode as bwdParameterNode
from src.core.partitioned import PartitionedStructure, PartitionedCluster, PartitionedGraph, PartitionedNode

# ************
# * HierarchicalComputationNode *
# ************
class HierarchicalComputationNode(base.Node):
    def __init__(self,
            name,
            main_target = None,
            info : VariableInfo = None, # TO REMOVE USELESS ??
            sub_cluster = None,
            is_fwd = True,
            _topological_number = -1,
            hierarchical_graph = None):
        super().__init__(main_target,
            parent_structure_with_id_generator=hierarchical_graph)
        self.name = name  # e.g. Fwd_1
        self.info = info
        self.sub_cluster = sub_cluster
        self.is_fwd = is_fwd
        self.is_leaf = bool(main_target is None)
        self.deps = set()  # HAN set
        self.users = set()  # HAN set
        self.deps_through_artifacts = set()  # HCN set
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
    def __init__(self, name, h_cluster = None):
        super().__init__()
        # /!\ All the HCN's and HAN's should be in the same level. /!\ 
        self.name = name
        self.dict_nodes = dict()  #  name -> HN
        self.list_HCNs = []  #  toposorted
        self.list_HANs = []  #  including interface HANs
        self.input_data_HANs = set()  # HAN set -> inputs' data
        self.output_data_HANs = set()  # HAN set -> outputs' data
        self.output_grad_HANs = set()  # HAN set -> outputs' grad
        self.input_grad_HANs = set()  # HAN set -> inputs' grad
        self.loss_hcn = None
        if h_cluster is not None:
            self.cluster = h_cluster
            self.all_clusters = set([h_cluster])
        else:
            self.cluster = None
            self.all_clusters = set()

    def make_users(self):
        for hn in self.dict_nodes.values():
            for req_hn in hn.deps:
                req_hn.users.add(hn)

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



# *************
# * HierarchicalCluster *
# *************
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
        if p_node is not None:
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
            all_anodes.update(input_data_anodes)
            all_anodes.update(input_grad_anodes)

            # 4) In case user of global source node
            if fwd_cnode in fb_graph.source_data_anode.users_real:
                input_data_anodes.add(fb_graph.source_data_anode)
            if fb_graph.source_grad_anode in bwd_cnode.users:
                input_grad_anodes.add(fb_graph.source_grad_anode)
            self.list_anodes = list(all_anodes)

            # 5) useful attributes
            self.make_dict_nodes()
            self.interfaces = {
                "input_data_anodes" : input_data_anodes,
                "output_data_anodes" : {data_anode},
                "output_grad_anodes" : {grad_anode},
                "input_grad_anodes"  : input_grad_anodes
            }


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
            # 5) Translator
            self.translator = p_cluster.translator
            self.translator.enrich_with_cnodes_and_anodes(self)

            # 6) Partitionings
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









    def make_dict_nodes(self):
        self.dict_nodes = dict(
            [(anode.name,anode) for anode in self.list_anodes]
          + [(cnode.name,cnode) for cnode in self.list_cnodes]
        )

    @property
    def all_interfaces(self):
        return set().union(*self.interfaces.values())
    



class HierarchicalStructure():
    def __init__(self,
            partitioned_structure : PartitionedStructure,
            forward_and_backward_graph : ForwardAndBackwardGraph):
        self.dict_info = partitioned_structure.dict_info
        # Build all the clusters:
        self.main_cluster = HierarchicalCluster(
            p_cluster = partitioned_structure.main_cluster,
            fb_graph = forward_and_backward_graph)
        # Useful attributes :
        self.dict_cluster_nb_to_cluster = dict(
            (cluster_nb,p_cluster.h_cluster)
            for (cluster_nb,p_cluster)
            in partitioned_structure.dict_cluster_nb_to_cluster.items()
        )
        self.all_clusters = set(
            p_cluster.h_cluster 
            for p_cluster in partitioned_structure.all_clusters)
        self.all_unique_clusters = set(
            p_cluster.h_cluster 
            for p_cluster in partitioned_structure.all_unique_clusters)
        self.all_unique_graphs = set().union(
            *[set(h_cluster.partitionings) 
            for h_cluster in self.all_unique_clusters])
        self.all_bottom_clusters = set()
        for hg in self.all_unique_graphs:
            for hcn in hg.list_HCNs:
                if hcn.sub_cluster is not None and hcn.sub_cluster.is_bottom:
                    self.all_bottom_clusters.add(hcn.sub_cluster)
        for cluster in self.all_unique_clusters: cluster.list_schedules = []
        for cluster in self.all_bottom_clusters: cluster.list_schedules = []
        

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

        


def PartitionedGraph_to_HierarchicalGraph(
        pg : PartitionedGraph, 
        h_cluster : HierarchicalCluster,
        kg : ForwardAndBackwardGraph):
    hg = HierarchicalGraph(pg.name.replace("pg","hg"),h_cluster)

    # -> Useful for interfaces
    dict_mt_to_han_data = dict()
    dict_mt_to_han_grad = dict()

    dict_han_to_anode : dict[HierarchicalAllocationNode, AllocationNode] = dict()
    #  A han represents exactly one anode
    dict_cnode_to_hcn : dict[ComputationNode, HierarchicalComputationNode] = dict()
    #  At a fixed level of depth, a cnode can be found in only one hcn
    #  -> These two dict make it super easy to build edges

    for pn in pg.nodes:
        sub_cluster = PartitionedNode_to_HierarchicalCluster(pn,kg)
        if pn.is_leaf:
            # === Bottom level ===
            mt = pn.mt
            # ** HCN_fwd **
            cnode_fwd = kg.dict_cnode_fwd[mt]
            hcn_fwd = HierarchicalComputationNode(
                cnode_fwd.name,
                main_target = mt,
                info = cnode_fwd.info,
                sub_cluster = sub_cluster, # = None if no grad
                is_fwd = True,
                _topological_number = cnode_fwd._topological_number)
            dict_cnode_to_hcn[cnode_fwd] = hcn_fwd

            # ** HAN_data **
            if mt not in dict_mt_to_han_data: # interfaces of sub clusters overlap
                anode_data = kg.dict_anode_data[mt]
                han_data = HierarchicalAllocationNode(anode_data)
                dict_han_to_anode[han_data] = anode_data
                dict_mt_to_han_data[mt] = han_data
            
            if mt in kg.dict_cnode_bwd:
                # ** HCN_bwd **
                cnode_bwd = kg.dict_cnode_bwd[mt]
                hcn_bwd = HierarchicalComputationNode(
                    cnode_bwd.name,
                    main_target = mt,
                    info = cnode_bwd.info,
                    sub_cluster = sub_cluster,
                    is_fwd = False,
                    _topological_number = cnode_bwd._topological_number)
                dict_cnode_to_hcn[cnode_bwd] = hcn_bwd

                # ** HAN_grad **
                if mt not in dict_mt_to_han_grad: # interfaces of sub clusters overlap
                    anode_grad = kg.dict_anode_grad[mt]
                    han_grad = HierarchicalAllocationNode(anode_grad)
                    dict_han_to_anode[han_grad] = anode_grad
                    dict_mt_to_han_grad[mt] = han_grad

        else:
            # === NOT bottom ===
            # ** HCN_fwd and bwd **
            hcn_fwd_num = 999999
            hcn_bwd_num = -1
            for cnode in sub_cluster.list_cnode:
                if not hasattr(cnode,"_topological_number"): continue
                if cnode.is_fwd:
                    hcn_fwd_num = min(hcn_fwd_num,cnode._topological_number)
                else:
                    hcn_bwd_num = max(hcn_bwd_num,cnode._topological_number)
            hcn_fwd = HierarchicalComputationNode(
                f"fwd_{pn.name}",
                sub_cluster = sub_cluster,
                is_fwd = True,
                _topological_number = hcn_fwd_num)
            hcn_bwd = HierarchicalComputationNode(
                f"bwd_{pn.name}",
                sub_cluster = sub_cluster,
                is_fwd = False,
                _topological_number = hcn_bwd_num)
            for cnode in sub_cluster.list_cnode:
                dict_cnode_to_hcn[cnode] = hcn_fwd if cnode.is_fwd else hcn_bwd

            # ** HANs **
            all_interfaces = sub_cluster.all_interfaces
            for anode in all_interfaces:
                han = HierarchicalAllocationNode(anode)
                if han.is_data:
                    if anode.mt not in dict_mt_to_han_data:
                        dict_han_to_anode[han] = anode
                        dict_mt_to_han_data[han.mt] = han
                else:
                    if anode.mt not in dict_mt_to_han_grad:
                        dict_han_to_anode[han] = anode
                        dict_mt_to_han_grad[han.mt] = han

    # ** loss_hcn **
    hg.loss_hcn = loss_hcn = HierarchicalComputationNode(
        f"Loss_hcn_of_{hg.name}",
        main_target = "loss"
        )
    
    # ** missing HierarchicalAllocationNodes -> inputs of bottom sub nodes**
    for inp_mt in h_cluster.p_cluster.inputs_mt:
        if inp_mt not in dict_mt_to_han_data:
            anode_data = kg.dict_anode_data[inp_mt]
            han_data = HierarchicalAllocationNode(anode_data)
            dict_han_to_anode[han_data] = anode_data
            dict_mt_to_han_data[han_data.mt] = anode_data
            if inp_mt in kg.dict_anode_grad:
                anode_grad = kg.dict_anode_grad[inp_mt]
                if anode_grad in h_cluster.all_interfaces:
                    han_grad = HierarchicalAllocationNode(anode_grad)
                    dict_han_to_anode[han_grad] = anode_grad
                    dict_mt_to_han_grad[han_grad.mt] = anode_grad
    
    # ** register nodes **
    hg.list_han = list(dict_han_to_anode.keys())
    hg.list_hcn = list(dict_cnode_to_hcn.values())
    hg.list_hcn.append(loss_hcn)
    hg.dict_nodes = dict(
        [(han.name,han) for han in hg.list_han]
      + [(hcn.name,hcn) for hcn in hg.list_hcn]
    )

    # ** interfaces **
    dict_anode_to_han = dict((anode,han) for (han,anode) in dict_han_to_anode.items())
    hg.input_data_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["input_data_anodes"])
    hg.output_data_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["output_data_anodes"])
    hg.input_grad_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["input_grad_anodes"])
    hg.output_grad_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["output_grad_anodes"])
    
    
    # === Build the edges ===
    for han in hg.list_han:
        anode = dict_han_to_anode[han]
        for cnode in anode.deps if anode.mt != constants.init_target_string else anode.deps_global:
            if cnode is not kg.loss_cnode:
                if cnode in dict_cnode_to_hcn:
                    hcn = dict_cnode_to_hcn[cnode]
                    han.deps.add(hcn)
                    hcn.users.add(han)
        for cnode in anode.users_real if anode.mt != constants.init_target_string else anode.users_global:
            if cnode is not kg.loss_cnode:
                if cnode in dict_cnode_to_hcn:
                    hcn = dict_cnode_to_hcn[cnode]
                    if not (hcn in han.deps):
                        han.users.add(hcn)
                        hcn.deps.add(han)

    # -> loss edges
    for han in hg.output_data_HANs:
        han.users.add(loss_hcn)
        loss_hcn.deps.add(han)
    for han in hg.output_grad_HANs:
        han.deps.add(loss_hcn)
        loss_hcn.users.add(han)

    # -> artifacts edges
    for cnode,hcn in dict_cnode_to_hcn.items():
        for req_via_art_cnode in cnode.deps_through_artifacts:
            if req_via_art_cnode in dict_cnode_to_hcn:
                req_via_art_hcn = dict_cnode_to_hcn[req_via_art_cnode]
                if req_via_art_hcn is not hcn:
                    hcn.deps_through_artifacts.add(
                        req_via_art_hcn
                    )

    hg.list_HCNs = hg.get_sorted_nodes_by_following_deps_relation()
    return hg








# ==========================
# === printing functions ===
# ==========================

color_hcn_fwd = "blue"
color_hcn_bwd = "blueviolet"
color_special = "green"
color_han = "olive"
color_edge = "black"


def get_color(hn):
    if hn.main_target == "loss":
        return color_special
    elif isinstance(hn, HierarchicalAllocationNode):
        return color_han
    elif hn.is_fwd:
        return color_hcn_fwd
    else:
        return color_hcn_bwd

def aux_print_HierarchicalGraph_message(hg : HierarchicalGraph):
    return (
        f"HierarchicalGraph - Hierarchical forward+backward graph, "\
        f"{len(hg.list_hcn)} HierarchicalComputationNodes; {len(hg.list_han)} HierarchicalAllocationNodes"
    )
def aux_print_HierarchicalGraph_name(hg,name=None):
    if name is not None: return name
    else: return "Hierarchical_HierarchicalGraph"

def aux_print_HierarchicalCluster_message(hc : HierarchicalCluster):
    partitionings = hc.representee_cluster.partitionings
    return f"{hc.name}, with {len(partitionings)} possible HierarchicalGraphs"
def aux_print_HierarchicalCluster_names(hc : HierarchicalCluster,name=None):
    if name is None: name = hc.name
    nb_hg = len(hc.representee_cluster.partitionings)
    return [f"HierarchicalGraph_{i}_of_{name}" for i in range(nb_hg)]

def print_HierarchicalGraph(hg: HierarchicalGraph, name=None, open=True, render_format="svg",dot=None,uniq_num=0):
    # ----- init -----
    if dot is None:
        render = True
        if name is None: name = aux_print_HierarchicalGraph_name(hg)
        dot = graphviz.Digraph(name,comment=name)
    else:
        render = False
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,**kwargs): dot.edge(uni(i1),uni(i2), **kwargs)
    # ----- Core -----
    # * nodes *
    for hcn in hg.list_hcn:
        node(hcn.name,hcn.name,color=get_color(hcn))
    for han in hg.list_han:
        node(han.name,han.name,
            color=get_color(han),
            tooltip=f"Mem {irotor.MemSize(han.mem)}")

    # * edges *
    for hcn in hg.list_hcn:
        for req_han in hcn.deps:
            edge(req_han.name, hcn.name, color=color_edge)
        for user_han in hcn.users:
            edge(hcn.name, user_han.name, color=color_edge)

    #  ----- render -----
    if render:
        small_fcts.graph_render(dot, open, "H", render_format)

