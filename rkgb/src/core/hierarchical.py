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
    list_cnodes : list[ComputationNode] = None
    list_anodes : list[AllocationNode] = None
    loss_cnode : ComputationNode = None
    loss_idx : int = None
    dict_kn : dict = None
    interfaces : dict[str, set[AllocationNode]]
    translator : anonymize.ClusterTranslator = None
    p_cluster : PartitionedCluster = None
    p_node : PartitionedNode = None
    representee_cluster = None # : HierarchicalCluster
    partitionings : list[HierarchicalGraph] = None
    list_sched = None #: list[Op_sched] LATER 

    def __init__(self,name,is_bottom):
        self.name = name
        self.loss_cnode = ComputationNode("loss")
        self.is_bottom = is_bottom

    def make_dict_kn(self):
        self.dict_kn = dict(
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

        


def PartitionedCluster_to_HierarchicalCluster(p_cluster : PartitionedCluster, kg : ForwardAndBackwardGraph):
    if hasattr(p_cluster,"h_cluster"): return p_cluster.h_cluster
    h_cluster = HierarchicalCluster(p_cluster.name.replace("P","H"),False)
    h_cluster.p_cluster = p_cluster
    p_cluster.h_cluster = h_cluster
    all_computation_mt = set(sn.mt for sn in p_cluster.s_nodes)

    # Collect list_cnode and add the fresh loss_cnode along the way
    loss_cnode = h_cluster.loss_cnode
    list_cnode = h_cluster.list_cnode = []
    for cnode in kg.list_cnode:
        if cnode.mt in all_computation_mt:
            list_cnode.append(cnode)
        if cnode == kg.loss_cnode:
            h_cluster.loss_idx = len(list_cnode)
            list_cnode.append(loss_cnode)

    set_anode = set(anode for anode in kg.list_anode if anode.mt in all_computation_mt)
    # `set` because we need to add the inputs_anode

    h_cluster.interfaces = interfaces = dict()
    interfaces["inputs_anode_data" ] = inputs_anode_data  = set()
    interfaces["outputs_anode_data"] = outputs_anode_data = set()
    interfaces["inputs_anode_grad" ] = inputs_anode_grad  = set()
    interfaces["outputs_anode_grad"] = outputs_anode_grad = set()

    # ** inputs **
    for input_mt in p_cluster.inputs_mt:
        input_data = kg.dict_anode_data[input_mt]
        inputs_anode_data.add(input_data)
        set_anode.add(input_data)
        if input_mt in kg.dict_anode_grad:
            input_grad = kg.dict_anode_grad[input_mt]
            if input_grad.deps_global != set():
                inputs_anode_grad.add(input_grad)
                set_anode.add(input_grad)

    # ** outputs **
    for output_mt in p_cluster.outputs_mt:
        output_data = kg.dict_anode_data[output_mt]
        outputs_anode_data.add(output_data)
        loss_cnode.deps_real.add(output_data)
        if output_mt in kg.dict_anode_grad:   
            output_grad = kg.dict_anode_grad[output_mt]
            outputs_anode_grad.add(output_grad)
            loss_cnode.users.add(output_grad)

    h_cluster.list_anode = list(set_anode)
    h_cluster.make_dict_kn()

    # ** translator **
    h_cluster.translator = translator = p_cluster.translator
    dict_mt_to_ano = translator.dict_mt_to_ano_pair
    dict_cnode_to_ano = translator.dict_cnode_to_ano_triplet = dict()
    dict_anode_to_ano = translator.dict_anode_to_ano_triplet = dict()
    dict_ano_to_cnode = translator.dict_ano_triplet_to_cnode = dict()
    dict_ano_to_anode = translator.dict_ano_triplet_to_anode = dict()
    for cnode in h_cluster.list_cnode:
        if cnode is loss_cnode:
            ano_triplet = ("loss",0,0)
        else:
            ano_pair = dict_mt_to_ano[cnode.mt]
            ano_triplet = ("fwd" if cnode.is_fwd else "bwd",) + ano_pair
        dict_cnode_to_ano[cnode] = ano_triplet
        dict_ano_to_cnode[ano_triplet] = cnode
    for anode in h_cluster.list_anode:
        ano_pair = dict_mt_to_ano[anode.mt]
        ano_triplet = (anode.allocation_type,) + ano_pair
        dict_anode_to_ano[anode] = ano_triplet
        dict_ano_to_anode[ano_triplet] = anode
    translator.dict_name_to_ano_triplet = dict(
        [(kn.name,ano) for (kn,ano) in dict_cnode_to_ano.items()]
    +   [(kn.name,ano) for (kn,ano) in dict_anode_to_ano.items()]
    )
    translator.dict_ano_triplet_to_name = dict(
        (anode,han) for (han,anode) in translator.dict_name_to_ano_triplet.items()
    )

    # ** partitionings and representee **
    if p_cluster is p_cluster.representee_cluster:
        h_cluster.representee_cluster = h_cluster
        h_cluster.partitionings = partitionings = []
        h_cluster.list_sched = []
        for pg in p_cluster.possible_partitioning:
            hg = PartitionedGraph_to_HierarchicalGraph(pg,h_cluster,kg)
            partitionings.append(hg)
    else:
        h_cluster.representee_cluster \
            = representee \
            = PartitionedCluster_to_HierarchicalCluster(p_cluster.representee_cluster,kg)
        # === make list_anode order match ===
        assert(len(representee.list_anode) == len(h_cluster.list_anode))
        old_list = h_cluster.list_anode
        ano_list_anode = [representee.translator.dict_anode_to_ano_triplet[anode] for anode in representee.list_anode]
        h_cluster.list_anode = [h_cluster.translator.dict_ano_triplet_to_anode[ano] for ano in ano_list_anode]
        assert(set(old_list) == set(h_cluster.list_anode))
        

    return h_cluster


def PartitionedNode_to_HierarchicalCluster(pn : PartitionedNode, kg : ForwardAndBackwardGraph):
    if pn.sub_cluster is not None:
        return PartitionedCluster_to_HierarchicalCluster(pn.sub_cluster, kg)
    elif pn.mt not in kg.dict_cnode_bwd:
        return None
    else:
        h_cluster = HierarchicalCluster(f"H_Cluster_bottom_{pn.mt}",True)
        h_cluster.p_node = pn
        h_cluster.list_sched = []
        h_cluster.representee_cluster = h_cluster
        loss_cnode = h_cluster.loss_cnode
        mt = pn.mt
        # -- list_cnode part --
        cnode_fwd : ComputationNode = kg.dict_cnode_fwd[mt]
        cnode_bwd : ComputationNode = kg.dict_cnode_bwd[mt]
        anode_data = kg.dict_anode_data[mt]
        anode_grad = kg.dict_anode_grad[mt]
        loss_cnode.deps_real = set([anode_data])
        loss_cnode.users = set([anode_grad])
        h_cluster.list_cnode = [cnode_fwd,loss_cnode,cnode_bwd]
        h_cluster.loss_idx = 1

        # -- list_anode part --
        set_anode = set([anode_data,anode_grad])
        if mt in kg.dict_anode_phantoms:
            set_anode.add(kg.dict_anode_phantoms[mt])
        inputs_data = set(
            req_anode for req_anode \
            in (cnode_fwd.deps_global - cnode_fwd.deps_fake))
        inputs_grad = set(user_anode for user_anode in cnode_bwd.users_global)
        set_anode.update(inputs_data)
        set_anode.update(inputs_grad)
        h_cluster.list_anode = list(set_anode)
        h_cluster.make_dict_kn()

        h_cluster.interfaces = {
            "inputs_anode_data"  : inputs_data,
            "outputs_anode_data" : set([anode_data]),
            "outputs_anode_grad" : set([anode_grad]),
            "inputs_anode_grad"  : inputs_grad
        }

        return h_cluster




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
    hg.input_data_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["inputs_anode_data"])
    hg.output_data_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["outputs_anode_data"])
    hg.input_grad_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["inputs_anode_grad"])
    hg.output_grad_HANs = set(dict_anode_to_han[anode] for anode in h_cluster.interfaces["outputs_anode_grad"])
    
    
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

