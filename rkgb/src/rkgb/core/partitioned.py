# ==========================
# ====== P structure =======
# ==========================

import warnings
import inspect
import math
import torch
from dataclasses import dataclass

from rkgb.utils.utils import Counter
from rkgb.lowlevel import anonymize
from rkgb.core import base
from rkgb.core.simplified import SimplifiedGraph, SimplifiedNode


class PartitionedNode(base.Node):
    is_protected_from_unwrap = None
    memory_occupied_by_all_outputs = None
    def __init__(self,
            main_graph,
            sub_cluster = None,
            main_target = None,
            sub_graph   = None, # FOR DYNAMIC PARTITIONING
            simplified_node = None):
        sub_cluster : PartitionedCluster
        sub_graph : PartitionedGraph
        super().__init__(parent_structure_with_id_generator=main_graph)
        self.main_graph  = main_graph
        self.sub_cluster : PartitionedCluster = sub_cluster
        self.main_target = main_target
        self.sub_graph   = sub_graph
        self.simplified_node : SimplifiedNode = simplified_node
        # => used for .deps/.users to compute io_targets
        if (int(main_target is not None) 
        + int(sub_graph is not None) 
        + int(sub_cluster is not None)) != 1: # hand made logical xor
            raise Exception(
                "A PartitionedNode is usually defined either by a main_target "\
                "or a sub_cluster, never both.\nFor dynamic partitioning, "\
                "you can instead giving a sub_graph, but it must be "\
                "temporary, during the partitioning."
            )
        if sub_cluster is not None:
            self.name = f"PN({sub_cluster.name})"
            self.is_leaf = False
        elif sub_graph is not None:
            self.name = f"PN({sub_graph.name})"
            self.is_leaf = False
            self.is_protected_from_unwrap = False
        else:
            self.name = f"BottomNode({main_target})"
            self.is_leaf = True
            self.is_protected_from_unwrap = True

        self.deps  = set()
        self.users = set()
        self.deps_through_artifacts  = set()
        self.users_through_artifacts = set()
        # FOR DYNAMIC PARTITIONING
        self.deps_global  = set() 
        self.users_global = set()
        self.deps_through_artifacts_global = set()
        self.users_through_artifacts_global = set()
        # Global edges contain ALL deps/users, of any depth

    def get_all_standard_deps(self):
        return self.deps.union(self.deps_through_artifacts)
    def get_all_standard_users(self):
        return self.users.union(self.users_through_artifacts)
    
    def does_require_grad(self):
        if not self.is_leaf:
            return True
        elif self.simplified_node is not None:
            return self.simplified_node.does_require_grad()
        else:
            return False

    @property
    def size(self):
        if self.is_leaf or self.is_protected_from_unwrap: return 1
        elif not (self.sub_cluster is None): return self.sub_cluster.size
        else: return len(self.sub_graph.nodes)
    @property
    def total_size(self):
        if self.is_leaf: return 1
        elif not (self.sub_cluster is None): return self.sub_cluster.size
        else: return self.sub_graph.total_size



class PartitionedGraph(base.Graph):
    node_class = PartitionedNode
    pn_wrapping_it : PartitionedNode = None
    # -> pn representing self in an upper graph
    _first_nodes : list[PartitionedNode] = None
    # -> in case we precise what are the first_nodes manually
    # otherwise, they are computed by @property self.first_nodes
    cluster = None # useful for debugging + printing
    without_artifacts = False
    list_of_blocks_indices = None # Special case, that we want to keep track
    def __init__(self,
            partitioned_cluster = None,
            parent_objet = None,
            is_main_graph = False,
            list_of_blocks_indices : list[tuple[int,int]] = None):
        # 1) Initialize simple attributes
        super().__init__(parent_objet)
        self.is_main_graph = is_main_graph
        self.output_nodes = set()
        if parent_objet is None:
            parent_objet = partitioned_cluster
        if hasattr(parent_objet,"counter_nb_graphs"):
            counter : Counter = parent_objet.counter_nb_graphs
            graph_nb = counter.count()
            self.counter_nb_graphs = counter
        else:
            graph_nb = -1
        self.graph_nb = graph_nb
        self.name = f"PartitionedGraph({graph_nb})"

        self.cluster = partitioned_cluster
        # Real constructors based on a partitioner_cluster
        if partitioned_cluster is not None:
            partitioned_cluster : PartitionedCluster
            dict_info = partitioned_cluster.p_structure.dict_info
            # 2) FIRST CONSTRUCTOR:
            # - Based on a list of blocks:
            if list_of_blocks_indices:
                self.list_of_blocks_indices = list_of_blocks_indices # useful to remember how it was built
                nb_blocks = len(list_of_blocks_indices)
                self.nodes = []
                self._first_nodes = set()
                self.output_nodes = set()
                s_nodes = partitioned_cluster.s_nodes
                list_blocks_s_nodes = []
                dict_sn_to_pn = dict()
                for block_i in range(nb_blocks):
                    (start_i,end_i) = list_of_blocks_indices[block_i]
                    # 'start' is included in the block; 'end' isn't
                    # 1) Create the PartitionedNode
                    if start_i == end_i - 1:
                        sn = s_nodes[start_i]
                        block_s_nodes = [sn]
                        block_pn = PartitionedNode(
                            main_graph=self,
                            main_target=sn.mt,
                            simplified_node=sn)
                        block_pn.memory_occupied_by_all_outputs = dict_info[sn.mt].memsize
                    else:
                        block_s_nodes = s_nodes[start_i:end_i]
                        sub_cluster = PartitionedCluster(
                            block_s_nodes,
                            partitioned_cluster.p_structure)
                        block_pn = PartitionedNode(
                            main_graph=self,
                            sub_cluster=sub_cluster)
                        block_pn.memory_occupied_by_all_outputs = dict_info[block_s_nodes[-1].mt].memsize
                    
                    # 2) Store the PNode in self.nodes/output_nodes/_first_nodes 
                    self.nodes.append(block_pn)
                    for sn in block_s_nodes:
                        if sn.mt in partitioned_cluster.outputs_mt:
                            self.output_nodes.add(block_pn)
                            break
                    for sn in block_s_nodes:
                        if sn.mt in partitioned_cluster.firsts_mt:
                            self._first_nodes.add(block_pn)
                            break

                    # 3) Edges with the previous block
                    for sn in block_s_nodes:
                        dict_sn_to_pn[sn] = block_pn
                    list_blocks_s_nodes.append(set(block_s_nodes))
                    if block_i > 0:
                        for sn in block_s_nodes:
                            # deps:
                            for req_sn in sn.deps:
                                if req_sn in s_nodes and req_sn not in block_s_nodes:
                                    req_pn : PartitionedNode = dict_sn_to_pn[req_sn]
                                    if req_pn not in block_pn.deps:
                                        block_pn.deps.add(req_pn)
                                        block_pn.deps_global.add(req_pn)
                                        req_pn.users.add(block_pn)
                                        req_pn.users_global.add(block_pn)
                            # deps through artifacts:
                            for req_sn in sn.deps_through_artifacts:
                                if req_sn not in block_s_nodes:
                                    req_pn : PartitionedNode = dict_sn_to_pn[req_sn]
                                    if req_pn not in block_pn.deps_through_artifacts:
                                        block_pn.deps_through_artifacts.add(req_pn)
                                        block_pn.deps_through_artifacts_global.add(req_pn)
                                        req_pn.users_through_artifacts.add(block_pn)
                                        req_pn.users_through_artifacts_global.add(block_pn)
                        # artifact edges are redundant with classic ones -> useless

            # 2) SECOND CONSTRUCTOR:
            # - Default constructor translating partitioner_cluster.s_nodes
            else:
                self.nodes = []
                dict_mt_to_pn = dict()
                s_nodes = partitioned_cluster.s_nodes
                for sn in s_nodes:
                    sn : SimplifiedNode
                    pn = PartitionedNode(
                        main_target = sn.mt,
                        main_graph = self,
                        simplified_node = sn)
                    pn.memory_occupied_by_all_outputs = dict_info[sn.mt].memsize
                    self.nodes.append(pn)
                    dict_mt_to_pn[sn.mt] = pn
                    for req_sn in sn.deps:
                        if req_sn in s_nodes:
                            req_pn = dict_mt_to_pn[req_sn.mt]
                            pn.deps.add(req_pn)
                            req_pn.users.add(pn)
                    for req_sn in sn.deps_through_artifacts:
                        if req_sn in s_nodes:
                            req_pn = dict_mt_to_pn[req_sn.mt]
                            pn.deps_through_artifacts.add(req_pn)
                            req_pn.users_through_artifacts.add(pn)
                    
                for pn in self.nodes:
                    pn.deps_global = set(pn.deps)
                    pn.users_global = set(pn.users)
                    pn.deps_through_artifacts_global = set(pn.deps_through_artifacts)
                    pn.users_through_artifacts_global = set(pn.users_through_artifacts)

                self._first_nodes = set(
                    [dict_mt_to_pn[first_mt] 
                    for first_mt in partitioned_cluster.firsts_mt])
                self.output_nodes = set(
                    [dict_mt_to_pn[out_mt] 
                    for out_mt in partitioned_cluster.outputs_mt])


    @property
    def size(self):
        return len(self.nodes)
    @property
    def total_size(self):
        return sum(pn.total_size for pn in self.nodes)

    @property
    def input_nodes(self):
        if self.pn_wrapping_it is None:
            return set()
        else:
            # deps at level above :
            wrapping_pn = self.pn_wrapping_it
            input_nodes = set(wrapping_pn.deps)
            # deps higher :
            if wrapping_pn.main_graph is not None:
                higher_g = wrapping_pn.main_graph
                if wrapping_pn in higher_g.first_nodes:
                    input_nodes.update(
                        higher_g.input_nodes.intersection(
                        wrapping_pn.deps_global
                        )
                    )
            return input_nodes

    @property
    def first_nodes(self): # -> users of at least one input
        if self._first_nodes is not None:
            return self._first_nodes
        else:
            input_nodes = self.input_nodes
            spn = set(self.nodes)
            first_nodes = set()
            for inp_pn in input_nodes:
                first_nodes.update(spn.intersection(inp_pn.users_global))
            return first_nodes

    def all_p_nodes_inside(self): # FOR DYNAMIC PARTITIONING
        all_p = set(self.nodes)
        for pn in self.nodes:
            if pn.sub_graph is not None:
                all_p.update(pn.sub_graph.all_p_nodes_inside())
        return all_p
    
    def derive_local_artifact_edges_from_global(self):  # FOR DYNAMIC PARTITIONING
        local_nodes = set(self.nodes)
        for pn in self.nodes:
            pn.deps_through_artifacts = pn.deps_through_artifacts_global.intersection(local_nodes)
            pn.users_through_artifacts = pn.users_through_artifacts_global.intersection(local_nodes)

    def set_all_protected_to_false(self): # FOR DYNAMIC PARTITIONING
        for pn in self.nodes:
            if not pn.is_leaf:
                pn.is_protected_from_unwrap = False
                pn.sub_graph.set_all_protected_to_false()

    def make_attribute_all_snodes(self): # FOR DYNAMIC PARTITIONING
        # Assume a dynamic structure
        self._all_snodes = all_snodes = set()
        for pn in self.nodes:
            if pn.sub_graph is not None:
                pn.sub_graph.make_attribute_all_snodes()
                all_snodes.update(pn.sub_graph._all_snodes)
            else:
                assert(pn.simplified_node is not None) # DYNAMIC MOD
                all_snodes.add(pn.simplified_node)

    def fix_redundant_clusters(self):
        for pn in self.nodes:
            pn : PartitionedNode
            if pn.sub_cluster is not None:
                self_or_equal_cluster = pn.sub_cluster.self_or_strictly_equal_cluster
                if self_or_equal_cluster is not pn.sub_cluster:
                    # => It's a redundancy of an other cluster
                    pn.sub_cluster = self_or_equal_cluster
                else:
                    # => Fine we continue
                    self_or_equal_cluster.fix_redundant_clusters()

    def make_temporary_global_root_node_to_deps_relation(self):
        # Overwrite base.Graph's method:
        if len(self.output_nodes) == 0:
            raise Exception(
                f"{self} doesn't have a single output node so we "\
                f"fail to create a root node to deps relation and "\
                f"trace deps (either to sort or find_cutting_points)")
        fresh_root = PartitionedNode(self,main_target="tmp_root_node_to_deps_relation")
        fresh_root.deps = set(self.output_nodes)
        for out_node in self.output_nodes:
            out_node.users.add(fresh_root)
        return fresh_root
    
    # = print and render =
    def __str__(self):
        return (
            f"{self.name}:\nGraph composed of {self.size} top level nodes,\n"\
            f"containing a total of {self.total_size} basic nodes.")
    
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            include_artifact_edges=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        color_leaf = "blue"
        color_edge = color_leaf
        color_sub_graph = "blueviolet"
        color_special   = "green"
        # 1) nodes and edges
        pn : PartitionedNode
        for pn in self.nodes:
            if pn.is_leaf:
                if only_function_name:
                    code = pn.simplified_node.main_fct
                    fontsize = '14'
                else:
                    code = pn.simplified_node.get_code()
                    fontsize = '10'
                label = f"{pn.name}\n{code}"
                dot.node(pn.name,label,color=color_leaf,fontsize=fontsize)
            else:
                label = f"{pn.name}\nCluster size: {pn.size}"
                dot.node(pn.name,label,color=color_sub_graph)
            for req_pn in pn.deps:
                dot.edge(req_pn.name,pn.name,color=color_edge)
            if include_artifact_edges:
                for req_pn in pn.deps_through_artifacts:
                    dot.edge(req_pn.name,pn.name,color=color_edge,style="dotted")
        
        # 2) inputs
        first_nodes = list(self.first_nodes)
        kwargs = dict(color = color_special,style="dashed")
        if first_nodes != []:
            label = "INPUTS:\n"+"\n".join(self.cluster.inputs_mt)
            dot.node("inputs",label,**kwargs)
            for first_pn in first_nodes:
                dot.edge("inputs",first_pn.name,**kwargs)

        # 3) outputs
        label = "OUTPUTS:\n"+"\n".join(self.cluster.outputs_mt)
        dot.node("outputs",label,**kwargs)
        for output_pn in self.output_nodes:
            dot.edge(output_pn.name,"outputs",**kwargs)
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )



class Partitioner():
    class Config():
        def __init__(self):
            pass
    def __init__(self):
        self.config : self.__class__.Config = None
    def __call__(self, cluster):
        return PartitionedGraph(cluster)



class PartitionedCluster():
    s_nodes = None
    p_structure = None
    inputs = None
    outputs = None
    inputs_mt = None
    firsts_mt = None
    outputs_mt = None
    translator : anonymize.ClusterTranslator = None
    self_or_strictly_equal_cluster = None
    cluster_hash = None
    ano_hash = None
    ano_cluster_id = None
    name = None
    representee_cluster = None
    partitionings = None
    partitioners_already_used = None
    # Tmp attributes :
    input_snodes = None
    first_snodes = None
    output_snodes = None
    dict_input_sn_to_users_sn = None
    dict_first_sn_to_required_inputs_sn = None
    dict_first_mt_to_required_inputs_mt = None
    dict_output_mt_to_targets_sent = None
    # Later :
    h_cluster = None

    def __init__(self,
            group_simplified_nodes,
            partitioned_structure):
        self.p_structure : PartitionedStructure = partitioned_structure
        self.counter_nb_graphs = partitioned_structure.counter_nb_graphs
        self.s_nodes = base.Node.sort_nodes(group_simplified_nodes)
        # => Following their ._topological_number = their index in sg.nodes

        # As we can use several partitioners at the same time,
        # it could happen that we already had the exact same cluster
        # ie the same set of s_nodes. So to be efficient, we store 
        # a handmade __repr__/__hash__ of each cluster:
        cluster_hash = hash(tuple(self.s_nodes))
        self.cluster_hash = cluster_hash
        dict_hash_to_cluster = self.p_structure.dict_cluster_hash_to_cluster
        if cluster_hash in dict_hash_to_cluster:
            self.self_or_strictly_equal_cluster = dict_hash_to_cluster[cluster_hash]
            self.name = self.self_or_strictly_equal_cluster.name
        else:
            cluster_nb = self.p_structure.counter_nb_clusters.count()
            self.cluster_nb = cluster_nb
            self.self_or_strictly_equal_cluster = self
            dict_hash_to_cluster[cluster_hash] = self
            self.compute_interfaces()
            self.make_ano_cluster_id()
            self.name = f"P_Cluster_{cluster_nb}_Ano_id_{self.ano_cluster_id}"

    @property
    def size(self):
        return len(self.s_nodes)
    # ======================
    
    # =================
    def compute_interfaces(self):
        """
        Build the following attributes related to inputs/outputs:
        self.inputs ; outputs ; inputs_mt ; outputs_mt
        first_snodes, output_nodes, dict_first_sn_to_targets_used
        """
        sg : SimplifiedGraph = self.p_structure.sg
        inputs = set()
        outputs = set()
        inputs_mt = []
        firsts_mt = []
        outputs_mt = []
        input_snodes = []
        first_snodes = []
        output_snodes = []
        self.dict_input_sn_to_users_sn = dict_input_sn_to_users_sn = dict()
        self.dict_first_sn_to_required_inputs_sn = dict_first_sn_to_req_inputs_sn = dict()
        self.dict_output_mt_to_targets_sent = dict_outputs_sent = dict()
        # == for each sn, check its interfaces outside the cluster ==
        for sn_to_proceed in self.s_nodes:
            for req_sn in sn_to_proceed.deps:
                used_targets = sg.dict_of_labels_on_edges[(req_sn,sn_to_proceed)]
                if req_sn not in self.s_nodes:
                    inputs.update(used_targets)
                    if sn_to_proceed.mt not in firsts_mt:
                        firsts_mt.append(sn_to_proceed.mt)
                        first_snodes.append(sn_to_proceed)
                        dict_first_sn_to_req_inputs_sn[sn_to_proceed] = [req_sn]
                    else:
                        dict_first_sn_to_req_inputs_sn[sn_to_proceed].append(req_sn)
                    if req_sn.mt not in inputs_mt:
                        inputs_mt.append(req_sn.mt)
                        input_snodes.append(req_sn)
                        dict_input_sn_to_users_sn[req_sn] = [sn_to_proceed]
                    else:
                        dict_input_sn_to_users_sn[req_sn].append(sn_to_proceed)
            for user_sn in sn_to_proceed.users:
                used_targets = sg.dict_of_labels_on_edges[(sn_to_proceed,user_sn)]
                if not (user_sn in self.s_nodes):
                    outputs.update(used_targets)
                    if sn_to_proceed.mt not in outputs_mt:
                        outputs_mt.append(sn_to_proceed.mt)
                        output_snodes.append(sn_to_proceed)
                        dict_outputs_sent[sn_to_proceed.mt] = set(used_targets)
                    else:
                        dict_outputs_sent[sn_to_proceed.mt].update(used_targets)

        # == check for interfaces between sg.init_node and the cluster ==
        init_node = sg.init_node
        for user_sn in init_node.users:
            if user_sn in self.s_nodes:
                used_targets = sg.dict_of_labels_on_edges[(init_node,user_sn)]
                inputs.update(used_targets)
                if init_node.mt not in inputs_mt:
                    inputs_mt.append(init_node.mt)
                    input_snodes.append(init_node)
                    dict_input_sn_to_users_sn[init_node] = [user_sn]
                else:
                    dict_input_sn_to_users_sn[init_node].append(user_sn)
                if user_sn.mt not in firsts_mt:
                    firsts_mt.append(user_sn.mt)
                    first_snodes.append(user_sn)
                    dict_first_sn_to_req_inputs_sn[user_sn] = [init_node]
                else:
                    dict_first_sn_to_req_inputs_sn[user_sn].append(init_node)

        # == check if cluster contains sg.output_nodes ==
        for output_node in sg.output_nodes:
            if output_node in self.s_nodes:
                output_targets = sg.dict_output_mt_to_targets_sent[output_node.mt]
                outputs.update(output_targets)
                if output_node.mt not in outputs_mt:
                    outputs_mt.append(output_node.mt)
                    output_snodes.append(output_node)
                    dict_outputs_sent[output_node.mt] = set(output_targets)
                else:
                    dict_outputs_sent[output_node.mt].update(output_targets)

        self.dict_first_mt_to_required_inputs_mt = dict(
            (first_sn.mt,[req_inp_sn.mt for req_inp_sn in req_inputs_sn]) 
            for (first_sn,req_inputs_sn) 
            in self.dict_first_sn_to_required_inputs_sn.items())
        self.inputs = list(inputs) ; self.inputs.sort(key=base.Node.get_num_tar)
        self.outputs = list(outputs) ; self.outputs.sort(key=base.Node.get_num_tar)
        self.first_snodes = first_snodes ; first_snodes.sort(key=base.Node.get_num)
        self.output_snodes = output_snodes ; output_snodes.sort(key=base.Node.get_num)
        self.input_snodes = anonymize.sort_inputs_mt(self,input_snodes)
        self.firsts_mt = [first_sn.mt for first_sn in self.first_snodes]
        self.outputs_mt = [output_sn.mt for output_sn in self.output_snodes]
        self.inputs_mt = [input_sn.mt for input_sn in self.input_snodes]
        # get_num_tar isn't fully accurate for anonymized graph recognition
        # so it's better to rely on get_num, except for inputs_mt.
        # inputs' topo-number, and so position, should impact 
        # graph recognition, instead we check the targets sent by input_nodes
    # =========================
    

    # =============================
    def make_ano_cluster_id(self):
        ano_hash = anonymize.AnonymousHash.hash(self)
        self.ano_hash = ano_hash
        dict_ano_hash_to_ano_id = self.p_structure.dict_cluster_ano_hash_to_ano_cluster_id
        dict_ano_id_to_representee_cluster = self.p_structure.dict_cluster_ano_id_to_representee_cluster
        if ano_hash in dict_ano_hash_to_ano_id:
            ano_id = dict_ano_hash_to_ano_id[ano_hash]
            self.ano_cluster_id = ano_id
            self.representee_cluster = dict_ano_id_to_representee_cluster[ano_id]
        else:
            self.partitionings = []
            self.partitioners_already_used = []
            ano_id = self.p_structure.counter_nb_unique_clusters.count()
            self.ano_cluster_id = ano_id
            dict_ano_hash_to_ano_id[ano_hash] = ano_id
            self.representee_cluster = self
            dict_ano_id_to_representee_cluster[ano_id] = self
    # =============================

    def partition(self,partitioner : Partitioner):
        if self.self_or_strictly_equal_cluster is not self:
            self.self_or_strictly_equal_cluster.partition(partitioner)
        elif self.representee_cluster is not self:
            self.representee_cluster.partition(partitioner)
        elif partitioner in self.partitioners_already_used:
            pass
        elif self.size < self.p_structure.min_size_to_trigger_partitioning:
            pass
        else:
            pg = partitioner(self)
            if pg is not None: 
                # we found something interesting 
                # => some Partitioners returns None if failed, eg PartitionerSequence 
                self.partitionings.append(pg)
                self.partitioners_already_used.append(partitioner)

    def fix_redundant_clusters(self):
        if self.partitionings is not None:
            for pg in self.partitionings:
                pg.fix_redundant_clusters()

    
    def __repr__(self):
        return self.name

    def __str__(self):
        cluster = self.representee_cluster
        s = f"{self.name}:\nCluster containing a total of {self.size} basic nodes,\n"
        nb_partitionings = len(cluster.partitionings)
        if nb_partitionings == 0:
            return s+"Without any partitioning ~ PartitionedGraph"
        elif nb_partitionings == 1:
            pg = cluster.partitionings[0]
            return s+ f"With one partitioning, which has {pg.size} top level nodes."
        else:
            return s + f"with {nb_partitionings} possible partitionings."
        
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
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
                    "ie corresponding PartitionedGraph: use cluster.partition()")
            if cluster is not self:
                print(f"Warning : render an equivalent cluster "\
                      f"{cluster.cluster_nb} (the representee)")
        for i,pg in enumerate(cluster.partitionings):
            if name is not None:
                graph_name = f"{i}-th partitioning of {name}"
            else:
                graph_name = None
            pg.render(graph_name,
                view,
                only_function_name,
                include_artifact_edges,
                directory,
                render_format,
                render,
                dot)




class PartitionedStructure():
    main_cluster : PartitionedCluster = None
    sg : SimplifiedGraph = None
    dict_info : dict = None
    dict_target_ano_id : dict[str, int] = None
    dict_mt_to_sn_ano_material : dict[str, anonymize.SimplifiedNodeAnonymizationMaterial] = None
    dict_cluster_hash_to_cluster : dict[str, PartitionedCluster] = None
    dict_cluster_ano_hash_to_ano_cluster_id : dict[str,int] = None
    dict_cluster_ano_id_to_representee_cluster : dict[int,PartitionedCluster] = None
    min_size_to_trigger_partitioning = 4
    def __init__(self,
            simplified_graph : SimplifiedGraph,
            partitioners : list[Partitioner],
            dict_target_ano_id,
            dict_mt_to_sn_ano_material):
        # 1) Initialize the structure
        self.sg = simplified_graph
        self.dict_info = simplified_graph.dict_info
        self.counter_nb_graphs = Counter()
        self.counter_nb_clusters = Counter()
        self.node_unique_id_generator = base.Node_unique_id_generator()

        # - Anonymizing stuff => to recognize equivalent clusters
        self.counter_nb_unique_clusters = Counter()
        self.dict_target_ano_id = dict_target_ano_id
        self.dict_mt_to_sn_ano_material = dict_mt_to_sn_ano_material
       
        self.dict_cluster_hash_to_cluster = dict()
        self.dict_cluster_ano_hash_to_ano_cluster_id = dict()
        self.dict_cluster_ano_id_to_representee_cluster = dict()

        self.main_cluster = PartitionedCluster(
            list(simplified_graph.nodes),self)

        # 2) Run the partitioners
        for partitioner in partitioners:
            self.main_cluster.partition(partitioner)
        self.main_cluster.fix_redundant_clusters()
        self.all_clusters = set(self.dict_cluster_hash_to_cluster.values())
        self.all_unique_clusters = set(self.dict_cluster_ano_id_to_representee_cluster.values())
        self.dict_cluster_nb_to_cluster = dict((c.cluster_nb,c) for c in self.all_clusters)

        # Give clean names to graphs
        for cluster in self.all_unique_clusters:
            for nb,pg in enumerate(cluster.partitionings):
                pg.name = f"Partitioning n°{nb} of {cluster.name}"
        # For anonymizing stuff: build a translator for each graph
        for cluster in self.all_clusters:
            cluster.translator = anonymize.ClusterTranslator(cluster)

        # render
        self.render = self.main_cluster.render

    def __str__(self):
        return "PartitionedStructure whose main cluster is:\n"+str(self.main_cluster)
    


class PartitionedDynamicManipulation(): # only contains staticmethod

    @staticmethod
    def prepare_dynamic_setup(pg : PartitionedGraph,cluster : PartitionedCluster):
        """
        In this setup, we assume pg is the highest level graph / standalone.
        But we need global relations : inputs received and outputs sent.
        Therefore we create a last_wrapping_graph, of which sg
        will be a sub_graph. With some fresh input nodes in
        last_wrapping_graph connected to pg's wrapping node.
        And one "sink" node at the end of last_wrapping_graph
        to serve as the user of pg's outputs.
        """
        last_wrapping_graph = PartitionedGraph(parent_objet=cluster.p_structure)
        pg_wrapping_pn = PartitionedNode(
            last_wrapping_graph,
            sub_graph=pg
        ) # => The node representing pg as a sub part of last_wrapping_graph
        # 1) Inputs:
        inputs_pn = []
        dict_input_mt_to_pn = dict()
        # Create the fresh input nodes in last_wrapping_graph
        for inp_mt in cluster.inputs_mt:
            inp_pn = PartitionedNode(last_wrapping_graph,main_target="InputInLastWrappingGraph "+inp_mt)
            inputs_pn.append(inp_pn)
            dict_input_mt_to_pn[inp_mt] = inp_pn
            inp_pn.users = set([pg_wrapping_pn])
        pg_wrapping_pn.deps = set(inputs_pn)
        # Connect pg's first_nodes to these new input_nodes via global deps/users
        first_nodes = pg._first_nodes
        for fst_node in first_nodes:
            req_inputs_mt = cluster.dict_first_mt_to_required_inputs_mt[fst_node.mt]
            for req_inp_mt in req_inputs_mt:
                req_inp_pn = dict_input_mt_to_pn[req_inp_mt]
                fst_node.deps_global.add(req_inp_pn)
                req_inp_pn.users_global.add(fst_node)
        pg._first_nodes = None
        # As the graph will dynamically evolve, we will recompute pg.first_nodes
        # when needed, whereas _first_nodes were the fixed original ones

        # 2) Outputs:
        sink_pn = PartitionedNode(
            last_wrapping_graph,
            main_target = "last_wrapping_graph_sink"
        )
        pg_wrapping_pn.users = set([sink_pn])
        for out_pn in pg.output_nodes:
            out_pn.users_global.add(sink_pn)
            sink_pn.deps_global.add(out_pn)

        pg.pn_wrapping_it = pg_wrapping_pn
        last_wrapping_graph.nodes = inputs_pn + [pg_wrapping_pn,sink_pn]

    # ********
    # * WRAP *
    # ********
    @staticmethod
    def wrap(group : list,main_pg : PartitionedGraph):
        """
        wrap a 'group' of nodes, currently living in 'main_pg',
        in a new sub graph 'new_pg', adding one level of depth.
        'new_pg' is represented by 'new_pg_wrapping_pn' in 'main_pg'
        """
        new_pg = PartitionedGraph(parent_objet=main_pg)
        new_pg.nodes = group
        new_pg_wrapping_pn = PartitionedNode(
            main_graph = main_pg,
            sub_graph  = new_pg,
        )
        new_pg.pn_wrapping_it = new_pg_wrapping_pn
        set_group = set(group)
        new_pg.output_nodes = output_nodes = set()
        for pn_in_group in group:
            # ** link new_pg_wrapping_pn with global edges **
            new_pg_wrapping_pn.deps_global.update(pn_in_group.deps_global)
            new_pg_wrapping_pn.users_global.update(pn_in_group.users_global)
            new_pg_wrapping_pn.deps_through_artifacts_global.update(pn_in_group.deps_through_artifacts_global)
            new_pg_wrapping_pn.users_through_artifacts_global.update(pn_in_group.users_through_artifacts_global)
            # -> remove edges to inside at the end
            # -> reciprocal at the end

            # -> change pn_in_group.main_graph
            pn_in_group.main_graph = new_pg

            # Deps outside the group are now considered as inputs
            deps_outside = pn_in_group.deps - set_group
            for req_pn in deps_outside:
                req_pn.users.discard(pn_in_group)
                req_pn.users.add(new_pg_wrapping_pn)
                pn_in_group.deps.discard(req_pn)
                new_pg_wrapping_pn.deps.add(req_pn)
            
            # ** outputs **
            if (pn_in_group in main_pg.output_nodes
            or (not pn_in_group.users.issubset(set_group))):
                output_nodes.add(pn_in_group)
                user_outside = pn_in_group.users - set_group
                for user_pn in user_outside:
                    user_pn.deps.discard(pn_in_group)
                    user_pn.deps.add(new_pg_wrapping_pn)
                    pn_in_group.users.discard(user_pn)
                    new_pg_wrapping_pn.users.add(user_pn)

        # ** memory_occupied_by_all_outputs **
        new_pg_wrapping_pn.memory_occupied_by_all_outputs = sum(
            out_pn.memory_occupied_by_all_outputs 
            for out_pn in output_nodes)

        # ** global edges must not include edge to nodes inside new_pg_wrapping_pn **
        all_p_nodes_inside = new_pg.all_p_nodes_inside()
        new_pg_wrapping_pn.deps_global -= all_p_nodes_inside
        new_pg_wrapping_pn.users_global -= all_p_nodes_inside
        new_pg_wrapping_pn.deps_through_artifacts_global -= all_p_nodes_inside
        new_pg_wrapping_pn.users_through_artifacts_global -= all_p_nodes_inside

        # ** reciprocal global edges **
        for global_req_pn in new_pg_wrapping_pn.deps_global:
            global_req_pn.users_global.add(new_pg_wrapping_pn)
        for global_user_pn in new_pg_wrapping_pn.users_global:
            global_user_pn.deps_global.add(new_pg_wrapping_pn)
        for artifact_global_req_pn in new_pg_wrapping_pn.deps_through_artifacts_global:
            artifact_global_req_pn.users_through_artifacts_global.add(new_pg_wrapping_pn)
        for artifact_global_user_pn in new_pg_wrapping_pn.users_through_artifacts_global:
            artifact_global_user_pn.deps_through_artifacts_global.add(new_pg_wrapping_pn)
        
        # ** update main_pg.nodes **
        main_lpn = main_pg.nodes
        main_lpn[main_lpn.index(group[0])] = new_pg_wrapping_pn
        for pn in group[1:]:
            main_lpn.remove(pn)

        # ** update main_pg outputs **
        main_out = main_pg.output_nodes
        if not main_out.isdisjoint(set_group):
            main_out.add(new_pg_wrapping_pn)
        main_pg.output_nodes -= set_group

        return new_pg_wrapping_pn


    # **********
    # * UNWRAP *
    # **********
    # unwrap 'pn' in its main graph
    @staticmethod
    def unwrap(pn_to_unwrap : PartitionedNode):
        pg      : PartitionedGraph = pn_to_unwrap.sub_graph
        main_pg : PartitionedGraph = pn_to_unwrap.main_graph
        if pn_to_unwrap.is_protected_from_unwrap: return ()
        group = list(pg.nodes)

        # 1) unplug pn_to_unwrap/pg
        # -> global edges
        for global_req_pn in pn_to_unwrap.deps_global:
            global_req_pn.users_global.remove(pn_to_unwrap)
        for global_user_pn in pn_to_unwrap.users_global:
            global_user_pn.deps_global.remove(pn_to_unwrap)
        for artifact_global_req_pn in pn_to_unwrap.deps_through_artifacts_global:
            artifact_global_req_pn.users_through_artifacts_global.remove(pn_to_unwrap)
        for artifact_global_user_pn in pn_to_unwrap.users_through_artifacts_global:
            artifact_global_user_pn.deps_through_artifacts_global.remove(pn_to_unwrap)
        # local edges (ie non global ones) will be overwritten anyway

        # ** plug back the group **
        # -> fix main_pg.nodes
        main_lpn = main_pg.nodes
        i = main_lpn.index(pn_to_unwrap)
        main_pg.nodes = main_lpn[:i] + group + main_lpn[i+1:]    
        # -> fix sub_pn.main_graph
        for sub_pn in group:
            sub_pn.main_graph = main_pg
        # -> use the property : deps = deps_global inter nodes 
        main_spn = set(main_pg.nodes)
        all_p_nodes_inside = main_pg.all_p_nodes_inside()
        to_update = group + list(pn_to_unwrap.deps) + list(pn_to_unwrap.users)
        for sub_pn in to_update:
            sub_pn.deps  = sub_pn.deps_global.intersection(main_spn)
            sub_pn.users = sub_pn.users_global.intersection(main_spn)
            if not sub_pn.users_global.issubset(all_p_nodes_inside):
                main_pg.output_nodes.add(sub_pn)

        if pn_to_unwrap in main_pg.output_nodes:
            main_pg.output_nodes.remove(pn_to_unwrap)
        return ()


    # *********
    # * MERGE *
    # *********
    # Merging replaces a group of nodes living in 'main_pg'
    # by a new node 'new_pn', with : .nodes = \sum sub_pn.nodes for sub_pn in group
    # thus it creates one wrapper, but flats the first depth level
    # -> To do so, wrap and then unwrap each sub_node
    @staticmethod
    def merge(group : list, main_pg : PartitionedGraph):
        new_pn = PartitionedDynamicManipulation.wrap(group,main_pg)
        new_pg = new_pn.sub_graph
        original_lpn = new_pg.nodes
        for sub_pn in original_lpn:
            PartitionedDynamicManipulation.unwrap(sub_pn)
        return new_pn
    
    # **********
    # * FREEZE *
    # **********
    # Freeze the dynamic structure : sub_graph -> sub_cluster
    @staticmethod
    def freeze(
            pg : PartitionedGraph,
            p_structure : PartitionedStructure,
            partitioner : Partitioner,):
        # return list of all nodes in pg for recursive purpose
        pg._first_nodes = pg.first_nodes # comment this if one day want to restart Dynamic
        for pn in pg.nodes:
            pn : PartitionedNode
            if pn.sub_graph is not None:
                sub_graph : PartitionedGraph = pn.sub_graph
                sub_cluster = PartitionedCluster(sub_graph._all_snodes,p_structure)
                sub_cluster = sub_cluster.self_or_strictly_equal_cluster
                pn.sub_cluster = sub_cluster
                pn.name = f"PN({sub_cluster.name})"
                pn.sub_graph = None # no longer dynamic
                if sub_cluster.representee_cluster is sub_cluster:
                    sub_graph.cluster = sub_cluster
                    sub_cluster.partitionings.append(sub_graph)
                    sub_cluster.partitioners_already_used.append(partitioner)
                    PartitionedDynamicManipulation.freeze(sub_graph,p_structure,partitioner)
                # otherwise -> We won't keep this sub_graph
                # -> we are only interested in partitioning representee
            else:
                if pn.simplified_node is None:
                    raise Exception(
                        f"PartitionedNode which is_leaf should have a self.simplified_node "\
                        f"(except special nodes, but there shouldn't be any "\
                        f"special node here). Here : pn.name : {pn.name}."
                    )


def make_partitioner(description):
    if description == "bottom_to_top":
        return PartitionerBottomToTop(main_graph_as_any_other=True)
    elif description == "repetitive":
        return PartitionerRecognizeRepetitivePattern()
    else:
        print("[Warning] make_partitioner: unknown sub_partitioner", self.config.sub_partitioner,
              "possible values:", "bottom_to_top", "repetitive")
        return PartitionerBottomToTop(main_graph_as_any_other=True)


class PartitionerBottomToTop(Partitioner):
    class Option():
        def __init__(self,group):
            self.group = group # list
            self.set_group = set(group)
        @property
        def size(self):
            return sum(pn.size for pn in self.group)
        @property
        def nb_sub_graphs(self):
            return sum(pn.sub_graph is not None for pn in self.group)
        
        @staticmethod
        def utils_is_seq(list_nodes):
            for i in range(len(list_nodes)-1):
                pn = list_nodes[i]
                next_pn = list_nodes[i+1]
                if len(pn.users_global) != 1: return False
                if list(pn.users_global)[0] is not next_pn: return False
                if len(next_pn.deps_global) != 1: return False
                # We check global edges, since a chain with external deps isn't a chain
            return True

        def is_seq(self):
            # self.group must be in the correct order
            if any((not pn.is_protected_from_unwrap) for pn in self.group):
                return False
            return self.__class__.utils_is_seq(self.group)
        
        def __eq__(self,opt2):
            return (self.set_group == opt2.set_group)
        def __hash__(self):
            return id(self)

    @dataclass
    class Config:
        max_len_seq: int = 99
        max_estimate_for_main_graph: int = 30
        max_estimate_per_sub_graph: int = 20
        min_size_per_sub_graph: int = 3
        main_graph_as_any_other: bool = False
        can_use_rotor: bool = True
        estimate_coeff_size: int = 1
        estimate_coeff_sub_graph: int = 1
        value_coeff_input_interfaces: int = 1
        value_coeff_output_interfaces: int = 1
        value_power_total_size: float = 0.5
        old_value_fct: bool = False
        old_value_fct_value_power_not_last: float = 1.1

        def __post_init__(self):
            if self.main_graph_as_any_other:
                self.max_estimate_for_main_graph = self.max_estimate_per_sub_graph
            if self.old_value_fct:
                self.option_value_fct = self.old_default_option_value_fct
            else:
                self.option_value_fct = self.default_option_value_fct
            self.option_stop_fct = self.default_option_stop_fct
            self.option_estimate_fct = self.default_estimate_fct
            self.is_top_graph_ok = self.default_is_top_graph_ok

        def default_estimate_fct(self,option):
            option : self.__class__.Option
            return (
                option.size * self.estimate_coeff_size
                + option.nb_sub_graphs * self.estimate_coeff_sub_graph
            )

        def old_default_option_value_fct(self,option):
            option : self.__class__.Option
            not_last_nodes = [
                pn for pn in option.group if pn.users.issubset(option.group)
            ]
            tot_mem_internal = sum(pn.memory_occupied_by_all_outputs for pn in not_last_nodes)
            if len(not_last_nodes)==0:
                value = 0
            else: 
                value = (tot_mem_internal
                * len(not_last_nodes)**-self.old_value_fct_value_power_not_last)
            # effort for determinism -> we break ties
            num_determinism = min(pn.unique_id for pn in option.group)
            return (value,num_determinism)
        
        def default_option_value_fct(self, option):
            option : self.__class__.Option
            inputs_pn = set().union(
                *[pn.deps_global - option.set_group 
                  for pn in option.group]
            )
            outputs_pn = set(
                pn for pn in option.group
                if not pn.users_global.issubset(option.set_group)
            )
            inputs_mem = sum(pn.memory_occupied_by_all_outputs for pn in inputs_pn if pn.simplified_node is not None)
            outputs_mem = sum(pn.memory_occupied_by_all_outputs for pn in outputs_pn if pn.simplified_node is not None)
            total_size = sum(pn.total_size for pn in option.group)
            # /!\ NEGATIVE VALUE
            # -> We will take the max -> = the less negative one
            value = - (
                    (   inputs_mem * self.value_coeff_input_interfaces
                    +  outputs_mem * self.value_coeff_output_interfaces)
                *
                    total_size**self.value_power_total_size
            )
            # effort for determinism -> we break ties
            num_determinism = min(pn.unique_id for pn in option.group)
            return (value,num_determinism)
        
        def default_option_stop_fct(self, option):
            option : self.__class__.Option
            if len(option.group)==1:
                return True
            if self.can_use_rotor and option.is_seq():
                return False
            else:
                return (self.option_estimate_fct(option) 
                        > self.max_estimate_per_sub_graph)
        
        def default_is_top_graph_ok(self,pg : PartitionedGraph):
            if (self.can_use_rotor
            and PartitionerBottomToTop.Option.utils_is_seq(pg.nodes)):
                return True
            else:
                size = len(pg.nodes)
                nb_sub_graphs = sum(pn.sub_graph is not None for pn in pg.nodes)
                estimate = (
                        size * self.estimate_coeff_size
                    +   nb_sub_graphs * self.estimate_coeff_sub_graph
                )
                return estimate <= self.max_estimate_for_main_graph
            

    def __init__(self, **kwargs):
        self.config = self.__class__.Config(**kwargs)

    # === FIRST WAY TO FIND OPTIONS : SEQUENCES ===
    def find_seq_options(self,pg : PartitionedGraph):
        # ** Find the sequences **
        tot_nb_seq = 0
        dict_seq_nb = dict() # name -> a seq nb
        dict_sequences = dict() # seq nb -> list of nodes in the seq
        for pn in pg.nodes:
            pn : PartitionedNode
            pn_users = pn.get_all_standard_users()
            if len(pn_users) == 1:
                user_pn : PartitionedNode = list(pn_users)[0]
                if len(user_pn.get_all_standard_deps()) == 1:
                    user_name = user_pn.name
                    if pn.name in dict_seq_nb:
                        seq_nb = dict_seq_nb[pn.name]
                        dict_seq_nb[user_name] = seq_nb
                        dict_sequences[seq_nb].append(user_pn)
                    else:
                        tot_nb_seq += 1
                        dict_seq_nb[pn.name] = tot_nb_seq
                        dict_seq_nb[user_name] = tot_nb_seq
                        dict_sequences[tot_nb_seq] = [pn,user_pn]

        # ** split too long sequences **
        # -> rely on config.max_len_seq
        all_sequences = list(dict_sequences.items())
        for seq_nb,sequence in all_sequences:
            seq_len = len(sequence)
            if seq_len > self.config.max_len_seq:
                del dict_sequences[seq_nb]
                nb_seq = math.ceil(seq_len/self.config.max_len_seq)
                sub_seq_len = math.ceil(seq_len/nb_seq)
                for first in range(0,seq_len,sub_seq_len):
                    end = min(first + sub_seq_len,seq_len)
                    sub_seq = sequence[first:end]
                    tot_nb_seq += 1
                    sub_seq_nb = tot_nb_seq
                    dict_sequences[sub_seq_nb] = sub_seq
                    for pn in sub_seq:
                        dict_seq_nb[pn.main_target] = sub_seq_nb

        return [
            self.__class__.Option(seq) \
            for seq in dict_sequences.values()
        ]

    # === SECOND WAY TO FIND OPTIONS : FLOWS ===
    # the flow of a pn are nodes 
    def find_flow_options(self,pg : PartitionedGraph):
        # === GENERALIZED VERSION OF base.Graph.find_cutting_points ===
        # for each node we need to find where its flow converge back
        # -> flow of a pn is defined as nodes in
        # -> `to_be_visited` which are descendants of pn

        # ATTENTION here source/sink are taken from .deps
        # relation perspective, e.g. outputs are sources.
        # /!\ Note that its merging here, not grouping /!\

        dict_nb_usages = dict(
            [(pn, len(pn.get_all_standard_users())) 
             for pn in pg.nodes])
        to_be_visited = []
        dict_flow = dict()
        # for a pn already visited -> its descendants in to_be_visited
        # if len(flow_size) = 0 => the flow converged
        # its a generalization of "seen" in cut_based_on_deps
        dict_total_flow = dict()
        # when a node is popped out of to_be_visited, 
        # its removed from dict_flow but dict_total_flow 
        # is a record of all the nodes which were in the flow
        # note that a node is in its own total_flow
        # also, total_flow is a list, whereas the current_flow is a set
        dict_which_flow = dict()
        # for a pn in to_be_visited -> all the flows he is part of
        # ie a list of PartitionedNodes, representing there flow
        # reciprocal of dict_flow 
        dict_end_of_flow = dict()
        # any pn -> where its flow converged back
        # this is what we want to build

        # ** Add a temporary global sink to deps relation **
        tmp_global_sink_pn = PartitionedNode(main_graph=pg,main_target="tmp_sink")
        tmp_global_sink_pn.users = first_nodes = pg.first_nodes
        for first_pn in first_nodes:
            first_pn.deps.add(tmp_global_sink_pn)
        dict_nb_usages[tmp_global_sink_pn] = len(first_nodes)

        # ** init **
        for pn in pg.nodes:
            pn : SimplifiedNode
            if len(pn.get_all_standard_users()) == 0:
                to_be_visited.append(pn)
                dict_which_flow[pn] = set()

        # ** search **
        while to_be_visited != []:
            pn = to_be_visited.pop()
            current_flows = dict_which_flow[pn]
            continuing_flows = set([pn])
            dict_flow[pn] = set()
            dict_total_flow[pn] = [pn]
            # * check if end of flows *
            for flow_pn in current_flows:
                flow = dict_flow[flow_pn]
                flow.remove(pn)
                # equivalent to "seen.remove(n)" in RK_get_1_separators
                if flow == set():
                    dict_end_of_flow[flow_pn] = pn
                else:
                    continuing_flows.add(flow_pn)
            # * visit pn *
            for req_pn in pn.get_all_standard_deps():
                # equivalent to seen.add(req_n) :
                for flow_pn in continuing_flows:
                    tot_flow = dict_total_flow[flow_pn]
                    flow     = dict_flow[flow_pn]
                    flow.add(req_pn)
                    if (not (req_pn is tmp_global_sink_pn)
                    and not (req_pn in tot_flow)):
                        tot_flow.append(req_pn)
                if req_pn in dict_which_flow:
                    dict_which_flow[req_pn].update(continuing_flows)
                else:
                    dict_which_flow[req_pn] = set(continuing_flows)
                dict_nb_usages[req_pn]-=1
                if dict_nb_usages[req_pn]==0:
                    to_be_visited.append(req_pn)

        # ** remove the temporary global sink **
        for first_pn in first_nodes:
            first_pn.deps.remove(tmp_global_sink_pn)

        # === SECOND ===
        # For each flow we have 4 options :
        # -> include the source or not
        # -> include the sink or not

        options = []
        for source_pn,sink_pn in dict_end_of_flow.items():
            flow = dict_total_flow[source_pn]
            flow.reverse()
            # Option 1 : with source and sink
            options.append(self.__class__.Option(flow))
            # Option 2 : without source
            if len(flow)>2:
                flow_ = list(flow) ; flow_.remove(source_pn)
                options.append(self.__class__.Option(flow_))
            # Option 3 : without sink
            if len(flow)>2 and not (sink_pn is tmp_global_sink_pn):
                flow_ = list(flow) ; flow_.remove(sink_pn)
                options.append(self.__class__.Option(flow_))
            # Option 4 : without source and sink
            if len(flow)>3 and not (sink_pn is tmp_global_sink_pn):
                flow_ = list(flow)
                flow_.remove(source_pn) ; flow_.remove(sink_pn)
                options.append(self.__class__.Option(flow_))

        return options
    

    # === ROUND OF DYNAMIC PARTITIONING ===
    def round_of_partitioning(self, pg : PartitionedGraph):
        config = self.config
        for pn in pg.nodes:
            pn.is_protected_from_unwrap = True
        pg.derive_local_artifact_edges_from_global()
        all_options = (
                self.find_seq_options(pg)
            +   self.find_flow_options(pg)
        )

        _all_options = list(all_options)
        for opt in _all_options:
            if not any(pn.does_require_grad()
                   for pn in opt.group):
                all_options.remove(opt)
            elif all(pn.deps_global.issubset(opt.set_group)
                     for pn in opt.group):
                all_options.remove(opt)


        dict_options_pn_is_part_of = dict((pn,set()) for pn in pg.nodes)
        # PartitionedNode -> Option set
        # After each merge, we actualize all the options which 
        # include some nodes concerned by the merge.
        for option in all_options:
            for pn in option.group:
                dict_options_pn_is_part_of[pn].add(option)

        def filter_options(all_options):
            filtered_list = []
            for option in all_options:
                if not config.option_stop_fct(option):
                    if option not in filtered_list:
                        filtered_list.append(option)
                        # -> rely on Option.__eq__ which check for set_group eq
            return filtered_list
                    
        while not config.is_top_graph_ok(pg) and all_options != []:
            all_options = filter_options(all_options)
            if all_options == []: break
            best_option = max(all_options,key=config.option_value_fct)
            all_options.remove(best_option)
            best_group = list(best_option.group)
            new_pn = PartitionedDynamicManipulation.merge(best_group,pg)
            updated_opts = set()
            for pn in best_group:
                opts = list(dict_options_pn_is_part_of[pn])
                for opt in opts:
                    if opt not in all_options: continue
                    # Case 1: one element of this group has already been replaced
                    if new_pn in opt.group: 
                        opt.group.remove(pn)
                        if len(opt.group) < 2: # too small
                            all_options.remove(opt)
                    # Case 2: replace pn by new_pn
                    else:
                        opt.group[opt.group.index(pn)] = new_pn
                        opt.set_group.add(new_pn)
                    opt.set_group.remove(pn) # anyway
                    if opt in all_options: updated_opts.add(opt)
                del dict_options_pn_is_part_of[pn]
            dict_options_pn_is_part_of[new_pn] = updated_opts


    def __call__(self, cluster: PartitionedCluster):
        pg = PartitionedGraph(cluster)
        config = self.config
        if config.is_top_graph_ok(pg): return pg

        # === Prepare dynamic setup ===
        PartitionedDynamicManipulation.prepare_dynamic_setup(pg,cluster)

        # === FIRST : Dynamic partitioning ===
        previous_size = -1
        while not config.is_top_graph_ok(pg) and len(pg.nodes) != previous_size:
            previous_size = len(pg.nodes)
            self.round_of_partitioning(pg)
        pg.set_all_protected_to_false()

        if len(pg.nodes) == previous_size: 
            warnings.warn(
                f"Partitioning of cluster '{cluster.name}' with "\
                f"{self.__class__} early stop, because it"\
                f"shrink more. Thus maybe to big, size : {previous_size}."
            )

        # === SECOND : freeze ===
        pg.make_attribute_all_snodes()
        if pg._all_snodes != set(cluster.s_nodes):
            raise Exception(
                f"BUG in {self.__class__}. When collecting all the "\
                f"SimplifiedNodes at the end, we don't find cluster.s_nodes. "\
                f"We lost some nodes...\n Original nb of nodes : {len(cluster.s_nodes)}; "\
                f"Nb of nodes at the end : {len(pg._all_snodes)}"
            )
        PartitionedDynamicManipulation.freeze(pg,cluster.p_structure,self.__class__)
        return pg




class PartitionerSequence(Partitioner):
    @dataclass
    class Config:
        sub_partitioner: str = "bottom_to_top"

    def __init__(self, *args, **kwargs):
        self.config = self.__class__.Config(*args, **kwargs)
        self.sub_partitioner = make_partitioner(self.config.sub_partitioner)

    def __call__(self, cluster : PartitionedCluster):
        # cluster only contains the list of concerned s_nodes, 
        # but to search for cutting_points we need a graph structure
        # hence we create a first temporary PartitionedGraph
        # which transposes cluster.s_nodes to PNode, with deps/users
        tmp_pg = PartitionedGraph(cluster)
        # To call base.Graph.find_cutting_points we first need
        # to add a sink to the .deps relation. An equivalent
        # of sg.init_node, without any deps, one clear first node.
        tmp_sink_pn = PartitionedNode(main_graph=tmp_pg,main_target="tmp_sink")
        tmp_pg.nodes.insert(0,tmp_sink_pn)
        first_nodes = tmp_pg.first_nodes
        for first_pn in first_nodes:
            if first_pn.deps == set():
                first_pn.deps.add(tmp_sink_pn)
                tmp_sink_pn.users.add(first_pn)

        seps_pn = tmp_pg.find_cutting_points()
        # Remove the temporary sink
        for first_pn in first_nodes:
            first_pn.deps.discard(tmp_sink_pn)
        tmp_pg.nodes.remove(tmp_sink_pn)
        # Could do: del tmp_pg

        seps_mt = [sep.mt for sep in seps_pn]
        seps_sn = [sn for sn in cluster.s_nodes if sn.mt in seps_mt]
        seps_index = [cluster.s_nodes.index(sep) for sep in seps_sn]
        nb_blocks = len(seps_index)
        seps_index.insert(0,-1)
        blocks = [
            (seps_index[i]+1,seps_index[i+1]+1) # (start,end)
            for i in range(nb_blocks)]
        # 'start' is included in the block; 'end' isn't

        if len(blocks)==1:
            # We didn't find any structure
            # => directly use the sub_partitioner instead
            cluster.partition(self.sub_partitioner)
            return None
        else:
            pg = PartitionedGraph(cluster,list_of_blocks_indices=blocks)
            # sub partition:
            for block_pn in pg.nodes:
                if block_pn.sub_cluster is not None:
                    sub_cluster : PartitionedCluster = block_pn.sub_cluster
                    sub_cluster.partition(self.sub_partitioner)
            return pg



class PartitionerRecognizeRepetitivePattern(Partitioner):
    @dataclass
    class Config:
        sub_partitioner: str = "bottom_to_top"
        split_patterns_in_two: bool = False
        recognize_simply_by_main_fct_not_whole_ano_material: bool = True
        strict_max_number_of_top_level_nodes: int = 50
        max_number_of_patterns: int = 40
        min_number_of_patterns: int = 2
        min_percentage_covered_required: float = 0.75
        put_intermediates_with_preceding_block: bool = True
        put_inputs_with_first_block: bool = False
        put_outputs_with_last_block: bool = False
    def __init__(self, *args, **kwargs):
        self.config = self.__class__.Config(*args, **kwargs)
        self.sub_partitioner = make_partitioner(self.config.sub_partitioner)

    def __call__(self, cluster : PartitionedCluster):
        list_nodes_hash = self.hash_cluster_nodes(cluster)
        patterns_indices = self.find_repetitive_patterns(list_nodes_hash)
        if patterns_indices is None or len(patterns_indices)==0:
            # We didn't find any structure 
            # => directly use the sub_partitioner instead
            cluster.partition(self.sub_partitioner)
            return None

        else:
            blocks_indices = self.build_blocks_based_on_patterns(
                patterns_indices,len(cluster.s_nodes))
            pg = PartitionedGraph(
                partitioned_cluster=cluster,
                list_of_blocks_indices=blocks_indices 
            )
            if self.config.split_patterns_in_two:
                new_blocks = self.split_patterns_in_two_parts(
                    blocks_indices,pg)
                if new_blocks == blocks_indices:
                    pass # keep pg
                else:
                    # We will rebuild the graph with the new blocks
                    # so first we need to clean the global directory: 
                    # ie remove mentions of the old clusters
                    dict_cl_hash_to_cl = cluster.p_structure.dict_cluster_hash_to_cluster
                    dict_cl_ano_hash_to_ano_id = cluster.p_structure.dict_cluster_ano_hash_to_ano_cluster_id
                    dict_cl_ano_id_to_repr = cluster.p_structure.dict_cluster_ano_id_to_representee_cluster
                    for pn in pg.nodes:
                        pn : PartitionedNode
                        if pn.sub_cluster is not None:
                            pn_cl = pn.sub_cluster
                            if pn_cl.self_or_strictly_equal_cluster is pn_cl:
                                del dict_cl_hash_to_cl[pn_cl.cluster_hash]
                                if pn_cl.representee_cluster is pn_cl:
                                    del dict_cl_ano_hash_to_ano_id[pn_cl.ano_hash]
                                    del dict_cl_ano_id_to_repr[pn_cl.ano_cluster_id]
                    pg = PartitionedGraph(
                        partitioned_cluster=cluster,
                        list_of_blocks_indices=new_blocks)
            
            # Sub partition:
            for block_pn in pg.nodes:
                if block_pn.sub_cluster is not None:
                    sub_cluster : PartitionedCluster = block_pn.sub_cluster
                    sub_cluster.partition(self.sub_partitioner)
            return pg
                

        

    def build_blocks_based_on_patterns(self,patterns_indices,total_length):
        """Delimits the blocks/sub clusters based on patterns_indices"""
        if self.config.put_intermediates_with_preceding_block:
            # CASE 1: we want to avoid having intermediate nodes at the top level
            # Note: We keep inputs and outputs outside any pattern
            separators = [start for (start,end) in patterns_indices]
            # Inputs block:
            if separators[0] != 0:
                if self.config.put_inputs_with_first_block:
                    separators[0] = 0
                else:
                    separators.insert(0,0)
            # Outputs block:
            if (patterns_indices[-1][1] != total_length
            and not self.config.put_outputs_with_last_block):
                separators.append(patterns_indices[-1][1])
            separators.append(total_length)
            blocks_indices = [(separators[i],separators[i+1]) for i in range(len(separators)-1)]

        else:
            # CASE 2: we keep the intermediate nodes between the patterns
            # separators = [0] if patterns_indices[0][0] != 0 else []
            blocks_indices = [(-1,0)] # to start the loop, removed at the end
            for start,end in patterns_indices:
                prev_end = blocks_indices[-1][1]
                if blocks_indices[-1][1] != start:
                    blocks_indices.append((prev_end,start))
                blocks_indices.append((start,end))
            blocks_indices.pop(0)
            # Inputs block:
            if (self.config.put_inputs_with_first_block
            and blocks_indices[0][0] != patterns_indices[0][0]):
                blocks_indices.pop(0)
                blocks_indices[0][0] = 0
            # Outputs block:
            if blocks_indices[-1][1] != total_length:
                if self.config.put_outputs_with_last_block:
                    blocks_indices[-1][1] = total_length
                else:
                    blocks_indices.append(
                        (blocks_indices[-1][1],total_length))
        return blocks_indices


    def hash_cluster_nodes(self,cluster : PartitionedCluster):
        # Give a simple hash number to each node
        if self.config.recognize_simply_by_main_fct_not_whole_ano_material:
            s_nodes_main_fcts = [sn.main_fct for sn in cluster.s_nodes]
            dict_main_fct_to_nb = dict()
            nb = 0
            for fct in s_nodes_main_fcts:
                if fct not in dict_main_fct_to_nb:
                    dict_main_fct_to_nb[fct] = nb
                    nb += 1
            return [
                dict_main_fct_to_nb[fct] 
                for fct in s_nodes_main_fcts]
        else:
            dict_mt_to_sn_ano_material = cluster.p_structure.dict_mt_to_sn_ano_material
            return [
                dict_mt_to_sn_ano_material[sn.mt].anonymous_id
                for sn in cluster.s_nodes
            ]


    def find_repetitive_patterns(self,list_nodes_hash):
        # TO IMPROVE: Currently O(n²) with some tricks
        # I think there exist super fancy algorithms to do this,
        # See Knuth-Morris-Pratt related algorithms
        # 0) Parameters to tune
        total_length = len(list_nodes_hash)
        min_interesting_pattern_length \
            = math.ceil(total_length / self.config.max_number_of_patterns * self.config.min_percentage_covered_required)
        max_interesting_pattern_length = int(total_length/self.config.min_number_of_patterns)
        min_nb_nodes_covered_by_patterns = int(total_length*self.config.min_percentage_covered_required)

        # 0) Store current best solution
        current_best_solution = None
        current_best_nb_patterns = 0

        for pattern_length in range(
                min_interesting_pattern_length,
                max_interesting_pattern_length+1):
            # 0) If we already found a solution and if it's no longer possible to find better one
            if int(total_length/pattern_length) < current_best_nb_patterns:
                return current_best_solution
            # 1) Hash all the possible patterns
            dict_hashes = dict()
            for start in range(total_length - pattern_length +1):
                end = start + pattern_length
                pattern_hash = hash(tuple(list_nodes_hash[start:end]))
                if pattern_hash in dict_hashes:
                    equivalent_start_indices = dict_hashes[pattern_hash]
                    if start >= equivalent_start_indices[-1]+pattern_length:
                        equivalent_start_indices.append(start)
                else:
                    dict_hashes[pattern_hash] = [start]
            # 2) See if a pattern is repeated
            for pattern_hash,start_indices in dict_hashes.items():
                if len(start_indices)*pattern_length >= min_nb_nodes_covered_by_patterns:
                    patterns_indices =  [
                        (start,start+pattern_length) 
                        for start in start_indices]
                    blocks_indices = self.build_blocks_based_on_patterns(patterns_indices,total_length)
                    if (
                        len(blocks_indices)
                        <= self.config.strict_max_number_of_top_level_nodes
                     and len(patterns_indices) >= current_best_nb_patterns):
                        current_best_solution = patterns_indices
                        current_best_nb_patterns = len(current_best_solution)
        return current_best_solution
    

    def split_patterns_in_two_parts(
            self,
            blocks_indices,
            current_pg : PartitionedGraph):
        new_blocks = []
        for block,block_pn in zip(blocks_indices,current_pg.nodes):
            block_pn : PartitionedNode
            start,end = block
            # if not (start in pattern_starts or end in pattern_ends):
            if end-start < 14:
                new_blocks.append(block)
                # e.g. intermediate blocks / inputs / outputs
            else:
                # We look for the best cut
                assert block_pn.sub_cluster is not None
                block_cluster : PartitionedCluster = block_pn.sub_cluster
                dict_sn_to_index = dict(
                    (sn,i) for (i,sn) in enumerate(block_cluster.s_nodes)
                )
                # 1) in the second part we don't want dependencies to previous blocks
                input_users_indices = [dict_sn_to_index[sn] for sn in block_cluster.first_snodes]
                min_start_part2 = max(max(input_users_indices)+1,5)
                # 2) none of the outputs should come from the first part
                outputs_indices = [dict_sn_to_index[sn] for sn in block_cluster.output_snodes]
                block_length = end-start
                max_start_part2 = min(min(outputs_indices),block_length-5)
                if min_start_part2>max_start_part2:
                    new_blocks.append(block)
                else:
                    first_part_snodes = block_cluster.s_nodes[:min_start_part2]
                    interface_edges_given_user = dict() # : v -> list[u]
                    interface_edges_given_req = dict() # : u -> list[v]
                    for sn in first_part_snodes:
                        for user_sn in sn.users:
                            user_sn_index = dict_sn_to_index[user_sn]
                            if user_sn_index >= min_start_part2:
                                if user_sn in interface_edges_given_user:
                                    interface_edges_given_user[user_sn].append(sn)
                                else:
                                    interface_edges_given_user[user_sn] = [sn]
                                if sn in interface_edges_given_req:
                                    interface_edges_given_req[sn].append(user_sn)
                                else:
                                    interface_edges_given_req[sn] = [user_sn]
                    # interface_mem = sum(sn.info.memsize for sn in interface_edges_given_req)
                    interface_size = len(interface_edges_given_req)
                    best_cut_index = min_start_part2
                    # best_interface_mem = interface_mem
                    best_interface_size = interface_size
                    part1_length = best_cut_index
                    part2_length = block_length-part1_length
                    for cut_index in range(min_start_part2+1,max_start_part2+1):
                        # move s_nodes[cut_index-1] to part 1
                        # 1) remove the incoming edges
                        cut_node = block_cluster.s_nodes[cut_index-1]
                        if cut_node in interface_edges_given_user:
                            for req_sn in interface_edges_given_user[cut_node]:
                                req_sn_users = interface_edges_given_req[req_sn]
                                req_sn_users.remove(cut_node)
                                if req_sn_users == []:
                                    # interface_mem -= req_sn.info.memsize
                                    interface_size -= 1
                                    del interface_edges_given_req[req_sn]
                            del interface_edges_given_user[cut_node]
                        # 2) add the outgoing edges
                        out_edges = []
                        for user_sn in cut_node.users:
                            if user_sn in block_cluster.s_nodes:
                                out_edges.append(user_sn)
                                if user_sn in interface_edges_given_user:
                                    interface_edges_given_user[user_sn].append(cut_node)
                                else:
                                    interface_edges_given_user[user_sn] = [cut_node]
                        if out_edges != []:
                            interface_edges_given_req[cut_node] = out_edges
                            # interface_mem += cut_node.info.memsize
                            interface_size += 1
                        # 3) Check if it's the best cut
                        new_part1_length = cut_index
                        new_part2_length = block_length-new_part1_length
                        if ((interface_size < best_interface_size)
                        or (interface_size == best_interface_size
                        and min(new_part1_length,new_part2_length) > min(part1_length,part2_length))):
                            best_interface_size = interface_size
                            best_cut_index = cut_index
                            part1_length = new_part1_length
                            part2_length = new_part2_length
                            
                    # Cut:
                    cut_index = best_cut_index + start
                    if best_cut_index == start or best_cut_index == end:
                        new_blocks.append(block)
                    else:
                        new_blocks.append((start,cut_index))
                        new_blocks.append((cut_index,end))
        return new_blocks
                
                

