# ==========================
# ====== P structure =======
# ==========================

import warnings
import math
import torch
from src.utils.utils import Counter
from src.core import base
from src.core.simplified import SimplifiedGraph, SimplifiedNode


class PartitionedNode(base.Node):
    is_protected_from_unwrap = None
    mem_out = None
    def __init__(self,
            main_graph,
            sub_cluster = None,
            main_target = None,
            sub_graph   = None, # FOR DYNAMIC PARTITIONING
            simplified_node = None):
        sub_cluster : PartitionedCluster
        sub_graph : PartitionedGraph
        super().__init__(other_obj=main_graph)
        self.main_graph  = main_graph
        self.sub_cluster = sub_cluster
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
            self.name = "PNode: "+sub_cluster.name
            self.is_leaf = False
        elif sub_graph is not None:
            self.name = "PNode: "+sub_graph.name
            self.is_leaf = False
            self.is_protected_from_unwrap = False
        else:
            self.name = f"Var_{main_target}"
            self.is_leaf = True
            self.is_protected_from_unwrap = True

        self.deps         = set()
        self.users        = set()
        self.deps_global  = set() # FOR DYNAMIC PARTITIONING
        self.users_global = set() # FOR DYNAMIC PARTITIONING
        # Global edges contain ALL deps/users, of any depth

    def get_all_standard_deps(self):
        return self.deps
    def get_all_standard_users(self):
        return self.users
    
    def does_requires_grad(self):
        if not self.is_leaf:
            return True
        else:
            return self.simplified_node.does_requires_grad()

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
    pn_wrapping_it : PartitionedNode = None
    # -> pn representing self in an upper graph
    _first_nodes : list[PartitionedNode] = None
    # -> in case we precise what are the first_nodes manually
    # otherwise, they are computed by @property self.first_nodes
    cluster = None # useful for debugging + printing
    without_artifacts = False
    def __init__(self,
            parent_objet = None,
            is_main_graph = False):
        super().__init__(parent_objet)
        self.is_main_graph = is_main_graph
        self.output_nodes = set()
        if hasattr(parent_objet,"counter_nb_graphs"):
            counter : Counter = parent_objet.counter_nb_graphs
            graph_nb = counter.count()
            self.counter_nb_graphs = counter
        else:
            graph_nb = -1
        self.name = f"PartitionedGraph {graph_nb}"

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
            pnw = self.pn_wrapping_it
            # deps at level above :
            input_nodes = set(pnw.deps)
            # deps higher :
            if pnw.main_graph is not None:
                higher_g = pnw.main_graph
                if pnw in higher_g.first_nodes:
                    input_nodes.update(
                        higher_g.input_nodes.intersection(
                        pnw.deps_global
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

    def set_all_protected_to_false(self):
        for pn in self.nodes:
            if not pn.is_leaf:
                pn.is_protected_from_unwrap = False
                pn.sub_graph.set_all_protected_to_false()

    def make_sub_cluster_original(self):
        for pn in self.nodes:
            pn : PartitionedNode
            if pn.sub_cluster is not None:
                sub_c = pn.sub_cluster.original_cluster
                if sub_c is pn.sub_cluster:
                    sub_c.make_sub_cluster_original()
                else:
                    pn.sub_cluster = sub_c

    def recompute_edges(self,sg : SimplifiedGraph):
        # NOT OPTIMAL AT ALL -> To improve if to slow
        # find edges_via_artifacts in PartitionedGraph
        # Moreover, recomputing edges is not necessary
        # it's just that it's cleaner to have an accurate PartitionedGraph
        # at the end
        if not self.without_artifacts:
            self.without_artifacts = True
            all_related_to_artifacts = set().union(
                set(e[0] for e in sg.edges_via_artifacts),
                set(e[1] for e in sg.edges_via_artifacts),
            )
            dict_where = dict()
            for sn in all_related_to_artifacts:
                for pn in self.nodes:
                    pn : PartitionedNode
                    if pn.simplified_node is sn:
                        dict_where[sn] = pn
                    elif (pn.sub_cluster is not None
                        and sn in pn.sub_cluster.s_nodes):
                        dict_where[sn] = pn
            
            # search edges we might have to delete
            suspicious_edges : set[tuple[PartitionedNode,PartitionedNode]] = set()
            for (req_sn,user_sn,_) in sg.edges_via_artifacts:
                if req_sn in dict_where and user_sn in dict_where:
                    req_pn = dict_where[req_sn]
                    user_pn = dict_where[user_sn]
                    if req_pn is not user_pn:
                        suspicious_edges.add((req_pn,user_pn))

            # for suspicious edges, we need to see if there is any
            # S edge from one to the other (all artifact edges have
            # been deleted, that's why we need to recompute P edges)
            for req_pn,user_pn in suspicious_edges:
                if req_pn.sub_cluster is not None:
                    all_sn_req_pn = req_pn.sub_cluster.s_nodes
                else:
                    all_sn_req_pn = [req_pn.simplified_node]
                if user_pn.sub_cluster is not None:
                    all_sn_user_pn = user_pn.sub_cluster.s_nodes
                else:
                    all_sn_user_pn = [user_pn.simplified_node]
                bool_keep_edge = False
                for sn_in_req_pn in all_sn_req_pn:
                    for user_sn in sn_in_req_pn.users:
                        if user_sn in all_sn_user_pn:
                            bool_keep_edge = True
                if not bool_keep_edge:
                    req_pn.users.remove(user_pn)
                    user_pn.deps.remove(req_pn)

            # recursive call
            for pn in self.nodes:
                if pn.sub_cluster is not None:
                    pn.sub_cluster.recompute_all_interfaces_and_edges()
        


class Partitioner():
    class Config():
        def __init__(self):
            pass
    def __init__(self):
        self.config : self.__class__.Config = None
    def __call__(self, cluster):
        return cluster.init_PartitionedGraph()
        # raise Exception(
            # "Base class of partitioners. __call__ method "\
            # "must be overwritten by all subclasses")



class PartitionedCluster():
    s_nodes = None
    p_structure = None
    inputs = None
    outputs = None
    inputs_mt = None
    firsts_mt = None
    outputs_mt = None
    translator = None
    original_cluster = None
    ano_cluster_id = None
    name = None
    representee_cluster = None
    possible_partitioning = None
    partitioners_already_used = None
    # Tmp attributes :
    first_nodes = None
    output_nodes = None
    dict_input_mt_to_inputs_sent = None
    dict_first_mt_to_inputs_used = None
    dict_first_mt_to_inputs_used_mt = None
    dict_output_mt_to_outputs_sent = None

    def __init__(self,group_s_nodes,p_structure):
        self.p_structure : PartitionedStructure = p_structure
        sg = p_structure.sg
        # == get the toposort for s_nodes ==
        # group_s_nodes is an unordered iterable
        self.s_nodes = [sn for sn in sg.nodes if sn in group_s_nodes]

        # == Check if the exact same cluster already exists ==
        charac_string = str([sn.get_num() for sn in self.s_nodes])
        ps_dict_charac_to_cluster = self.p_structure.dict_cluster_charac_string_to_cluster
        if charac_string in ps_dict_charac_to_cluster:
            self.original_cluster = ps_dict_charac_to_cluster[charac_string]
            self.name = self.original_cluster.name
        else:
            cluster_nb = self.p_structure.counter_nb_clusters.count()
            self.original_cluster \
                = ps_dict_charac_to_cluster[charac_string] \
                = self
            self.make_io()
            self.make_ano_cluster_id()
            self.name = f"P_Cluster_{cluster_nb}_Ano_id_{self.ano_cluster_id}"

    @property
    def size(self):
        return len(self.s_nodes)
    # ======================
    
    # ======================
    def init_PartitionedGraph(self):
        assert(self.original_cluster is self)
        pg = PartitionedGraph(parent_objet=self.p_structure)
        pg.cluster = self
        pg.nodes = p_nodes = []
        dict_info = self.p_structure.dict_info
        dict_mt_to_pn = dict()
        for sn in self.s_nodes:
            pn = PartitionedNode(
                main_graph  = pg,
                main_target = sn.mt,
                sn = sn)
            pn.mem_out = dict_info[sn.mt].memsize
            p_nodes.append(pn)
            dict_mt_to_pn[sn.mt] = pn
            for req_sn in sn.deps.keys():
                if req_sn in self.s_nodes:
                    req_pn = dict_mt_to_pn[req_sn.mt]
                    pn.deps.add(req_pn)
                    req_pn.users.add(pn)
        for pn in p_nodes:
            pn.deps_global = set(pn.deps)
            pn.users_global = set(pn.users)

        pg._first_nodes = set([dict_mt_to_pn[first_mt] for first_mt in self.firsts_mt])
        pg.output_nodes = set([dict_mt_to_pn[out_mt] for out_mt in self.outputs_mt])
        return pg
    # =================
            
    # =================
    def make_io(self):
        # ATTRIBUTEs NEEDED : s_nodes, p_structure
        # DO: inputs, outputs, inputs_mt, outputs_mt
        # DO TMP: first_nodes, output_nodes, dict_first_sn_to_inputs_used
        inputs = set()
        outputs = set()
        inputs_mt = []
        firsts_mt = []
        outputs_mt = []
        first_nodes = []
        output_nodes = []
        self.dict_input_mt_to_inputs_sent = dict_inputs_sent = dict()
        self.dict_first_mt_to_inputs_used = dict_inputs_used = dict()
        self.dict_first_mt_to_inputs_used_mt = dict_inputs_used_mt = dict()
        self.dict_output_mt_to_outputs_sent = dict_outputs_sent = dict()
        # == for each sn, check its interfaces outside the cluster ==
        for sn in self.s_nodes:
            for req_sn,used_targets in sn.deps.items():
                if not (req_sn in self.s_nodes):
                    inputs.update(used_targets)
                    if sn.mt not in firsts_mt:
                        firsts_mt.append(sn.mt)
                        first_nodes.append(sn)
                        dict_inputs_used[sn.mt] = set(used_targets)
                        dict_inputs_used_mt[sn.mt] = set([req_sn.mt])
                    else:
                        dict_inputs_used[sn.mt].update(used_targets)
                        dict_inputs_used_mt[sn.mt].add(req_sn.mt)
                    if req_sn.mt not in inputs_mt:
                        inputs_mt.append(req_sn.mt)
                        dict_inputs_sent[req_sn.mt] = set(used_targets)
                    else:
                        dict_inputs_sent[req_sn.mt].update(used_targets)
            for user_sn,used_targets in sn.users.items():
                if not (user_sn in self.s_nodes):
                    outputs.update(used_targets)
                    if sn.mt not in outputs_mt:
                        outputs_mt.append(sn.mt)
                        output_nodes.append(sn)
                        dict_outputs_sent[sn.mt] = set(used_targets)
                    else:
                        dict_outputs_sent[sn.mt].update(used_targets)

        # == check for interfaces between sg.init_node and the cluster ==
        sg : SimplifiedGraph = self.p_structure.sg
        ino = sg.init_node
        for user_sn,used_targets in ino.users.items():
            if user_sn in self.s_nodes:
                inputs.update(used_targets)
                if ino.mt not in inputs_mt:
                    inputs_mt.append(ino.mt)
                    dict_inputs_sent[ino.mt] = set(used_targets)
                else:
                    dict_inputs_sent[ino.mt].update(used_targets)
                if user_sn.mt not in firsts_mt:
                    firsts_mt.append(user_sn.mt)
                    first_nodes.append(user_sn)
                    dict_inputs_used[user_sn.mt] = set(used_targets)
                    dict_inputs_used_mt[user_sn.mt] = set([ino.mt])
                else:
                    dict_inputs_used[user_sn.mt].update(used_targets)
                    dict_inputs_used_mt[user_sn.mt].add(ino.mt)

        # == check if cluster contains sg.output_nodes ==
        if sg.wrapper_output_node is None: # sg has only one output_node
            out_node = sg.output_nodes[0]
            if out_node in self.s_nodes:
                outputs.update(sg.outputs)
                if out_node.mt not in outputs_mt:
                    outputs_mt.append(out_node.mt)
                    output_nodes.append(out_node)
                    dict_outputs_sent[out_node.mt] = set(sg.outputs)
                else:
                    dict_outputs_sent[out_node.mt].update(set(sg.outputs))
        else:
            for out_node in sg.output_nodes:
                if out_node in self.s_nodes:
                    out_targets = sg.wrapper_output_node.deps[out_node]
                    outputs.update(out_targets)
                    if out_node.mt not in outputs_mt:
                        outputs_mt.append(out_node.mt)
                        output_nodes.append(out_node)
                        dict_outputs_sent[out_node.mt] = set(out_targets)
                    else:
                        dict_outputs_sent[out_node.mt].update(out_targets)

        self.inputs = list(inputs) ; self.inputs.sort(key=base.Node.get_num_tar)
        self.outputs = list(outputs) ; self.outputs.sort(key=base.Node.get_num_tar)
        self.firsts_mt = firsts_mt ; firsts_mt.sort(key=base.Node.get_num_tar)
        self.inputs_mt = inputs_mt ; inputs_mt.sort(key=base.Node.get_num_tar)
        self.outputs_mt = outputs_mt ; outputs_mt.sort(key=base.Node.get_num_tar)
        self.first_nodes = first_nodes ; first_nodes.sort(key=base.Node.get_num)
        self.output_nodes = output_nodes ; output_nodes.sort(key=base.Node.get_num)
    # =========================
    
    # =========================
    def make_translator(self):
        # ATTRIBUTES NEEDED : s_nodes, p_structure
        # DO : translator
        dict_mt_to_ano_info = self.p_structure.dict_mt_to_ano_sn_info
        dict_tar_to_ano_tar_id = self.p_structure.dict_tar_to_ano_tar_id 
        self.translator = translator = ClusterTranslator()
        translator.dict_mt_to_ano_pair = dict()
        translator.dict_sn_to_ano_pair = dict()
        translator.dict_ano_pair_to_sn = dict()
        dict_ano_id_to_nb_seen = dict()

        for sn in self.s_nodes:
            ano_id = dict_mt_to_ano_info[sn.mt].ano_id
            if ano_id not in dict_ano_id_to_nb_seen:
                dict_ano_id_to_nb_seen[ano_id] = placement = 1
            else:
                dict_ano_id_to_nb_seen[ano_id] = placement \
                    = dict_ano_id_to_nb_seen[ano_id]+1
            pair = (ano_id,placement)
            translator.dict_mt_to_ano_pair[sn.mt] = pair
            translator.dict_sn_to_ano_pair[sn] = pair
            translator.dict_ano_pair_to_sn[pair] = sn

        # Experimental
        inputs_mt = list(self.inputs_mt) 
        inputs_mt.sort(key = base.Node.get_num_tar)
        for inp_nb,inp_mt in enumerate(inputs_mt):
            assert(inp_mt not in translator.dict_mt_to_ano_pair)
            inputs_sent = self.dict_input_mt_to_inputs_sent[inp_mt]
            inputs_num = [dict_tar_to_ano_tar_id[inp] for inp in inputs_sent]
            inputs_num.sort()
            translator.dict_mt_to_ano_pair[inp_mt] \
                = (str(inputs_num),f"input_{inp_nb}")
    # =============================

    # =============================
    def make_ano_cluster_id(self):
        # ATTRIBUTES NEEDED : s_nodes, p_structure, io
        # DO : ano_cluster_id, representee_cluster
        dict_mt_to_ano_info = self.p_structure.dict_mt_to_ano_sn_info
        dict_inputs_used = self.dict_first_mt_to_inputs_used
        dict_outputs_sent = self.dict_output_mt_to_outputs_sent
        charac_list = []
        for sn in self.s_nodes:
            ano_info : Ano_SimplifiedNode_Info = dict_mt_to_ano_info[sn.mt]
            charac_edges = []
            sn_deps = list(sn.deps.items())
            sn_deps.sort(key = lambda c : dict_mt_to_ano_info[c[0].mt].ano_id)
            for req_sn,used_targets in sn_deps:
                if req_sn not in self.s_nodes: continue
                charac_used_targets = []
                sort_key = lambda tar : ano_info.dict_tar_to_ano_nb[tar]
                used_targets = list(used_targets)
                used_targets.sort(key=sort_key)
                for used_target in used_targets:
                    charac_used_targets.append(
                        req_sn.all_targets.index(used_target)
                    )
                charac_edges.append((
                    dict_mt_to_ano_info[req_sn.mt].ano_id, # TO CHANGE : take index in self.s_nodes instead
                    charac_used_targets
                ))
            charac_inputs = []
            if sn in self.first_nodes:
                for input_used in dict_inputs_used[sn.mt]:
                    charac_inputs.append(self.inputs.index(input_used))
            charac_outputs = []
            if sn in self.output_nodes:
                for output_used in dict_outputs_sent[sn.mt]:
                    charac_outputs.append(self.outputs.index(output_used))
            charac_list.append((
                ano_info.ano_id,
                charac_edges,
                charac_inputs,
                charac_outputs
            ))
        charac_string = str(charac_list)
        dict_cluster_to_ano_id = \
            self.p_structure.dict_cluster_ano_charac_string_to_ano_cluster_id
        if charac_string in dict_cluster_to_ano_id:
            self.ano_cluster_id = ano_id = dict_cluster_to_ano_id[charac_string]
            self.representee_cluster \
                = self.p_structure.dict_ano_cluster_id_to_representee_cluster[ano_id]
        else:
            self.possible_partitioning = []
            self.partitioners_already_used = []
            self.ano_cluster_id \
                = ano_id \
                = dict_cluster_to_ano_id[charac_string] \
                = self.p_structure.counter_nb_unique_clusters.count()
            self.representee_cluster \
                = self.p_structure.dict_ano_cluster_id_to_representee_cluster[ano_id] \
                = self
    # =============================

    def partition(self,partitioner : Partitioner):
        if not (self.original_cluster is self):
            self.original_cluster.partition(partitioner)
        elif not (self.representee_cluster is self):
            self.representee_cluster.partition(partitioner)
        elif partitioner in self.partitioners_already_used:
            pass
        elif self.size < self.p_structure.min_size_to_trigger_partitioning:
            pass
        else:
            self.partitioners_already_used.append(partitioner)
            self.possible_partitioning.append(partitioner(self))

    def make_sub_cluster_original(self):
        if self.possible_partitioning is not None:
            for pg in self.possible_partitioning:
                pg.make_sub_cluster_original()

    def recompute_all_interfaces_and_edges(self):
        # It's useless to compute translator before
        # so we do it only now, and it help us know
        # if we already run this method on self
        if self.translator is None:
            self.make_io() # recompute interfaces
            self.make_translator()
            if self.representee_cluster is self:
                for pg in self.possible_partitioning:
                    pg.recompute_edges(self.p_structure.sg)






class PartitionedStructure():
    main_cluster : PartitionedCluster = None
    sg : SimplifiedGraph = None
    dict_info : dict = None
    dict_tar_to_ano_tar_id : dict[str, int] = None
    dict_mt_to_ano_sn_info : dict[str, Ano_SimplifiedNode_Info] = None
    dict_cluster_charac_string_to_cluster : dict[str, PartitionedCluster] = None
    dict_cluster_ano_charac_string_to_ano_cluster_id : dict[str,int] = None
    dict_ano_cluster_id_to_representee_cluster : dict[int,PartitionedCluster] = None
    min_size_to_trigger_partitioning : int = None
    def __init__(self,sg : SimplifiedGraph, min_size_to_trigger_partitioning = 4):
        self.sg = sg
        self.dict_info = sg.dict_info
        self.counter_nb_graphs = Counter()
        self.counter_nb_clusters = Counter()
        self.counter_nb_unique_clusters = Counter
        self.dict_tar_to_ano_tar_id = dict()
        self.dict_mt_to_ano_sn_info = dict()
        self.dict_cluster_charac_string_to_cluster = dict()
        self.dict_cluster_ano_charac_string_to_ano_cluster_id = dict()
        self.dict_ano_cluster_id_to_representee_cluster = dict()
        self.node_unique_id_generator = base.Node_unique_id_generator()
        self.min_size_to_trigger_partitioning = min_size_to_trigger_partitioning 

    # ==========================================================
    def make_dict_mt_to_ano_info(self,original_mod : torch.nn.Module):
        # ATTRIBUTES NEEDED : sg
        # DO : dict_mt_to_ano_sn_info
        # -> generate ano_sn_info for all the s_nodes
        self.dict_sn_charac_string_to_ano_id = dict_sn_charac_string_to_ano_id = dict()
        self.dict_tar_to_ano_tar_id = dict_tar_to_ano_tar_id = dict()
        dict_charac_info_to_ano_id = dict()
        nb_unique_sns = 0
        for sn in [self.sg.init_node] + self.sg.nodes:
            ano_sn_info = Ano_SimplifiedNode_Info(sn,self.sg,original_mod)
            sn_charac_string = ano_sn_info.make_charac_string()
            if sn_charac_string in dict_sn_charac_string_to_ano_id:
                ano_sn_info.ano_id \
                    = dict_sn_charac_string_to_ano_id[sn_charac_string]
            else:
                ano_sn_info.ano_id \
                    = dict_sn_charac_string_to_ano_id[sn_charac_string] \
                    = nb_unique_sns \
                    = nb_unique_sns + 1
            self.dict_mt_to_ano_sn_info[sn.mt] = ano_sn_info

        nb_unique_tar = 0
        for tar,info in self.sg.dict_info.items():
            charac_info = str(Ano_SimplifiedNode_Info.make_charac_info(info))
            if charac_info in dict_charac_info_to_ano_id:
                dict_tar_to_ano_tar_id[tar] \
                    = dict_charac_info_to_ano_id[charac_info]
            else:
                dict_tar_to_ano_tar_id[tar] \
                    = dict_charac_info_to_ano_id[charac_info] \
                    = nb_unique_tar \
                    = nb_unique_tar +1
        

    def make_graphs_names(self):
        for cluster in self.dict_ano_cluster_id_to_representee_cluster.values():
            for nb,pg in enumerate(cluster.possible_partitioning):
                pg.name = f"Possible_pg_{nb}_of_{cluster.name}"
                




class PartitionedDynamicManipulation(): # only contains staticmethod

    @staticmethod
    def prepare_dynamic_setup(pg : PartitionedGraph,cluster : PartitionedCluster):
        first_nodes = pg._first_nodes
        pg._first_nodes = None
        last_wrapping_graph = PartitionedGraph(parent_objet=cluster.p_structure)
        main_pn = PartitionedNode(
            last_wrapping_graph,
            sub_graph=pg
        )
        # ** inputs **
        inputs_pn = []
        dict_input_mt_to_pn = dict()
        for inp_mt in cluster.inputs_mt:
            inp_pn = PartitionedNode(last_wrapping_graph,main_target=inp_mt)
            inputs_pn.append(inp_pn)
            dict_input_mt_to_pn[inp_mt] = inp_pn
            inp_pn.users = set([main_pn])
        main_pn.deps = set(inputs_pn)
        for fst_node in first_nodes:
            inputs_used = cluster.dict_first_mt_to_inputs_used_mt[fst_node.mt]
            for inp_mt in inputs_used:
                inp_pn = dict_input_mt_to_pn[inp_mt]
                fst_node.deps_global.add(inp_pn)
                inp_pn.users_global.add(fst_node)
        # ** outputs **
        sink_pn = PartitionedNode(
            last_wrapping_graph,
            main_target = "last_wrapping_graph_output_node_sink"
        )
        main_pn.users = set([sink_pn])
        for out_pn in pg.output_nodes:
            out_pn.users_global.add(sink_pn)
            sink_pn.deps_global.add(out_pn)

        pg.pn_wrapping_it = main_pn
        last_wrapping_graph.nodes = inputs_pn + [main_pn,sink_pn]

    # ********
    # * WRAP *
    # ********
    # wrap a 'group' of nodes, currently living in 'main_pg'
    # into a new node 'new_pn', adding one level of depth.
    @staticmethod
    def wrap(group : list,main_pg : PartitionedGraph):
        new_pg = PartitionedGraph(parent_objet=main_pg)
        new_pg.nodes = group
        new_pn = PartitionedNode(
            main_graph = main_pg,
            sub_graph  = new_pg,
        )
        new_pg.pn_wrapping_it = new_pn
        set_group = set(group)
        new_pg.output_nodes = output_nodes = set()
        for pn in group:
            # ** link new_pn with global edges **
            new_pn.deps_global.update(pn.deps_global)
            new_pn.users_global.update(pn.users_global)
            # -> remove edges to inside at the end
            # -> reciprocal at the end

            # -> change pn.main_graph
            pn.main_graph = new_pg

            # ** inputs ** 
            if not pn.deps.issubset(set_group):
                deps_outside = pn.deps - set_group
                for req_pn in deps_outside:
                    req_pn.users.discard(pn)
                    req_pn.users.add(new_pn)
                    pn.deps.discard(req_pn)
                    new_pn.deps.add(req_pn)
            
            # ** outputs **
            if (pn in main_pg.output_nodes
            or (not pn.users.issubset(set_group))):
                output_nodes.add(pn)
                user_outside = pn.users - set_group
                for user_pn in user_outside:
                    user_pn.deps.discard(pn)
                    user_pn.deps.add(new_pn)
                    pn.users.discard(user_pn)
                    new_pn.users.add(user_pn)

        # ** mem_out **
        new_pn.mem_out = sum(out_pn.mem_out for out_pn in output_nodes)

        # ** global edges must not include edge to nodes inside new_pn **
        all_p_nodes_inside = new_pg.all_p_nodes_inside()
        new_pn.deps_global -= all_p_nodes_inside
        new_pn.users_global -= all_p_nodes_inside

        # ** reciprocal global edges **
        for req_g_pn in new_pn.deps_global:
            req_g_pn.users_global.add(new_pn)
        for user_g_pn in new_pn.users_global:
            user_g_pn.deps_global.add(new_pn)
        
        # ** update main_pg.nodes **
        main_lpn = main_pg.nodes
        main_lpn[main_lpn.index(group[0])] = new_pn
        for pn in group[1:]:
            main_lpn.remove(pn)

        # ** update main_pg outputs **
        main_out = main_pg.output_nodes
        if not main_out.isdisjoint(set_group):
            main_out.add(new_pn)
        main_pg.output_nodes -= set_group

        return new_pn


    # **********
    # * UNWRAP *
    # **********
    # unwrap 'pn' in its main graph
    @staticmethod
    def unwrap(pn : PartitionedNode):
        pg      : PartitionedGraph = pn.sub_graph
        main_pg : PartitionedGraph = pn.main_graph
        if pn.is_protected_from_unwrap: return ()
        group = list(pg.nodes)

        # ** unplug pn/pg **
        # -> global edges
        for req_g_pn in pn.deps_global: req_g_pn.users_global.remove(pn)
        for user_g_pn in pn.users_global: user_g_pn.deps_global.remove(pn)
        # not global edges will be overwritten anyway

        # ** plug back the group **
        # -> fix main_pg.nodes
        main_lpn = main_pg.nodes
        i = main_lpn.index(pn)
        main_pg.nodes = main_lpn[:i] + group + main_lpn[i+1:]    
        # -> fix sub_pn.main_graph
        for sub_pn in group:
            sub_pn.main_graph = main_pg
        # -> use the property : deps = deps_global inter nodes 
        main_spn = set(main_pg.nodes)
        all_p_nodes_inside = main_pg.all_p_nodes_inside()
        to_update = group + list(pn.deps) + list(pn.users)
        for sub_pn in to_update:
            sub_pn.deps  = sub_pn.deps_global.intersection(main_spn)
            sub_pn.users = sub_pn.users_global.intersection(main_spn)
            if not sub_pn.users_global.issubset(all_p_nodes_inside):
                main_pg.output_nodes.add(sub_pn)

        if pn in main_pg.output_nodes:
            main_pg.output_nodes.remove(pn)
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
    def freeze(pg : PartitionedGraph,p_structure : PartitionedStructure,partitioner : Partitioner):
        # return list of all nodes in pg for recursive purpose
        pg._first_nodes = pg.first_nodes # comment this if one day want to restart Dynamic
        all_snodes = set()
        for pn in pg.nodes:
            pn : PartitionedNode
            if pn.sub_graph is not None:
                sub_g = pn.sub_graph
                sub_snodes = PartitionedDynamicManipulation.freeze(sub_g,p_structure,partitioner)
                sub_c = PartitionedCluster(sub_snodes,p_structure)
                original_c = sub_c.original_cluster
                if original_c.representee_cluster is original_c:
                    original_c.partitioners_already_used.append(partitioner)
                    original_c.possible_partitioning.append(sub_g)
                # otherwise -> We won't keep this sub_graph
                # -> we are only interested in partitioning representee
                sub_g.cluster = original_c
                pn.sub_cluster = original_c
                pn.name = sub_c.name
                pn.sub_graph = None # comment this if one day want to restart Dynamic
                all_snodes.update(sub_snodes)
            else:
                if pn.simplified_node is None:
                    raise Exception(
                        f"PartitionedNode which is_leaf should have a self.simplified_node "\
                        f"(except special nodes, but there shouldn't be any "\
                        f"special node here). Here : pn.name : {pn.name}."
                    )
                all_snodes.add(pn.simplified_node)
        return all_snodes



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
            if not all(pn.is_protected_from_unwrap for pn in self.group):
                return False
            return self.__class__.utils_is_seq(self.group)
        
        def __eq__(self,opt2):
            return (self.set_group == opt2.set_group)
        def __hash__(self):
            return id(self)
    
    class Config():
        def __init__(self,
                max_len_seq = 99,
                max_estimate_for_main_graph = 30,
                max_estimate_per_sub_graph = 20,
                min_size_per_sub_graph = 3,
                main_graph_as_any_other = False,
                can_use_rotor = True,
                estimate_coeff_size = 1,
                estimate_coeff_sub_graph = 1,
                value_coeff_input_interfaces = 1,
                value_coeff_output_interfaces = 1,
                value_power_total_size = 0.5,
                old_value_fct = False,
                old_value_fct_value_power_not_last = 1.1,
        ):
            self.max_len_seq = max_len_seq
            self.can_use_rotor = can_use_rotor
            self.min_size_per_sub_graph = min_size_per_sub_graph
            self.max_estimate_per_sub_graph \
                = max_sub \
                = max_estimate_per_sub_graph
            self.max_estimate_for_main_graph \
                = max_sub if main_graph_as_any_other \
                else max_estimate_for_main_graph
            # -- estimate_fct --
            self.estimate_coeff_size = estimate_coeff_size
            self.estimate_coeff_sub_graph = estimate_coeff_sub_graph
            self.option_estimate_fct = self.default_estimate_fct
            # -- value_fct --
            self.old_value_fct_value_power_not_last = old_value_fct_value_power_not_last
            self.value_coeff_input_interfaces = value_coeff_input_interfaces
            self.value_coeff_output_interfaces = value_coeff_output_interfaces
            self.value_power_total_size = value_power_total_size
            if old_value_fct:
                self.option_value_fct = self.old_default_option_value_fct
            else:
                self.option_value_fct = self.default_option_value_fct
            self.option_stop_fct = self.default_option_stop_fct
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
            tot_mem_internal = sum(pn.mem_out for pn in not_last_nodes)
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
            inputs_mem = sum(pn.mem_out for pn in inputs_pn if pn.simplified_node is not None)
            outputs_mem = sum(pn.mem_out for pn in outputs_pn if pn.simplified_node is not None)
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
            

    config : Config = None
    def __init__(self, **kwargs):
        self.config = self.__class__.Config(**kwargs)

    # === FIRST WAY TO FIND OPTIONS : SEQUENCES ===
    def find_seq_options(self,pg : PartitionedGraph):
        # ** Find the sequences **
        tot_nb_seq = 0
        dict_seq_nb = dict() # name -> a seq nb
        dict_sequences = dict() # seq nb -> list of nodes in the seq
        for pn in pg.nodes:
            if len(pn.users) == 1 and len(list(pn.users)[0].deps) == 1:
                name = pn.name
                user_pn = list(pn.users)[0]
                user_name = user_pn.name
                if name in dict_seq_nb:
                    seq_nb = dict_seq_nb[name]
                    dict_seq_nb[user_name] = seq_nb
                    dict_sequences[seq_nb].append(user_pn)
                else:
                    tot_nb_seq += 1
                    dict_seq_nb[name] = tot_nb_seq
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
        # === GENERALIZED VERSION OF RK_get_1_separators ===
        # for each node we need to find where its flow converge back
        # -> flow of a pn is defined as nodes in
        # -> `to_be_visited` which are descendants of pn

        # ATTENTION here source/sink are taken from .deps
        # relation perspective, e.g. outputs are sources.
        # /!\ Note that its merging here, not grouping /!\

        dict_nb_usages = dict([(pn, len(pn.users)) for pn in pg.nodes])
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
            if len(pn.users) == 0:
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
            for req_pn in pn.deps:
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
        all_options = (
                self.find_seq_options(pg)
            +   self.find_flow_options(pg)
        )

        _all_options = list(all_options)
        for opt in _all_options:
            if all(not pn.does_requires_grad()
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
        pg : PartitionedGraph = cluster.init_PartitionedGraph()
        pg.cluster = cluster
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
        all_snodes = PartitionedDynamicManipulation.freeze(
            pg,cluster.p_structure,self.__class__)
        if all_snodes != set(cluster.s_nodes):
            raise Exception(
                f"BUG in {self.__class__}. When collecting all the "\
                f"SimplifiedNodes at the end, we don't find cluster.s_nodes. We probably "\
                f"lost some nodes...\n Original nb of nodes : {len(cluster.s_nodes)}; "\
                f"Nb of nodes at the end : {len(all_snodes)}"
            )
        return pg




class PartitionerSequence(Partitioner):
    class Config():
        def __init__(self,sub_partitioner : Partitioner = None):
            if sub_partitioner is None:
                sub_partitioner = PartitionerBottomToTop(
                    main_graph_as_any_other = True
                )
            self.sub_partitioner = sub_partitioner 

    config : Config = None
    def __init__(self, **kwargs):
        self.config = self.__class__.Config(**kwargs)

    def __call__(self, cluster : PartitionedCluster):
        pg : PartitionedGraph = cluster.init_PartitionedGraph()
        pg.cluster = cluster
        dict_info = cluster.p_structure.dict_info

        # ** Add a temporary global source before get separators**
        tmp_global_source_pn = PartitionedNode(main_graph=pg,main_target="tmp_source")
        pg.nodes.insert(0,tmp_global_source_pn)
        first_nodes = pg.first_nodes
        for first_pn in first_nodes:
            if first_pn.deps == set():
                first_pn.deps.add(tmp_global_source_pn)
                tmp_global_source_pn.users.add(first_pn)

        seps_pn = pg.find_cutting_points()

        # -> remove tmp_source
        for first_pn in first_nodes:
            first_pn.deps.discard(tmp_global_source_pn)
        pg.nodes.remove(tmp_global_source_pn)

        # -> multiple output_nodes
        if not (seps_pn[-1] is pg.nodes[-1]):
            seps_pn.append(pg.nodes[-1])
        seps_mt = [sep.mt for sep in seps_pn]
        seps_sn = [sn for sn in cluster.s_nodes if sn.mt in seps_mt]
        seps_index = [-1]+[cluster.s_nodes.index(sep) for sep in seps_sn]
        pg.nodes = p_nodes = []
        for block_nb in range(len(seps_sn)):
            first_i = seps_index[block_nb]
            last_i = seps_index[block_nb+1]
            if last_i - first_i == 1:
                sn = cluster.s_nodes[last_i]
                block_pn = PartitionedNode(pg,main_target=sn.mt,sn=sn)
                block_pn.mem_out = dict_info[sn.mt].memsize
            else:
                block_s_nodes = cluster.s_nodes[first_i+1:last_i+1]
                sub_cluster = PartitionedCluster(block_s_nodes,cluster.p_structure)
                block_pn = PartitionedNode(pg,sub_cluster=sub_cluster)
                block_pn.mem_out = dict_info[cluster.s_nodes[last_i].mt].memsize
            p_nodes.append(block_pn)
            if block_nb > 0:
                prev_pn : PartitionedNode = p_nodes[-2]
                block_pn.deps.add(prev_pn)
                block_pn.deps_global.add(prev_pn)
                prev_pn.users.add(block_pn)
                prev_pn.users_global.add(block_pn)
                # global edges are useless since not dynamic
        pg._first_nodes = set([p_nodes[0]])
        pg.output_nodes = set([p_nodes[-1]])
        # -- sub partition --
        for block_pn in p_nodes:
            if block_pn.sub_cluster is not None:
                sub_cluster : PartitionedCluster = block_pn.sub_cluster
                sub_cluster.partition(self.config.sub_partitioner)
        return pg


def S_to_P(
    sg : SimplifiedGraph,
    original_mod : torch.nn.Module,
    partitioners = [
        Partitioner(),
        Partitioner_OLD_bottom_to_top(),
        PartitionerBottomToTop(),
        PartitionerSequence()
    ],
    min_size_to_trigger_partitioning = 4):
    # sg = copy_SimplifiedGraph(sg)
    sg.discard_all_artifacts()
    p_structure = PartitionedStructure(sg,
        min_size_to_trigger_partitioning = min_size_to_trigger_partitioning)
    p_structure.make_dict_mt_to_ano_info(original_mod)
    p_structure.main_cluster = main_cluster \
        = PartitionedCluster(list(sg.nodes),p_structure)
    for partitioner in partitioners:
        main_cluster.partition(partitioner)
    main_cluster.make_sub_cluster_original()
    p_structure.make_graphs_names()
    sg.delete_edges_via_artifacts()
    main_cluster.recompute_all_interfaces_and_edges()
    return p_structure





# ==========================
# === printing functions ===
# ==========================

color_leaf     = "blue"
color_sub_graph = "blueviolet"
color_special  = "green"
color_edge     = color_leaf

def aux_print_PartitionedGraph_message(pg : PartitionedGraph):
    return f"PartitionedGraph - Partitioned forward graph : of size {len(pg.nodes)}"
def aux_print_PartitionedGraph_name(pg,name=None):
    if name is not None: return name
    else: return "Partitioned_Forward_PartitionedGraph"

def aux_print_PartitionedCluster_message(pc : PartitionedCluster):
    possible_pg = pc.representee_cluster.possible_partitioning
    return f"{pc.name}, with {len(possible_pg)} possible partitioning"
def aux_print_PartitionedCluster_names(pc : PartitionedCluster,name=None):
    if name is None: name = pc.name
    parti_used = pc.representee_cluster.partitioners_already_used
    return [f"PartitionedGraph_{i}_via_{type(parti)}_of_{name}" for i,parti in enumerate(parti_used)]

def print_PartitionedGraph(pg : PartitionedGraph,name=None,open=True,render_format="svg",dot=None,uniq_num=0):
    # ----- init -----
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,**kwargs): dot.edge(uni(i1),uni(i2), **kwargs)
    # ----- Core -----
    cluster : PartitionedCluster = pg.cluster
    for pn in pg.nodes:
        if pn.is_leaf:
            node(pn.name,pn.name,color=color_leaf)
        else:
            node(
                pn.name,
                f"{pn.name}\nCluster size: {pn.size}",
                color=color_sub_graph)
        for req_pn in pn.deps:
            edge(req_pn.name,pn.name,color=color_edge)
    
    # -> input
    first_nodes = list(pg.first_nodes)
    kwargs = dict(color = color_edge,style="dashed")
    if first_nodes != []:
        node(
            "inputs",
            f"INPUTS:\n"+"\n".join(cluster.inputs_mt),
            color=color_special, style="dashed")
        for pn in first_nodes:
            edge("inputs",pn.name,**kwargs)

    # -> output
    node(
        "outputs",
        f"OUTPUTS:\n"+"\n".join(cluster.outputs_mt),
        color=color_special, style="dashed")
    for pn in pg.output_nodes:
        edge(pn.name,"outputs",**kwargs)
    # ----- render -----
