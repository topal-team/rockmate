# ==========================
# ====== P structure =======
# ==========================

# TODO TODO ensure that a sub_graph always requires_grad

# ** Graph partitioning **

from .utils import *
from .Stools import S_graph, S_node, copy_S_graph
from .Ktools import K_C_node, K_D_node


# *************
# * P_cluster *
# *************

class Ano_S_node_Info():
    ano_id : int = None
    ano_str_code : str = None
    dict_tar_to_ano_nb : dict[str, int] = None
    dict_tar_to_ano_tar : dict[str, str] = None
    dict_cst_to_ano_cst : dict[str, str] = None
    dict_param_to_ano_param : dict[str, str] = None
    dict_ano_tar_to_basic_info : dict[str, def_info.Var_info] = None
    dict_ano_cst_to_basic_info : dict[str, def_info.Var_info] = None
    dict_ano_param_to_basic_info : dict[str, def_info.Var_info] = None

    # =====================================================================
    def __init__(self,sn : S_node, sg : S_graph, model : torch.nn.Module):
        # DO: everything except self.ano_id and self.charac_string
        # -> Similar to Atools_for_S_and_K.Graph_translator.__init__
        # =============================
        # === FIRST : read the code ===
        all_real_vars   = []
        all_real_cst    = []
        all_real_params = []
        def handle_str(real_str):
            if (real_str[:2] == "__"
            and not real_str in all_real_vars):
                all_real_vars.append(real_str)
            elif (real_str[:5] == "self."
            or real_str[:5] == "self["
            or real_str[:13] == "getattr(self."
            and not real_str in all_real_params):
                all_real_params.append(real_str)
            elif (real_str[:5] == "_cst_"
            and not real_str in all_real_cst):
                all_real_cst.append(real_str)
        def search_through(a):
            if isinstance(a,ast.AST):
                if isinstance(a,ast.Name):
                    handle_str(a.id)
                else:
                    for s in a._fields:
                        try: search_through(getattr(a,s))
                        except: pass
            elif isinstance(a,str): handle_str(a)
            elif hasattr(a,"__iter__"):
                for sub_a in a: search_through(sub_a)

        search_through(sn.get_code_ast())

        # ===============================================
        # === SECOND : build anonymized tgt/cst/param ===
        self.dict_tar_to_ano_nb = dict_tar_anb = dict()
        self.dict_tar_to_ano_tar = dict_tar_atar = dict()
        self.dict_cst_to_ano_cst = dict_cst_acst = dict()
        self.dict_param_to_ano_param = dict_param_aparam = dict()
        self.dict_ano_tar_to_basic_info = dict_atar_info = dict()
        self.dict_ano_cst_to_basic_info = dict_acst_info = dict()
        self.dict_ano_param_to_basic_info = dict_aparam_info = dict()
        # Build ano targets + info
        all_real_vars = sorted(all_real_vars,key = RK_node.get_num_tar)
        nb_var = 0
        for real_name in all_real_vars:
            nb_var += 1
            atar = f"__{nb_var}_ano"
            dict_tar_atar[real_name] = atar
            dict_tar_anb[real_name] = nb_var
            dict_atar_info[atar] = sg.dict_info[real_name]
            # -> We will keep only basic attributes of Var_info

        # Build ano constants + info
        all_real_cst = sorted(all_real_cst,key = RK_node.get_num_cst)
        nb_cst = 0
        for cst_real_name in all_real_cst:
            value = sg.dict_constants[cst_real_name]
            nb_cst += 1
            acst = f"_cst_{nb_cst}_ano"
            dict_cst_acst[cst_real_name] = acst
            dict_acst_info[acst] = def_info.Var_info(value)

        # Build ano params + info
        nb_param = 0
        for param_full_name in all_real_params: # strings
            # -> e.g. param_full_name = "self.layer1.weight"
            param_value = eval(param_full_name,{"self":model},{})
            nb_param += 1
            aparam = f"self.param_{nb_param}"
            dict_param_aparam[param_full_name] = aparam
            dict_aparam_info[aparam] = def_info.Var_info(param_value)
                
        # =============================
        # === THIRD: build ano code ===
        str_code = sn.get_code()
        for tar,atar in dict_tar_atar.items():
            str_code = str_code.replace(tar,atar)
        for cst,acst in dict_cst_acst.items():
            str_code = str_code.replace(cst,acst)
        for param,aparam in dict_param_aparam.items():
            str_code = str_code.replace(param,aparam)
        self.ano_code = str_code
    # ============================


    # ============================
    @staticmethod
    def make_charac_info(info : def_info.Var_info):
        if info.ttype is tuple or info.ttype is list:
            return (
                info.ttype,
                [Ano_S_node_Info.make_charac_info(sub) for sub in info.sub_info]
            )
        else:
            return (
                info.ttype,
                info.dtype if hasattr(info,"dtype") else None,
                info.tsize if hasattr(info,"tsize") else None,
                info.requires_grad if hasattr(info,"requires_grad") else None,
                info.memsize if hasattr(info,"memsize") else None,
            )
    # ============================

    # ============================
    def make_charac_string(self):
        charac_list = [self.ano_code]
        for atar,info in self.dict_ano_tar_to_basic_info.items():
            charac_list.append((atar,Ano_S_node_Info.make_charac_info(info)))
        for acst,info in self.dict_ano_cst_to_basic_info.items():
            charac_list.append((acst,Ano_S_node_Info.make_charac_info(info)))
        for aparam,info in self.dict_ano_param_to_basic_info.items():
            charac_list.append((aparam,Ano_S_node_Info.make_charac_info(info)))
        return str(charac_list)
    # ============================


class Cluster_translator():
    dict_mt_to_ano_pair : dict[str, tuple[int,int]] = None
    dict_sn_to_ano_pair : dict[S_node, tuple[int,int]] = None
    dict_ano_pair_to_sn : dict[tuple[int,int], S_node] = None
    dict_kcn_to_ano_triplet : dict[K_C_node, tuple[str,int,int]] = None
    dict_kdn_to_ano_triplet : dict[K_D_node, tuple[str,int,int]] = None
    dict_ano_triplet_to_kcn : dict[tuple[str,int,int], K_C_node] = None
    dict_ano_triplet_to_kdn : dict[tuple[str,int,int], K_D_node] = None
    dict_name_to_ano_triplet : dict = None
    dict_ano_triplet_to_name : dict = None
    def __init__(self):
        pass


class Partitioner():
    class Config():
        def __init__(self):
            pass
    def __init__(self):
        self.config : self.__class__.Config = None
    def __call__(self, cluster):
        return cluster.init_P_graph()
        # raise Exception(
            # "Base class of partitioners. __call__ method "\
            # "must be overwritten by all subclasses")


class P_node(RK_node):
    is_protected_from_unwrap = None
    mem_out = None
    def __init__(self,
            main_graph,
            sub_cluster = None,
            main_target = None,
            sub_graph   = None, # FOR DYNAMIC PARTITIONING
            sn          = None):
        super().__init__("P",other_obj=main_graph)
        self.main_graph  = main_graph
        self.sub_cluster = sub_c = sub_cluster
        self.main_target = mt = main_target
        self.sub_graph   = sub_g = sub_graph
        self.sn = sn # used for .deps/.user to compute io_targets
        if int(mt is not None) + int(sub_g is not None) + int(sub_c is not None) != 1:
            raise Exception(
                "A P_node is usually defined either by a main_target "\
                "or a sub_cluster, never both.\nFor dynamic partitioning, "\
                "you can instead giving a sub_graph, but it must be "\
                "temporary, during the partitioning."
            )
        if sub_c is not None:
            self.name = sub_c.name
            self.is_leaf = False
        elif sub_g is not None:
            self.sub_graph_id = sid = sub_g.graph_id
            self.name        = f"sub_graph_{sid}"
            self.is_leaf     = False
            self.is_protected_from_unwrap = False
        else:
            self.sub_graph_id = None
            self.name        = f"Var_{mt}"
            self.is_leaf     = True
            self.is_protected_from_unwrap = True

        self.deps         = set()
        self.users        = set()
        self.deps_global  = set() # FOR DYNAMIC PARTITIONING
        self.users_global = set() # FOR DYNAMIC PARTITIONING
        # Global edges contain ALL the deps/users, of any depth

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


class P_graph(RK_graph):
    name : str = None
    pn_wrapping_it : P_node = None # -> pn representing self in an upper graph
    _first_nodes : list[P_node] = None
    cluster = None # just for debug + printing
    without_artefacts = False
    def __init__(self,graph_id,p_structure=None,other_pg=None,dict_info=None):
        if p_structure is None and other_pg is None and dict_info is None:
            raise Exception(
                "P_graph.__init__ needs a dict_info, you can give it "\
                "a p_structure or an other_pg (since they both have a dict_info)."
            )
        other_obj = p_structure if other_pg is None else other_pg
        super().__init__("P",other_obj)
        self.dict_info = other_obj.dict_info if dict_info is None else dict_info
        self.output_nodes = set()
        # Note: self.nodes contains only the nodes of self
        self.graph_id = graph_id
        self.next_sub_graph_id = 1 # for new sub_graphs' id

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
    @property
    def is_main_graph(self):
        return bool(self.graph_id == "0")

    def all_p_nodes_inside(self): # FOR DYNAMIC PARTITIONING
        all_p = set(self.nodes)
        for pn in self.nodes:
            if pn.sub_graph is not None:
                all_p.update(pn.sub_graph.all_p_nodes_inside())
        return all_p

    def make_sub_graph_id(self,new_graph_id):
        self.graph_id = new_graph_id
        next_id = 1
        for pn in self.nodes:
            if not pn.is_leaf:
                new_sg_id = f"{new_graph_id}_{next_id}"
                next_id  += 1
                pn.sub_graph.make_sub_graph_id(new_sg_id)
                pn.sub_graph_id = new_sg_id
                pn.name        = f"sub_graph_{new_sg_id}"
        self.next_sub_graph_id = next_id

    def set_all_protected_to_false(self):
        for pn in self.nodes:
            if not pn.is_leaf:
                pn.is_protected_from_unwrap = False
                pn.sub_graph.set_all_protected_to_false()

    def make_sub_cluster_original(self):
        for pn in self.nodes:
            pn : P_node
            if pn.sub_cluster is not None:
                sub_c = pn.sub_cluster.original_cluster
                if sub_c is pn.sub_cluster:
                    sub_c.make_sub_cluster_original()
                else:
                    pn.sub_cluster = sub_c

    def recompute_edges(self,sg : S_graph):
        # NOT OPTIMAL AT ALL -> To improve if to slow
        # find artefact_edges in P_graph
        # Moreover, recomputing edges is not necessary
        # it's just that it's cleaner to have an accurate P_graph
        # at the end
        if not self.without_artefacts:
            self.without_artefacts = True
            all_related_to_artefacts = set().union(
                set(e[0] for e in sg.artefact_edges),
                set(e[1] for e in sg.artefact_edges),
            )
            dict_where = dict()
            for sn in all_related_to_artefacts:
                for pn in self.nodes:
                    pn : P_node
                    if pn.sn is sn:
                        dict_where[sn] = pn
                    elif (pn.sub_cluster is not None
                        and sn in pn.sub_cluster.s_nodes):
                        dict_where[sn] = pn
            
            # search edges we might have to delete
            suspicious_edges : set[tuple[P_node,P_node]] = set()
            for (req_sn,user_sn,_) in sg.artefact_edges:
                if req_sn in dict_where and user_sn in dict_where:
                    req_pn = dict_where[req_sn]
                    user_pn = dict_where[user_sn]
                    if req_pn is not user_pn:
                        suspicious_edges.add((req_pn,user_pn))

            # for suspicious edges, we need to see if there is any
            # S edge from one to the other (all artefact edges have
            # been deleted, that's why we need to recompute P edges)
            for req_pn,user_pn in suspicious_edges:
                if req_pn.sub_cluster is not None:
                    all_sn_req_pn = req_pn.sub_cluster.s_nodes
                else:
                    all_sn_req_pn = [req_pn.sn]
                if user_pn.sub_cluster is not None:
                    all_sn_user_pn = user_pn.sub_cluster.s_nodes
                else:
                    all_sn_user_pn = [user_pn.sn]
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
        

class P_cluster():
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
        self.p_structure = p_structure
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
            cluster_nb \
                = self.p_structure.nb_clusters \
                = self.p_structure.nb_clusters +1
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
    def init_P_graph(self):
        assert(self.original_cluster is self)
        pg = P_graph("0",self.p_structure)
        pg.cluster = self
        pg.nodes = p_nodes = []
        dict_info = self.p_structure.dict_info
        dict_mt_to_pn = dict()
        for sn in self.s_nodes:
            pn = P_node(
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
        sg : S_graph = self.p_structure.sg
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
        if sg.special_output_node is None: # sg has only one output_node
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
                    out_targets = sg.special_output_node.deps[out_node]
                    outputs.update(out_targets)
                    if out_node.mt not in outputs_mt:
                        outputs_mt.append(out_node.mt)
                        output_nodes.append(out_node)
                        dict_outputs_sent[out_node.mt] = set(out_targets)
                    else:
                        dict_outputs_sent[out_node.mt].update(out_targets)

        self.inputs = list(inputs) ; self.inputs.sort(key=RK_node.get_num_tar)
        self.outputs = list(outputs) ; self.outputs.sort(key=RK_node.get_num_tar)
        self.firsts_mt = firsts_mt ; firsts_mt.sort(key=RK_node.get_num_tar)
        self.inputs_mt = inputs_mt ; inputs_mt.sort(key=RK_node.get_num_tar)
        self.outputs_mt = outputs_mt ; outputs_mt.sort(key=RK_node.get_num_tar)
        self.first_nodes = first_nodes ; first_nodes.sort(key=RK_node.get_num)
        self.output_nodes = output_nodes ; output_nodes.sort(key=RK_node.get_num)
    # =========================
    
    # =========================
    def make_translator(self):
        # ATTRIBUTES NEEDED : s_nodes, p_structure
        # DO : translator
        dict_mt_to_ano_info = self.p_structure.dict_mt_to_ano_sn_info
        dict_tar_to_ano_tar_id = self.p_structure.dict_tar_to_ano_tar_id 
        self.translator = translator = Cluster_translator()
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
        inputs_mt.sort(key = RK_node.get_num_tar)
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
            ano_info : Ano_S_node_Info = dict_mt_to_ano_info[sn.mt]
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
                    dict_mt_to_ano_info[req_sn.mt].ano_id,
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
                = self.p_structure.nb_unique_clusters \
                = self.p_structure.nb_unique_clusters + 1
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






class P_structure():
    main_cluster : P_cluster = None
    sg : S_graph = None
    dict_info : dict = None
    dict_tar_to_ano_tar_id : dict[str, int] = None
    dict_mt_to_ano_sn_info : dict[str, Ano_S_node_Info] = None
    dict_cluster_charac_string_to_cluster : dict[str, P_cluster] = None
    dict_cluster_ano_charac_string_to_ano_cluster_id : dict[str,int] = None
    dict_ano_cluster_id_to_representee_cluster : dict[int,P_cluster] = None
    nb_clusters : int = None
    nb_unique_clusters : int = None
    min_size_to_trigger_partitioning : int = None
    def __init__(self,sg : S_graph, min_size_to_trigger_partitioning = 4):
        self.sg = sg
        self.dict_info = sg.dict_info
        self.nb_clusters = 0
        self.nb_unique_clusters = 0
        self.dict_tar_to_ano_tar_id = dict()
        self.dict_mt_to_ano_sn_info = dict()
        self.dict_cluster_charac_string_to_cluster = dict()
        self.dict_cluster_ano_charac_string_to_ano_cluster_id = dict()
        self.dict_ano_cluster_id_to_representee_cluster = dict()
        self.node_unique_id_generator = Node_unique_id_generator()
        self.min_size_to_trigger_partitioning = min_size_to_trigger_partitioning 

    # ==========================================================
    def make_dict_mt_to_ano_info(self,model : torch.nn.Module):
        # ATTRIBUTES NEEDED : sg
        # DO : dict_mt_to_ano_sn_info
        # -> generate ano_sn_info for all the s_nodes
        self.dict_sn_charac_string_to_ano_id = dict_sn_charac_string_to_ano_id = dict()
        self.dict_tar_to_ano_tar_id = dict_tar_to_ano_tar_id = dict()
        dict_charac_info_to_ano_id = dict()
        nb_unique_sns = 0
        for sn in [self.sg.init_node] + self.sg.nodes:
            ano_sn_info = Ano_S_node_Info(sn,self.sg,model)
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
            charac_info = str(Ano_S_node_Info.make_charac_info(info))
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
                

class P_Dynamic_manipulation(): # only contains staticmethod

    @staticmethod
    def prepare_dynamic_setup(pg : P_graph,cluster : P_cluster):
        first_nodes = pg._first_nodes
        pg._first_nodes = None
        last_wrapping_graph = P_graph("-1",cluster.p_structure)
        main_pn = P_node(
            last_wrapping_graph,
            sub_graph=pg
        )
        # ** inputs **
        inputs_pn = []
        dict_input_mt_to_pn = dict()
        for inp_mt in cluster.inputs_mt:
            inp_pn = P_node(last_wrapping_graph,main_target=inp_mt)
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
        sink_pn = P_node(
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
    def wrap(group : list,main_pg : P_graph):
        group_nb = main_pg.next_sub_graph_id
        main_pg.next_sub_graph_id += 1
        new_pg = P_graph(
            graph_id=f"{main_pg.graph_id}_{group_nb}",
            other_pg=main_pg
        )
        new_pg.nodes = group
        new_pn = P_node(
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
    def unwrap(pn : P_node):
        pg      : P_graph = pn.sub_graph
        main_pg : P_graph = pn.main_graph
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
    def merge(group : list, main_pg : P_graph):
        new_pn = P_Dynamic_manipulation.wrap(group,main_pg)
        new_pg = new_pn.sub_graph
        original_lpn = new_pg.nodes
        for sub_pn in original_lpn:
            P_Dynamic_manipulation.unwrap(sub_pn)
        main_pg.make_sub_graph_id(main_pg.graph_id)
        return new_pn
    
    # **********
    # * FREEZE *
    # **********
    # Freeze the dynamic structure : sub_graph -> sub_cluster
    @staticmethod
    def freeze(pg : P_graph,p_structure : P_structure,partitioner : Partitioner):
        # return list of all nodes in pg for recursive purpose
        pg._first_nodes = pg.first_nodes # comment this if one day want to restart Dynamic
        all_snodes = set()
        for pn in pg.nodes:
            pn : P_node
            if pn.sub_graph is not None:
                sub_g = pn.sub_graph
                sub_snodes = P_Dynamic_manipulation.freeze(sub_g,p_structure,partitioner)
                sub_c = P_cluster(sub_snodes,p_structure)
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
                if pn.sn is None:
                    raise Exception(
                        f"P_node which is_leaf should have a self.sn : S_node "\
                        f"(except special nodes, but there shouldn't be any "\
                        f"special node here). Here : pn.name : {pn.name}."
                    )
                all_snodes.add(pn.sn)
        return all_snodes
            

class Partitioner_OLD_bottom_to_top(Partitioner): 
    # /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\
    # /!\ NOT DETERMINISTIC AND SOON NO LONGER MAINTAINED /!\
    # /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\
    class Option():
        def __init__(self,group):
            self.group = group
        @property
        def nb_nodes(self):
            return len(self.group)
        @property
        def nb_subnodes(self):
            return sum(pn.size for pn in self.group)

    class Config():
        def __init__(self,
                max_nodes_for_main_graph = 10,
                max_nodes_per_sub_graph = 10,
                min_nodes_per_sub_graph = 3,
                option_value_fct = None,
                merge_flow_stop_fct = None,
                ):
            self.min_nodes_per_sub_graph = min_nodes_per_sub_graph  
            self.max_nodes_per_sub_graph = max_nodes_per_sub_graph
            self.max_nodes_for_main_graph = max_nodes_for_main_graph

            self.option_value_fct = option_value_fct \
                if option_value_fct is not None \
                else self.get_default_option_value_fct()
            self.merge_flow_stop_fct = merge_flow_stop_fct \
                if merge_flow_stop_fct is not None \
                else self.get_default_merge_flow_stop_fct()

        def get_default_option_value_fct(self,
            importance_nb_nodes = 1,
            importance_nb_subnodes = -1/4,
            constant_value = 5):
            def value_fct(option):
                if option.nb_subnodes > self.max_nodes_per_sub_graph:
                    return -10
                else:
                    val = (option.nb_nodes * importance_nb_nodes
                        + option.nb_subnodes * importance_nb_subnodes
                        + constant_value)
                    val = math.exp(val)
                    return val
            return value_fct
        
        def get_default_merge_flow_stop_fct(self):
            def merge_flow_stop_condition(pg,best_option): # 'True' means stop
                pg : P_graph
                tot_nb_nodes = len(pg.nodes)
                if pg.is_main_graph:
                    if tot_nb_nodes <= self.max_nodes_for_main_graph: return True
                else:
                    if tot_nb_nodes <= self.max_nodes_per_sub_graph: return True
                if best_option.nb_subnodes > self.max_nodes_per_sub_graph: return True
                else:
                    value = self.option_value_fct(best_option)
                    limit = math.sqrt(tot_nb_nodes) * math.sqrt(self.max_nodes_per_sub_graph)
                    return value <= limit
            return merge_flow_stop_condition

    config : Config = None
    def __init__(self, **kwargs):
        self.config = self.__class__.Config(**kwargs)

    # === RULE : GROUP SEQUENCE OF NODES TOGETHER ===
    def rule_group_sequences(self,pg : P_graph):
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
        all_sequences = list(dict_sequences.items())

        for seq_nb,sequence in all_sequences:
            if not pg.does_node_requires_grad(sequence[-1]):
                del dict_sequences[seq_nb]

        # ** Group each sequence **
        for seq_nb,sequence in dict_sequences.items():
            if len(sequence) >= self.config.min_nodes_per_sub_graph:
                new_pn = P_Dynamic_manipulation.wrap(sequence,pg)

        pg.make_sub_graph_id(pg.graph_id)

    # === RULE : MERGE NODES WITH A UNIQUE COMMON ANCESTOR ===
    # the flow of pn are nodes in `to_be_visited` which are descendants of pn
    def rule_merge_small_flows(self,pg : P_graph):
        # === FIRST ===
        # for each node we need to find where its flow converge back
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
        # ie a list of P_nodes, representing there flow
        # reciprocal of dict_flow 
        dict_end_of_flow = dict()
        # any pn -> where its flow converged back
        # this is what we want to build

        # ** Add a temporary global source **
        tmp_global_source_pn = P_node(main_graph=pg,main_target="tmp_source")
        tmp_global_source_pn.users = first_nodes = pg.first_nodes
        for first_pn in first_nodes:
            first_pn.deps.add(tmp_global_source_pn)
        dict_nb_usages[tmp_global_source_pn] = len(first_nodes)

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
                # equivalent to "seen.remove(n)" in Rk_get_1_separators
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
                    if (not (req_pn is tmp_global_source_pn)
                    and not (req_pn in tot_flow)):
                        tot_flow.append(req_pn)
                if req_pn in dict_which_flow:
                    dict_which_flow[req_pn].update(continuing_flows)
                else:
                    dict_which_flow[req_pn] = set(continuing_flows)
                dict_nb_usages[req_pn]-=1
                if dict_nb_usages[req_pn]==0:
                    to_be_visited.append(req_pn)

        # ** remove the temporary global source **
        for first_pn in first_nodes:
            first_pn.deps.remove(tmp_global_source_pn)

        # === SECOND ===
        # For each flow we have 4 options :
        # -> include the source or not
        # -> include the sink or not
        # But I will consider only 2 :
        # -> We always include the source
        # ATTENTION here sink/source are taken from .deps
        # relation perspective, e.g. outputs are sources.

        # /!\ Note that its merging here, not grouping /!\

        all_options : set[self.__class__.Option] = set()
        dict_options_pn_is_part_of = dict()
        # P_node -> merge_flow_option set
        # After each simplification, we actualize all 
        # the options which use some of the simplified nodes

        # ** init **
        for source_pn,sink_pn in dict_end_of_flow.items():
            flow = dict_total_flow[source_pn]
            flow.reverse()
            options = [self.__class__.Option(flow)]
            if (not (sink_pn is tmp_global_source_pn) and len(flow)>2):
                flow_ = list(flow) ; flow_.remove(sink_pn)
                options.append(self.__class__.Option(flow_))
            for opt in options:
                if len(opt.group) <= 1: continue
                all_options.add(opt)
                for pn in opt.group:
                    if pn in dict_options_pn_is_part_of:
                        dict_options_pn_is_part_of[pn].add(opt)
                    else:
                        dict_options_pn_is_part_of[pn] = set([opt])

        _all_options = set(all_options)
        dict_info = pg.dict_info
        for opt in _all_options:
            set_group = set(opt.group)
            if all(not pn.does_requires_grad(dict_info)
                   for pn in opt.group):
                all_options.remove(opt)
            elif all(pn.deps.issubset(set_group)
                     for pn in opt.group):
                all_options.remove(opt)

        # ** main loop **
        while all_options != set():
            best_option = max(all_options,key=self.config.option_value_fct)
            all_options.remove(best_option)
            if self.config.merge_flow_stop_fct(pg,best_option):
                break
            else:
                best_group = list(best_option.group)
                new_pn = P_Dynamic_manipulation.merge(best_group,pg)
                updated_opts = set()
                for pn in best_group:
                    opts = list(dict_options_pn_is_part_of[pn])
                    for opt in opts:
                        if opt not in all_options: continue
                        group = opt.group
                        # Case 1: one element of this group has already been replaced
                        if new_pn in group: 
                            group.remove(pn)
                            if len(group) < 2: # too small
                                all_options.discard(opt)
                        # Case 2: replace pn by new_pn
                        else:
                            group[group.index(pn)] = new_pn
                        if opt in all_options: updated_opts.add(opt)
                    del dict_options_pn_is_part_of[pn]
                dict_options_pn_is_part_of[new_pn] = updated_opts

    def __call__(self, cluster: P_cluster):
        pg : P_graph = cluster.init_P_graph()
        pg.cluster = cluster
        if cluster.size < self.config.max_nodes_per_sub_graph:
            return pg
        # === Prepare dynamic setup ===
        P_Dynamic_manipulation.prepare_dynamic_setup(pg,cluster)

        # === FIRST : Dynamic partitioning ===
        previous_size = -1
        while (len(pg.nodes) > self.config.max_nodes_for_main_graph
            and len(pg.nodes) != previous_size):
            previous_size = len(pg.nodes)
            self.rule_group_sequences(pg)
            self.rule_merge_small_flows(pg)
            for pn in pg.nodes:
                pn.is_protected_from_unwrap = True
        pg.set_all_protected_to_false()
        pg.make_sub_graph_id("0")
        if len(pg.nodes) == previous_size: 
            warnings.warn(
                f"Partitioning of cluster '{cluster.name}' with "\
                f"{self.__class__} early stop, because it"\
                f"shrink more. Thus maybe to big, size : {previous_size}."
            )

        # === SECOND : freeze ===
        all_snodes = P_Dynamic_manipulation.freeze(
            pg,cluster.p_structure,self.__class__)
        if all_snodes != set(cluster.s_nodes):
            raise Exception(
                f"BUG in {self.__class__}. When collecting all the "\
                f"S_nodes at the end, we don't find cluster.s_nodes. We probably "\
                f"lost some nodes...\n Original nb of nodes : {len(cluster.s_nodes)}; "\
                f"Nb of nodes at the end : {len(all_snodes)}"
            )
        return pg



class Partitioner_bottom_to_top(Partitioner):
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
                if len(pn.users) != 1: return False
                if list(pn.users)[0] is not next_pn: return False
                if len(next_pn.deps) != 1: return False
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
                * len(not_last_nodes)**-self.value_power_not_last)
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
            inputs_mem = sum(pn.mem_out for pn in inputs_pn if pn.sn is not None)
            outputs_mem = sum(pn.mem_out for pn in outputs_pn if pn.sn is not None)
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
        
        def default_is_top_graph_ok(self,pg : P_graph):
            if (self.can_use_rotor
            and Partitioner_bottom_to_top.Option.utils_is_seq(pg.nodes)):
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
    def find_seq_options(self,pg : P_graph):
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
    def find_flow_options(self,pg : P_graph):
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
        # ie a list of P_nodes, representing there flow
        # reciprocal of dict_flow 
        dict_end_of_flow = dict()
        # any pn -> where its flow converged back
        # this is what we want to build

        # ** Add a temporary global sink to deps relation **
        tmp_global_sink_pn = P_node(main_graph=pg,main_target="tmp_sink")
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
    def round_of_partitioning(self, pg : P_graph):
        config = self.config
        for pn in pg.nodes:
            pn.is_protected_from_unwrap = True
        all_options = (
                self.find_seq_options(pg)
            +   self.find_flow_options(pg)
        )

        _all_options = list(all_options)
        dict_info = pg.dict_info
        for opt in _all_options:
            if all(not pn.does_requires_grad(dict_info)
                   for pn in opt.group):
                all_options.remove(opt)
            elif all(pn.deps_global.issubset(opt.set_group)
                     for pn in opt.group):
                all_options.remove(opt)


        dict_options_pn_is_part_of = dict((pn,set()) for pn in pg.nodes)
        # P_node -> Option set
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
            new_pn = P_Dynamic_manipulation.merge(best_group,pg)
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


    def __call__(self, cluster: P_cluster):
        pg : P_graph = cluster.init_P_graph()
        pg.cluster = cluster
        config = self.config
        if config.is_top_graph_ok(pg): return pg

        # === Prepare dynamic setup ===
        P_Dynamic_manipulation.prepare_dynamic_setup(pg,cluster)

        # === FIRST : Dynamic partitioning ===
        previous_size = -1
        while not config.is_top_graph_ok(pg) and len(pg.nodes) != previous_size:
            previous_size = len(pg.nodes)
            self.round_of_partitioning(pg)
        pg.set_all_protected_to_false()
        pg.make_sub_graph_id("0")

        if len(pg.nodes) == previous_size: 
            warnings.warn(
                f"Partitioning of cluster '{cluster.name}' with "\
                f"{self.__class__} early stop, because it"\
                f"shrink more. Thus maybe to big, size : {previous_size}."
            )

        # === SECOND : freeze ===
        all_snodes = P_Dynamic_manipulation.freeze(
            pg,cluster.p_structure,self.__class__)
        if all_snodes != set(cluster.s_nodes):
            raise Exception(
                f"BUG in {self.__class__}. When collecting all the "\
                f"S_nodes at the end, we don't find cluster.s_nodes. We probably "\
                f"lost some nodes...\n Original nb of nodes : {len(cluster.s_nodes)}; "\
                f"Nb of nodes at the end : {len(all_snodes)}"
            )
        return pg




class Partitioner_seq(Partitioner):
    class Config():
        def __init__(self,sub_partitioner : Partitioner = None):
            if sub_partitioner is None:
                sub_partitioner = Partitioner_bottom_to_top(
                    main_graph_as_any_other = True
                )
            self.sub_partitioner = sub_partitioner 

    config : Config = None
    def __init__(self, **kwargs):
        self.config = self.__class__.Config(**kwargs)

    def __call__(self, cluster : P_cluster):
        pg : P_graph = cluster.init_P_graph()
        pg.cluster = cluster
        dict_info = cluster.p_structure.dict_info

        # ** Add a temporary global source before get separators**
        tmp_global_source_pn = P_node(main_graph=pg,main_target="tmp_source")
        pg.nodes.insert(0,tmp_global_source_pn)
        first_nodes = pg.first_nodes
        for first_pn in first_nodes:
            if first_pn.deps == set():
                first_pn.deps.add(tmp_global_source_pn)
                tmp_global_source_pn.users.add(first_pn)

        seps_pn = RK_get_1_separators(pg)

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
                block_pn = P_node(pg,main_target=sn.mt,sn=sn)
                block_pn.mem_out = dict_info[sn.mt].memsize
            else:
                block_s_nodes = cluster.s_nodes[first_i+1:last_i+1]
                sub_cluster = P_cluster(block_s_nodes,cluster.p_structure)
                block_pn = P_node(pg,sub_cluster=sub_cluster)
                block_pn.mem_out = dict_info[cluster.s_nodes[last_i].mt].memsize
            p_nodes.append(block_pn)
            if block_nb > 0:
                prev_pn : P_node = p_nodes[-2]
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
                sub_cluster : P_cluster = block_pn.sub_cluster
                sub_cluster.partition(self.config.sub_partitioner)
        return pg


def S_to_P(
    sg : S_graph,
    model : torch.nn.Module,
    partitioners = [
        Partitioner(),
        Partitioner_OLD_bottom_to_top(),
        Partitioner_bottom_to_top(),
        Partitioner_seq()
    ],
    min_size_to_trigger_partitioning = 4):
    sg = copy_S_graph(sg)
    sg.discard_all_artefacts()
    p_structure = P_structure(sg,
        min_size_to_trigger_partitioning = min_size_to_trigger_partitioning)
    p_structure.make_dict_mt_to_ano_info(model)
    p_structure.main_cluster = main_cluster \
        = P_cluster(list(sg.nodes),p_structure)
    for partitioner in partitioners:
        main_cluster.partition(partitioner)
    main_cluster.make_sub_cluster_original()
    p_structure.make_graphs_names()
    sg.delete_artefact_edges()
    main_cluster.recompute_all_interfaces_and_edges()
    return p_structure





# ==========================
# === printing functions ===
# ==========================

color_leaf     = "blue"
color_sub_graph = "blueviolet"
color_special  = "green"
color_edge     = color_leaf

def aux_print_P_graph_message(pg : P_graph):
    return f"P_graph - Partitioned forward graph : of size {len(pg.nodes)}"
def aux_print_P_graph_name(pg,name=None):
    if name is not None: return name
    else: return "Partitioned_Forward_P_graph"

def aux_print_P_cluster_message(pc : P_cluster):
    possible_pg = pc.representee_cluster.possible_partitioning
    return f"{pc.name}, with {len(possible_pg)} possible partitioning"
def aux_print_P_cluster_names(pc : P_cluster,name=None):
    if name is None: name = pc.name
    parti_used = pc.representee_cluster.partitioners_already_used
    return [f"P_graph_{i}_via_{type(parti)}_of_{name}" for i,parti in enumerate(parti_used)]

def print_P_graph(pg : P_graph,name=None,open=True,render_format="svg",dot=None,uniq_num=0):
    # ----- init -----
    if dot is None:
        render = True
        if name is None: name = aux_print_P_graph_name(pg)
        dot = graphviz.Digraph(name,comment=name)
    else:
        render = False
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,**kwargs): dot.edge(uni(i1),uni(i2), **kwargs)
    # ----- Core -----
    cluster : P_cluster = pg.cluster
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
    if render:
        small_fcts.graph_render(dot,open,"P",render_format)
