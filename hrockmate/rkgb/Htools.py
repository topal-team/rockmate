# ==========================
# ====== H structure =======
# ==========================

from .utils import *
from .Ptools import P_structure, P_cluster, P_graph, P_node, Cluster_translator 
from .Ktools import K_graph, K_C_node, K_D_node


# ************
# * H_C_node *
# ************
class H_C_node(RK_node):
    def __init__(self,
            name,
            main_target = None, 
            sub_cluster = None,
            is_fwd = True,
            number = -1):
        super().__init__("HC",mt = main_target)
        self.name = name  # e.g. Fwd_1
        self.sub_cluster = sub_cluster
        self.is_fwd = is_fwd
        self.is_leaf = bool(main_target is None)
        self.deps = set()  # HDN set
        self.users = set()  # HDN set
        self.deps_through_size_artefacts = set()  # HCN set
        self.number = number
        # About self.number :
        # When toposorting list_hcn we want to respect as much as possible
        # the original order, so list_hcn toposort consists of:
        # -> inverse Breath First Search (as any toposort)
        # -> if equal, then choose the one which comes first in list_kcn.
        # And that's what self.number is used for: Rk_node.get_num
        # (for D, S and K nodes we can get the number from main_target,
        # but some H_nodes don't have a main_target).


# ************
# * H_D_node *
# ************
class H_D_node(RK_node):
    kdn : K_D_node = None
    main_target : str = None
    name : str = None
    is_data : bool = None
    deps : set[H_C_node] = None
    users : set[H_C_node] = None
    def __init__(self,kdn : K_D_node = None):
        self.deps = set()
        self.users = set()
        if kdn is None:
            super().__init__("HD")
        else:
            super().__init__("HD",mt = kdn.mt)
            self.kdn = kdn
            self.name = H_D_node.make_name_from_kdn(kdn)
            self.mem = kdn.mem
            if kdn.kdn_type == "data":
                self.is_data = True
            elif kdn.kdn_type == "grad":
                self.is_data = False
            else:
                assert(kdn.kdn_type == "phantoms")
                raise Exception("An HDN cannot represent a KDN of type 'phantoms'")
        
    @staticmethod
    def make_name_from_kdn(kdn):
        return f"{kdn.kdn_type}_{kdn.mt}"


# ***********
# * H_graph *
# ***********
class H_graph(RK_graph):
    def __init__(self, name, h_cluster = None, other_obj = None):
        super().__init__("H",other_obj)
        # /!\ All the HCN's and HDN's should be in the same level. /!\ 
        self.name = name
        self.dict_hn = dict()  #  name -> HN
        self.list_hcn = []  #  toposorted
        self.list_hdn = []  #  including interface HDNs
        self.inputs_hdn_data = set()  # HDN set -> inputs' data
        self.outputs_hdn_data = set()  # HDN set -> outputs' data
        self.outputs_hdn_grad = set()  # HDN set -> outputs' grad
        self.inputs_hdn_grad = set()  # HDN set -> inputs' grad
        self.loss_hcn = None
        if h_cluster is not None:
            self.cluster = h_cluster
            self.all_clusters = set([h_cluster])
        else:
            self.cluster = None
            self.all_clusters = set()

    def make_users(self):
        for hn in self.dict_hn.values():
            for req_hn in hn.deps:
                req_hn.users.add(hn)

    def sort_list_hcn(self):
        # -> similar to K_graph.sort_list_kcn
        l1 = list(self.list_hcn)
        root_hcn = H_C_node("")
        root_hcn.deps = set(self.inputs_hdn_grad) # Not enough
        # -> Not enough, some hdn_grad aren't returned
        # -> We need all the root of the 'deps' relation
        leaves_hcn = set()
        for hcn in self.list_hcn:
            if not hcn.is_fwd:
                if len(hcn.get_users())==0:
                    leaves_hcn.add(hcn)
        root_hdn = H_D_node()
        root_hdn.deps = leaves_hcn
        root_hcn.deps.add(root_hdn)
        self.list_hcn = l = RK_sort_based_on_deps(root_hcn)
        l.remove(root_hcn)
        if set(l1) != set(l):
            print([hcn.name for hcn in l1 if hcn not in l])
            raise Exception(
                "Problem with H_nodes edges, set(list_hcn) "\
                "has changed after RK_sort_based_on_deps.\n"\
                "Which means some edges outside the graph "\
                "or some missing edges inside."
            )


# *************
# * H_cluster *
# *************
class H_cluster():
    is_bottom : bool = None
    list_kcn : list[K_C_node] = None
    list_kdn : list[K_D_node] = None
    loss_kcn : K_C_node = None
    loss_idx : int = None
    dict_kn : dict = None
    interfaces : dict[str, set[K_D_node]]
    translator : Cluster_translator = None
    p_cluster : P_cluster = None
    p_node : P_node = None
    all_clusters : set = None
    representee_cluster = None # : H_cluster
    possible_hg : list[H_graph] = None
    list_sched = None #: list[Op_sched] LATER 

    def __init__(self,name,is_bottom):
        self.name = name
        self.loss_kcn = K_C_node("loss")
        self.is_bottom = is_bottom
        self.all_clusters = set([self])

    def make_dict_kn(self):
        # ATTRIBUTES NEEDED: list_kcn, list_kdn
        self.dict_kn = dict(
            [(kdn.name,kdn) for kdn in self.list_kdn]
          + [(kcn.name,kcn) for kcn in self.list_kcn]
        )

    @property
    def all_interfaces(self):
        return set().union(*self.interfaces.values())


def P_cluster_to_H_cluster(p_cluster : P_cluster, kg : K_graph):
    if hasattr(p_cluster,"h_cluster"): return p_cluster.h_cluster
    h_cluster = H_cluster(p_cluster.name.replace("P","H"),False)
    h_cluster.p_cluster = p_cluster
    p_cluster.h_cluster = h_cluster
    all_computation_mt = set(sn.mt for sn in p_cluster.s_nodes)

    # Collect list_kcn and add the fresh loss_kcn along the way
    loss_kcn = h_cluster.loss_kcn
    list_kcn = h_cluster.list_kcn = []
    for kcn in kg.list_kcn:
        if kcn.mt in all_computation_mt:
            list_kcn.append(kcn)
        if kcn == kg.loss_kcn:
            h_cluster.loss_idx = len(list_kcn)
            list_kcn.append(loss_kcn)

    set_kdn = set(kdn for kdn in kg.list_kdn if kdn.mt in all_computation_mt)
    # `set` because we need to add the inputs_kdn

    h_cluster.interfaces = interfaces = dict()
    interfaces["inputs_kdn_data" ] = inputs_kdn_data  = set()
    interfaces["outputs_kdn_data"] = outputs_kdn_data = set()
    interfaces["inputs_kdn_grad" ] = inputs_kdn_grad  = set()
    interfaces["outputs_kdn_grad"] = outputs_kdn_grad = set()

    # ** inputs **
    for input_mt in p_cluster.inputs_mt:
        input_data = kg.dict_KDN_data[input_mt]
        inputs_kdn_data.add(input_data)
        set_kdn.add(input_data)
        if input_mt in kg.dict_KDN_grad:
            input_grad = kg.dict_KDN_grad[input_mt]
            if input_grad.deps_global != set():
                inputs_kdn_grad.add(input_grad)
                set_kdn.add(input_grad)

    # ** outputs **
    for output_mt in p_cluster.outputs_mt:
        output_data = kg.dict_KDN_data[output_mt]
        outputs_kdn_data.add(output_data)
        loss_kcn.deps_real.add(output_data)
        if output_mt in kg.dict_KDN_grad:   
            output_grad = kg.dict_KDN_grad[output_mt]
            outputs_kdn_grad.add(output_grad)
            loss_kcn.users.add(output_grad)

    h_cluster.list_kdn = list(set_kdn)
    h_cluster.make_dict_kn()

    # ** translator **
    h_cluster.translator = translator = p_cluster.translator
    dict_mt_to_ano = translator.dict_mt_to_ano_pair
    dict_kcn_to_ano = translator.dict_kcn_to_ano_triplet = dict()
    dict_kdn_to_ano = translator.dict_kdn_to_ano_triplet = dict()
    dict_ano_to_kcn = translator.dict_ano_triplet_to_kcn = dict()
    dict_ano_to_kdn = translator.dict_ano_triplet_to_kdn = dict()
    for kcn in h_cluster.list_kcn:
        if kcn is loss_kcn:
            ano_triplet = ("loss",0,0)
        else:
            ano_pair = dict_mt_to_ano[kcn.mt]
            ano_triplet = ("fwd" if kcn.is_fwd else "bwd",) + ano_pair
        dict_kcn_to_ano[kcn] = ano_triplet
        dict_ano_to_kcn[ano_triplet] = kcn
    for kdn in h_cluster.list_kdn:
        ano_pair = dict_mt_to_ano[kdn.mt]
        ano_triplet = (kdn.kdn_type,) + ano_pair
        dict_kdn_to_ano[kdn] = ano_triplet
        dict_ano_to_kdn[ano_triplet] = kdn
    translator.dict_name_to_ano_triplet = dict(
        [(kn.name,ano) for (kn,ano) in dict_kcn_to_ano.items()]
    +   [(kn.name,ano) for (kn,ano) in dict_kdn_to_ano.items()]
    )
    translator.dict_ano_triplet_to_name = dict(
        (kdn,hdn) for (hdn,kdn) in translator.dict_name_to_ano_triplet.items()
    )

    # ** possible_hg and representee **
    if p_cluster is p_cluster.representee_cluster:
        h_cluster.representee_cluster = h_cluster
        h_cluster.possible_hg = possible_hg = []
        h_cluster.list_sched = []
        for pg in p_cluster.possible_partitioning:
            hg = P_graph_to_H_graph(pg,h_cluster,kg)
            h_cluster.all_clusters.update(hg.all_clusters)
            possible_hg.append(hg)
    else:
        h_cluster.representee_cluster \
            = representee \
            = P_cluster_to_H_cluster(p_cluster.representee_cluster,kg)
        # === make list_kdn order match ===
        assert(len(representee.list_kdn) == len(h_cluster.list_kdn))
        old_list = h_cluster.list_kdn
        ano_list_kdn = [representee.translator.dict_kdn_to_ano_triplet[kdn] for kdn in representee.list_kdn]
        h_cluster.list_kdn = [h_cluster.translator.dict_ano_triplet_to_kdn[ano] for ano in ano_list_kdn]
        assert(set(old_list) == set(h_cluster.list_kdn))
        

    return h_cluster


def P_node_to_H_cluster(pn : P_node, kg : K_graph):
    if pn.sub_cluster is not None:
        return P_cluster_to_H_cluster(pn.sub_cluster, kg)
    elif pn.mt not in kg.dict_KCN_bwd:
        return None
    else:
        h_cluster = H_cluster(f"H_Cluster_bottom_{pn.mt}",True)
        h_cluster.p_node = pn
        h_cluster.list_sched = []
        h_cluster.representee_cluster = h_cluster
        loss_kcn = h_cluster.loss_kcn
        mt = pn.mt
        # -- list_kcn part --
        kcn_fwd : K_C_node = kg.dict_KCN_fwd[mt]
        kcn_bwd : K_C_node = kg.dict_KCN_bwd[mt]
        kdn_data = kg.dict_KDN_data[mt]
        kdn_grad = kg.dict_KDN_grad[mt]
        loss_kcn.deps_real = set([kdn_data])
        loss_kcn.users = set([kdn_grad])
        h_cluster.list_kcn = [kcn_fwd,loss_kcn,kcn_bwd]
        h_cluster.loss_idx = 1

        # -- list_kdn part --
        set_kdn = set([kdn_data,kdn_grad])
        if mt in kg.dict_KDN_phantoms:
            set_kdn.add(kg.dict_KDN_phantoms[mt])
        inputs_data = set(
            req_kdn for req_kdn \
            in (kcn_fwd.deps_global - kcn_fwd.deps_fake))
        inputs_grad = set(user_kdn for user_kdn in kcn_bwd.users_global)
        set_kdn.update(inputs_data)
        set_kdn.update(inputs_grad)
        h_cluster.list_kdn = list(set_kdn)
        h_cluster.make_dict_kn()

        h_cluster.interfaces = {
            "inputs_kdn_data"  : inputs_data,
            "outputs_kdn_data" : set([kdn_data]),
            "outputs_kdn_grad" : set([kdn_grad]),
            "inputs_kdn_grad"  : inputs_grad
        }

        return h_cluster




def P_graph_to_H_graph(
        pg : P_graph, 
        h_cluster : H_cluster,
        kg : K_graph):
    hg = H_graph(pg.name.replace("pg","hg"),h_cluster)

    # -> Useful for interfaces
    dict_mt_to_hdn_data = dict()
    dict_mt_to_hdn_grad = dict()

    dict_hdn_to_kdn : dict[H_D_node, K_D_node] = dict()
    #  A hdn represents exactly one kdn
    dict_kcn_to_hcn : dict[K_C_node, H_C_node] = dict()
    #  At a fixed level of depth, a kcn can be found in only one hcn
    #  -> These two dict make it super easy to build edges

    for pn in pg.nodes:
        sub_cluster = P_node_to_H_cluster(pn,kg)
        if sub_cluster is not None:
            hg.all_clusters.update(sub_cluster.all_clusters)
        if pn.is_leaf:
            # === Bottom level ===
            mt = pn.mt
            # ** HCN_fwd **
            kcn_fwd = kg.dict_KCN_fwd[mt]
            hcn_fwd = H_C_node(
                kcn_fwd.name,
                main_target = mt,
                sub_cluster = sub_cluster, # = None if no grad
                is_fwd = True,
                number = kcn_fwd._number)
            dict_kcn_to_hcn[kcn_fwd] = hcn_fwd

            # ** HDN_data **
            if mt not in dict_mt_to_hdn_data: # interfaces of sub clusters overlap
                kdn_data = kg.dict_KDN_data[mt]
                hdn_data = H_D_node(kdn_data)
                dict_hdn_to_kdn[hdn_data] = kdn_data
                dict_mt_to_hdn_data[mt] = hdn_data
            
            if mt in kg.dict_KCN_bwd:
                # ** HCN_bwd **
                kcn_bwd = kg.dict_KCN_bwd[mt]
                hcn_bwd = H_C_node(
                    kcn_bwd.name,
                    main_target = mt,
                    sub_cluster = sub_cluster,
                    is_fwd = False,
                    number = kcn_bwd._number)
                dict_kcn_to_hcn[kcn_bwd] = hcn_bwd

                # ** HDN_grad **
                if mt not in dict_mt_to_hdn_grad: # interfaces of sub clusters overlap
                    kdn_grad = kg.dict_KDN_grad[mt]
                    hdn_grad = H_D_node(kdn_grad)
                    dict_hdn_to_kdn[hdn_grad] = kdn_grad
                    dict_mt_to_hdn_grad[mt] = hdn_grad

        else:
            # === NOT bottom ===
            # ** HCN_fwd and bwd **
            hcn_fwd_num = 999999
            hcn_bwd_num = -1
            for kcn in sub_cluster.list_kcn:
                if not hasattr(kcn,"_number"): continue
                if kcn.is_fwd:
                    hcn_fwd_num = min(hcn_fwd_num,kcn._number)
                else:
                    hcn_bwd_num = max(hcn_bwd_num,kcn._number)
            hcn_fwd = H_C_node(
                f"fwd_{pn.name}",
                sub_cluster = sub_cluster,
                is_fwd = True,
                number = hcn_fwd_num)
            hcn_bwd = H_C_node(
                f"bwd_{pn.name}",
                sub_cluster = sub_cluster,
                is_fwd = False,
                number = hcn_bwd_num)
            for kcn in sub_cluster.list_kcn:
                dict_kcn_to_hcn[kcn] = hcn_fwd if kcn.is_fwd else hcn_bwd

            # ** HDNs **
            all_interfaces = sub_cluster.all_interfaces
            for kdn in all_interfaces:
                hdn = H_D_node(kdn)
                if hdn.is_data:
                    if kdn.mt not in dict_mt_to_hdn_data:
                        dict_hdn_to_kdn[hdn] = kdn
                        dict_mt_to_hdn_data[hdn.mt] = hdn
                else:
                    if kdn.mt not in dict_mt_to_hdn_grad:
                        dict_hdn_to_kdn[hdn] = kdn
                        dict_mt_to_hdn_grad[hdn.mt] = hdn

    # ** loss_hcn **
    hg.loss_hcn = loss_hcn = H_C_node(
        f"Loss_hcn_of_{hg.name}",
        main_target = "loss"
        )
    
    # ** missing H_D_nodes -> inputs of bottom sub nodes**
    for inp_mt in h_cluster.p_cluster.inputs_mt:
        if inp_mt not in dict_mt_to_hdn_data:
            kdn_data = kg.dict_KDN_data[inp_mt]
            hdn_data = H_D_node(kdn_data)
            dict_hdn_to_kdn[hdn_data] = kdn_data
            dict_mt_to_hdn_data[hdn_data.mt] = kdn_data
            if inp_mt in kg.dict_KDN_grad:
                kdn_grad = kg.dict_KDN_grad[inp_mt]
                if kdn_grad in h_cluster.all_interfaces:
                    hdn_grad = H_D_node(kdn_grad)
                    dict_hdn_to_kdn[hdn_grad] = kdn_grad
                    dict_mt_to_hdn_grad[hdn_grad.mt] = kdn_grad
    
    # ** register nodes **
    hg.list_hdn = list(dict_hdn_to_kdn.keys())
    hg.list_hcn = list(dict_kcn_to_hcn.values())
    hg.list_hcn.append(loss_hcn)
    hg.dict_hn = dict(
        [(hdn.name,hdn) for hdn in hg.list_hdn]
      + [(hcn.name,hcn) for hcn in hg.list_hcn]
    )

    # ** interfaces **
    dict_kdn_to_hdn = dict((kdn,hdn) for (hdn,kdn) in dict_hdn_to_kdn.items())
    hg.inputs_hdn_data = set(dict_kdn_to_hdn[kdn] for kdn in h_cluster.interfaces["inputs_kdn_data"])
    hg.outputs_hdn_data = set(dict_kdn_to_hdn[kdn] for kdn in h_cluster.interfaces["outputs_kdn_data"])
    hg.inputs_hdn_grad = set(dict_kdn_to_hdn[kdn] for kdn in h_cluster.interfaces["inputs_kdn_grad"])
    hg.outputs_hdn_grad = set(dict_kdn_to_hdn[kdn] for kdn in h_cluster.interfaces["outputs_kdn_grad"])
    
    
    # === Build the edges ===
    for hdn in hg.list_hdn:
        kdn = dict_hdn_to_kdn[hdn]
        for kcn in kdn.deps if kdn.mt != "sources" else kdn.deps_global:
            if kcn is not kg.loss_kcn:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    hdn.deps.add(hcn)
                    hcn.users.add(hdn)
        for kcn in kdn.users_real if kdn.mt != "sources" else kdn.users_global:
            if kcn is not kg.loss_kcn:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    if not (hcn in hdn.deps):
                        hdn.users.add(hcn)
                        hcn.deps.add(hdn)

    # -> loss edges
    for hdn in hg.outputs_hdn_data:
        hdn.users.add(loss_hcn)
        loss_hcn.deps.add(hdn)
    for hdn in hg.outputs_hdn_grad:
        hdn.deps.add(loss_hcn)
        loss_hcn.users.add(hdn)

    # -> artefacts edges
    for kcn,hcn in dict_kcn_to_hcn.items():
        for req_via_art_kcn in kcn.deps_through_size_artefacts:
            if req_via_art_kcn in dict_kcn_to_hcn:
                req_via_art_hcn = dict_kcn_to_hcn[req_via_art_kcn]
                if req_via_art_hcn is not hcn:
                    hcn.deps_through_size_artefacts.add(
                        req_via_art_hcn
                    )

    hg.sort_list_hcn()
    return hg


def P_and_K_to_H(ps : P_structure, kg : K_graph):
    kg.make_kcns_number()
    return P_cluster_to_H_cluster(ps.main_cluster,kg)



# =================
# = Utils to test =
def nb_clusters(main_cluster : H_cluster):
    nb = 0
    nb_unique = 0
    biggest = -1
    for c in main_cluster.all_clusters:
        if "bottom" in c.name: continue
        nb += 1
        if c.representee_cluster is c:
            nb_unique += 1
            if c is not main_cluster:
                for hg in c.p_cluster.possible_partitioning:
                    biggest = max(len(hg.nodes),biggest)
    return nb,nb_unique,biggest


# ==========================
# === printing functions ===
# ==========================

color_hcn_fwd = "blue"
color_hcn_bwd = "blueviolet"
color_special = "green"
color_hdn = "olive"
color_edge = "black"


def get_color(hn):
    if hn.main_target == "loss":
        return color_special
    elif isinstance(hn, H_D_node):
        return color_hdn
    elif hn.is_fwd:
        return color_hcn_fwd
    else:
        return color_hcn_bwd

def aux_print_H_graph_message(hg : H_graph):
    return (
        f"H_graph - Hierarchical forward+backward graph, "\
        f"{len(hg.list_hcn)} H_C_nodes; {len(hg.list_hdn)} H_D_nodes"
    )
def aux_print_H_graph_name(hg,name=None):
    if name is not None: return name
    else: return "Hierarchical_H_graph"

def aux_print_H_cluster_message(hc : H_cluster):
    possible_hg = hc.representee_cluster.possible_hg
    return f"{hc.name}, with {len(possible_hg)} possible H_graphs"
def aux_print_H_cluster_names(hc : H_cluster,name=None):
    if name is None: name = hc.name
    nb_hg = len(hc.representee_cluster.possible_hg)
    return [f"H_graph_{i}_of_{name}" for i in range(nb_hg)]

def print_H_graph(hg: H_graph, name=None, open=True, render_format="svg",dot=None,uniq_num=0):
    # ----- init -----
    if dot is None:
        render = True
        if name is None: name = aux_print_H_graph_name(hg)
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
    for hdn in hg.list_hdn:
        node(hdn.name,hdn.name,
            color=get_color(hdn),
            tooltip=f"Mem {irotor.MemSize(hdn.mem)}")

    # * edges *
    for hcn in hg.list_hcn:
        for req_hdn in hcn.deps:
            edge(req_hdn.name, hcn.name, color=color_edge)
        for user_hdn in hcn.users:
            edge(hcn.name, user_hdn.name, color=color_edge)

    #  ----- render -----
    if render:
        small_fcts.graph_render(dot, open, "H", render_format)

