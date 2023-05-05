# ==========================
#  ====== H structure =======
# ==========================

from rkgb.utils import *
from rkgb.Ptools import P_structure, P_cluster, P_graph, P_node, Cluster_translator 
from rkgb.Ktools import K_graph, K_C_node, K_D_node


# ************
# * H_C_node *
# ************
class H_C_node:
    """ previous attrs TO REMOVE OR MOVE TO SOLVER DIR
    fwd_time = 0
    fwd_overhead = 0
    ff_op_list = []
    """
    def __init__(self, name):
        self.name = name  # e.g. Fwd_1
        self.deps = set()  # HDN set
        self.users = set()  # HDN set
        self.sub_cluster = None
        self.main_target = None  # Bottom level
        self.is_fwd = True
        self.is_leaf = False
        self.number = -1
        #  When toposorting the HCNs we want to respect the same order
        # as the one found for KCNs, ie follow variable number
        # (get_num_tar) AND add some special deps for artifacts
        #  To make it easy, we give a number to each hcn based on K_graph's
        # toposort : hcn.is_leaf -> hcn.number = kcn index in list_kcn
        # else -> hcn.number = min (sub_hcn.number in hcn.subgraph)


# ************
# * H_D_node *
# ************
class H_D_node:
    def __init__(self, name, main_target):
        self.name = name
        self.main_target = main_target
        self.deps = set()  # HCN set
        self.users = set()  # HCN set
        self.mem = 0
        self.is_data = True  #  otherwise, grad
        self.kdn = None  # temporary attribute


# ***********
# * H_graph *
# ***********
class H_graph:
    """ previous attrs TO REMOVE OR MOVE TO SOLVER DIR
    list_sched = []
    """
    def __init__(self, name):
        # /!\ All the HCN's and HDN's should be in the same level. /!\ 
        self.name = name
        self.dict_hn = dict()  #  name -> HN
        self.dict_hg = dict()  #  name -> sub_Hgraph
        self.list_hcn = []  #  toposorted
        self.list_hdn = []  #  including interface HDNs
        self.inputs_hdn_data = set()  # HDN set -> inputs' data
        self.outputs_hdn_data = set()  # HDN set -> outputs' data
        self.outputs_hdn_grad = set()  # HDN set -> outputs' grad
        self.inputs_hdn_grad = set()  # HDN set -> inputs' grad
        self.loss_hcn = None
        self.all_kcn_inside = set()  # temporary attribute

    def make_users(self):
        for hn in self.dict_hn.values():
            for req_hn in hn.deps:
                req_hn.users.add(hn)

    def sort_list_hcn(self):
        # -> similar to K_graph.sort_list_kcn
        l1 = list(self.list_hcn)
        leaves_hcn = set()
        for hcn in self.list_hcn:
            if not hcn.is_fwd:
                if len(hcn.users) == 0:
                    leaves_hcn.add(hcn)
                else:
                    really_useful = False
                    for user_hdn in hcn.users:
                        if len(user_hdn.users) != 0:
                            really_useful = True
                    if not really_useful:
                        leaves_hcn.add(hcn)
        root_hdn = H_D_node("", "")
        root_hcn = H_C_node("")
        root_hdn.deps = leaves_hcn
        root_hcn.deps = set(self.inputs_hdn_grad)  # Not enough
        root_hcn.deps.add(root_hdn)
        self.list_hcn = l = RK_sort_based_on_deps(root_hcn)
        l.remove(root_hcn)
        if set(l1) != set(l):
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
    list_kcn : list[K_C_node] = None
    list_kdn : list[K_D_node] = None
    loss_kcn : K_C_node = None
    loss_idx : int = None
    dict_kn : dict = None
    interfaces : dict[str, set[K_D_node]]
    translator : Cluster_translator = None
    representee_cluster = None # : H_cluster
    possible_hg : list[H_graph] = None
    solutions = None #: list[H_sched] LATER 
    solvers_already_used = None #: list[Solver] LATER

    def __init__(self):
        self.loss_kcn = K_C_node("loss")

    def make_dict_kn(self):
        # ATTRIBUTES NEEDED: list_kcn, list_kdn
        self.dict_kn = dict(
            [(kdn.name,kdn) for kdn in self.list_kdn]
          + [(kcn.name,kcn) for kcn in self.list_kdn]
        )


def P_cluster_to_H_cluster(p_cluster : P_cluster, kg : K_graph):
    if hasattr(p_cluster,"h_cluster"): return p_cluster.h_cluster
    h_cluster = H_cluster()
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
        ano_pair = dict_mt_to_ano[kcn.mt]
        ano_triplet = ("fwd" if kcn.is_fwd else "bwd",) + ano_pair
        dict_kcn_to_ano[kcn] = ano_triplet
        dict_ano_to_kcn[ano_triplet] = kcn
    for kdn in h_cluster.list_kdn:
        ano_pair = dict_mt_to_ano[kdn.mt]
        ano_triplet = (kdn.kdn_type,) + ano_pair
        dict_kdn_to_ano[kdn] = ano_triplet
        dict_ano_to_kdn[ano_triplet] = kdn

    # ** possible_hg and representee **
    if p_cluster is p_cluster.representee_cluster:
        h_cluster.representee_cluster = h_cluster
        h_cluster.possible_hg = possible_hg = []
        for pg in p_cluster.possible_partitioning:
            possible_hg.append(P_graph_to_H_graph(pg,kg))
    else:
        h_cluster.representee_cluster \
            = P_cluster_to_H_cluster(p_cluster.representee_cluster)

    return h_cluster


def P_node_to_H_cluster(pn : P_node, kg : K_graph):
    if pn.sub_cluster is not None:
        return P_cluster_to_H_cluster(pn.sub_cluster, kg)
    elif pn.mt not in kg.dict_KCN_bwd:
        return None
    else:
        h_cluster = H_cluster()
        loss_kcn = h_cluster.loss_kcn
        mt = pn.mt
        # -- list_kcn part --
        kcn_fwd : K_C_node = kg.dict_KCN_fwd[mt]
        kcn_bwd : K_C_node = kg.dict_KCN_fwd[mt]
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
        inputs_data = set(req_kdn for req_kdn in kcn_fwd.deps_real)
        inputs_grad = set(user_kdn for user_kdn in kcn_bwd.users)
        set_kdn.update(inputs_data)
        set_kdn.update(inputs_grad)
        h_cluster.list_kdn = list(set_kdn)
        h_cluster.make_dict_kn()

        h_cluster.interfaces = {
            "inputs_kdn_data"  : inputs_data,
            "outputs_kdn_data" : set([kdn_data]),
            "inputs_kdn_grad"  : set([kdn_grad]),
            "outputs_kdn_grad" : inputs_grad
        }

        h_cluster.representee_cluster = h_cluster
        return h_cluster


def P_graph_to_H_graph(pg : P_graph, kg : K_graph):
    pass # TODO TODO TODO
    """
    #  -> This function is recursive in 'pg'
    hg = H_graph(f"Hg_{pg.graph_id}")
    # ** useful dicts **
    dict_mt_to_hdn_data = dict()
    dict_mt_to_hdn_grad = dict()
    #  -> to easily find the inputs and outputs
    dict_hdn_to_kdn = dict()
    #  A hdn represents exactly one kdn
    dict_kcn_to_hcn = dict()
    #  At a fixed level of depth, a kcn can be found in only one hcn
    #  -> These two dict make it super easy to build edges

    #  ** small functions to extract K_nodes from kg **
    dict_kn = kg.dict_kn
    dict_kn[kg.input_kdn_data.name] = kg.input_kdn_data
    dict_kn[kg.input_kdn_grad.name] = kg.input_kdn_grad
    has_bwd = lambda mt: f"bwd_{mt}" in dict_kn
    has_grad = lambda mt: f"{mt} grad" in dict_kn
    has_phantoms = lambda mt: f"{mt} phantoms" in dict_kn
    get_kcn_fwd = lambda mt: dict_kn[f"fwd_{mt}"]
    get_kcn_bwd = lambda mt: dict_kn[f"bwd_{mt}"]
    get_kdn_data = lambda mt: dict_kn[f"{mt} data"]
    get_kdn_grad = lambda mt: dict_kn[f"{mt} grad"]
    get_kdn_phantoms = lambda mt: dict_kn[f"{mt} phantoms"]
    
    # ==* First, build the H_nodes *==
    for pn in pg.list_nodes:
        if pn.is_leaf:
            # ** Bottom level **
            mt = pn.main_target
            hcn_fwd = H_C_node(f"Fwd_{mt}")
            hdn_data = H_D_node(f"Data_{mt}", mt)
            kcn_fwd = get_kcn_fwd(mt)
            kdn_data = get_kdn_data(mt)
            dict_kcn_to_hcn[kcn_fwd] = hcn_fwd
            dict_hdn_to_kdn[hdn_data] = kdn_data
            hcn_fwd.is_leaf = True
            hcn_fwd.main_target = mt
            hcn_fwd.number = kg.list_kcn.index(kcn_fwd)
            hcn_fwd.fwd_time = kcn_fwd.time
            hcn_fwd.fwd_overhead = kcn_fwd.overhead
            hcn_fwd.ff_op_list = [
                # H_op(hcn_fwd.name, hcn_fwd, is_fwd=True, is_del=False)
                Op(kcn_fwd, fast_forward=True)
            ]
            hdn_data.mem = kdn_data.mem
            hcns = [hcn_fwd]
            hdns = [hdn_data]
            #  ** bwd part **
            if has_bwd(mt):
                hcn_bwd = H_C_node(f"Bwd_{mt}")
                hdn_grad = H_D_node(f"Grad_{mt}", mt)
                kcn_bwd = get_kcn_bwd(mt)
                kdn_grad = get_kdn_grad(mt)
                dict_kcn_to_hcn[kcn_bwd] = hcn_bwd
                dict_hdn_to_kdn[hdn_grad] = kdn_grad
                hcn_bwd.is_leaf = True
                hcn_bwd.is_fwd = False
                hcn_bwd.main_target = mt
                hcn_bwd.number = kg.list_kcn.index(kcn_bwd)
                # hcn_bwd.fwd_time = kcn_bwd.time
                # hcn_bwd.fwd_overhead = kcn_bwd.overhead
                hdn_grad.is_data = False
                hdn_grad.mem = kdn_grad.mem
                hcns.append(hcn_bwd)
                hdns.append(hdn_grad)
                #  ** last level graph **
                sub_hg = H_graph(f"Hg_{mt}")
                hcn_fwd.sub_graph = hcn_bwd.sub_graph = sub_hg
            else:
                sub_hg = None
        else:
            #  ** Recursive **
            sub_hg = P_and_K_to_H(pn.subgraph, kg)
            hcn_fwd = H_C_node(f"Fwd_{pn.name}")
            hcn_bwd = H_C_node(f"Bwd_{pn.name}")
            hcn_fwd.sub_graph = hcn_bwd.sub_graph = sub_hg
            hcn_bwd.is_fwd = False
            # number :
            nb_min_fwd = 9999999
            nb_min_bwd = 9999999
            for sub_hcn in sub_hg.list_hcn:
                if sub_hcn.is_fwd:
                    nb_min_fwd = min(sub_hcn.number, nb_min_fwd)
                else:
                    nb_min_bwd = min(sub_hcn.number, nb_min_bwd)
            hcn_fwd.number = nb_min_fwd
            hcn_bwd.number = nb_min_bwd
            hcn_fwd.fwd_time = sum(
                sub_hcn.fwd_time for sub_hcn in sub_hg.list_hcn
            )
            (
                hcn_fwd.fwd_overhead,
                hcn_fwd.ff_op_list,
            ) = sub_hg.fast_fwd_overhead()
            # fwd_time and overhead are for fast forward so bwd node has none

            for kcn in sub_hg.all_kcn_inside:
                dict_kcn_to_hcn[kcn] = hcn_fwd if kcn.is_fwd else hcn_bwd

            hcns = [hcn_fwd, hcn_bwd]
            hdns = []
            io_of_sub_hg = sub_hg.outputs_hdn_data.union(
                sub_hg.outputs_hdn_grad
            )  # We don't create the nodes for the inputs
            for hdn_io_in_sub_hg in io_of_sub_hg:
                mt = hdn_io_in_sub_hg.main_target
                mem = hdn_io_in_sub_hg.mem
                kdn_io = hdn_io_in_sub_hg.kdn
                if kdn_io.kdn_type == "grad":
                    hdn_io = H_D_node(f"Grad_in_{hg.name}_{mt}", mt)
                    hdn_io.is_data = False
                else:
                    hdn_io = H_D_node(f"Data_in_{hg.name}_{mt}", mt)
                hdn_io.mem = mem
                hdns.append(hdn_io)
                dict_hdn_to_kdn[hdn_io] = kdn_io

        # * register everything *
        for hn in hcns + hdns:
            hg.dict_hn[hn.name] = hn
        hg.list_hcn.extend(hcns)
        if not (sub_hg is None):
            hg.dict_hg[sub_hg.name] = sub_hg
        for hdn in hdns:
            if hdn.is_data:
                dict_mt_to_hdn_data[hdn.main_target] = hdn
            else:
                dict_mt_to_hdn_grad[hdn.main_target] = hdn

    #  ==* Inputs / Outputs *==
    hg.outputs_hdn_data = set(
        dict_mt_to_hdn_data[inp] for inp in pg.output_targets
    )
    hg.outputs_hdn_grad = set(
        dict_mt_to_hdn_grad[inp] for inp in pg.output_targets
    )
    hg.inputs_hdn_data = inputs_data = set()
    hg.inputs_hdn_grad = inputs_grad = set()
    for inp_mt in pg.input_targets:
        assert inp_mt not in dict_mt_to_hdn_data
        assert inp_mt not in dict_mt_to_hdn_grad
        kdn_data = get_kdn_data(inp_mt)
        hdn_data = H_D_node(f"Data_{inp_mt}_in_{hg.name}", inp_mt)
        hdn_data.mem = kdn_data.mem
        dict_mt_to_hdn_data[inp_mt] = hdn_data
        dict_hdn_to_kdn[hdn_data] = kdn_data
        inputs_data.add(hdn_data)
        hg.list_hdn.append(hdn_data)
        hg.dict_hn[hdn_data.name] = hdn_data
        if has_grad(inp_mt):
            kdn_grad = get_kdn_grad(inp_mt)
            hdn_grad = H_D_node(f"Grad_{inp_mt}_in_{hg.name}", inp_mt)
            hdn_grad.is_data = False
            hdn_grad.mem = kdn_grad.mem
            dict_mt_to_hdn_grad[inp_mt] = hdn_grad
            dict_hdn_to_kdn[hdn_grad] = kdn_grad
            inputs_grad.add(hdn_grad)
            hg.list_hdn.append(hdn_grad)
            hg.dict_hn[hdn_grad.name] = hdn_grad
    hg.interfaces = (
        hg.inputs_hdn_data
        | hg.inputs_hdn_grad
        | hg.outputs_hdn_data
        | hg.outputs_hdn_grad
    )

    #  =* loss_hcn *=
    hg.loss_hcn = loss_hcn = H_C_node(f"Loss_hcn_of_{hg.name}")
    hg.loss_hcn.main_target = "loss"
    hg.list_hcn.append(hg.loss_hcn)

    hg.list_hdn = list(dict_hdn_to_kdn.keys())

    #  =* register hdn.kdn and hg.all_kcn_inside *=
    for hdn, kdn in dict_hdn_to_kdn.items():
        hdn.kdn = kdn
    hg.all_kcn_inside = set(dict_kcn_to_hcn.keys())

    #  ===* Second, build the edges *===
    for hdn in hg.list_hdn:
        kdn = dict_hdn_to_kdn[hdn]
        for kcn in kdn.deps:
            if kcn is not kg.loss_kcn:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    hdn.deps.add(hcn)
                    hcn.users.add(hdn)
        for kcn in kdn.users_real:
            if kcn is not kg.loss_kcn:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    if not (hcn in hdn.deps):
                        hdn.users.add(hcn)
                        hcn.deps.add(hdn)

    for hdn in hg.outputs_hdn_data:
        hdn.users.add(loss_hcn)
        loss_hcn.deps.add(hdn)
    for hdn in hg.outputs_hdn_grad:
        hdn.deps.add(loss_hcn)
        loss_hcn.users.add(hdn)

    # ** special input "source" **
    if "sources" in pg.input_targets:
        hdn_data = dict_mt_to_hdn_data["sources"]
        kdn_data = dict_hdn_to_kdn[hdn_data]
        for kcn in kdn_data.users_global:
            if kcn in dict_kcn_to_hcn:
                hcn = dict_kcn_to_hcn[kcn]
                hdn_data.users.add(hcn)
                hcn.deps.add(hdn_data)
        hdn_grad = dict_mt_to_hdn_grad["sources"]
        kdn_grad = dict_hdn_to_kdn[hdn_grad]
        if kdn_grad.deps_global == set():
            hg.list_hdn.remove(hdn_grad)
        else:
            for kcn in kdn_grad.deps_global:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    hdn_grad.deps.add(hcn)
                    hcn.users.add(hdn_grad)
    hg.sort_list_hcn()

    return hg
    """


def P_and_K_to_H(ps : P_structure, kg : K_graph):
    return H_cluster(ps.main_cluster,kg)

    


# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================
# =====================================================



# ==========================


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
    if isinstance(hn, H_D_node):
        return color_hdn
    if hn.is_fwd:
        return color_hcn_fwd
    return color_hcn_bwd


def print_H_graph(hg: H_graph, name=None, open=True, render_format="svg"):
    # ----- init -----
    print(
        f"Hierarchical graph : \n"
        f"{len(hg.list_hcn)} H_C_nodes,\n"
        f"{len(hg.list_hdn)} H_D_nodes"
    )
    if name is None:
        name = "Hierarchical_graph"
    dot = graphviz.Digraph(name, comment="H_graph = Hierarchical graph")
    # ----- Core -----
    # * nodes *
    def print_hcn(hcn: H_C_node):
        mt = hcn.main_target
        dot.node(
            hcn.name,
            hcn.name,
            color=get_color(hcn),
            tooltip=(
                f"Fast Forward Time : {hcn.fwd_time}"
                f"Fast Forward Memory Overhead : "
                f"{irotor.MemSize(hcn.fwd_overhead)}"
            )
            if hcn.is_fwd
            else "",
        )

    def print_hdn(kdn):
        dot.node(
            kdn.name,
            kdn.name,
            color=get_color(kdn),
            tooltip=f"Mem {irotor.MemSize(kdn.mem)}",
        )

    for hcn in hg.list_hcn:
        print_hcn(hcn)
    for hdn in hg.list_hdn:
        print_hdn(hdn)

    # * edges *
    for hcn in hg.list_hcn:
        for req_hdn in hcn.deps:
            dot.edge(req_hdn.name, hcn.name, color=color_edge)
        for user_hdn in hcn.users:
            dot.edge(hcn.name, user_hdn.name, color=color_edge)

    #  ----- render -----
    small_fcts.graph_render(dot, open, "H", render_format)

