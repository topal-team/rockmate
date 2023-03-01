# ==========================
# ====== P structure =======
# ==========================

# ** Graph partitioning **

from rkgb.utils import *

# **********
# * P_node *
# **********

class P_node():
    def __init__(self,
            subgraph    = None,
            main_target = None,
            unique_id_generator = None):
        self.subgraph    = sgp = subgraph
        self.main_target = mt  = main_target
        assert((mt is not None) != (sgp is not None))
        if sgp is not None:
            self.subgraph_id = sid = sgp.graph_id
            self.name        = f"Subgraph_{sid}"
            self.is_leaf     = False
        else:
            self.subgraph_id = None
            self.name        = f"Var_{mt}"
            self.is_leaf     = True
        self.deps         = set()
        self.users        = set()
        self.deps_only_global  = set()
        self.users_only_global = set()
        self.unique_id = small_fcts.use_generator(unique_id_generator,self)

    def __eq__(self,pn2,force_order=False,raise_exception=False):
        pn1 = self
        b = small_fcts.check_attr(pn1,pn2,
            ["name","subgraph","main_target"],
            raise_exception)
        mmt = lambda nl : [pn.main_target for pn in nl]
        s = shared_methods.sort_nodes if force_order else (lambda s : s)
        for attr in ["deps","users","deps_only_global","users_only_global"]:
            b *= mmt(s(getattr(pn1,attr))) == mmt(s(getattr(pn2,attr)))
            if not b and raise_exception:
                raise Exception(f"P_nodes differ on attr {attr}")
        return bool(b)
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self)


# ***********
# * P_graph *
# ***********

class P_graph():
    def __init__(self,graph_id,unique_id_generator):
        self.graph_id = graph_id
        self.unique_id_generator = unique_id_generator
        self.dict_leaf_node_address = dict() 
        # contains all the leaf nodes of self and sub graphs
        # dict : main_target -> subgraph_id
        self.list_nodes = [] # contains only the nodes of self
        self.input_nodes  = [] # do NOT belong to self -> NOT in list_nodes
        self.output_nodes = [] # do belong to self -> in list_nodes

    def __eq__(self,pg2,force_order=False,raise_exception=False):
        pg1 = self
        b = small_fcts.check_attr(pg1,pg2,
            ["graph_id","dict_leaf_node_address"],
            raise_exception=raise_exception)
        mmt = lambda nl : [pn.main_target for pn in nl]
        s = shared_methods.sort_nodes if force_order else (lambda s : s)
        for attr in ["input_nodes","output_nodes"]:
            b *= mmt(s(getattr(pg1,attr))) == mmt(s(getattr(pg2,attr)))
            if not b and raise_exception:
                raise Exception(f"P_graphs differ on attr {attr}")
        b *= (len(pg1.list_nodes) == len(pg2.list_nodes))
        if not b and raise_exception:
            raise Exception("P_graphs differ on list_nodes length")
        if b:
            for pn1,pn2 in zip(pg1.list_nodes,pg2.list_nodes):
                b *= pn1.__eq__(pn2,force_order,raise_exception)
        return bool(b)

    
    def make_dict_leaf_node_address(self):
        dict_address = self.dict_leaf_node_address
        for pn in self.list_nodes:
            if pn.is_leaf:
                dict_address[pn.main_target] = self.graph_id
            else:
                sub_pg = pn.subgraph
                sub_pg.make_dict_leaf_node_address()
                sub_dict = sub_pg.dict_leaf_node_address
                dict_address.update(sub_dict)


# ************
# * P_config *
# ************

class P_config():
    def __init__(self,
            min_nodes_per_graph = 3,
            max_nodes_per_graph = 25):
        self.min_nodes_per_graph = min_nodes_per_graph  
        self.max_nodes_per_graph = max_nodes_per_graph  

default_config = P_config()

# ==========================



# =============================================
# === Generate a P_graph based on a S_graph ===
# =============================================

def S_to_P_init(sg):
    unique_id_generator = [0]
    pg = P_graph("0",unique_id_generator)
    pg.list_nodes = lpn = []
    dict_mt_to_pn = dict()
    for sn in sg.nodes:
        mt = sn.main_target
        pn = P_node(
            main_target=mt,
            unique_id_generator=unique_id_generator)
        lpn.append(pn)
        dict_mt_to_pn[mt] = pn
        for req_sn in sn.deps.keys():
            if not (req_sn is sg.init_node):
                req_pn = dict_mt_to_pn[req_sn.main_target]
                pn.deps.add(req_pn)
                req_pn.users.add(pn)

    # ** input ** 
    input_pn = P_node(
        main_target="sources",
        unique_id_generator=unique_id_generator
    )
    pg.input_nodes = [input_pn]
    for user_sn in sg.init_node.users:
        user_pn = dict_mt_to_pn[user_sn.main_target]
        user_pn.deps_only_global.add(input_pn)
        input_pn.users_only_global.add(user_pn)
    
    # ** output **
    output_pn = dict_mt_to_pn[sg.output_node.main_target]
    pg.output_nodes = [output_pn]
    return pg

# ==========================



# =====================================================
# === FIRST RULE : GROUP SEQUENCE OF NODES TOGETHER ===
# =====================================================

def rule_group_sequences(pg : P_graph,config : P_config = default_config):
    # ** Find the sequences **
    tot_nb_seq = 0
    dict_seq_nb = dict() # mt -> a seq nb
    dict_sequences = dict() # seq nb -> list of nodes in the seq
    for pn in pg.list_nodes:
        if len(pn.users) == 1 and len(list(pn.users)[0].deps) == 1:
            mt = pn.main_target
            user_pn = list(pn.users)[0]
            user_mt = user_pn.main_target
            if mt in dict_seq_nb:
                seq_nb = dict_seq_nb[mt]
                dict_seq_nb[user_mt] = seq_nb
                dict_sequences[seq_nb].append(user_pn)
            else:
                tot_nb_seq += 1
                dict_seq_nb[mt] = tot_nb_seq
                dict_seq_nb[user_mt] = tot_nb_seq
                dict_sequences[tot_nb_seq] = [pn,user_pn]

    # ** split too long sequences **
    all_sequences = list(dict_sequences.items())
    for seq_nb,sequence in all_sequences:
        seq_len = len(sequence)
        if seq_len > config.max_nodes_per_graph:
            del dict_sequences[seq_nb]
            nb_seq = math.ceil(seq_len/config.max_nodes_per_graph)
            sub_seq_len = math.ceil(seq_len/nb_seq)
            for first in range(0,seq_len,sub_seq_len):
                end = min(first + sub_seq_len,seq_len)
                sub_seq = sequence[first:end]
                tot_nb_seq += 1
                sub_seq_nb = tot_nb_seq
                dict_sequences[sub_seq_nb] = sub_seq
                for pn in sub_seq:
                    dict_seq_nb[pn.main_target] = sub_seq_nb

    # ** Group the node **
    for seq_nb,sequence in dict_sequences.items():
        if len(sequence) < config.min_nodes_per_graph:
            continue
        sub_pg = P_graph(
            graph_id=f"{pg.graph_id}_{seq_nb}",
            unique_id_generator=pg.unique_id_generator
        )
        sub_pg.list_nodes = sequence
        new_pn = P_node(
            subgraph = sub_pg,
            unique_id_generator=pg.unique_id_generator
        )
        # * unplug the first node *
        first_pn = sequence[0]
        sub_pg.input_nodes = list(first_pn.deps)
        for req_g_pn in first_pn.deps_only_global:
            # req_g_pn.users_only_global.discard(first_pn)
            # req_g_pn.users_only_global.add(new_pn)
            new_pn.deps_only_global.add(req_g_pn)
        first_pn.deps_only_global = set()
        for req_pn in first_pn.deps:
            req_pn.users.discard(first_pn)
            req_pn.users.add(new_pn)
            new_pn.deps.add(req_pn)
            # req_pn.users_only_global.add(first_pn)
            first_pn.deps_only_global.add(req_pn)
        first_pn.deps = set()

        # * unplug the last node *
        last_pn = sequence[-1]
        sub_pg.output_nodes = [last_pn]
        for user_g_pn in last_pn.users_only_global:
            # user_g_pn.deps_only_global.discard(last_pn)
            # user_g_pn.deps_only_global.add(new_pn)
            new_pn.users_only_global.add(user_g_pn)
        last_pn.users_only_global = set()
        for user_pn in last_pn.users:
            user_pn.deps.discard(last_pn)
            user_pn.deps.add(new_pn)
            new_pn.users.add(user_pn)
            # user_pn.deps_only_global.add(last_pn)
            last_pn.users_only_global.add(user_pn)
        last_pn.users = set()

        # * update the main_pg's list_nodes *
        lpn = pg.list_nodes
        lpn[lpn.index(first_pn)] = new_pn
        for pn in sequence[1:]:
            lpn.remove(pn)

# ==========================



# ===============================================================
# === SECOND RULE : MERGE NODES WITH A UNIQUE COMMON ANCESTOR ===
# ===============================================================

# the flow of pn is the list of nodes 
# to_be_visited which are descendants of pn

def rule_merge_small_flows(pg : P_graph,config : P_config = default_config):
    # === FIRST ===
    # for each node we need to find where its flow converge back
    dict_nb_usages = dict([(pn, len(pn.users)) for pn in pg.list_nodes])
    to_be_visited = []
    dict_flow = dict()
    # for a pn already visited -> its descendants in to_be_visited
    # if len(flow_size) = 0 => the flow converged
    # its a generalisation of "seen" in cut_based_on_deps
    dict_which_flow = dict()
    # for a pn in to_be_visited -> all the flows he is part of
    # ie a list of P_nodes, representing there flow
    # reciprocal of dict_flow 
    dict_end_of_flow = dict()
    # any pn -> where its flow converged back
    # this is what we want to build
    for pn in pg.list_nodes:
        if len(pn.users) == 0:
            to_be_visited.append(pn)
            dict_which_flow[pn] = set()

    while to_be_visited != []:
        pn = to_be_visited.pop()
        current_flows = dict_which_flow[pn]
        continuing_flows = set([pn])
        dict_flow[pn] = set()
        # * check if end of flows *
        for flow_pn in current_flows:
            flow = dict_flow[flow_pn]
            flow.remove(pn)
            # equivalent to "seen.remove(n)" in cut_based_on_deps
            if flow == set():
                dict_end_of_flow[flow_pn] = pn
            else:
                continuing_flows.add(flow_pn)
        # * visit pn *
        for req_pn in pn.deps:
            # equivalent to seen.add(req_n) :
            for flow_pn in continuing_flows:
                flow = dict_flow[flow_pn]
                flow.add(req_pn)
            if req_pn in dict_which_flow:
                dict_which_flow[req_pn].update(continuing_flows)
            else:
                dict_which_flow[req_pn] = continuing_flows
            dict_nb_usages[req_pn]-=1
            if dict_nb_usages[req_pn]==0:
                to_be_visited.append(req_pn)

    # === SECOND ===
    # For each flow we have 4 options :
    # -> include the source or not
    # -> include the sink or not
    return dict_end_of_flow
    
            



        
        

            







# ==========================



# ==========================
# === printing functions ===
# ==========================

color_leaf     = "blue"
color_subgraph = "blueviolet"
color_special  = "green"
color_edge     = color_leaf

def print_P_graph(pg : P_graph,name=None,open=True,render_format="svg"):
    # ----- init -----
    print(f"Partitioned forward graph : {len(pg.list_nodes)} nodes")
    if name is None: name = "Partitioned_forward_graph"
    dot = graphviz.Digraph(
        name,
        comment="P_graph = Partitioned forward graph")
    # ----- Core -----
    for input_pn in pg.input_nodes:
        dot.node(
            input_pn.name,
            f"INPUT {input_pn.main_target}",
            color=color_special, style="dashed")
    for pn in pg.list_nodes:
        if pn.is_leaf:
            dot.node(pn.name,pn.name,color=color_leaf)
        else:
            dot.node(
                pn.name,
                f"{pn.name}\n{len(pn.subgraph.list_nodes)} nodes",
                color=color_subgraph)
        for req_pn in pn.deps:
            dot.edge(req_pn.name,pn.name,color=color_edge)
        for req_pn in pn.deps_only_global:
            dot.edge(req_pn.name,pn.name,color=color_special)
    # ----- render -----
    small_fcts.graph_render(dot,open,"P",render_format)
