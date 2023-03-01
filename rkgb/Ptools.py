# ==========================
# ====== P structure =======
# ==========================

# ** Graph partitioning **

from rkgb.utils import *

class P_node():
    def __init__(self,
            subgraph    = None,
            subgraph_id = None,
            main_target = None,
            unique_id_generator = None):
        self.subgraph    = subgraph
        self.subgraph_nb = snb = subgraph_id
        self.main_target = mt  = main_target
        if self.subgraph_nb is not None:
            self.name = f"Subgraph_{snb}"
        else:
            self.name = f"Var_{mt}"
        self.deps         = set()
        self.users        = set()
        self.deps_global  = set()
        self.users_global = set()
        self.unique_id = small_fcts.use_generator(unique_id_generator,self)

    def __eq__(self,pn2,force_order=False,raise_exception=False):
        pn1 = self
        b = small_fcts.check_attr(pn1,pn2,
            ["name","subgraph","main_target"],
            raise_exception)
        mmt = lambda nl : [pn.main_target for pn in nl]
        s = shared_methods.sort_nodes if force_order else (lambda s : s)
        for attr in ["deps","users","deps_global","users_global"]:
            c = mmt(s(getattr(pn1,attr))) == mmt(s(getattr(pn2,attr)))
            b *= c
            if not c and raise_exception:
                raise Exception(f"P_nodes differ on attr {attr}")
        return bool(b)
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self)


class P_graph():
    def __init__(self,graph_id,unique_id_generator):
        self.graph_id = graph_id
        self.dict_leaf_node_address = dict() 
        # contains all the leaf nodes of self and sub graphs
        # dict : main_target -> subgraph_id

        self.list_nodes = [] 
        # contains only the nodes of self
        self.input_names  = []
        self.input_nodes  = [] # do NOT belong to self -> NOT in list_nodes
        self.output_names = []
        self.output_nodes = [] # do belong to self -> in list_nodes

        self.unique_id_generator = unique_id_generator
    
    def make_dict_leaf_node_address(self):
        pass
        # TODO TODO


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
    for sn in sg.nodes:
        mt = sn.main_target
        pn = dict_mt_to_pn[mt]
        for req_sn in sn.deps.keys():
            req_pn = dict_mt_to_pn[req_sn.main_target]
            pn.deps.add(req_pn)
            req_pn.users.add(pn)

    # ** input ** 
    input_pn = P_node(
        main_target="sources",
        unique_id_generator=unique_id_generator
    )
    pg.input_names = [input_pn.name]
    pg.input_nodes = [input_pn]
    for user_sn in sg.init_node.users:
        user_pn = dict_mt_to_pn[user_sn.main_target]
        user_pn.deps_global.add(input_pn)
        input_pn.users_global.add(user_pn)
    
    # ** output **
    output_pn = dict_mt_to_pn[sg.output_node.main_target]
    pg.output_names = [output_pn.name]
    pg.output_nodes = [output_pn]




