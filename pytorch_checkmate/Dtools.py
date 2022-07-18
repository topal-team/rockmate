from .root import *
from . import Btools # -> B structure

# ==========================
# ====== D structure =======
# ==========================

class D_node(Btools.B_node):
    def __init__(self,target="",code=None,fct=""):
        # "code" must be an AST, "fct" is a string
        super().__init__(target,code,fct)
        self.used_by = set()

class D_graph():
    def __init__(self):
        self.inputs = [] # str list
        self.nodes  = [] # D_node list -> topo sorted
        self.output = None # str
        #self.output_node = None # D_node
        self.dict_rand = {}
        self.dict_info = {} # target -> FWD_info

# ==========================


# ==========================
# = Move from B to D graph =
# ==========================

def sort_nodes(g : Btools.B_graph): # -> B_node list 
    # use output's node and trace everything
    # /!\ never trust B_graph.nodes
    o_var = g.output
    if not o_var.has_node: return []
    else: return sort_based_on_req(o_var.node)


def get_info(x) -> FWD_info:
    # for FWD_info see root.py
    info = FWD_info()
    if (isinstance(x,int) or
        (isinstance(x,torch.Tensor) and x.shape==torch.Size([]))):
        tt = torch.Size
    else:
        tt = type(x)
    info.ttype = tt
    if tt==torch.Size:
        info.tsize = int(x)
        info.requires_grad = False
    elif tt==torch.Tensor:
        info.tsize = x.shape
        info.dtype = x.dtype
        info.requires_grad = x.requires_grad
    elif tt==tuple or tt==list:
        info.sub_info = [get_info(y) for y in x]
    else:
        raise Exception(f"The type {tt} is unknown")
    return info

def generate_tmp_local(g,dict_info,n):
    tmp_local = {}
    for sub_n in n.req:
        sub_info = dict_info[sub_n.target]
        sub_x = generate_val(sub_info,device) # from root.py
        tmp_local[sub_n.target] = sub_x
    if n.is_rand:
        for sub_r in n.req_rand:
            exec(g.dict_rand[sub_r],our_global,tmp_local)
    return tmp_local

# ==========================

# ===== Main function ======

def B_to_D(bg : Btools.B_graph,nn_mod,dict_inputs,D_device=None):
    # -> D_graph:
    # -- device --
    global device
    if D_device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:   device = D_device
    nn_mod.to(device)
    for (k,x) in dict_inputs.items():
        dict_inputs[k] = x.to(device)
    # --

    # --- init and sort ---
    dg = D_graph()
    inputs       = dg.inputs
    d_nodes      = dg.nodes
    dict_info    = dg.dict_info
    dg.dict_rand = bg.dict_rand
    dict_nodes   = {}
    b_nodes      = sort_nodes(bg)

    # --- translate B node to D and make dict_info ---
    # -> to make dict_info we need to run the forward !
    # -> we handle nodes one by one : 
    # 1) generate random input vectors
    # 2) exec the code to generate the node value
    # 3) extract the info for this node, and forget the tensors
    our_global = globals().copy()
    our_global["self"] = nn_mod
    our_global["device"] = device

    for n in b_nodes:
        # -- translate B node to D --
        dn = D_node(n.target,n.ast_code,n.fct)
        if n.is_input:
            inputs.append(n.target)
            dn.is_input = True
            dict_info[n.target] = get_info(dict_inputs[n.target])
        for sub_n in n.req:
            sub_dn = dict_nodes[sub_n]
            dn.req.add(sub_dn)
            sub_dn.used_by.add(dn)
        dict_nodes[n] = dn
        d_nodes.append(dn)

        # -- compute the forward to get info --
        if not n.is_input:
            tmp_local = generate_tmp_local(dg,dict_info,n)
            exec(n.get_code(), our_global, tmp_local)
            dict_info[n.target] = get_info(tmp_local[n.target])
            del tmp_local

    # --- translate output ---
    o_var = bg.output
    assert(isinstance(o_var.val,ast.Name))
    str_val = o_var.val.id
    #if o_var.has_node:
    #    dg.output_node = dict_nodes[o_var.node]
    dg.output = str_val

    return dg

# ==========================



# ==========================
# === printing functions ===
# ==========================

def print_info(info : FWD_info):
    print(f"\tttype = {info.ttype}")
    print(f"\ttsize = {info.tsize}")
    print(f"\tdtype = {info.dtype}")
    print(f"\trequires_grad = {info.requires_grad}")
    print(f"\tsub_info = {info.sub_info}")

def print_all_fw_nodes(g,print_ast=True):
    # g : B or D graph
    print(g.dict_rand)
    for n in g.nodes:
        if print_ast:
            print(ast.dump(n.ast_code,indent=4))
        else:
            print(f"({n.target}) : [{n.fct}] : {n.get_code()}")
    if isinstance(g,D_graph):
        print("dict_info : ")
        for (tar,info) in g.dict_info.items():
            print(f"{tar} info :")
            print_info(info)

def print_fw_code(g : D_graph):
    print(g.dict_rand)
    str_input = ','.join(g.inputs)
    print(f"def main({str_input}):")
    for n in g.nodes:
        if not n.is_input: print(f"\t{n.get_code()}")
    print(f"\treturn {g.output}")

def print_D_graph(g : D_graph,name=None,open=True):
    print(len(g.nodes))
    if name is None:
        name = "forward D-graph"
    dot = graphviz.Digraph(name,comment="D_graph = forward graph")
    for n in g.nodes:
        if n.is_input:
            dot.node(n.target,n.get_code(),color="blue")
        elif n.target == g.output:
            dot.node(n.target,n.get_code(),color="red")
        else: dot.node(n.target,n.get_code())
    for n in g.nodes:
        for sub_n in n.req:
            dot.edge(sub_n.target,n.target)
    graph_render(dot,open,"D") # from root.py

# ==========================



# ==========================
# === test forward code ====
# ==========================

def test_fw_code(g : D_graph,nn_mod,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = nn_mod
    for inp in g.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for v in g.dict_rand:
        exec(g.dict_rand[v], globals(), loc_dict)
    for n in g.nodes:
        if n.is_rand:
            for sub_t in n.req_rand:
                exec(g.dict_rand[sub_t])
        if not n.is_input:
            exec(n.get_code(), globals(), loc_dict)
    return loc_dict[g.output]

# ==========================
