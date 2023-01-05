from .utils import *
from .Btools import B_node,B_graph

# ==========================
# ====== D structure =======
# ==========================

class D_node(B_node):
    def __init__(self,target="",code=None,fct="",
            is_rand=False,deps_rand=None):
        """ attributes :
        .target    : str  : the name of the only var defined in the node
        .ast_code  : AST  : right part of the assigning code
        .fct       : str  : the function used in .ast_code
        .is_input  : bool : inputs are represented by nodes wth dummy code
        .is_rand   : bool : ? .fct involves randomness
        .deps      : D_node set : required nodes to run .ast_code
        .deps_rand : D_node set : required random nodes
        .users     : D_node set : reciprocal of .deps
        .protected : bool : ? self is a 1-separator of the graph
        """
        super().__init__(target,code,fct)
        self.is_rand = is_rand
        self.deps_rand = deps_rand if deps_rand else set()
        self.users = set()
        self.protected = False
        self.num = get_num(self)
    def __eq__(self,dn2):
        dn1 = self
        b = check_attr(dn1,dn2,
            ["protected","target","fct","is_rand","deps_rand"])
        mkstr = lambda nl : [rn.target for rn in nl]
        b = (b
            and (mkstr(dn1.deps) == mkstr (dn2.deps))
            and (mkstr(dn1.users) == mkstr (dn2.users))
            and (dn1.get_code() == dn2.get_code()))
        return b
    def __hash__(self):
        return self.num
        #return id(self) # __eq__ => need __hash__

class D_graph():
    def __init__(self):
        self.inputs = [] # str list
        self.nodes  = [] # D_node list -> topo sorted
        self.output = None # str
        self.output_node = None # D_node
        self.dict_rand = {}
        self.dict_info = {} # target -> FWD_info
    def __eq__(self,g2):
        return check_attr(self,g2,
            ["inputs","output","dict_info","nodes"])
    def __hash__(self):
        return id(self)

    def prepare_cut(self):
        # in case, after simplifications, we will cut / sequentialize
        # we need to protect the separators from simplifications
        # but in case of chain of separators, we only protect
        # the last one (we will keep a good structure, while reducing
        # the number of blocs)
        all_sep = cut_based_on_deps(self) # utils.py : sep from inp to output
        important_sep = []
        for i in range(len(all_sep)-1):
            sep = all_sep[i]
            if sep.users != set([all_sep[i+1]]):
                important_sep.append(sep)
        important_sep.append(all_sep[-1])
        print_debug([sep.target for sep in important_sep])
        for sep in important_sep: sep.protected = True

# ==========================


# ==========================
# = Move from B to D graph =
# ==========================

def sort_nodes(g : B_graph): # -> B_node list 
    # use output's node and trace everything
    # /!\ never trust B_graph.nodes
    o_var = g.output
    if not o_var.has_node: return []
    else: return sort_based_on_deps(o_var.node)


def get_info(x,is_inplace=False,inplace_real_name=None) -> FWD_info:
    # for FWD_info see utils.py
    info = FWD_info()
    info.is_inplace=is_inplace
    if is_inplace:
        info.inplace_real_name = inplace_real_name
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

def generate_tmp_local(g,dict_info,dn,our_global):
    tmp_local = {}
    for req_dn in dn.deps:
        req_dn_info = dict_info[req_dn.target]
        req_x = generate_val(req_dn_info,device) # from utils.py
        if isinstance(req_x,torch.Tensor):
            req_x = req_x.clone()
        tmp_local[req_dn.target] = req_x
    for req_rd in dn.deps_rand:
        code = make_str_assign(req_rd,g.dict_rand[req_rd])
        exec(code,our_global,tmp_local)
    return tmp_local

# ==========================

# ===== Main function ======

def B_to_D(bg : B_graph,model,dict_inputs,device=None):
    if not device:
        device = get_device_and_check_all_same_device(model,dict_inputs)
    globals()["device"] = device

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
    our_global["self"] = model
    our_global["device"] = device

    for bn in b_nodes:
        # -- translate B node to D --
        dn = D_node(bn.target,bn.ast_code,bn.fct,
                is_rand = bn.is_rand,
                deps_rand = set(bn.deps_rand))
        if bn.is_input:
            inputs.append(bn.target)
            dn.is_input = True
            dict_info[bn.target] = get_info(dict_inputs[bn.target])
        for req_bn in bn.deps:
            req_dn = dict_nodes[req_bn]
            dn.deps.add(req_dn)
            req_dn.users.add(dn)
        dict_nodes[bn] = dn
        d_nodes.append(dn)

        # -- compute the forward to get info --
        if not bn.is_input:
            tmp_local = generate_tmp_local(dg,dict_info,bn,our_global)
            exec(bn.get_code(), our_global, tmp_local)
            # - detect inplace operation -
            bn_value = tmp_local[bn.target]
            is_inplace = False # by default
            inplace_real_name = None
            if isinstance(bn_value,torch.Tensor):
                for o_name,o_value in tmp_local.items():
                    if (o_name != bn.target
                    and o_name in dict_info
                    and o_value is bn_value):
                        o_info = dict_info[o_name]
                        is_inplace = True
                        if o_info.is_inplace:
                            inplace_real_name = o_info.inplace_real_name
                        else:
                            inplace_real_name = o_name
                        break
            dict_info[bn.target] = get_info(
                bn_value,is_inplace,inplace_real_name)
            del tmp_local

    # --- translate output ---
    o_var = bg.output
    assert(isinstance(o_var.val,ast.Name))
    str_val = o_var.val.id
    if o_var.has_node:
        dg.output_node = dict_nodes[o_var.node]
    dg.output = str_val

    # -- prepares the sequencing --
    dg.prepare_cut()

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
            print(ast.dump(
                make_ast_assign((n.target,n.ast_code)),indent=4))
        else:
            print(f"({n.target}) : [{n.fct}] : {n.get_code()}")
    if isinstance(g,D_graph):
        print("dict_info : ")
        for (tar,info) in g.dict_info.items():
            print(f"{tar} info :")
            print_info(info)

def print_fw_code(dg : D_graph):
    print(dg.dict_rand)
    str_input = ','.join(dg.inputs)
    print(f"def main({str_input}):")
    for dn in dg.nodes:
        if not dn.is_input: print(f"\t{dn.get_code()}")
    print(f"\treturn {dg.output}")

def print_D_graph(dg : D_graph,name=None,open=True):
    print(len(dg.nodes))
    if name is None:
        name = "forward D-graph"
    dot = graphviz.Digraph(name,comment="D_graph = forward graph")
    for dn in dg.nodes:
        if dn.is_input:
            dot.node(dn.target,dn.get_code(),color="blue")
        elif dn.target == dg.output:
            dot.node(dn.target,dn.get_code(),color="red")
        else: dot.node(dn.target,dn.get_code())
    for dn in dg.nodes:
        for req_dn in dn.deps:
            dot.edge(req_dn.target,dn.target)
    graph_render(dot,open,"D") # from utils.py

# ==========================



# ==========================
# === test forward code ====
# ==========================

def test_fw_code(dg : D_graph,model,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = model
    for inp in dg.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for rd_tar,ast_code in dg.dict_rand.items():
        code = make_str_assign(rd_tar,ast_code)
        exec(code, globals(), loc_dict)
    for dn in dg.nodes:
        for req_rd in dn.deps_rand:
            code = make_str_assign(req_rd,dg.dict_rand[req_rd])
            exec(code,globals(),loc_dict)
        if not dn.is_input:
            exec(dn.get_code(), globals(), loc_dict)
    return loc_dict[dg.output]

# ==========================

