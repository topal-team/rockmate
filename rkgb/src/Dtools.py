from rkgb.utils import *
from rkgb.Btools import B_node,B_graph

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
        .is_rand   : bool : whether .fct involves randomness
        .deps      : D_node set : required nodes to run .ast_code
        .deps_rand : str set : required random targets
        .users     : D_node set : reciprocal of .deps
        .protected : bool : whether self is a 1-separator of the graph
        """
        super().__init__(target,code,fct)
        self.is_rand = is_rand
        self.deps_rand = deps_rand if deps_rand else set()
        self.users = set()
        self.protected = False
        self.num = shared_methods.get_num(self)
    def __eq__(self,dn2,force_order=False,raise_exception=False):
        dn1 = self
        b = small_fcts.check_attr(dn1,dn2,
            ["protected","target","fct","is_rand","deps_rand"],
            raise_exception=raise_exception)
        mkstr = lambda nl : [rn.target for rn in nl]
        s = shared_methods.sort_nodes if force_order else (lambda s : s)
        b = (b
            and (mkstr(s(dn1.deps)) == mkstr(s(dn2.deps)))
            and (mkstr(s(dn1.users)) == mkstr(s(dn2.users)))
            and (dn1.get_code() == dn2.get_code()))
        return b
    def __hash__(self):
        if hasattr(self,"num"): return self.num
        else: return id(self)
        # __eq__ => need __hash__

class D_graph():
    def __init__(self):
        self.inputs = [] # str list
        self.nodes  = [] # D_node list -> topo sorted
        self.output = None # str
        self.output_node = None # D_node
        self.dict_info = dict() # target -> FWD_info
        self.dict_rand = dict()
        self.dict_constants = dict()
    def __eq__(self,g2,force_order=False,raise_exception=False):
        g1 = self
        b = small_fcts.check_attr(g1,g2,
            ["inputs","output","dict_info"],
            raise_exception=raise_exception)
        mt = lambda l : [dn.target for dn in l]
        b *= (mt(g1.nodes) == mt(g2.nodes))
        if raise_exception and not b:
            raise Exception("D_graphs' differ on nodes order or length")
        if b:
            for dn1,dn2 in zip(g1.nodes,g2.nodes):
                b *= dn1.__eq__(dn2,force_order,raise_exception)
        return b
    def __hash__(self):
        return id(self)

    def prepare_cut(self):
        # in case, after simplifications, we will cut / sequentialize
        # we need to protect the separators from simplifications
        # but in case of chain of separators, we only protect
        # the last one (we will keep a good structure, while reducing
        # the number of blocs)
        all_sep = shared_methods.cut_based_on_deps(self)
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
    else: return shared_methods.sort_based_on_deps(o_var.node)

def generate_deep_tmp_local(bg,dict_info,bn,our_global):
    tmp_local = dict()
    done = set()
    ready = set()
    todo = list(bn.deps)
    while todo != []:
        req_bn = todo[-1]
        req_tar = req_bn.target
        if req_tar in done:
            todo.pop()
        else:
            req_info = dict_info[req_tar]
            if (req_info.is_inplace 
            or  req_info.is_view
            or  req_bn.fct == "getattr"):
                if req_tar in ready:
                    for req_rd in req_bn.deps_rand:
                        if not req_rd in done:
                            code = ast_add_on.make_str_assign(
                                (req_rd,bg.dict_rand[req_rd]))
                            exec(code,our_global,tmp_local)
                            done.add(req_rd)
                    exec(req_bn.get_code(),our_global,tmp_local)
                    done.add(req_tar)
                    todo.pop()
                else:
                    todo.extend(list(req_bn.deps))
                    ready.add(req_tar)
            else:
                req_x = def_info.generate_val(req_info,our_global["device"])
                if isinstance(req_x,torch.Tensor):
                    req_x = req_x.clone()
                tmp_local[req_tar] = req_x
                done.add(req_tar)
                todo.pop()
    return tmp_local

# ==========================

# ===== Main function ======

def B_to_D(bg : B_graph,model,dict_inputs,device=None,dont_build_dict_info=False):
    if not device:
        device = small_fcts.get_device_and_check_all_same_device(
            model,dict_inputs)
    # globals()["device"] = device

    # --- init and sort ---
    dg = D_graph()
    inputs       = dg.inputs
    d_nodes      = dg.nodes
    dict_info    = dg.dict_info
    dg.dict_rand = bg.dict_rand
    dg.dict_constants = bg.dict_constants
    dict_nodes   = dict()
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
    our_global.update(bg.dict_constants)

    for bn in b_nodes:
        # -- translate B node to D --
        dn = D_node(bn.target,bn.ast_code,bn.fct,
                is_rand = bn.is_rand,
                deps_rand = set(bn.deps_rand))
        if bn.is_input:
            inputs.append(bn.target)
            dn.is_input = True
            dict_info[bn.target] = def_info.Var_info(
                dict_inputs[bn.target],
                data_owner_name = bn.target)
        for req_bn in bn.deps:
            req_dn = dict_nodes[req_bn.target]
            dn.deps.add(req_dn)
            req_dn.users.add(dn)
        dict_nodes[bn.target] = dn
        d_nodes.append(dn)

        # -- compute the forward to get info --
        if dont_build_dict_info:
            dict_info[bn.target] = def_info.Var_info()
        elif not bn.is_input:
            tmp_local = generate_deep_tmp_local(dg,dict_info,bn,our_global)
            try:
                exec(
                    bn.get_code(force_special_kwargs=True), 
                    our_global, tmp_local)
            except:
                # Something bad happened on jit.trace's Python code
                # -> for instance a dtype has been replaced by an integer 
                # we will try to fix this
                fixed = [False]
                def try_to_fix(sub_code):
                    if isinstance(sub_code,ast.Call):
                        args_ast = list(sub_code.args)
                        for i,arg in enumerate(args_ast):
                            if (ast_add_on.is_constant(arg)
                            and isinstance(arg.value,int)):
                                save_value = arg.value
                                sub_code.args[i] = (
                                    ast_add_on.make_ast_constant(
                                    global_vars.get_torchscript_dtype(arg.value)
                                ) )
                                try:
                                    exec(bn.get_code(), our_global, tmp_local)
                                    fixed[0] = True
                                    break
                                except:
                                    sub_code.args[i] = save_value
                            else: try_to_fix(arg)
                code = bn.ast_code
                try_to_fix(code)
                if not fixed[0]: raise Exception(
                    f"Sorry there are some errors in the code generated :"\
                    f"using jit which make it impossible to exec, the code "\
                    f"is : {ast_add_on.ast_to_str(code)}"
                )
                
            # - detect inplace operation -
            bn_value = tmp_local[bn.target]
            is_view    = False # by default
            is_inplace = False # by default
            data_parents = set()
            if small_fcts.has_a_data_ptr(bn_value):
                bn_data_ptr = small_fcts.get_data_ptr(bn_value)
                for o_name,o_value in tmp_local.items():
                    if (o_name != bn.target
                    and o_name in dict_info
                    and small_fcts.has_a_data_ptr(o_value)
                    and small_fcts.get_data_ptr(o_value) == bn_data_ptr):
                        data_parents.add(o_name)
                        data_owner_name = o_name
                        if o_value is bn_value: is_inplace = True
                        else: is_view = True
            if is_inplace or is_view:
                bn_deps_names = set(req_bn.target for req_bn in bn.deps)
                data_direct_parents = bn_deps_names & data_parents
                if len(data_direct_parents) == 0: raise Exception(
                    f"{bn.target} is an inplace or view op, it doesn't "\
                    f"share its data with any of its deps ?!")
                data_direct_parent_name = data_direct_parents.pop()
                o_info = dict_info[data_direct_parent_name]
                data_owner_name = o_info.data_owner_name
                # -> we must protect the data_owner from cheap simplification
                if is_inplace:
                    data_owner = dict_nodes[data_owner_name]
                    data_owner.protected = True

            else:
                data_owner_name = bn.target
                data_direct_parent_name = bn.target
            dict_info[bn.target] = def_info.Var_info(
                bn_value,
                is_view    = is_view,
                is_inplace = is_inplace,
                data_owner_name = data_owner_name,
                data_direct_parent_name = data_direct_parent_name)
            del tmp_local

    # --- translate output ---
    o_var = bg.output
    assert(isinstance(o_var.val,ast.Name))
    str_val = o_var.val.id
    if o_var.has_node:
        dg.output_node = dict_nodes[str_val]
    dg.output = str_val

    # -- prepares the sequencing --
    dg.prepare_cut()

    return dg

# ==========================



# ==========================
# === printing functions ===
# ==========================

def print_all_fw_nodes(g,print_ast=True):
    # g : B or D graph
    print(g.dict_rand)
    for n in g.nodes:
        if print_ast:
            print(ast.dump(ast_add_on.make_ast_assign(
                (n.target,n.ast_code)),indent=4))
        else:
            print(f"({n.target}) : [{n.fct}] : {n.get_code()}")
    if isinstance(g,D_graph):
        print("dict_info : ")
        for (tar,info) in g.dict_info.items():
            print(f"{tar} info : {info}")

def print_fw_code(dg : D_graph):
    print(dg.dict_rand)
    str_input = ','.join(dg.inputs)
    print(f"def main({str_input}):")
    for dn in dg.nodes:
        if not dn.is_input: print(f"\t{dn.get_code()}")
    print(f"\treturn {dg.output}")

def print_D_graph(dg : D_graph,name=None,open=True,render_format="svg"):
    print(len(dg.nodes))
    if name is None:
        name = "Forward_graph"
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
    small_fcts.graph_render(dot,open,"D",render_format)

# ==========================


""" NOT maintained
# ==========================
# === test forward code ====
# ==========================

def test_fw_code(dg : D_graph,model,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = model
    for inp in dg.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for rd_tar,ast_code in dg.dict_rand.items():
        code = ast_add_on.make_str_assign(rd_tar,ast_code)
        exec(code, globals(), loc_dict)
    for dn in dg.nodes:
        for req_rd in dn.deps_rand:
            code = ast_add_on.make_str_assign((req_rd,dg.dict_rand[req_rd]))
            exec(code,globals(),loc_dict)
        if not dn.is_input:
            exec(
                dn.get_code(force_special_kwargs=True), 
                globals(), loc_dict)
    return loc_dict[dg.output]

# ==========================
"""
