# ==========================
# ====== D structure =======
# ==========================

from torch import Tensor
from src.lowlevel import ast_add_on
from src.lowlevel import variable_info
from src.core import base
from src.core.raw import RawNode,RawGraph

# **********
# * ForwardNode *
# **********

class ForwardNode(base.Node):
    def __init__(self,
            target,
            code_ast=None,
            fct="",
            is_rand=False,
            deps_rand=None,
            forward_graph=None):
        """ attributes :
        .target    : str  : the name of the only var defined in the node
        .code_ast  : AST  : right part of the assigning code
        .fct       : str  : the function used in .code_ast
        .is_input  : bool : inputs are represented by nodes wth dummy code
        .is_rand   : bool : whether .fct involves randomness
        .deps      : ForwardNode set : required nodes to run .code_ast
        .deps_rand : str set : required random targets
        .users     : ForwardNode set : reciprocal of .deps
        .protected : bool : whether self is a 1-separator of the graph
        """
        super().__init__("F",target,
            parent_structure_with_id_generator=forward_graph)
        if code_ast is None:
            code_ast = ast_add_on.make_ast_constant("/!\\ not defined /!\\")
        self.code_ast = code_ast
        self.fct = fct
        self.is_input = False
        self.is_rand = is_rand
        self.deps = set()
        self.users = set()
        self.deps_rand = deps_rand if deps_rand else set()
        self.protected = False

    def get_all_standard_deps(self):
        return self.deps
    def get_all_standard_users(self):
        return self.users


# ***********
# * ForwardGraph *
# ***********

class ForwardGraph(base.Graph):
    def __init__(self):
        super().__init__("F")


    def generate_deep_tmp_local(self,raw_node,our_global):
        # To generate an environment where to run raw_node's code,
        # we generate its dependencies, either using the info 
        # (about shape, dtype etc) we previously collected, 
        # or by running their code in case of view or inplace nodes, 
        # in which case we first (i) generate their dependencies, 
        # using dict_info; and also (ii) its random dependencies.
        tmp_local = dict()
        done = set()
        ready = set()
        todo = list(raw_node.deps)
        while todo != []:
            req_rn = todo[-1]
            req_tar = req_rn.target
            if req_tar in done:
                todo.pop()
            else:
                req_info = self.dict_info[req_tar]
                if (req_info.is_inplace 
                or  req_info.is_view
                or  req_rn.fct == "getattr"):
                    if req_tar in ready:
                        for req_rd in req_rn.deps_rand:
                            if not req_rd in done:
                                code = ast_add_on.make_str_assign(
                                    (req_rd,self.dict_rand[req_rd]))
                                exec(code,our_global,tmp_local)
                                done.add(req_rd)
                        exec(req_rn.get_code(),our_global,tmp_local)
                        done.add(req_tar)
                        todo.pop()
                    else:
                        todo.extend(list(req_rn.deps))
                        ready.add(req_tar)
                else:
                    req_x = req_info.generate_value(our_global["device"])
                    if isinstance(req_x,Tensor):
                        req_x = req_x.clone()
                    tmp_local[req_tar] = req_x
                    done.add(req_tar)
                    todo.pop()
        return tmp_local


    def prepare_cut(self):
        # in case, after simplifications, we will cut / sequentialize
        # we need to protect the separators from "cheap" simplifications
        seps = RK_get_1_separators(self)
        for sep in seps: sep.protected = True

# ==========================


# ==========================
# = Move from B to D graph =
# ==========================

def sort_nodes(g : B_graph): # -> B_node list 
    # use output's node and trace everything
    # /!\ never trust B_graph.nodes
    o_var = g.output_var
    if not o_var.has_node: return []
    else: return RK_sort_based_on_deps(o_var.node)



# ==========================

# ===== Main function ======

def B_to_D(bg : B_graph,model,dict_inputs,device=None,dont_build_dict_info=False):
    if not device:
        device = small_fcts.get_device_and_check_all_same_device(
            model,dict_inputs)

    # --- init and sort ---
    dg = ForwardGraph()
    dg.inherit_base_attributes(bg)
    inputs       = dg.inputs
    d_nodes      = dg.nodes
    dict_info    = dg.dict_info
    dict_nodes   = dict()
    b_nodes      = sort_nodes(bg)

    # -- Fix B_nodes without users --
    to_insert_back = [
        bn for bn in bg.nodes
        if (bn not in b_nodes
        and len(bn.deps)!=0)]
    to_insert_back.sort(key=base.Node.get_num)
    to_insert_back = to_insert_back[::-1]
    while to_insert_back != []:
        retry_list = []
        for bn in to_insert_back:
            index_deps = []
            fail = False
            for req_bn in bn.deps:
                if req_bn not in b_nodes:
                    fail = True
                    break
                else:
                    index_deps.append(b_nodes.index(req_bn))
            if fail:
                retry_list.append(bn)
                continue
            else:
                max_index = max(index_deps)
                b_nodes.insert(max_index+1,bn)
        if retry_list == to_insert_back:
            to_insert_back = [] # -> Give up
        else:
            to_insert_back = retry_list
    # ------
            
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
    
    dict_inplace_ops = dict() # dict : main target -> set targets of inplace stuff

    sources_req_grad = False
    for bn in b_nodes:
        # -- translate B node to D --
        dn = ForwardNode(bn.target,bn.code_ast,bn.fct,
                is_rand = bn.is_rand,
                deps_rand = set(bn.deps_rand),
                other_obj=dg)
        if bn.is_input:
            inputs.append(bn.target)
            dn.is_input = True
            dict_info[bn.target] = input_info = def_info.VariableInfo(
                dict_inputs[bn.target],
                data_owner_name = bn.target)
            if input_info.requires_grad:
                sources_req_grad = True
        for req_bn in bn.deps:
            req_dn = dict_nodes[req_bn.target]
            dn.deps.add(req_dn)
            req_dn.users.add(dn)
        dict_nodes[bn.target] = dn
        d_nodes.append(dn)

        # -- run local forward to get info --
        if dont_build_dict_info:
            dict_info[bn.target] = def_info.VariableInfo()
        elif not bn.is_input:
            tmp_local = generate_deep_tmp_local(dg,bn,our_global)
            try:
                exec(
                    bn.get_code(force_special_kwargs=True), 
                    our_global, tmp_local)
            except:
                # Something bad happened on jit.trace's Python code
                # -> for instance a dtype has been replaced by an integer 
                # we will try to fix this
                fixed = [False]
                def try_to_fix(sub_code): # fix inplace
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
                code = bn.code_ast
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

            # === FIRST WAY TO RECOGNIZE A VIEW ===
            # -> data_ptr
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

            # === SECOND WAY TO RECOGNIZE A VIEW ===
            # -> main_fct is a view/inplace function
            if not (is_inplace or is_view):
                if (bn.fct in global_vars.list_view_fct
                or bn.fct in global_vars.list_inplace_fct):
                    data_parents = set()
                    for req_bn in bn.deps:
                        req_info = dict_info[req_bn.mt]
                        if req_info.variable_type is Tensor:
                            data_parents.add(req_bn.mt)
                    if data_parents != set():
                        if bn.fct in global_vars.list_inplace_fct:
                            is_inplace = True
                        else:
                            is_view = True

            # === register ===
            if is_inplace or is_view:
                bn_deps_names = set(req_bn.target for req_bn in bn.deps)
                data_direct_parents = bn_deps_names & data_parents
                if len(data_direct_parents) == 0:
                    for req_bn in bn.deps:
                        for req_req_bn in req_bn.deps:
                            req_req_name = req_req_bn.target
                            if req_req_name in data_parents:
                                data_direct_parents.add(req_req_name)
                if len(data_direct_parents) == 0: raise Exception(
                    f"{bn.target} is an inplace or view op, it doesn't "\
                    f"share its data with any of its deps ?! (even deps of deps)")
                data_direct_parent_name = data_direct_parents.pop()
                o_info = dict_info[data_direct_parent_name]
                data_owner_name = o_info.data_owner_name
                # -> we must protect the data_owner from cheap simplification
                if is_inplace:
                    data_owner = dict_nodes[data_owner_name]
                    data_owner.protected = True
                # -> If several inplace operations 
                # -> We must ensure we compute them in the original order
                if is_inplace:
                    for other_inplace_op in dict_inplace_ops[data_owner_name]:
                        other_dn = dict_nodes[other_inplace_op]
                        other_dn.users.add(dn)
                        dn.deps.add(other_dn)
                    dict_inplace_ops[data_owner_name].add(dn.mt)
                    
            else:
                data_owner_name = bn.target
                data_direct_parent_name = bn.target
                dict_inplace_ops[bn.mt] = set()
            dict_info[bn.target] = info = def_info.VariableInfo(
                bn_value,
                is_view    = is_view,
                is_inplace = is_inplace,
                data_owner_name = data_owner_name,
                data_direct_parent_name = data_direct_parent_name)
            # ** Correct req_grad of data_parent **
            if info.requires_grad:
                dict_info[info.data_owner_name].requires_grad = True

            del tmp_local

    dg.sources_req_grad = sources_req_grad 


    # --- translate output ---
    o_var = bg.output_var
    if not isinstance(o_var.val,ast.Name):
        warnings.warn(
            f"According to Btools module's return isn't a variable."\
            f"Thus we assume it's a constant. \n"\
            f"AST type of the output: {type(o_var.val)}")
        raise global_vars.ExceptionModuleDoesNotReqGrad
    str_val = o_var.val.id
    if not o_var.has_node:
        warnings.warn(
            f"Btools hasn't attached any node to the output."\
            f"Thus we assume it's a constant.")
        raise global_vars.ExceptionModuleDoesNotReqGrad
    else:
        output_node = dict_nodes[str_val]
        output_info = dict_info[output_node.mt]
        if not output_info.requires_grad:
            warnings.warn(
                "None of the outputs require grad. "\
                "Thus there is nothing to do.")
            raise global_vars.ExceptionModuleDoesNotReqGrad
        else:
            dg.output_nodes = [output_node]
    dg.outputs = [str_val] # maybe just a tuple constructor...
    dg.whole_module_output = str_val


    # --- missing edges for inplace operations ---
    # -> May change the output
    dict_dn = dict((dn.mt,dn) for dn in d_nodes)
    for index_dn,dn in enumerate(d_nodes):
        if len(dn.users)==0 and dn.mt != str_val:
            info : def_info.VariableInfo = dict_info[dn.mt]
            assert(info.is_view or info.is_inplace)
            data_owner_name = info.data_owner_name
            data_owner_dn = dict_dn[data_owner_name]
            if data_owner_name is dg.whole_module_output:
                # discard
                if data_owner_name in dg.outputs:
                    dg.outputs.remove(data_owner_name)
                dg.outputs.append(dn.mt)
            for user_dn in data_owner_dn.users:
                index_user = d_nodes.index(user_dn)
                if index_user > index_dn:
                    user_dn.deps.add(dn)
                    dn.users.add(dn)


    # --- Correct requires_grad ---
    for dn in d_nodes:
        info = dict_info[dn.mt]
        if dict_info[info.data_owner_name].requires_grad:
            info.requires_grad = True

    # -> If none of users req_grad -> useless to req_grad
    for dn in d_nodes[::-1]:
        if dn.mt not in dg.outputs:
            info = dict_info[dn.mt]
            if info.requires_grad:
                one_user_req_grad = False
                for user_dn in dn.users:
                    if dict_info[user_dn.mt].requires_grad:
                        one_user_req_grad = True
                        break
                if not one_user_req_grad:
                    info.requires_grad = False



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
    for n in g:
        if print_ast:
            print(ast.dump(ast_add_on.make_ast_assign(
                (n.target,n.code_ast)),indent=4))
        else:
            print(f"({n.target}) : [{n.fct}] : {n.get_code()}")
    if isinstance(g,ForwardGraph):
        print("dict_info : ")
        for (tar,info) in g.dict_info.items():
            print(f"{tar} info : {info}")

def print_fw_code(dg : ForwardGraph):
    print(dg.dict_rand)
    str_input = ','.join(dg.inputs)
    print(f"def main({str_input}):")
    for dn in dg.nodes:
        if not dn.is_input: print(f"\t{dn.get_code()}")
    print(f"\treturn {dg.output}")

def aux_print_ForwardGraph_message(dg : ForwardGraph):
    return f"ForwardGraph - Forward graph : of size {len(dg.nodes)}"
def aux_print_ForwardGraph_name(dg,name=None):
    if name is not None: return name
    else: return "Forward_ForwardGraph"

def print_ForwardGraph(dg : ForwardGraph,name=None,open=True,render_format="svg",dot=None,uniq_num=0):
    if dot is None:
        render = True
        if name is None: name = aux_print_ForwardGraph_name(dg)
        dot = graphviz.Digraph(name,comment=name)
    else:
        render = False
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,**kwargs): dot.edge(uni(i1),uni(i2), **kwargs)
    for dn in dg.nodes:
        if dn.is_input:
            node(dn.target,dn.get_code(),color="blue")
        elif dn.target in dg.outputs:
            node(dn.target,dn.get_code(),color="red")
        else: node(dn.target,dn.get_code())
    for dn in dg.nodes:
        for req_dn in dn.deps:
            edge(req_dn.target,dn.target)
    if render:
        small_fcts.graph_render(dot,open,"D",render_format)

# ==========================


""" NOT maintained
# ==========================
# === test forward code ====
# ==========================

def test_fw_code(dg : ForwardGraph,model,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = model
    for inp in dg.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for rd_tar,code_ast in dg.dict_rand.items():
        code = ast_add_on.make_str_assign(rd_tar,code_ast)
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
