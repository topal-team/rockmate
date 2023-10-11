# ==========================
# ====== D structure =======
# ==========================

import ast
import warnings
from torch import Tensor
from src.lowlevel import ast_add_on
from src.lowlevel import jit_patch
from src.lowlevel import constants
from src.lowlevel import variable_info
from src.lowlevel import preprocess_samples
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

# To explain in the doc :
# -> to make dict_info we need to run the forward !
# -> we handle nodes one by one : 
# 1) generate random input vectors
# 2) exec the code to generate the node value
# 3) extract the info for this node, and forget the tensors
class ForwardGraph(base.Graph):
    def __init__(self,
        raw_graph : RawGraph,
        model,
        dict_inputs : preprocess_samples.DictInputs,
        device,
        build_dict_info=True,
    ):
        super().__init__("F")
        self.inherit_base_attributes(raw_graph) # => dict_rand / dict_constants
        self.sources_req_grad = False # by default

        raw_nodes = raw_graph.get_toposorted_list_of_non_useless_nodes()
        dict_nodes = dict()
        dict_inplace_ops = dict() 
        # dict : main target -> set targets of related inplace operations

        our_global = globals().copy()
        our_global["self"] = model
        our_global["device"] = device
        our_global.update(self.dict_constants)

        # Translate each node one by one following the topo-order
        rn : RawNode
        for rn in raw_nodes:
            fn = ForwardNode(rn.target,rn.code_ast,rn.fct,
                    is_rand = rn.is_rand,
                    deps_rand = set(rn.deps_rand),
                    forward_graph=self)
            # inputs:
            if rn.is_input:
                self.inputs.append(rn.target)
                fn.is_input = True
                self.dict_info[rn.target] \
                    = input_info = variable_info.VariableInfo(
                        dict_inputs[rn.target],
                        data_owner_name = rn.target)
                if input_info.requires_grad:
                    self.sources_req_grad = True
            # deps:
            for req_rn in rn.deps:
                req_fn = dict_nodes[req_rn.target]
                fn.deps.add(req_fn)
                req_fn.users.add(fn)
            dict_nodes[rn.target] = fn
            self.nodes.append(fn)

            # info :
            if not build_dict_info:
                self.dict_info[fn.target] = variable_info.VariableInfo()
            elif not rn.is_input:
                # 1) Run node's code to generate the value
                tmp_local = self.generate_deep_tmp_local(rn,our_global)
                rn_code_str = rn.get_code(force_special_kwargs=True)
                try: exec(rn_code_str,our_global,tmp_local)
                except:
                    jit_patch.try_to_fix_dtype_in_returned_ast_code(
                        rn_code_str,our_global,tmp_local
                    )

            self.dict_info[rn.target] = self.detect_inplace_or_view(
                rn,tmp_local,dict_nodes,dict_inplace_ops)
            del tmp_local
        
        output_target, output_forward_node \
            = self.get_output_node_and_check_if_requires_grad(
                raw_graph.output_raw_var,dict_nodes
            )
        self.output_nodes = [output_forward_node]
        self.outputs = [output_target]
        self.whole_module_output = output_target


    # --- missing edges for inplace operations ---
    # -> May change the output
    dict_dn = dict((dn.mt,dn) for dn in d_nodes)
    for index_dn,dn in enumerate(d_nodes):
        if len(dn.users)==0 and dn.mt != output_target:
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
    

    def detect_inplace_or_view(self,
            current_raw_node,
            current_forward_node,
            tmp_local,
            dict_nodes,
            dict_inplace_ops):
        # - detect inplace operation -
        current_rn_value = tmp_local[current_raw_node.target]
        is_view    = False # by default
        is_inplace = False # by default
        data_parents = set() # variables which have the same data_ptr

        # === FIRST WAY TO RECOGNIZE A VIEW ===
        # -> data_ptr
        if variable_info.VariableInfo.has_a_data_ptr(current_rn_value):
            current_rn_data_ptr = variable_info.VariableInfo.get_data_ptr(current_rn_value)
            for o_name,o_value in tmp_local.items():
                if (o_name != current_raw_node.target
                and o_name in self.dict_info
                and variable_info.VariableInfo.has_a_data_ptr(o_value)
                and variable_info.VariableInfo.get_data_ptr(o_value) == current_rn_data_ptr):
                    data_parents.add(o_name)
                    data_owner_name = o_name
                    if o_value is current_rn_value: is_inplace = True
                    else: is_view = True

        # === SECOND WAY TO RECOGNIZE A VIEW ===
        # -> main_fct is a view/inplace function
        if not (is_inplace or is_view):
            if (current_raw_node.fct in constants.list_view_fct
            or current_raw_node.fct in constants.list_inplace_fct):
                data_parents = set()
                for req_rn in current_raw_node.deps:
                    req_info = self.dict_info[req_rn.mt]
                    if req_info.variable_type is Tensor:
                        data_parents.add(req_rn.mt)
                if data_parents != set():
                    if current_raw_node.fct in constants.list_inplace_fct:
                        is_inplace = True
                    else:
                        is_view = True

        # === register ===
        if is_inplace or is_view:
            current_rn_deps_names = set(
                req_rn.target for req_rn in current_raw_node.deps)
            data_direct_parents = current_rn_deps_names & data_parents
            if len(data_direct_parents) == 0:
                for req_rn in current_raw_node.deps:
                    for req_req_rn in req_rn.deps:
                        req_req_name = req_req_rn.target
                        if req_req_name in data_parents:
                            data_direct_parents.add(req_req_name)
            if len(data_direct_parents) == 0: raise Exception(
                f"{current_raw_node.target} is an inplace or view op, it doesn't "\
                f"share its data with any of its deps ?! (even deps of deps)")
            data_direct_parent_name = data_direct_parents.pop()
            o_info = self.dict_info[data_direct_parent_name]
            data_owner_name = o_info.data_owner_name
            # -> we must protect the data_owner from cheap simplification
            if is_inplace:
                data_owner = dict_nodes[data_owner_name]
                data_owner.protected = True
            # -> If several inplace operations 
            # -> We must ensure we compute them in the original order
            if is_inplace:
                for other_inplace_op in dict_inplace_ops[data_owner_name]:
                    other_fn = dict_nodes[other_inplace_op]
                    other_fn.users.add(current_forward_node)
                    current_forward_node.deps.add(other_fn)
                dict_inplace_ops[data_owner_name].add(current_forward_node.mt)
                
        else:
            data_owner_name = current_raw_node.target
            data_direct_parent_name = current_raw_node.target
            dict_inplace_ops[current_raw_node.mt] = set()
        info = variable_info.VariableInfo(
            current_rn_value,
            is_view    = is_view,
            is_inplace = is_inplace,
            data_owner_name = data_owner_name,
            data_direct_parent_name = data_direct_parent_name)
        # ** Correct req_grad of data_parent **
        if info.requires_grad:
            self.dict_info[info.data_owner_name].requires_grad = True
        return info
    

    def get_output_node_and_check_if_requires_grad(
            self,output_raw_var,dict_nodes):
        if not isinstance(output_raw_var.val,ast.Name):
            warnings.warn(
                f"According to Btools module's return isn't a variable."\
                f"Thus we assume it's a constant. \n"\
                f"AST type of the output: {type(output_raw_var.val)}")
            raise constants.ExceptionModuleDoesNotReqGrad
        output_target = output_raw_var.val.id
        if not output_raw_var.has_node:
            warnings.warn(
                f"Btools hasn't attached any node to the output."\
                f"Thus we assume it's a constant.")
            raise constants.ExceptionModuleDoesNotReqGrad
        else:
            output_node = dict_nodes[output_target]
            output_info = self.dict_info[output_node.mt]
            if not output_info.requires_grad:
                warnings.warn(
                    "None of the outputs require grad. "\
                    "Thus there is nothing to do.")
                raise constants.ExceptionModuleDoesNotReqGrad
            else:
                return output_target,output_node


    def prepare_cut(self):
        # in case, after simplifications, we will cut / sequentialize
        # we need to protect the separators from "cheap" simplifications
        seps = RK_get_1_separators(self)
        for sep in seps: sep.protected = True

# ==========================

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
