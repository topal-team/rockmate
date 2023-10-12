# ==========================
# ====== D structure =======
# ==========================

from typing import Union
import ast
import warnings
import torch
from src.lowlevel import ast_add_on
from src.lowlevel import jit_patch
from src.lowlevel import constants
from src.lowlevel import preprocess_samples
from src.lowlevel.variable_info import VariableInfo
from src.core import base
from src.core.raw import RawNode,RawVar,RawGraph

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
        self.info : VariableInfo = None

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
        build_variable_info=True,
    ):
        super().__init__("F")
        self.inherit_base_attributes(raw_graph) # => dict_rand / dict_constants
        self.sources_req_grad = False # by default

        raw_nodes = raw_graph.get_toposorted_list_of_non_useless_nodes()
        dict_forward_nodes = dict()
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
                input_info = VariableInfo(
                    dict_inputs.dict[rn.target],
                    data_owner_name = rn.target)
                fn.info = self.dict_info[rn.target] = input_info
                if input_info.requires_grad:
                    self.sources_req_grad = True
            # deps:
            for req_rn in rn.deps:
                req_fn = dict_forward_nodes[req_rn.target]
                fn.deps.add(req_fn)
                req_fn.users.add(fn)
            dict_forward_nodes[rn.target] = fn
            self.nodes.append(fn)

            # info :
            if not build_variable_info:
                fn.info = self.dict_info[fn.target] = VariableInfo()
            elif not rn.is_input:
                # 1) Run node's code to generate the value
                tmp_local = self.generate_deep_tmp_local(rn,our_global)
                rn_code_str = rn.get_code(force_special_kwargs=True)
                try: exec(rn_code_str,our_global,tmp_local)
                except:
                    jit_patch.try_to_fix_dtype_in_returned_ast_code(
                        rn_code_str,our_global,tmp_local
                    )
                fn.info = self.dict_info[rn.target] \
                    = self.detect_inplace_or_view(
                    rn,fn,tmp_local,dict_forward_nodes,dict_inplace_ops)
                del tmp_local
        
        # get the output_node: if not requires_grad => raise Exception
        # which is catch in Rockmate, since it implies no Backward
        output_target = self.get_output_node_and_check_if_requires_grad(
            raw_graph.output_raw_var)
        self.outputs = [output_target]
        self.whole_module_output = output_target

        self.fix_missing_edges_for_inplace_operations(dict_forward_nodes)
        # -> Might change self.outputs
        self.output_nodes = [
            dict_forward_nodes[output_tar] 
            for output_tar in self.outputs]

        self.fix_requires_grad()

        # Protect some node in case we want to cut:
        cutting_points = self.find_cutting_points()
        for cut_point in cutting_points: cut_point.protected = True



    def generate_deep_tmp_local(self,raw_node,our_global):
        # To generate an environment where to run raw_node's code,
        # we generate its dependencies, either using the info 
        # (about shape, dtype etc) we previously collected, 
        # or by running their code in case of view or inplace nodes, 
        # in which case we first (i) generate their dependencies, 
        # using previously collected info; and (ii) its random dependencies.
        tmp_local = dict()
        done = set()
        ready = set()
        todo = list(raw_node.deps)
        while todo != []:
            req_rn = todo[-1]
            req_target = req_rn.target
            if req_target in done:
                todo.pop()
            else:
                req_rn_info = self.dict_info[req_target]
                if (req_rn_info.is_inplace 
                or  req_rn_info.is_view
                or  req_rn.fct == "getattr"):
                    if req_target in ready:
                        for req_rd in req_rn.deps_rand:
                            if not req_rd in done:
                                code = ast_add_on.make_str_assign(
                                    (req_rd,self.dict_rand[req_rd]))
                                exec(code,our_global,tmp_local)
                                done.add(req_rd)
                        exec(req_rn.get_code(),our_global,tmp_local)
                        done.add(req_target)
                        todo.pop()
                    else:
                        todo.extend(list(req_rn.deps))
                        ready.add(req_target)
                else:
                    req_x = req_rn_info.generate_value(our_global["device"])
                    if isinstance(req_x,torch.Tensor):
                        req_x = req_x.clone()
                    tmp_local[req_target] = req_x
                    done.add(req_target)
                    todo.pop()
        return tmp_local
    

    def detect_inplace_or_view(self,
            current_raw_node,
            current_forward_node,
            tmp_local,
            dict_forward_nodes,
            dict_inplace_ops):
        # - detect inplace operation -
        current_rn_value = tmp_local[current_raw_node.target]
        is_view    = False # by default
        is_inplace = False # by default
        data_parents = set() # variables which have the same data_ptr

        # === FIRST WAY TO RECOGNIZE A VIEW ===
        # -> data_ptr
        if VariableInfo.has_a_data_ptr(current_rn_value):
            current_rn_data_ptr = VariableInfo.get_data_ptr(current_rn_value)
            for o_name,o_value in tmp_local.items():
                if (o_name != current_raw_node.target
                and o_name in self.dict_info
                and VariableInfo.has_a_data_ptr(o_value)
                and VariableInfo.get_data_ptr(o_value) == current_rn_data_ptr):
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
                    req_rn_info = self.dict_info[req_rn.target]
                    if req_rn_info.variable_type is torch.Tensor:
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
            if len(data_direct_parents) == 0:
                raise Exception(
                    f"{current_raw_node.target} is an inplace or "\
                    f"view op, it doesn't share its data with any "\
                    f"of its deps ?! (even deps of deps)")
            data_direct_parent_name = data_direct_parents.pop()
            o_info = self.dict_info[data_direct_parent_name]
            data_owner_name = o_info.data_owner_name
            # -> we must protect the data_owner from cheap simplification
            if is_inplace:
                data_owner = dict_forward_nodes[data_owner_name]
                data_owner.protected = True
            # -> If several inplace operations 
            # -> We must ensure we compute them in the original order
            if is_inplace:
                for other_inplace_op in dict_inplace_ops[data_owner_name]:
                    other_fn = dict_forward_nodes[other_inplace_op]
                    other_fn.users.add(current_forward_node)
                    current_forward_node.deps.add(other_fn)
                dict_inplace_ops[data_owner_name].add(current_forward_node.mt)
                
        else:
            data_owner_name = current_raw_node.target
            data_direct_parent_name = current_raw_node.target
            dict_inplace_ops[current_raw_node.mt] = set()
        current_node_info = VariableInfo(
            current_rn_value,
            is_view    = is_view,
            is_inplace = is_inplace,
            data_owner_name = data_owner_name,
            data_direct_parent_name = data_direct_parent_name)
        # ** Correct req_grad of data_parent **
        if (current_node_info.requires_grad
        and current_raw_node.mt != data_owner_name):
            self.dict_info[data_owner_name].requires_grad = True
        return current_node_info
    

    def get_output_node_and_check_if_requires_grad(
            self,output_raw_var : RawVar):
        if not isinstance(output_raw_var.value_ast,ast.Name):
            warnings.warn( # TO CHANGE COMMENTS 
                f"According to Btools module's return isn't a variable."\
                f"Thus we assume it's a constant. \n"\
                f"AST type of the output: {type(output_raw_var.val)}")
            raise constants.ExceptionModuleDoesNotReqGrad
        output_target = output_raw_var.value_ast.id
        if not output_raw_var.has_node:
            warnings.warn(
                f"Btools hasn't attached any node to the output."\
                f"Thus we assume it's a constant.")
            raise constants.ExceptionModuleDoesNotReqGrad
        else:
            output_info = self.dict_info[output_target]
            if not output_info.requires_grad:
                warnings.warn(
                    "None of the outputs require grad. "\
                    "Thus there is nothing to do.")
                raise constants.ExceptionModuleDoesNotReqGrad
            else:
                return output_target


    def fix_missing_edges_for_inplace_operations(self,dict_forward_nodes):
        # example: a = f(x) ; b = inplace(a) ; c = g(a)
        # by default: b doesn't have users, and there is a 
        # direct edge from a to c, skipping b. We fix this.
        fn : ForwardNode
        for fn_index,fn in enumerate(self.nodes):
            if (len(fn.users)==0 
            and fn.main_target != self.whole_module_output):
                # no user and not output => inplace or view
                assert(fn.info.is_view or fn.info.is_inplace)

                # 1) In case fn is a view/inplace over self.whole_module_output
                # we might have to change self.outputs:
                # example: a=f(x) ; b1=view(a) ; b2=inplace(b1) ; c1=inplace(a)
                # then outputs=[a] => outputs=[b1,c1] => outputs=[b2,c1]
                if fn.info.data_owner_name is self.whole_module_output:
                    if fn.info.data_direct_parent_name in self.outputs:
                        self.outputs.remove(fn.info.data_direct_parent_name)
                    self.outputs.append(fn.main_target)
                    # => in the example:
                    # - "b1" replace "a"
                    # - "c1" is added to the list, but "a" was already discarded
                    # - "b2" replace "b1"

                # 2) Add some edges, in the first example a->b->c instead of a->c
                # for this we rely on the topo-order, since we respect the original
                # order in which operations were done, if the inplace operations "b"
                # took place before "c" then it will appear before in the topo-order
                # in which case we need to add an edge "b"->"c"
                data_owner_node = dict_forward_nodes[fn.info.data_owner_name]
                for user_fn in data_owner_node.users:
                    index_user = self.nodes.index(user_fn)
                    if index_user > fn_index:
                        user_fn.deps.add(fn)
                        fn.users.add(user_fn)


    def fix_requires_grad(self):
        # fix 1) 
        # if data_owner requires_grad => inplace/view of it should too
        fn : ForwardNode
        for fn in self.nodes:
            if self.dict_info[fn.info.data_owner_name].requires_grad:
                fn.info.requires_grad = True
        # fix 2) 
        # If none of users req_grad (even after fix 1) => useless to req_grad 
        for fn in self.nodes[::-1]:
            if (fn.main_target not in self.outputs
            and fn.info.requires_grad
            and not any(user_fn.info.requires_grad 
                        for user_fn in fn.users)):
                fn.info.requires_grad = False

    def __str__(self):
        return f"RawGraph with {len(self.nodes)} nodes."
    
    def render(self,
            name=None,
            view=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None
        ):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        for fn in self.nodes:
            if fn.is_input: color = "blue"
            elif fn.target in self.outputs: color = "red"
            else: color = None
            dot.node(fn.target,fn.get_code(),color=color)
        for fn in self.nodes:
            for req_fn in fn.deps:
                dot.edge(req_fn.target,fn.target)
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )

    def print_forward_code(self):
        print("def main({}):".format(','.join(self.inputs)))
        for fn in self.nodes:
            if not fn.is_input: print(f"\t{fn.get_code()}")
        print("\treturn {}".format(','.join(self.outputs)))

    def print_all_nodes(self,print_ast_not_str=True):
        for fn in self.nodes:
            if print_ast_not_str:
                print(ast.dump(ast_add_on.make_ast_assign(
                    (fn.target,fn.code_ast)),indent=4))
            else:
                print(f"({fn.target}) : [{fn.fct}] :\n{fn.get_code()}")
            print(fn.info)
        print("DICT RANDOM OPERATIONS :\n",self.dict_rand)

    def build_equivalent_torch_nn_module(
            self,model : torch.nn.Module,device):
        forward_graph = self
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameters = model.parameters
            def forward(self,*args,**kwargs):
                # 1) Prepare the environnement of exec
                dict_inputs = preprocess_samples.DictInputs(model,args,kwargs)
                tmp_local = dict_inputs.dict
                our_global = globals().copy()
                our_global["self"] = model
                our_global["device"] = device
                our_global.update(forward_graph.dict_constants)
                # 2) exec each node one by one
                fn : ForwardNode
                for fn in forward_graph.nodes:
                    for req_random in fn.deps_rand:
                        if req_random not in tmp_local:
                            code = ast_add_on.make_str_assign(
                                (req_random,forward_graph.dict_rand[req_random]))
                            exec(code,our_global,tmp_local)
                    if not fn.is_input:
                        exec(
                            fn.get_code(force_special_kwargs=True),
                            our_global,tmp_local
                        )
                # 3) return
                output_targets = forward_graph.outputs
                if len(output_targets)==1:
                    return tmp_local[output_targets[0]]
                else:
                    return tuple(tmp_local[out] for out in output_targets)
        return Module()
                