# ==========================
# ====== D structure =======
# ==========================

import ast
import warnings
import torch

from rkgb.lowlevel import ast_add_on
from rkgb.lowlevel import jit_patch
from rkgb.lowlevel import constants
from rkgb.lowlevel import preprocess_samples
from rkgb.lowlevel.execution_environments import EnvironmentGenerator
from rkgb.lowlevel.variable_info import VariableInfo
from rkgb.core import base
from rkgb.core.raw import RawNode,RawGraph

class ExceptionViewOverParameter(Exception):
    def __init__(self,view_value):
        super().__init__("")
        self.view_value = view_value

# **********
# * ForwardNode *
# **********

class ForwardNode(base.Node):
    def __init__(self,
            target=base.Node.no_target_string,
            code_ast=None,
            fct="",
            is_rand=False,
            required_random_tensors=None,
            forward_graph=None):
        """ attributes :
        .target    : str  : the name of the only var defined in the node
        .code_ast  : AST  : right part of the assigning code
        .fct       : str  : the function used in .code_ast
        .is_input  : bool : inputs are represented by nodes wth dummy code
        .is_rand   : bool : whether .fct involves randomness
        .deps      : ForwardNode set : required nodes to run .code_ast
        .required_random_tensors : str set : e.g. r=randn(); y=add(x,r)
        .users     : ForwardNode set : reciprocal of .deps
        """
        super().__init__(target,
            parent_structure_with_id_generator=forward_graph)
        if code_ast is None:
            code_ast = ast_add_on.make_ast_constant("/!\\ not defined /!\\")
        self.code_ast = code_ast
        self.fct = fct
        self.is_input = False
        self.is_rand = is_rand
        self.deps = set()
        self.users = set()
        self.required_random_tensors = required_random_tensors if required_random_tensors else set()
        self.required_parameter_nodes = set()
        self.info : VariableInfo = None


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
    node_class = ForwardNode
    def __init__(self,
        raw_graph : RawGraph,
        original_mod : torch.nn.Module,
        example_inputs : preprocess_samples.ExampleInputs,
        current_device,
        inspection_device,
        build_variable_info=True,
    ):
        super().__init__()
        self.inherit_base_attributes(raw_graph)
        # => dict_rand / dict_constants / output_targets 
        self.sources_req_grad = False # by default
        dict_forward_nodes = dict()
        our_global = EnvironmentGenerator.generate_global_env(
            self,current_device,inspection_device,original_mod)

        # Parameter nodes
        dict_param_str_to_node = dict()
        for rn in raw_graph.nodes:
            for param_str in rn.required_parameters: # e.g. self.layer[0].weight
                if param_str not in dict_param_str_to_node:
                    param_node = base.ParameterNode(param_str,
                        parent_structure_with_id_generator=self)
                    dict_param_str_to_node[param_str] = param_node
                    param_name = param_node.param_name # e.g. layer.0.weight
                    try: 
                        param_value = original_mod.get_parameter(param_name)
                        param_node.is_buffer = False
                    except:
                        param_value = original_mod.get_buffer(param_name)
                        param_node.is_buffer = True
                    param_node.info = param_info = VariableInfo(param_value)
                    param_node.requires_grad = param_info.requires_grad
                    param_node.mem = param_info.memsize
        self.parameter_nodes = dict_param_str_to_node.values()
        # -> to recognize views over parameters

        # === PART 1: Translate each node one by one following the topo-order ===
        rn : RawNode
        all_f_nodes_except_inputs = []
        for rn in raw_graph.nodes:
            fn = ForwardNode(rn.target,rn.code_ast,rn.fct,
                is_rand=rn.is_rand,
                required_random_tensors=set(rn.required_random_tensors),
                forward_graph=self)
            # inputs:
            if rn.is_input:
                fn.is_input = True
                input_info = VariableInfo(
                    example_inputs.dict[rn.target],
                    data_owner_name=rn.target)
                fn.info = self.dict_info[rn.target] = input_info
                if input_info.requires_grad:
                    self.sources_req_grad = True
                dict_forward_nodes[rn.target] = fn
                self.nodes.append(fn)
                continue
            else:
                all_f_nodes_except_inputs.append(fn)
            # deps:
            for req_rn in rn.deps:
                req_fn = dict_forward_nodes[req_rn.target]
                fn.deps.add(req_fn)
                req_fn.users.add(fn)
            dict_forward_nodes[rn.target] = fn
            # required parameters:
            fn.required_parameter_nodes = set(
                dict_param_str_to_node[param_str]
                for param_str in rn.required_parameters
            )
            for required_param_node in fn.required_parameter_nodes:
                required_param_node.users.add(fn)

        # all_f_nodes_except_inputs != self.nodes
        # as we will remove the views over parameters
                
        # == PART 2: Build fn's VariableInfo ==
        fn : ForwardNode
        for fn in all_f_nodes_except_inputs:
            # info :
            if not build_variable_info:
                fn.info = self.dict_info[fn.target] = VariableInfo()
            # 1) Generate local environment
            tmp_local = EnvironmentGenerator.generate_local_env(
                fn,self,our_global,original_mod,
                current_device,inspection_device)
            
            # 2) Execute fn's code
            fn_code_str = fn.get_code(force_special_kwargs=True)
            try: exec(fn_code_str,our_global,tmp_local)
            except:
                if self.tracer_used == "jit":
                    jit_patch.try_to_fix_dtype_in_returned_ast_code(
                        fn_code_str,our_global,tmp_local
                    )
                else:
                    raise Exception(
                        f"Sorry, we fail to execute the code we got from "\
                        f"the tracer ({self.tracer_used}):\n{fn_code_str}."
                    )

            # 3) Build the VariableInfo
            try:
                fn.info = self.dict_info[fn.target] \
                    = self.detect_inplace_or_view(
                    fn,tmp_local,
                    dict_forward_nodes)
                self.nodes.append(fn)

            # 4) Special case: view over a parameter
            except ExceptionViewOverParameter as exception_object:
                assert(fn.required_parameter_nodes != set())
                parent_param_node = fn.required_parameter_nodes.pop()
                # TO IMPROVE: in case of torch.expand_as => several
                # or torch.add(param1,param2) => should merge the param_nodes
                parent_param_node.view_targets.append(fn.target)
                parent_param_node.view_code.append((fn.target,fn.code_ast))
                # Unplug pn = view over a param:
                for user_fn in fn.users:
                    user_fn.deps.remove(fn)
                    user_fn.deps.update(fn.deps)
                    user_fn.required_parameter_nodes.add(parent_param_node)
                    parent_param_node.users.add(user_fn)
                for req_fn in fn.deps:
                    req_fn.users.remove(fn)
                    req_fn.users.update(fn.users)
                self.dict_info[fn.target] = VariableInfo(exception_object.view_value)
                # => In partitioned.py we need VariableInfo of all targets
            del tmp_local

        # == PART 3: finish ==
        self.fix_missing_edges_for_inplace_operations(dict_forward_nodes)
        # -> Might change self.output_targets (previously inherited)
        self.output_nodes = [
            dict_forward_nodes[output_tar] 
            for output_tar in self.output_targets]
        self.nodes = self.get_sorted_nodes_by_following_deps_relation()
        self.check_if_output_requires_grad()
        self.fix_requires_grad()


    def detect_inplace_or_view(self,
            current_forward_node,
            tmp_local,
            dict_forward_nodes):
        current_target = current_forward_node.target
        current_rn_value = tmp_local[current_target]
        all_parameters_data_ptr = [
            VariableInfo.get_data_ptr(param_value) 
            for param_value in tmp_local["all_parameters_values"]]
        is_view    = False # by default
        is_inplace = False
        data_parents = set() # variables which have the same data_ptr

        # === FIRST WAY TO RECOGNIZE A VIEW ===
        # -> data_ptr
        if (VariableInfo.has_a_data_ptr(current_rn_value)
        and not (current_forward_node.fct is
                constants.constructor_function_string)): # TO TEST
            current_rn_data_ptr = VariableInfo.get_data_ptr(current_rn_value)
            if current_rn_data_ptr in all_parameters_data_ptr:
                raise ExceptionViewOverParameter(current_rn_value)
            else:
                for o_name,o_value in tmp_local.items():
                    if (o_name != current_target
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
            main_fct = current_forward_node.fct.split(".")[0]
            if (main_fct in constants.list_view_functions
            or main_fct in constants.list_inplace_functions):
                data_parents = set()
                for req_rn in current_forward_node.deps:
                    req_rn_info = self.dict_info[req_rn.target]
                    if req_rn_info.variable_type is torch.Tensor:
                        data_parents.add(req_rn.mt)
                if data_parents != set():
                    if current_forward_node.fct in constants.list_inplace_functions:
                        is_inplace = True
                    else:
                        is_view = True

        # === register ===
        if is_inplace or is_view:
            current_rn_deps_names = set(
                req_rn.target for req_rn in current_forward_node.deps)
            data_direct_parents = current_rn_deps_names.intersection(data_parents)
            if len(data_direct_parents) == 0:
                for req_rn in current_forward_node.deps:
                    for req_req_rn in req_rn.deps:
                        req_req_name = req_req_rn.target
                        if req_req_name in data_parents:
                            data_direct_parents.add(req_req_name)
            if len(data_direct_parents) == 0:
                raise Exception(
                    f"{current_target} is an inplace or "\
                    f"view op, it doesn't share its data with any "\
                    f"of its deps ?! (even deps of deps)")
            data_direct_parent_name = data_direct_parents.pop()
            direct_parent_info = self.dict_info[data_direct_parent_name]
            data_owner_name = direct_parent_info.data_owner_name
            owner_info = self.dict_info[data_owner_name]

            if is_inplace:
                # If several inplace operations:
                # We must ensure we compute them in the original order
                # so we look for all the already registered inplace 
                # operations over the same data_owner_name, and we 
                # put the new one as a user of the others
                for other_inplace_op in owner_info.inplace_targets:
                    other_fn = dict_forward_nodes[other_inplace_op]
                    other_fn.users.add(current_forward_node)
                    current_forward_node.deps.add(other_fn)
                owner_info.inplace_targets.add(current_target)
            elif is_view:
                owner_info.view_targets.add(current_target)
        else:
            data_owner_name = current_target
            data_direct_parent_name = current_target

        current_node_info = VariableInfo(
            current_rn_value,
            is_view    = is_view,
            is_inplace = is_inplace,
            data_owner_name = data_owner_name,
            data_direct_parent_name = data_direct_parent_name)
        # Correct req_grad of data_parent: 
        # if current req grad, then its data_parent too
        if (current_node_info.requires_grad
        and current_target != data_owner_name):
            self.dict_info[data_owner_name].requires_grad = True
        return current_node_info


    def check_if_output_requires_grad(self):
        if not any(
            output_node.info.requires_grad
            for output_node in self.output_nodes
        ):
            warnings.warn(
                "None of the outputs require grad. "\
                "Thus there is nothing to do.")
            raise constants.ExceptionModuleDoesNotReqGrad


    def fix_missing_edges_for_inplace_operations(self,dict_forward_nodes):
        # example: a = f(x) ; b = inplace(a) ; c = g(a)
        # by default: b doesn't have users, and there is a 
        # direct edge from a to c, skipping b. We fix this.
        fn : ForwardNode
        for fn_index,fn in enumerate(self.nodes):
            if (len(fn.users)==0
            and fn.info.is_inplace # or is_view ? TO CHECK
            and fn.main_target not in self.output_targets):
                # no user and not output => inplace or view
                assert(fn.info.is_view or fn.info.is_inplace)

                # 1) In case fn is inplace over an output
                # we might have to change self.output_targets:
                # example: a=f(x) ; b1=view(a) ; b2=inplace(b1) ; c1=inplace(a)
                # then outputs=[a] => outputs=[c1] => outputs=[b2,c1]
                data_owner_name = fn.info.data_owner_name
                if (fn.info.is_inplace 
                and data_owner_name in self.original_mod_output_targets):
                    if data_owner_name in self.output_targets:
                        self.output_targets.remove(data_owner_name)
                    self.output_targets.append(fn.main_target)
                    # => in the example:
                    # - "c1" replace "a"
                    # - "b2" is added to the list, but "a" is already discarded

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
            if (fn.main_target not in self.output_targets
            and fn.info.requires_grad
            and not any(user_fn.info.requires_grad 
                        for user_fn in fn.users)):
                fn.info.requires_grad = False

    def __str__(self):
        return f"Forward Graph with {len(self.nodes)} nodes."
    
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            include_parameter_nodes=False,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None
        ):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        if include_parameter_nodes:
            for param_node in self.parameter_nodes:
                param_node : base.ParameterNode
                dot.node(
                    param_node.param_str,
                    param_node.param_str
                    if param_node.view_targets == []
                    else f"{param_node.param_str}\n{param_node.get_code()}",
                    style = "dashed")
        for fn in self.nodes:
            fn : ForwardNode
            if fn.is_input: color = "blue"
            elif fn.target in self.output_targets: color = "red"
            else: color = None
            if only_function_name: label = fn.fct
            else: label = fn.get_code()
            dot.node(fn.target,label,color=color)
            for req_fn in fn.deps:
                dot.edge(req_fn.target,fn.target)
            if include_parameter_nodes:
                for req_param in fn.required_parameter_nodes:
                    dot.edge(req_param.param_str,fn.target,style="dashed")
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )

    def print_forward_code(self):
        print("def main({}):".format(','.join(self.input_targets)))
        for fn in self.nodes:
            if not fn.is_input: print(f"\t{fn.get_code()}")
        print("\treturn {}".format(','.join(self.output_targets)))

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
            self,original_mod : torch.nn.Module,device):
        forward_graph = self
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Inherits all attributes from original_mod:
                for k, v in original_mod.__dict__.items():
                    if (
                        not "forward" in k
                        and not "backward" in k
                        and k not in ["training"]
                    ):
                        self.__dict__[k] = v
            def forward(self,*args,**kwargs):
                # 1) Prepare the environnement of exec
                our_global = forward_graph.make_simple_copy_of_globals(self,device)
                example_inputs = preprocess_samples.ExampleInputs(original_mod,args,kwargs)
                tmp_local = example_inputs.dict
                # 2) exec each node one by one
                fn : ForwardNode
                param_nodes_already_executed = set()
                for fn in forward_graph.nodes:
                    for req_random in fn.required_random_tensors:
                        if req_random not in tmp_local:
                            code = ast_add_on.make_str_assign(
                                (req_random,forward_graph.dict_rand[req_random]))
                            exec(code,our_global,tmp_local)
                    for req_param in fn.required_parameter_nodes:
                        if req_param not in param_nodes_already_executed:
                            param_nodes_already_executed.add(req_param)
                            exec(req_param.get_code(),our_global,tmp_local)
                    if not fn.is_input:
                        exec(
                            fn.get_code(force_special_kwargs=True),
                            our_global,tmp_local
                        )
                # 3) return
                output_targets = forward_graph.output_targets
                if len(output_targets)==1:
                    return tmp_local[output_targets[0]]
                else:
                    return tuple(tmp_local[out] for out in output_targets)
        return Module()
                