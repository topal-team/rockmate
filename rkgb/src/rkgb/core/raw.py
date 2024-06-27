# ==========================
# ====== B structure =======
# ==========================

# =====================================
# = EXTRACT TRACE CODE TO BUILD THE   =
# = GROUNDWORK OF BASIC FORWARD GRAPH =
# =====================================

# -> use torch.jit_trace.code
# -> use ast.parse
# -> extract the info from ast result using cross recursive functions
#    like a Functional programming language

# Do some simplifications :
# -> Remove some useless getattr
# -> Inline ALL operations
# -> Remove TorchScript's operations (e.g. ops.prim.NumToTensor)

# Each node consist of one assignment:
#  one target obtained with a primitive operation
#  .code attributes are AST objects

from typing import Union
import warnings
import inspect
import ast
import torch
from torch import Tensor

from rkgb.lowlevel import ast_add_on
from rkgb.lowlevel import constants
from rkgb.lowlevel import preprocess_samples
from rkgb.lowlevel import jit_patch
from rkgb.core import base


class RawNode(base.Node):
    def __init__(self, 
            target=base.Node.no_target_string,
            code_ast=None, 
            fct="", 
            is_input=False,
            raw_parser=None,
        ):
        """ attributes :
        .target   : str  : the name of the only var defined in the node
        .code_ast : AST  : right part of the assigning code
        .fct      : str  : the function used in .code_ast
        .is_input : bool : input vars are represented by nodes with dummy code
        .is_rand  : bool : whether .fct involves randomness
        .deps      : RawNode set : required nodes to run .code_ast
        .required_random_tensors : str set : required random fresh tensor;
        e.g. r = torch.randn(...); y = torch.add(x,r); 
        => RawNode[y].required_random_tensors = {"r"}
        """
        super().__init__(target,
            parent_structure_with_id_generator=raw_parser)
        if code_ast is None:
            self.code_ast = ast_add_on.make_ast_constant("/!\\ NO CODE /!\\")
        else:
            self.code_ast = code_ast
        self.fct = fct
        self.is_input = is_input
        self.is_rand = bool(fct in constants.list_random_functions)
        self.required_random_tensors : set[str] = set()
        self.required_parameters : set[str] = set()
        # TO REMOVE
        # self.deps : set[RawNode] = deps if deps is not None else set()
        self.deps = set()
        if raw_parser is not None: raw_parser.all_raw_nodes.append(self)


class RawGraph(base.Graph):
    """ Raw graph 
    => very few attributes = just what we get from jit or Dynamo"""
    node_class = RawNode
    def __init__(self,
            original_mod : torch.nn.Module,
            example_inputs : preprocess_samples.ExampleInputs,
            use_jit_instead_of_dynamo = False,
            jit_impose_device=True,
            dynamo_kwargs={}
        ):
        super().__init__()
        if use_jit_instead_of_dynamo:
            self._init_using_jit(
                original_mod,
                example_inputs,
                jit_impose_device
            )
        else:
            self._init_using_dynamo(
                original_mod,
                example_inputs,
                dynamo_kwargs=dynamo_kwargs
            )
        self.toposort_and_keep_only_useful_nodes()
        self.clear_redundancies_in_self_nodes()
        

    def _init_using_jit(self,
            original_mod : torch.nn.Module,
            example_inputs : preprocess_samples.ExampleInputs,
            impose_device):
        self.tracer_used = "jit"
        # Call TorchScript jit's tracer
        ordered_example_inputs = example_inputs.to_list_args(original_mod)
        with torch.no_grad():
            jit_result = torch.jit.trace_module(
                original_mod,
                {"forward": ordered_example_inputs}, 
                check_trace=False
            )
        # Parse
        parser = RawJitParser(impose_device)
        output_variable : RawJitParserVariable = parser.parse(
            jit_result, "self", "forward", [], is_main=True
        )
        self.original_mod_input_targets = self.input_targets = parser.input_targets
        self.nodes = parser.all_raw_nodes
        self.dict_rand = parser.dict_rand
        self.dict_constants = parser.dict_constants
        # Organize the output: we open the wrapper
        # in case 'output=[a,b]' => output_targets = [a,b]
        output_target,output_node \
            = RawJitParser.get_output_attributes(output_variable)
        if output_node.fct == constants.constructor_function_string:
            self.output_nodes = list(output_node.deps)
            self.output_targets = [out_node.target for out_node in self.output_nodes]
            # To be sure to respect the original order in which outputs are returned:
            self.original_mod_output_targets = []
            for output_ast in output_node.code_ast.elts:
                assert(isinstance(output_ast,ast.Name))
                # fail => return a constant ? TO IMPROVE
                self.original_mod_output_targets.append(output_ast.id)
        else:
            self.output_nodes = [output_node]
            self.output_targets = [output_target]
            self.original_mod_output_targets = [output_target]

    
    def _init_using_dynamo(self,
            original_mod : torch.nn.Module,
            example_inputs : preprocess_samples.ExampleInputs,
            dynamo_kwargs = {}):
        self.tracer_used = "dynamo"
        # Call Dynamo's export :
        dynamo_result : torch.export.ExportedProgram = torch.export.export(
            original_mod,
            args = example_inputs.args,
            kwargs=example_inputs.kwargs,
            **dynamo_kwargs
            )
        dynamo_graph = dynamo_result.graph
        dynamo_signature = dynamo_result.graph_signature
        whole_code_str = dynamo_graph.python_code("self").src
        whole_code_ast : ast.FunctionDef = ast.parse(whole_code_str).body[0]

        # I generate a parser to ensure the same naming system
        # as we are doing when parsing jit result,
        # e.g. "x" => "__32_x" (for jit)
        # e.g. "arg0_1" => "__32_arg0_1" (for dynamo / here)
        # Moreover the parser give a unique id to each node
        # in a deterministic way.
        parser = RawParser()
        dict_dynamo_name_to_raw_node = dict()

        # I) Process the "args" = which consists of all the inputs, 
        # parameters and buffers. Buffers are variables stored in
        # `self` that aren't parameters; e.g. BatchNorm's running_var
        dynamo_all_args = whole_code_ast.args.args
        dict_dynamo_name_to_correct_ast = dict()
        # e.g: "arg15_1" to AST("input_ids"), 
        # or "arg10_1" to AST("self.h[0]").

        # Useful dict to find back the parameters and buffers:
        dict_param_value_to_name = dict(
            (value,name) for (name,value) 
            in original_mod.named_parameters()) 
        dict_buffer_value_to_name = dict(
            (value,name) for (name,value)
            in original_mod.named_buffers())

        state_dict = {}
        for n,p in dynamo_result.named_parameters():
            state_dict[n] = p
        for n,b in dynamo_result.named_buffers():
            state_dict[n] = b

        for arg in dynamo_all_args:
            dynamo_arg_name = arg.arg # e.g. "arg15_1"
            # 1) Parameters:
            if dynamo_arg_name in dynamo_signature.inputs_to_parameters:
                dynamo_param_name = dynamo_signature.inputs_to_parameters[dynamo_arg_name]
                # e.g. L__self___embeddings_word_embeddings.weight
                # It's not as simple as changing the "_" by ".",
                # here we look for "self.embeddings.word_embeddings.weight"
                param_value = state_dict[dynamo_param_name]
                param_real_name = dict_param_value_to_name[param_value]
                dict_dynamo_name_to_correct_ast[dynamo_arg_name] \
                    = ast_add_on.make_ast_attribute_from_list(
                        ast.Name("self"),param_real_name.split(".")
                    )
            # 2) Buffers
            elif dynamo_arg_name in dynamo_signature.inputs_to_buffers:
                dynamo_buffer_name = dynamo_signature.inputs_to_buffers[dynamo_arg_name]
                buffer_value = state_dict[dynamo_buffer_name]
                buffer_real_name = dict_buffer_value_to_name[buffer_value]
                dict_dynamo_name_to_correct_ast[dynamo_arg_name] \
                    = ast_add_on.make_ast_attribute_from_list(
                        ast.Name("self"),buffer_real_name.split(".")
                    )
        # 3) Inputs
        dict_inputs = example_inputs.dict
        self.original_mod_input_targets = self.input_targets = list(
            arg for arg in inspect.signature(original_mod.forward).parameters
            if (arg in dict_inputs 
            and isinstance(dict_inputs[arg],torch.Tensor))
        )
        assert(len(dynamo_signature.user_inputs)==len(self.input_targets))
        for dynamo_input_name, real_input_name in zip(
                dynamo_signature.user_inputs,
                self.input_targets):
            dict_dynamo_name_to_correct_ast[dynamo_input_name] \
                = ast.Name(real_input_name)
            dict_dynamo_name_to_raw_node[dynamo_input_name] \
                = RawNode(
                    target=real_input_name,
                    code_ast=ast_add_on.make_ast_constant("INPUT"),
                    fct="INPUT",
                    is_input=True,
                    raw_parser=parser
                )

        # II) Process all the assignments
        assignment_codes = [
            code for code in whole_code_ast.body
            if (isinstance(code,ast.Assign)
            and not(ast_add_on.is_constant(code.value) and code.value.value is None))
        ]
        dict_code_target_to_code = dict()
        for code in assignment_codes:
            # So we can remove 'get_attr'
            assert(len(code.targets)==1)
            dict_code_target_to_code[code.targets[0].id] = code
        self.dict_constants = dict()

        dynamo_all_nodes = dynamo_graph.nodes
        dynamo_assignment_nodes = []
        dynamo_input_nodes = {}
        for dynamo_node in dynamo_all_nodes:
            if dynamo_node.target in dynamo_signature.user_inputs:
                dynamo_input_nodes[dynamo_node.target] = dynamo_node
            if dynamo_node.op == "call_function":
                dynamo_assignment_nodes.append(dynamo_node)
            elif dynamo_node.op == "get_attr":
                code = dict_code_target_to_code[dynamo_node.name]
                if not "self" in ast_add_on.ast_to_str(code):
                    raise Exception(
                        "Dynamo 'get_attr' node, but not over self ?? "\
                        "Please notify us about this case:\n",
                        ast_add_on.ast_to_str(code))
                assignment_codes.remove(code)
                assert(isinstance(code.value,ast.Attribute))
                attr = code.value.attr
                assert(hasattr(dynamo_result.graph_module,attr))
                cst_name = parser.get_constant_name(dynamo_node.name)
                self.dict_constants[cst_name] = getattr(dynamo_result.graph_module,attr)
                dict_dynamo_name_to_correct_ast[dynamo_node.name] = ast.Name(cst_name)
            elif dynamo_node.op in ["placeholder","output"]:
                continue # Nothing interesting
            else:
                raise Exception(f"Unknown Dynamo Node's operation type {dynamo_node.op}")
            
        assert(len(dynamo_assignment_nodes)==len(assignment_codes))
        self.dynamic_shapes = {}
        for dynamo_input_name, dynamo_input_node in dynamo_input_nodes.items():
            raw_node = dict_dynamo_name_to_raw_node[dynamo_input_name]
            self.dynamic_shapes[raw_node.target] = dynamo_input_node.meta["val"].shape
        set_dynamo_vars_that_are_constants = set() 
        for dynamo_node,node_code in zip(
                dynamo_assignment_nodes,
                assignment_codes):
            target = parser.make_name_unique(dynamo_node.name)
            dict_dynamo_name_to_correct_ast[dynamo_node.name] = ast.Name(target)
            dependency_dynamo_names = [
                req_dynamo_node.name
                for req_dynamo_node in dynamo_node.all_input_nodes
            ]
            code_with_correct_device = ast_add_on.substitute_device_call(
                node_code.value
            )
            substitution_dict = dict()
            for dep_id in dependency_dynamo_names:
                if not dep_id in dict_dynamo_name_to_correct_ast:
                    # To handle stuff like:
                    # For instance (example in Bloom)
                    # `lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_lifted_tensor_constant0)`
                    # where `_lifted_tensor_constant0` is undefined
                    for dynamo_key,cst_value in dynamo_result.tensor_constants.items():
                        if dep_id in dynamo_key:
                            cst_name = parser.get_constant_name(dep_id)
                            self.dict_constants[cst_name] = cst_value
                            dict_dynamo_name_to_correct_ast[dep_id] = ast.Name(cst_name)
                            set_dynamo_vars_that_are_constants.add(dep_id)
                            break
                    if not dep_id in dict_dynamo_name_to_correct_ast:
                        raise Exception(
                            f"{dep_id} is called in the line:\n"\
                            f"{ast_add_on.ast_to_str(node_code)}\n"\
                            f"but we don't know where it's defined.")
                substitution_dict[dep_id] = dict_dynamo_name_to_correct_ast[dep_id]
            code_with_correct_names = ast_add_on.substitute_with_dict(
                code_with_correct_device,
                {
                    dep_id : dict_dynamo_name_to_correct_ast[dep_id]
                    for dep_id in dependency_dynamo_names
                }
            ) # e.g. Substitute "arg2_0" by AST("__32_arg2_0) or AST("self.h[0]").
            raw_node = RawNode(
                target=target,
                code_ast=code_with_correct_names,
                fct=dynamo_node.target.__name__,
                raw_parser=parser)
            dict_dynamo_name_to_raw_node[dynamo_node.name] = raw_node
            val = dynamo_node.meta["val"]
            self.dynamic_shapes[raw_node.target] = val.shape if hasattr(val, "shape") else None
            # Deps :
            raw_node.deps = set(
                dict_dynamo_name_to_raw_node[dep_id]
                for dep_id in dependency_dynamo_names
                if dep_id in dict_dynamo_name_to_raw_node
            )
            # Required parameters :
            raw_node.required_parameters = set(
                ast_add_on.ast_to_str(
                    dict_dynamo_name_to_correct_ast[param_id])
                for param_id in dependency_dynamo_names
                if (param_id not in dict_code_target_to_code
                and param_id not in dict_dynamo_name_to_raw_node
                and param_id not in set_dynamo_vars_that_are_constants)
            )
        self.nodes = parser.all_raw_nodes

        # outputs:
        for output_dynamo_name in dynamo_signature.user_outputs:
            output_raw_node = dict_dynamo_name_to_raw_node[output_dynamo_name]
            self.output_nodes.append(output_raw_node)
            self.output_targets.append(output_raw_node.target)
        self.original_mod_output_targets = list(self.output_targets)
        
        


    def toposort_and_keep_only_useful_nodes(self):
        list_raw_nodes = self.get_sorted_nodes_by_following_deps_relation()
        # Reinsert some nodes:
        # In the toposorting function, we collect nodes by following 
        # the deps relation from the output_nodes all the way to the
        # first nodes. But some nodes corresponding to inplace operations
        # are missed, as they don't have users:
        # e.g. a = f(x) ; i = inplace(a) ; output = g(a)
        # TO IMPROVE / CHANGE ??? I think its kind greedy strategy...
        to_insert_back = [
            rn for rn in self.nodes
            if (rn not in list_raw_nodes
            and len(rn.deps)!=0)]
        to_insert_back.sort(key=base.Node.get_num)
        to_insert_back = to_insert_back[::-1]
        while to_insert_back != []:
            retry_list = []
            for rn in to_insert_back:
                index_deps = []
                fail = False
                for req_rn in rn.deps:
                    if req_rn not in list_raw_nodes:
                        fail = True
                        break
                    else:
                        index_deps.append(list_raw_nodes.index(req_rn))
                if fail:
                    retry_list.append(rn)
                    continue
                else:
                    max_index = max(index_deps)
                    list_raw_nodes.insert(max_index+1,rn)
            if retry_list == to_insert_back:
                to_insert_back = [] # -> Give up
            else:
                to_insert_back = retry_list
        self.nodes = list_raw_nodes
    

    def clear_redundancies_in_self_nodes(self):
        """
        A lot of very tiny nodes share the exact same code:
        e.g. v1 = [0,1]; v2 = [0,1], x1 = v1[1], x2 = v2[1]
        So we identify similar codes (v2 similar to v1)
        and we substitute v2 by v1 in other nodes' code,
        so x2 = v1[1], so we recognize x2 similar to x1 etc.
        """
        # I) Check if we can replace the constants
        # directly by their value; or at least avoid having 
        # a lot of nodes referring to the same value
        dict_constant_alias = dict()
        dict_constant_value_to_name = dict()
        dict_constant_number_to_name = dict()
        dict_constants_at_the_beginning = dict(self.dict_constants)
        self.dict_constants = dict()
        for cst_name,cst_value in dict_constants_at_the_beginning.items():
            cst_number = int(cst_name.split("_")[2])
            dict_constant_number_to_name[cst_number] = cst_name
            if type(cst_value) in [int,str,bool]:
                dict_constant_alias[cst_name] \
                    = ast_add_on.make_ast_constant(cst_value)
            elif cst_value in dict_constant_value_to_name:
                dict_constant_alias[cst_name] \
                    = ast.Name(dict_constant_value_to_name[cst_value])
            else:
                dict_constant_value_to_name[cst_value] = cst_name
                self.dict_constants[cst_name] = cst_value
        # II) clear redundancies in self.nodes' code
        dict_node_alias = dict()
        # node -> node; e.g. Node(v2) -> Node(v1)
        dict_code_to_node = dict() 
        # string code -> node; e.g. AST([0,1]) -> Node(v1)
        list_raw_nodes_without_redundancies = []
        node : RawNode
        for node in self.nodes:
            # 0) Don't touch to inputs
            if node.is_input:
                list_raw_nodes_without_redundancies.append(node)
                continue
            # 1) Correct code via aliases in deps
            new_deps = set()
            for req_node in node.deps:
                if req_node in dict_node_alias:
                    alias_of_req_node = dict_node_alias[req_node]
                    node.code_ast = ast_add_on.substitute(
                        node.code_ast,
                        req_node.target,
                        ast.Name(alias_of_req_node.target)
                    )
                    new_deps.add(alias_of_req_node)
                else:
                    new_deps.add(req_node)
            node.deps = new_deps
            # 2) Correct the constants
            code_str = node.get_code()
            # Look for "_cst_{number}" 
            for str_after_cst_ in code_str.split("_cst_")[1:]:
                cst_number = int(
                    str_after_cst_[:str_after_cst_.index("_")])
                cst_name = dict_constant_number_to_name[cst_number]
                if cst_name in dict_constant_alias:
                    node.code_ast = ast_add_on.substitute(
                        node.code_ast,
                        cst_name,
                        dict_constant_alias[cst_name]
                    )

            # 3) Check if an other node has the exam same code
            if node in self.output_nodes:
                list_raw_nodes_without_redundancies.append(node)
                continue # don't change the output
            code_str = ast_add_on.ast_to_str(node.code_ast)
            if code_str in dict_code_to_node:
                dict_node_alias[node] = dict_code_to_node[code_str]
            else:
                dict_code_to_node[code_str] = node
                list_raw_nodes_without_redundancies.append(node)
        self.nodes = list_raw_nodes_without_redundancies


    # Overwrite base.Graph methods => used to sort nodes by following 'deps'
    def make_temporary_global_root_node_to_deps_relation(self):
        fresh_root = RawNode()
        fresh_root.deps = set(self.output_nodes)
        return fresh_root
    def remove_temporary_global_root_node(self, fresh_root):
        ()

    def __str__(self):
        return f"Raw Graph with {len(self.nodes)} nodes."
    def render(self,
            name=None,
            view=True,
            only_function_name=False,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None
        ):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        for rn in self.nodes:
            if only_function_name: label = rn.fct
            else: label = rn.get_code()
            dot.node(rn.target,label)
        for rn in self.nodes:
            for req_rn in rn.deps:
                dot.edge(req_rn.target,rn.target)
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )

    def print_all_nodes(self,print_ast_not_str=True):
        for rn in self.nodes:
            if print_ast_not_str:
                print(ast.dump(ast_add_on.make_ast_assign(
                    (rn.target,rn.code_ast)),indent=4))
            else:
                print(f"({rn.target}) : [{rn.fct}] :\n{rn.get_code()}")
        print("DICT RANDOM OPERATIONS :\n",self.dict_rand)


class RawParser():
    def __init__(self):
        self.counter_unique_number = 0
        self.all_raw_nodes = []
        self.node_unique_id_generator = base.Node_unique_id_generator()
    def get_unique_number(self):
        self.counter_unique_number += 1
        return self.counter_unique_number
    def make_name_unique(self,name):
        return f"__{self.get_unique_number()}_{name}"
    def get_fresh_name(self):
        return f"__{self.get_unique_number()}_fresh"
    def get_unique_name(self,name = None):
        if name is None:
            return self.get_fresh_name()
        else:
            return self.make_name_unique(name)
    def get_constant_name(self,s):
        return f"_cst_{self.get_unique_number()}_{s}"
    

class RawJitParserVariable:
    def __init__(
            self,
            value_ast,
            raw_parser,
            node: RawNode = None,
            is_attr_of_self=False,
            real_value_as_an_attr_of_self=None,
        ):
        self.is_attr_of_self = is_attr_of_self
        self.real_value_as_an_attr_of_self = real_value_as_an_attr_of_self
        self._value_ast = value_ast
        self.has_node = False  # by default
        self.is_rand = False  # by default
        if node: 
            # compared to before, we don't automatically simplify 
            # when deps=empty. We do it only in case it's a random op
            if node.is_rand and node.deps == set() and not node.is_input:
                raw_parser.dict_rand[node.target] = node.code_ast
                self.is_rand = True
            else:
                self.has_node = True
                self.node = node

    def use_value_ast(self, calling_node : RawNode):
        """ Instead of self._value_ast, you must use this
        Take care of the "deps" relation
        """
        if self.is_attr_of_self:
            calling_node.required_parameters.add(self._value_ast.value.id)
        elif self.has_node:
            calling_node.deps.add(self.node)
        elif self.is_rand:
            calling_node.required_random_tensors.add(self._value_ast.id)
        return self._value_ast

    def inherits_self_attr(self, parent, list_attributes):  
        # for a getattr AND is_attr_of_self
        if parent.has_node:
            self.has_node = True
            self.node = parent.node
        obj = parent.real_value_as_an_attr_of_self
        for at in list_attributes:
            obj = getattr(obj,at)
        self.real_value_as_an_attr_of_self = obj


# Draft of the general comment about how the parser works:
# if the expr is simple (e.g. constant or self's attr)
# -> RawJitParserVariable.has_node == False
# otherwise, a node (= a piece of code) is created.
# The optional parameter  "target" imposes the name of the var created
class RawJitParser(RawParser):
    def __init__(self,impose_device):
        super().__init__()
        self.impose_device = impose_device
        self.dict_rand = dict()
        self.current_dict_raw_vars = dict()
        self.dict_constants = dict()
        self.current_jit_memory = dict()

    def aux_for_attribute(self, target, parent_raw_var, list_attributes):
        """ Used for:
         - `getattr(a,"b")` via an ast.Call
         - `a.b` via an ast.Attribute
        """
        if parent_raw_var.is_attr_of_self:
            parent_ast = parent_raw_var._value_ast
            new_ast = ast_add_on.make_ast_attribute_from_list(
                parent_ast,list_attributes)
            new_raw_var = RawJitParserVariable(new_ast,raw_parser=self,is_attr_of_self=True)
            new_raw_var.inherits_self_attr(parent_raw_var,list_attributes)
        else:
            new_id = self.get_unique_name(target)
            new_node = RawNode(target=new_id, fct="getattr",raw_parser=self)
            parent_ast = parent_raw_var.use_value_ast(calling_node=new_node)
            new_ast = ast_add_on.make_ast_attribute_from_list(
                parent_ast,list_attributes)
            new_node.code_ast = new_ast
            new_raw_var = RawJitParserVariable(new_ast,raw_parser=self,node=new_node)
        return new_raw_var


    def handle_ast_attribute(self, target : str, expr : ast.Attribute) -> RawJitParserVariable:
        parent_expr,list_attributes = ast_add_on.open_all_nested_attributes(expr)
        parent_raw_var = self.handle_expr(None,parent_expr)
        return self.aux_for_attribute(target,parent_raw_var,list_attributes)


    def aux_handle_ast_call_getattr(self,target,call_args):
        assert len(call_args) == 2 # getattr(arg1,arg2)
        assert ast_add_on.is_constant(call_args[1])
        # If fail: something like `getattr(a,"foo"+"bar")`
        # It's a nested operation, so I assume jit inlined it
        # If new tracer, then use handle_expr over call_args[1]
        parent_raw_var = self.handle_expr(None,call_args[0])
        attribute = call_args[1].value
        return self.aux_for_attribute(target,parent_raw_var,[attribute])
    
    def aux_handle_ast_call_sub_module(
            self,
            called_raw_var,
            rest_of_func_name,
            call_arg_raw_vars):
        jit_result_sub_module = called_raw_var.real_value_as_an_attr_of_self
        for attribute in rest_of_func_name[:-1]:
            jit_result_sub_module = getattr(jit_result_sub_module, attribute)
        sub_module_name = ast_add_on.ast_to_str(called_raw_var._value_ast)
        method_name = rest_of_func_name[-1]
        save_current_dict_raw_vars = self.current_dict_raw_vars
        save_current_jit_memory = self.current_jit_memory
        sub_module_output_raw_var = self.parse(
            jit_result_sub_module, sub_module_name, 
            method_name, call_arg_raw_vars
        )
        self.current_dict_raw_vars = save_current_dict_raw_vars
        self.current_jit_memory = save_current_jit_memory
        # Note: to parse sub_module, we give raw_vars of its inputs,
        # and it returns the raw_var which define its output
        # -> ie it creates the raw_var of the sub_module call result!
        # Which is exactly the objective of `RawJitParser.handle_ast` functions
        return sub_module_output_raw_var

    def handle_ast_call(self,target : str, expr : ast.Call) -> RawJitParserVariable:
        call_args = list(expr.args)
        ast_first_term,rest_of_func_name \
            = ast_add_on.open_all_nested_attributes(expr.func)
        # e.g. `torch.nn.functional(a,b)` :
        # -> call_args = [ast.Name("a"),ast.Name("b")]
        # -> ast_first_term = ast.Name("torch")
        # -> rest_of_func_name = ["nn","functional"]
        assert isinstance(ast_first_term,ast.Name)
        # If fail:
        # It means it's an ast.Call (more precisely a method call),
        # over an expr, something like `([1]+[2]).reverse()`
        # Which is clearly a nested operation, so I assume jit inline this.
        # In case one day we move to a need tracer which doesn't
        # inline, use 'self.handle_expr' on ast_first_part
        first_term_of_func_name = ast_first_term.id

        # getattr:
        if (first_term_of_func_name == "getattr" and not rest_of_func_name):
            return self.aux_handle_ast_call_getattr(target,call_args)

        # TorchScript's functions:
        # -> must be removed because some refer to TorchScript global var
        # -> their purpose is to make the type explicit, e.g. int_to_tensor
        elif first_term_of_func_name == "ops":
            assert len(call_args)==1
            # If fail:
            # All TorchScript's ops functions I found had 1 arg, and none
            # were useful; if you found one which has 2 args maybe it's
            # useful, to fix you just need to add "and len(call_args)==1"
            # in the if condition, which will avoid your new case and
            # letting the final else handle it.
            return self.handle_expr(target,call_args[0])
        elif first_term_of_func_name == "int":
            return self.handle_expr(target,call_args[0])
        elif first_term_of_func_name == "annotate":
            assert len(call_args) == 2
            # If fail: then I miss understood "annotate" functions
            # check comment 6 lines above.
            return self.handle_expr(target,call_args[1])
        elif (self.impose_device 
        and first_term_of_func_name == "torch"
        and rest_of_func_name[0] == "device"):
            return RawJitParserVariable(value_ast=ast.Name("device"),raw_parser=self)

        else:
            call_arg_raw_vars = [self.handle_expr(None,arg) for arg in call_args]
            # Method => sub module:
            if first_term_of_func_name in self.current_dict_raw_vars: # => a method
                called_raw_var = self.current_dict_raw_vars[first_term_of_func_name]
                assert called_raw_var.is_attr_of_self
                # If fail:
                # I assumed all method call are call to sub modules of `self`:
                # e.g. self.w.wpe(...)
                # Since jit seems to always avoid other method calls.
                # You can always use the method directly via the class:
                # e.g. instead of "x.size()" use "torch.Tensor.size(x)"
                # So even though we change the tracer, you can use such trick.
                return self.aux_handle_ast_call_sub_module(
                    called_raw_var,rest_of_func_name,call_arg_raw_vars
                )
            
            else: # Else = Call to a primitive function
                # Function name, fix jit error : `torch` -> `torch.ops.aten`
                if (first_term_of_func_name == "torch" 
                    and len(rest_of_func_name) == 1):
                    short_fct_name = rest_of_func_name[0]
                    fct_name = "torch.ops.aten."+short_fct_name
                else:
                    short_fct_name = fct_name \
                        = ".".join([first_term_of_func_name]+rest_of_func_name)
                target = self.get_unique_name(target)
                new_node = RawNode(
                    target=target,fct=short_fct_name,raw_parser=self)
                args_ast = [
                    arg_raw_var.use_value_ast(calling_node=new_node) 
                    for arg_raw_var in call_arg_raw_vars
                ]
                kwds_ast = []
                for kw in expr.keywords:
                    if self.impose_device and kw.arg == "device":
                        kwds_ast.append(
                            ast.keyword("device", ast.Name("device"))
                        )
                    elif kw.arg == "dtype" and ast_add_on.is_constant(kw.value):
                        kwds_ast.append(
                            ast.keyword(
                                "dtype",
                                ast_add_on.make_ast_constant(
                                    jit_patch.get_torchscript_dtype(kw.value.value)
                                ))
                        )
                    elif not (
                        (
                            (kw.arg == "dtype" or kw.arg == "layout")
                            and ast_add_on.is_constant(kw.value)
                            and isinstance(kw.value.value, int)  # WTF
                        )
                        or (kw.arg == "layout" and kw.value.value is None)
                    ):
                        kwds_ast.append(
                            ast.keyword(
                                kw.arg,
                                (self.handle_expr(None,kw.value))\
                                     .use_value_ast(new_node),
                            )
                        )
                    #else: we remove weird keywords jit adds sometimes
                new_node.code_ast = ast.Call(
                    func=ast.Name(fct_name), args=args_ast, keywords=kwds_ast
                )
                return RawJitParserVariable(ast.Name(target),raw_parser=self,node=new_node)
            

    def handle_ast_tuple_or_list(self,
            target : str, expr : Union[ast.List,ast.Tuple]) -> RawJitParserVariable:
        # Not simplified / inserted here, might change TO CHANGE ?
        # -> because I need to precise the calling_node...
        target = self.get_unique_name(target)
        new_node = RawNode(
            target=target,
            fct=constants.constructor_function_string,
            raw_parser=self)
        args_raw_vars = [self.handle_expr(None,e) for e in expr.elts]
        args_ast = [v.use_value_ast(calling_node=new_node) for v in args_raw_vars]
        if isinstance(expr,ast.List):
            new_node.code_ast = ast.List(args_ast)
        else:
            new_node.code_ast = ast.Tuple(args_ast)
        return RawJitParserVariable(ast.Name(target),raw_parser=self,node=new_node)


    def handle_expr(self, target : str, expr) -> RawJitParserVariable:
        if ast_add_on.is_constant(expr):
            return RawJitParserVariable(expr,raw_parser=self)
        elif isinstance(expr, ast.Name):
            if expr.id in self.current_dict_raw_vars:
                return self.current_dict_raw_vars[expr.id]
            else:
                raise Exception(
                    "Unknown variable encountered while parsing "\
                    "jit result, external/global var ?")
        elif (
            isinstance(expr, ast.Attribute)  # -> special constants
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "CONSTANTS"
        ):
            cst_name = self.get_constant_name(expr.attr)
            self.dict_constants[cst_name] = self.current_jit_memory[expr.attr]
            return RawJitParserVariable(ast.Name(cst_name),raw_parser=self)
        elif isinstance(expr, ast.Attribute):
            return self.handle_ast_attribute(target,expr)
        elif isinstance(expr, ast.Call):
            return self.handle_ast_call(target,expr)
        elif isinstance(expr, ast.List) or isinstance(expr,ast.Tuple):
            return self.handle_ast_tuple_or_list(target,expr)
        elif isinstance(expr, ast.UnaryOp):
            assert isinstance(expr.op, ast.USub)  # quick fix
            assert ast_add_on.is_constant(expr.operand)
            return RawJitParserVariable(expr,raw_parser=self)
        else:
            raise Exception(f"{type(expr)} unknown")


    def parse(self,
            jit_result_of_this_module, 
            module_name, method_name, 
            input_raw_vars, is_main=False) -> RawJitParserVariable:
        # jit_result : is the result for a specific module or sub module,
        # ex : module      = jit_tr_GPT2.wpe
        #      module_name = "self.wpe"
        #      method_name = "forward"
        # inputs_vars : RawJitParserVariables on which module's function is applied

        # 1) Get the code from jit
        if method_name == "forward":  # quick fix
            code, memory \
                = jit_result_of_this_module.code_with_constants
        else:
            result = getattr(jit_result_of_this_module, method_name)
            code, memory = result.code_with_constants
        if not isinstance(memory, dict):  # quick fix, due to a type error in jit
            memory = memory.const_mapping
        self.current_jit_memory = memory
        method_code_ast = (ast.parse(code)).body[0]

        # 2) Initiate the local env of raw vars with 1 var for "self"
        self.current_dict_raw_vars = dict()
        self.current_dict_raw_vars["self"] = RawJitParserVariable(
            value_ast=ast.Name(module_name), 
            raw_parser=self,
            is_attr_of_self=True, 
            real_value_as_an_attr_of_self=jit_result_of_this_module
        )

        # 3) Add the inputs to the local env
        input_names = [inp.arg for inp in method_code_ast.args.args]
        # Note: input_names[0] = "self"
        if is_main: # = top level
            self.input_targets = input_names[1:]
            for input_name in self.input_targets:
                input_node = RawNode(
                    target=input_name,
                    raw_parser=self,
                    code_ast=ast_add_on.make_ast_constant("INPUT"),
                    fct="INPUT",
                    is_input=True,
                )
                self.current_dict_raw_vars[input_name] \
                    = RawJitParserVariable(
                        ast.Name(input_name),
                        raw_parser=self,
                        node=input_node)
        else: # sub module => Called at higher level => inputs = the calling args
            assert len(input_names) == len(input_raw_vars) + 1
            for input_name,input_raw_var in \
                    zip(input_names[1:],input_raw_vars):
                self.current_dict_raw_vars[input_name] = input_raw_var
                # Link local inputs' names with higher level RawJitParserVariables

        # 4) Parse each line 1 by 1
        for line_of_code_ast in method_code_ast.body:
            if isinstance(line_of_code_ast, ast.Assign):
                # 1) Flat targets
                ast_targets = line_of_code_ast.targets[0]
                if isinstance(ast_targets, ast.Name):
                    list_targets = [ast_targets.id]
                    main_target = ast_targets.id
                elif (isinstance(ast_targets, ast.Tuple) 
                or    isinstance(ast_targets, ast.List)):
                    list_targets = [tar.id for tar in ast_targets.elts]
                    main_target = None
                else:
                    raise Exception(
                        f"ast.Assign's target neither name, tuple or list ?"
                        f"{type(ast_targets)} found"
                    )

                # 2) Parse the expression
                main_raw_var = self.handle_expr(
                    main_target,
                    line_of_code_ast.value,
                )

                # 3) Handle multiple targets
                # (a,b) = code => fresh_var = code ; a = fresh_var[0], b = ...[1]
                if main_target is not None:
                    self.current_dict_raw_vars[main_target] = main_raw_var
                else: # multiple 
                    for target_index, target_name in enumerate(list_targets):
                        target_unique_name = self.make_name_unique(target_name)
                        target_assigning_node = RawNode(
                            target=target_unique_name,
                            fct="getattr",
                            raw_parser=self)
                        target_assigning_node.code_ast = ast.Subscript(
                            main_raw_var.use_value_ast(
                                calling_node=target_assigning_node),
                            ast_add_on.make_ast_constant(target_index)
                        )
                        target_raw_var = RawJitParserVariable(
                            ast.Name(target_unique_name),
                            raw_parser=self,
                            node=target_assigning_node
                        )
                        self.current_dict_raw_vars[target_name] = target_raw_var

        # 5) end of the loop
            else:
                assert isinstance(line_of_code_ast, ast.Return)
                return self.handle_expr(
                    target=None,
                    expr=line_of_code_ast.value)

        raise Exception("No ast.Return found at the end of jit.code ??!")
    
    @staticmethod
    def get_output_attributes(output_raw_var : RawJitParserVariable):
        """
        get the output_node: if not requires_grad => raise Exception
        which is catch in Rockmate, since it implies no Backward
        """
        if not isinstance(output_raw_var._value_ast,ast.Name):
            warnings.warn( # TO CHANGE COMMENTS 
                f"The RawJitParser found that the output isn't an "\
                f"ast.Name, we assume it's a constant. \nAST type "\
                f"of the output: {type(output_raw_var.val)}")
            raise constants.ExceptionModuleDoesNotReqGrad
        if not output_raw_var.has_node:
            warnings.warn(
                f"RawJitParser hasn't attached any node to the output."\
                f"Thus we assume it's a constant.")
            raise constants.ExceptionModuleDoesNotReqGrad
        else:
            return (
                output_raw_var._value_ast.id,
                output_raw_var.node
            )
        