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
import ast
import torch
from torch import Tensor
from lowlevel import ast_add_on
from lowlevel import constants
from lowlevel import preprocess_samples
from lowlevel import jit_patch
from core import base


class RawNode(base.Node):
    def __init__(self, 
            target,
            code_ast=None, 
            fct="", 
            deps=None, 
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
        .deps_rand : str set : required random targets
        """
        super().__init__("R",target,parent_structure=raw_parser)
        if code_ast is None:
            self.code_ast = ast_add_on.make_ast_constant("/!\\ NO CODE /!\\")
        else:
            self.code_ast = code_ast
        self.fct = fct
        self.is_input = is_input
        self.is_rand = bool(fct in constants.list_rand_fct)
        self.deps_rand : set[str] = set()
        self.deps : set[RawNode] = deps if deps is not None else set()
        if raw_parser is not None: raw_parser.all_raw_nodes.append(self)

    def get_all_standard_deps(self):
        return self.deps



class RawVar:
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
        self.value_ast = value_ast
        self.has_node = False  # by default
        self.is_rand = False  # by default
        if node: # to improve via "multiple_outputs" branch
            if node.deps == set() and not node.is_input:
                if node.is_rand:
                    raw_parser.dict_rand[node.target] = node.code_ast
                    self.is_rand = True
                else:  # node without deps but neither input or rand
                    self.value_ast = node.code_ast
            else:
                self.has_node = True
                self.node = node

    def use_value_ast(self, calling_node):
        """ Instead of self.value_ast, you must use this
        Take care of the "deps" relation
        """
        if self.has_node:
            calling_node.deps.add(self.node)
        elif self.is_rand:
            calling_node.deps_rand.add(self.value_ast.id)
        return self.value_ast

    def inherits_self_attr(self, parent, list_attributes):  
        # for a getattr AND is_attr_of_self
        if parent.has_node:
            self.has_node = True
            self.node = parent.node
        obj = parent.real_value_as_an_attr_of_self
        for at in list_attributes:
            obj = getattr(obj,at)
        self.real_value_as_an_attr_of_self = obj



class RawGraph(base.Graph):
    """RawGraph
    -> tldr: raw graph => very few attributes
    -> The only important attribute is "output_raw_var"
       which gives the source of the "deps" relation
    """
    def __init__(self,
            model,
            dict_inputs : preprocess_samples.DictInputs,
            impose_device=True
        ):
        super().__init__("R")
        # - use jit -
        samples_for_jit = dict_inputs.to_list_args()
        with torch.no_grad():
            jit_result = torch.jit.trace_module(
                model, {"forward": samples_for_jit}, check_trace=False
            )
        # - parse -
        parser = RawParser(impose_device)
        self.output_raw_var = parser.parse(
            jit_result, "self", "forward", [], is_main=True
        )
        self.nodes = parser.all_raw_nodes
        self.dict_rand = parser.dict_rand
        self.dict_constants = parser.dict_constants


    def __str__(self):
        return (
            f"RawGraph with {len(self.nodes)} nodes "\
            f"(remember this list may contain garbage)")
    def render(self,
            name=None,
            view=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None
        ):
        name = base.Graph._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        for rn in self.nodes:
            dot.node(rn.target,rn.get_code())
        for rn in self.nodes:
            for req_rn in rn.deps:
                dot.edge(req_rn.target,rn.target)
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )

# Draft of the general comment about how the parser works:
# if the expr is simple (e.g. constant or self's attr)
# -> RawVar.has_node == False
# otherwise, a node (= a piece of code) is created.
# The optional parameter  "target" imposes the name of the var created
class RawParser():
    def __init__(self,impose_device):
        self.impose_device = impose_device
        self.all_raw_nodes = []
        self.dict_rand = dict()
        self.current_dict_raw_vars = dict()
        self.dict_constants = dict()
        self.current_jit_memory = dict()
        self.node_unique_id_generator = base.Node_unique_id_generator()
        self.counter_unique_number = 0

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


    def rebuild_ast_attribute(self,parent_ast,list_attributes):
        new_ast = parent_ast
        for attr in list_attributes:
            if attr.isdigit():
                new_ast = ast.Subscript(new_ast,
                    slice=ast_add_on.make_ast_constant(int(attr)))
            else:
                new_ast = ast.Attribute(new_ast,attr)
        return new_ast

    def aux_for_attribute(self, target, parent_raw_var, list_attributes):
        """ Used for:
         - `getattr(a,"b")` via an ast.Call
         - `a.b` via an ast.Attribute
        """
        if parent_raw_var.is_attr_of_self:
            parent_ast = parent_raw_var.value_ast
            new_ast = self.rebuild_ast_attribute(parent_ast,list_attributes)
            new_raw_var = RawVar(new_ast,raw_parser=self,is_attr_of_self=True)
            new_raw_var.inherits_self_attr(parent_raw_var,list_attributes)
        else:
            new_id = self.get_unique_name(target)
            new_node = RawNode(target=new_id, fct="getattr",raw_parser=self)
            parent_ast = parent_raw_var.use_value_ast(calling_node=new_node)
            new_ast = self.rebuild_ast_attribute(parent_ast,list_attributes)
            new_node.code_ast = new_ast
            new_raw_var = RawVar(new_ast,raw_parser=self,node=new_node)
        return new_raw_var


    def handle_ast_attribute(self, target : str, expr : ast.Attribute) -> RawVar:
        parent_expr,list_attributes = ast_add_on.open_all_nested_attributes(expr)
        parent_raw_var = self.handle_expr(self.get_fresh_name(),parent_expr)
        return self.aux_for_attribute(target,parent_raw_var,list_attributes)


    def aux_handle_ast_call_getattr(self,target,call_args):
        assert len(call_args) == 2 # getattr(arg1,arg2)
        assert ast_add_on.is_constant(call_args[1])
        # If fail: something like `getattr(a,"foo"+"bar")`
        # It's a nested operation, so I assume jit inlined it
        # If new tracer, then use handle_expr over call_args[1]
        parent_raw_var = self.handle_expr(call_args[0])
        attribute = call_args[1].value
        return self.aux_for_attribute(target,parent_raw_var,[attribute])
    
    def aux_handle_ast_call_sub_module(
            self,called_raw_var,rest_of_func_name ,call_arg_raw_vars):
        jit_result_sub_module = called_raw_var.real_value_as_an_attr_of_self
        for attribute in rest_of_func_name[:-1]:
            jit_result_sub_module = getattr(jit_result_sub_module, attribute)
        sub_module_name = ast_add_on.ast_to_str(called_raw_var.value_ast)
        method_name = rest_of_func_name[-1]
        save_current_dict_raw_vars = self.current_dict_raw_vars
        sub_module_output_raw_var = self.parse(
            jit_result_sub_module, sub_module_name, 
            method_name, call_arg_raw_vars
        )
        self.current_dict_raw_vars = save_current_dict_raw_vars
        # Note: to parse sub_module, we give raw_vars of its inputs,
        # and it returns the raw_var which define its output
        # -> ie it creates the raw_var of the sub_module call result!
        # Which is exactly the objective of `RawParser.handle_ast` functions
        return sub_module_output_raw_var

    """ TO REMOVE => I find a much better solution : torch.ops.aten
    def aux_for_call_find_function_name(
        first_term_of_func_name,rest_of_func_name):
        # Explanation:
        # When there is a torch function call, jit registers only
        # the name of the function and nothing about which PyTorch package
        # or class it comes from. Thus we have to test one by one: 
        # torch.<> ; torch.Tensor.<> ; torch.nn.functional.<> etc
        # to find where the function comes from.
        # Note: it's error prone, in case two files define functions
        # with the same name but different behaviors; but we have no choice.
        # We made a short list of possible packages
        #   torch.<.>
        #   torch.nn.functional.<.>
        #   torch.Tensor.<.>
        #   torch._C._nn.<.>
        #   torch._C._fft.<.>
        #   torch.ops.aten.<.>
        # Don't hesitate to append this list: 
        # -> rkgb.lowlevel.constants.list_pytorch_packages
        if (first_term_of_func_name == "torch" 
        and len(rest_of_func_name) == 1):
            last_term_of_func_name = rest_of_func_name[1]
            for package_name in constants.list_pytorch_packages:
                try:
                    exec(f"{package_name}.{last_term_of_func_name}")
                    fct_name = f"{package_name}.{last_term_of_func_name}"
                    return fct_name
                except:
                    pass

            # None of the packages match
            raise Exception(
                f"When there is a torch function call, jit "\
                f"registers only the name of the function and "\
                f"not where it comes from.\nFor instance only `gelu` "\
                f"instead of `torch.nn.functional.gelu`.\n Here we "\
                f"didn't manage to find where `{last_term_of_func_name}`"\
                f"comes from. If you know from which package it comes "\
                f"you can add it in "\
                f"`rkgb.lowlevel.constants.list_pytorch_packages`"
            )
        else:
            fct_name = ".".join([first_term_of_func_name]+rest_of_func_name)
            return fct_name
    """
        

    def handle_ast_call(self,target : str, expr : ast.Call) -> RawVar:
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
            return RawVar(value_ast=ast.Name("device"),raw_parser=self)

        else:
            call_arg_raw_vars = [self.handle_expr(None,arg) for arg in call_args]
            # Method => sub module:
            if first_term_of_func_name in self.dict_vars: # => a method
                called_raw_var = self.dict_vars[first_term_of_func_name]
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
                    fct_name = "torch.ops.aten."+rest_of_func_name[0]
                else:
                    fct_name = ".".join([first_term_of_func_name]+rest_of_func_name)
                target = self.get_unique_name(target)
                new_node = RawNode(target=target,fct=fct_name,raw_parser=self)
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
                                (self.handle_expr(kw.value)).use_value_ast(new_node),
                            )
                        )
                    #else: we remove weird keywords jit adds sometimes
                new_node.code_ast = ast.Call(
                    func=ast.Name(fct_name), args=args_ast, keywords=kwds_ast
                )
                return RawVar(ast.Name(target),raw_parser=self,node=new_node)
            

    def handle_ast_tuple_or_list(self,
            target : str, expr : Union[ast.List,ast.Tuple]) -> RawVar:
        # Not simplified / inserted here, might change TO CHANGE ?
        # -> because I need to precise the calling_node...
        target = self.get_unique_name(target)
        constr = "list" if isinstance(expr,ast.List) else "tuple"
        new_node = RawNode(
            target=target, fct=f"{constr} constructor",raw_parser=self)
        args_raw_vars = [self.handle_expr(e) for e in expr.elts]
        args_ast = [v.use_value_ast(calling_node=new_node) for v in args_raw_vars]
        if isinstance(expr,ast.List):
            new_node.code_ast = ast.List(args_ast)
        else:
            new_node.code_ast = ast.Tuple(args_ast)
        return RawVar(ast.Name(target),raw_parser=self,node=new_node)


    def handle_expr(self, target : str, expr) -> RawVar:
        if ast_add_on.is_constant(expr):
            return RawVar(expr,raw_parser=self)
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
            return RawVar(ast.Name(cst_name),raw_parser=self)
        elif isinstance(expr, ast.Attribute):
            return self.handle_ast_attribute(target,expr)
        elif isinstance(expr, ast.Call):
            return self.handle_ast_call(target,expr)
        elif isinstance(expr, ast.List) or isinstance(expr,ast.Tuple):
            return self.handle_ast_tuple_or_list(target,expr)
        elif isinstance(expr, ast.UnaryOp):
            assert isinstance(expr.op, ast.USub)  # quick fix
            assert ast_add_on.is_constant(expr.operand)
            return RawVar(expr,raw_parser=self)
        else:
            raise Exception(f"{type(expr)} unknown")


    def parse(self,
            jit_result_of_this_module, 
            module_name, method_name, 
            input_raw_vars, is_main=False) -> RawVar:
        # jit_result : is the result for a specific module or sub module,
        # ex : module      = jit_tr_GPT2.wpe
        #      module_name = "self.wpe"
        #      method_name = "forward"
        # inputs_vars : RawVars on which module's function is applied

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
        self.current_dict_raw_vars["self"] = RawVar(
            val=ast.Name(module_name), 
            is_attr_of_self=True, 
            real_value_as_an_attr_of_self=jit_result_of_this_module
        )

        # 3) Add the inputs to the local env
        input_names = [inp.arg for inp in method_code_ast.args.args]
        # Note: input_names[0] = "self"
        if is_main: # = top level
            for input_name in input_names[1:]:
                input_node = RawNode(
                    target=input_name,
                    raw_parser=self,
                    code=ast_add_on.make_ast_constant("INPUT"),
                    fct="INPUT",
                    is_input=True,
                )
                self.current_dict_raw_vars[input_name] \
                    = RawVar(ast.Name(input_name),
                             raw_parser=self,
                             node=input_node)
        else: # sub module => Called at higher level => inputs = the calling args
            assert len(input_names) == len(input_raw_vars) + 1
            for input_name,input_raw_var in \
                    zip(input_names[1:],input_raw_vars):
                self.current_dict_raw_vars[input_name] = input_raw_var
                # Link local inputs' names with higher level RawVars

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
                        target_raw_var = RawVar(
                            ast.Name(target_unique_name),
                            raw_parser=self,
                            node=target_assigning_node
                        )

        # 5) end of the loop
            else:
                assert isinstance(line_of_code_ast, ast.Return)
                return self.handle_expr(
                    target=None,
                    expr=line_of_code_ast.value)

        raise Exception("No ast.Return found at the end of jit.code ??!")