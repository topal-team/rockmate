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
            raw_parser,
            ast_code=None, 
            fct="", 
            deps=None, 
            is_input=False
        ):
        """ attributes :
        .target   : str  : the name of the only var defined in the node
        .ast_code : AST  : right part of the assigning code
        .fct      : str  : the function used in .ast_code
        .is_input : bool : input vars are represented by nodes with dummy code
        .is_rand  : bool : whether .fct involves randomness
        .deps      : RawNode set : required nodes to run .ast_code
        .deps_rand : str set : required random targets
        """
        super().__init__("R",target,parent_structure=raw_parser)
        if ast_code is None:
            self.ast_code = ast_add_on.make_ast_constant("/!\\ NO CODE /!\\")
        else:
            self.ast_code = ast_code
        self.fct = fct
        raw_parser.all_raw_nodes.append(self)
        raw_parser.deps[self] = deps if deps is not None else set()
        self.is_input = is_input
        self.is_rand = bool(fct in constants.list_rand_fct)
        self.deps_rand = set()

    def deps(self,parent_structure): # Either a RawParser or a RawGraph
        return parent_structure.deps[self]
    def indirect_deps(self,parent_structure):
        return parent_structure.deps[self]


class RawVar:
    def __init__(
            self,
            value_ast,
            raw_parser,
            node: RawNode = None,
            is_attr_of_self=False,
            real_value_as_an_attr_of_self=None,
        ):
        # "val" must be an AST
        self.is_attr_of_self = is_attr_of_self
        self.real_value_as_an_attr_of_self = real_value_as_an_attr_of_self
        self.ast = value_ast
        self.has_node = False  # by default
        self.is_rand = False  # by default
        if node: # to improve via "multiple_outputs" branch
            if node.deps == set() and not node.is_input:
                if node.is_rand:
                    raw_parser.dict_rand[node.target] = node.ast_code
                    self.is_rand = True
                else:  # node without deps but neither input or rand
                    self.ast = node.ast_code
            else:
                self.has_node = True
                self.node = node

    def get_ast(self, calling_node):
        """ Instead of self.ast, you must use this
        Take care of the "deps" relation
        """
        if self.has_node:
            calling_node.deps.add(self.node)
        elif self.is_rand:
            calling_node.deps_rand.add(self.ast.id)
        return self.ast

    def inherits(self, parent, list_attributes):  
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
    -> Attention: most attributes common to all graphs
       such as "inputs" or "outputs" are empty, even 
       "nodes" aren't toposorted and some are useless
    -> The only attribute important is "output_raw_var"
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



class RawParser():
    def __init__(self,impose_device):
        self.impose_device = impose_device
        self.all_raw_nodes = []
        self.dict_rand = dict()
        self.dict_constants = dict()
        self.current_dict_raw_vars = dict()
        self.node_unique_id_generator = base.Node_unique_id_generator()
        self.counter_unique_number = 0

    def get_unique_number(self):
        self.counter_unique_number += 1
        return self.counter_unique_number
    def make_name_unique(self,s):
        return f"__{self.get_unique_number()}_{s}"
    def get_fresh_name(self):
        return f"__{self.get_unique_number()}_fresh"
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

    def aux_for_attributes(self, target, parent_raw_var, list_attributes):
        """ Used for:
         - Via an ast.Call `getattr(a,"b")`
         - Via an ast.Attribute `a.b`
        """
        if parent_raw_var.is_attr_of_self:
            parent_ast = parent_raw_var.ast
            new_ast = self.rebuild_ast_attribute(parent_ast,list_attributes)
            new_raw_var = RawVar(new_ast, is_attr_of_self=True)
            new_raw_var.inherits(parent_raw_var,list_attributes)
        else:
            if target is None:
                new_id = self.get_fresh_name()
            else:
                new_id = self.make_name_unique(target)
            new_node = RawNode(target=new_id, fct="getattr",raw_parser=self)
            parent_ast = parent_raw_var.get_ast(calling_node=new_node)
            new_ast = self.rebuild_ast_attribute(parent_ast,list_attributes)
            new_node.ast_code = new_ast
            new_raw_var = RawVar(new_ast,node=new_node)
        return new_raw_var


    def handle_ast_attribute(self, target : str, expr : ast.Attribute) -> RawVar:
        parent_expr,list_attributes = ast_add_on.open_all_nested_attributes(expr)
        parent_raw_var = self.handle_expr(self.get_fresh_name(),parent_expr)
        return self.aux_for_attributes(target,parent_raw_var,list_attributes)


    def aux_handle_ast_call_getattr(self,target,call_args):
        assert len(call_args) == 2 # getattr(arg1,arg2)
        assert ast_add_on.is_constant(call_args[1])
        # If fail: something like `getattr(a,"foo"+"bar")`
        # It's a nested operation, so I assume jit inlined it
        # If new tracer, then use handle_expr over call_args[1]
        parent_raw_var = self.handle_expr(call_args[0])
        attribute = call_args[1].value
        return self.aux_for_attributes(target,parent_raw_var,[attribute])
    
    def aux_handle_ast_call_sub_module(
            self,called_raw_var,rest_of_func_name ,call_arg_raw_vars):
        sub_module = called_raw_var.real_value_as_an_attr_of_self
        for attribute in rest_of_func_name[:-1]:
            sub_module = getattr(sub_module, attribute)
        sub_module_name = ast_add_on.ast_to_str(called_raw_var.ast)
        method_name = rest_of_func_name[-1]
        sub_module_output_raw_var = self.parse(
            sub_module, sub_module_name, method_name, call_arg_raw_vars
        )
        # Note: to parse sub_module, we give raw_vars of its inputs,
        # and it returns the raw_var which define its output
        # -> ie it creates the raw_var of the sub_module call result!
        # Which is exactly the objective of `RawParser.handle_ast` functions
        return sub_module_output_raw_var
    
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
        

    def handle_call(self,target : str, expr : ast.Call) -> RawVar:
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
            return RawVar(value_ast = ast.Name("device"))

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
                fct_name = self.aux_for_call_find_function_name(
                    first_term_of_func_name,rest_of_func_name)
                if target is None:
                    target = self.get_fresh_name()
                new_node = RawNode(target=target,fct=fct_name,raw_parser=self)
                args_ast = [
                    v.get_value(calling_node=new_node) for v in args_Bvar
                ]
                kwds_ast = []
                for kw in expr.keywords:
                    if var_impose_device and kw.arg == "device":
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
                                (handle_expr(kw.value)).get_value(new_node),
                            )
                        )
                new_node.ast_code = ast.Call(
                    func=ast.Name(fct_name), args=args_ast, keywords=kwds_ast
                )
                return RawVar(ast.Name(target), node=new_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    def parse(self,sub_module, sub_module_name, method_name, inputs_raw_vars, is_main=False):
        # -> B_graph
        # ex : sub_mod     = jit_tr_GPT2.wpe
        #      sub_mod_str = "self.wpe"
        #      sub_fct     = "forward"
        # inputs_vars : RawVars on which the sub_fct is applied
        if sub_fct == "forward":  # quick fix
            code, memory = sub_mod.code_with_constants
        else:
            code, memory = getattr(sub_mod, sub_fct).code_with_constants
        if not isinstance(memory, dict):  # quick fix, due to a type error in jit
            memory = memory.const_mapping
        a = (ast.parse(code)).body[0]

        dict_vars = {}
        dict_vars["self"] = RawVar(
            val=ast.Name(sub_mod_str), 
            is_attr_of_self=True, 
            real_value_as_an_attr_of_self=sub_mod
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- Inputs --
        inputs = []
        for arg in a.args.args:
            inputs.append(arg.arg)
        nb_i = len(inputs)
        if is_main:  # /!\
            for i in range(1, nb_i):
                i_node = RawNode(
                    target=inputs[i],
                    code=ast_add_on.make_ast_constant("INPUT"),
                    fct="INPUT",
                    deps=set(),
                    is_input=True,
                )
                dict_vars[inputs[i]] = RawVar(ast.Name(inputs[i]), node=i_node)
        else:
            assert nb_i == len(inputs_vars) + 1
            for i in range(1, nb_i):  # inputs[0]="self"
                dict_vars[inputs[i]] = inputs_vars[i - 1]
                # Link local inputs' names with global vars
        # ~~~~~~~~~~~~~~~~~~~~~~~~~


        # ===== AUXILIARY FUNCTIONS =====

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- open list of targets e.g. tuple --
        # -> so that each node has only one target
        # (e.g. l = fct() ; a = l[0] ; b = l[1] instead of a,b=fct())
        def init_targets(list_tg):
            if len(list_tg) == 1:
                return make_unique(list_tg[0])
            else:
                return get_fresh_name()

        def handle_targets(list_tg, main_var):  # str list of len > 1
            for i, tg in enumerate(list_tg):
                new_tg_id = make_unique(tg)
                new_node = RawNode(target=new_tg_id, fct="getattr")
                main_val = main_var.get_value(calling_node=new_node)
                assert isinstance(main_val, ast.Name)
                # else : to much simplifications :/
                new_node.ast_code = ast.Subscript(
                    main_val, ast_add_on.make_ast_constant(i)
                )
                new_raw_var = RawVar(ast.Name(new_tg_id), node=new_node)
                dict_vars[tg] = new_raw_var

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # isinstance(expr, ast.List or ast.Tuple)
        # constr = "list" or "tuple"
        # in this version, I'm *not* inserting them here, I will do it later.
        # -> because I need to precise the calling_node...
        def aux_handle_tuple_or_list(expr, target, constr):
            if target is None:
                target = get_fresh_name()
            new_node = RawNode(target=target, fct=f"{constr} constructor")
            args_vars = [handle_expr(v) for v in expr.elts]
            args_ast = [v.get_value(calling_node=new_node) for v in args_vars]
            if constr == "list":
                c = ast.List(args_ast)
            else:
                c = ast.Tuple(args_ast)
            new_node.ast_code = c
            return RawVar(ast.Name(target), node=new_node)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- handle any expr -- return type -> RawVar
        # -> the main recursive fct to handle ast
        # if the expr is simple (e.g. constant or self's attr)
        # -> RawVar.has_node == False
        # otherwise, a node (= a piece of code) is created.
        # The optional parameter  "target" imposes the name of the var created
        # /!\ TorchScript's global constant vars must have been removed
        def handle_expr(expr, target: str = None) -> RawVar:
            if ast_add_on.is_constant(expr):
                return RawVar(expr)
            elif isinstance(expr, ast.Name):
                assert expr.id in dict_vars
                return dict_vars[expr.id]
            elif (
                isinstance(expr, ast.Attribute)  # -> special constants
                and isinstance(expr.value, ast.Name)
                and expr.value.id == "CONSTANTS"
            ):
                s = get_constant_name(expr.attr)
                dict_constants[s] = memory[expr.attr]
                return RawVar(ast.Name(s))
                #return RawVar(ast_add_on.make_ast_constant(memory[expr.attr]))
            elif isinstance(expr, ast.Attribute):
                return handle_attr(expr, target)  # may creates one node
            elif isinstance(expr, ast.Call):
                return handle_call(expr, target)
                # may creates nodes for arguments (+ for output=target)
            elif isinstance(expr, ast.List):
                return aux_handle_tuple_or_list(expr, target, "list")
            elif isinstance(expr, ast.Tuple):
                return aux_handle_tuple_or_list(expr, target, "tuple")
            elif isinstance(expr, ast.UnaryOp):
                assert isinstance(expr.op, ast.USub)  # quick fix
                assert ast_add_on.is_constant(expr.operand)
                return RawVar(expr)
            else:
                raise Exception(f"{type(expr)} unknown")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # =========================

        # == MAIN ==
        for ast_n in a.body:
            if isinstance(ast_n, ast.Assign):
                # -- targets --
                list_tg = []
                tg = ast_n.targets[0]
                if isinstance(tg, ast.Name):
                    list_tg = [tg.id]
                    target_id = tg.id
                elif isinstance(tg, ast.Tuple) or isinstance(tg, ast.List):
                    for e in tg.elts:
                        list_tg.append(e.id)
                    target_id = None
                else:
                    raise Exception(
                        f"ast.Call's target neither name, tuple or list ?"
                        f"{type(tg)} found"
                    )

                # -- main --
                main_id = init_targets(list_tg)
                main_var = handle_expr(ast_n.value, main_id)
                if len(list_tg) > 1:
                    handle_targets(list_tg, main_var)

                if target_id is not None:
                    dict_vars[target_id] = main_var

            else:
                assert isinstance(ast_n, ast.Return)
                ret_graph = B_graph()
                ret_graph.output_var = handle_expr(ast_n.value, target=None)
                return ret_graph

        raise Exception("No ast.Return found at the end of jit.code ??!")