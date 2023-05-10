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

from rkgb.utils import *

# **********
# * B_node *
# **********

class B_node(RK_node):
    def __init__(self, target="", code=None, fct="", deps=None, is_input=False):
        """ attributes :
        .target   : str  : the name of the only var defined in the node
        .ast_code : AST  : right part of the assigning code
        .fct      : str  : the function used in .ast_code
        .is_input : bool : input vars are represented by nodes with dummy code
        .is_rand  : bool : whether .fct involves randomness
        .deps      : B_node set : required nodes to run .ast_code
        .deps_rand : str set : required random targets
        """
        super().__init__("B",target=target,unique_id_generator=node_unique_id_generator)
        if not code:
            code = ast_add_on.make_ast_constant("/!\\ not defined /!\\")
        self.ast_code = code
        self.fct = fct
        self.deps = deps if not (deps is None) else set()
        self.is_input = is_input
        self.is_rand = bool(fct in global_vars.list_rand_fct)
        self.deps_rand = set()
        global all_nodes
        all_nodes.append(self)


# *********
# * B_Var *
# *********

class B_var:
    def __init__(
        self,
        val,
        node: B_node = None,
        is_attr_of_self=False,
        real_value_as_an_attr_of_self=None,
    ):
        # "val" must be an AST
        self.is_attr_of_self = is_attr_of_self
        self.real_value_as_an_attr_of_self = real_value_as_an_attr_of_self
        self.val = val
        self.has_node = False  # by default
        self.is_rand = False  # by default
        if node:
            if node.deps == set() and not node.is_input:
                if node.is_rand:
                    dict_rand[node.target] = node.ast_code
                    self.is_rand = True
                else:  # src neither input or rand
                    self.val = node.ast_code
            else:
                self.has_node = True
                self.node = node

    def get_value(self, calling_node):
        if self.has_node:
            calling_node.deps.add(self.node)
        elif self.is_rand:
            calling_node.deps_rand.add(self.val.id)
        return self.val

    def inherits(self, parent, l_attr):  
        # for a getattr AND is_attr_of_self
        if parent.has_node:
            self.has_node = True
            self.node = parent.node
        obj = parent.real_value_as_an_attr_of_self
        for at in l_attr:
            obj = getattr(obj,at)
        self.real_value_as_an_attr_of_self = obj


# **********
# * B_graph *
# **********

# /!\ Most of B_graph attributes are empty and .nodes shouldn't be trusted 
# -> not toposorted + contains a lot of useless nodes
class B_graph(RK_graph):
    def __init__(self):
        super().__init__("B")
        self.output_var = None 

# ==========================


# ==========================
# ====== Make B graph ======
# ==========================

dict_rand = dict()
node_unique_id_generator = Node_unique_id_generator()
all_nodes = []

def make_B(model, dict_inputs, verbose=None, impose_device=True, device=None):
    # -- global vars --
    global var_impose_device
    var_impose_device = impose_device
    if not (verbose is None):
        global_vars.ref_verbose[0] = verbose
    if not device:
        device = small_fcts.get_device_and_check_all_same_device(
            model, dict_inputs
        )
    # -- Receptacle variables --
    # -> Everything is done using cross recursive functions, 
    #    but we save some results in these general variables 
    #    rather than passing them as function arguments.
    global dict_rand ; dict_rand = dict()  # all random targets
    dict_constants = dict()
    global all_nodes ; all_nodes = [] # list of all the nodes generated
    fresh_var = 0 # counter to give a unique number to each target
    global node_unique_id_generator
    node_unique_id_generator = Node_unique_id_generator()

    sample_for_jit = small_fcts.order_dict_inputs(dict_inputs,model)

    with torch.no_grad():
        jit_result = torch.jit.trace_module(
            model, {"forward": sample_for_jit}, check_trace=False
        )

    # ===============================
    # === MAIN RECURSIVE FUNCTION ===
    def open_sub_module(sub_mod, sub_mod_str, sub_fct, inputs_vars, is_main=False):
        # -> B_graph
        # ex : sub_mod     = jit_tr_GPT2.wpe
        #      sub_mod_str = "self.wpe"
        #      sub_fct     = "forward"
        # inputs_vars : B_vars on which the sub_fct is applied
        if sub_fct == "forward":  # quick fix
            code, memory = sub_mod.code_with_constants
        else:
            code, memory = getattr(sub_mod, sub_fct).code_with_constants
        if not isinstance(memory, dict):  # quick fix, due to a type error in jit
            memory = memory.const_mapping
        a = (ast.parse(code)).body[0]

        dict_vars = {}
        dict_vars["self"] = B_var(
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
                i_node = B_node(
                    target=inputs[i],
                    code=ast_add_on.make_ast_constant("INPUT"),
                    fct="INPUT",
                    deps=set(),
                    is_input=True,
                )
                dict_vars[inputs[i]] = B_var(ast.Name(inputs[i]), node=i_node)
        else:
            assert nb_i == len(inputs_vars) + 1
            for i in range(1, nb_i):  # inputs[0]="self"
                dict_vars[inputs[i]] = inputs_vars[i - 1]
                # Link local inputs' names with global vars
        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -> variables' names must be unique through all the program
        def make_unique(s):
            nonlocal fresh_var
            fresh_var += 1
            return f"__{fresh_var}_{s}"
        
        # -> In case we add new lines :
        def get_fresh_var():
            nonlocal fresh_var
            fresh_var += 1
            return f"__{fresh_var}_fv"
        
        # -> In case its an external constant
        def get_constant(s):
            nonlocal fresh_var
            fresh_var += 1
            return f"_cst_{fresh_var}_{s}"

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ===== AUXILIARY FUNCTIONS =====
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- handle attribute --
        # -> explicit "getattr" or using "." (e.g. self.wpe)
        def aux_make_ast(p_val, format_fct, l_attr):  # -> AST
            if isinstance(p_val, ast.Name):
                new_val = format_fct(p_val)
            else:
                attr = ".".join(l_attr)
                new_val = ast.Call(
                    func=ast.Name("getattr"),
                    args=[p_val, ast_add_on.make_ast_constant(attr)],
                    keywords=[],
                )
            return new_val

        def aux_handle_attr(target, parent_var, format_fct, l_attr):
            if parent_var.is_attr_of_self:
                p_val = parent_var.val
                new_val = aux_make_ast(p_val, format_fct, l_attr)
                new_var = B_var(new_val, is_attr_of_self=True)
                new_var.inherits(parent_var, l_attr)
            else:
                if target is None:
                    new_id = get_fresh_var()
                else:
                    new_id = make_unique(target)
                new_node = B_node(target=new_id, fct="getattr")
                p_val = parent_var.get_value(calling_node=new_node)
                new_val = aux_make_ast(p_val, format_fct, l_attr)
                new_node.ast_code = new_val
                new_var = B_var(new_val, node=new_node)
            return new_var

        def handle_attr(expr: ast.Attribute, target: str):
            l_name = ast_add_on.open_attr_until_name(expr)
            if l_name[0] not in dict_vars:
                raise Exception(
                    f"Unknown global variable mentioned in the code "
                    f"extracted by jit {l_name[0]}."
                )
            parent_var = dict_vars[l_name[0]]
            attr = ".".join(l_name[1:])
            format_fct = lambda pv: ast.Name(pv.id + "." + attr)
            return aux_handle_attr(target, parent_var, format_fct, l_name[1:])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- open list of targets e.g. tuple --
        # -> so that each node has only one target
        # (e.g. l = fct() ; a = l[0] ; b = l[1] instead of a,b=fct())
        def init_targets(list_tg):
            if len(list_tg) == 1:
                return make_unique(list_tg[0])
            else:
                return get_fresh_var()

        def handle_targets(list_tg, main_var):  # str list of len > 1
            for i, tg in enumerate(list_tg):
                new_tg_id = make_unique(tg)
                new_node = B_node(target=new_tg_id, fct="getattr")
                main_val = main_var.get_value(calling_node=new_node)
                assert isinstance(main_val, ast.Name)
                # else : to much simplifications :/
                new_node.ast_code = ast.Subscript(
                    main_val, ast_add_on.make_ast_constant(i)
                )
                new_var = B_var(ast.Name(new_tg_id), node=new_node)
                dict_vars[tg] = new_var

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- handle a function call -- (cross recursive with handle_expr)
        def handle_call(expr: ast.Call, target) -> B_var:
            l_name = ast_add_on.open_attr_until_name(expr.func)  # full name
            args = list(expr.args)

            # == explicit getattr ==
            if len(l_name) == 1 and l_name[0] == "getattr":
                assert len(args) == 2
                assert ast_add_on.is_constant(args[1])
                # If fail: Why is there an expression to
                # refer to the attr we want to take ?!
                parent_var = handle_expr(args[0])
                attr = args[1].value
                if attr.isdigit():
                    format_fct = lambda pv: ast.Subscript(
                        value=pv, slice=ast_add_on.make_ast_constant(int(attr))
                    )
                else:
                    format_fct = lambda pv: ast.Call(
                        func=ast.Name("getattr"),
                        args=[pv, ast_add_on.make_ast_constant(attr)],
                        keywords=[],
                    )
                return aux_handle_attr(target, parent_var, format_fct, [attr])
                # might create one node

            # == TorchScript's functions ==
            # -> must be removed because some refer to TorchScript global var
            elif l_name[0] == "ops":
                assert len(args) == 1
                return handle_expr(args[0], target)
            elif l_name[0] == "int":
                return handle_expr(args[0], target)
            elif l_name[0] == "annotate":
                assert len(args) == 2
                return handle_expr(args[1], target)
            elif var_impose_device and l_name[0] == "torch" and l_name[1] == "device":
                return B_var(val = ast.Name("device"))

            else:  # -> real function
                args_Bvar = [handle_expr(ar, target=None) for ar in args]
                # == sub module ==
                if l_name[0] in dict_vars:
                    sub_var = dict_vars[l_name[0]]
                    print_debug(
                        f"In {sub_mod_str}.{sub_fct} try to sub open "
                        f"{ast_add_on.ast_to_str(sub_var.val)}.{l_name[1:]}"
                    )
                    assert sub_var.is_attr_of_self
                    sub_sub_mod = sub_var.real_value_as_an_attr_of_self
                    for at in l_name[1:-1]:
                        sub_sub_mod = getattr(sub_sub_mod, at)
                    sub_sub_str = ast_add_on.ast_to_str(sub_var.val)
                    sub_graph = open_sub_module(
                        sub_sub_mod, sub_sub_str, l_name[-1], args_Bvar
                    )
                    return sub_graph.output_var

                # == builtin functions ==
                else:
                    if target is None:
                        target = get_fresh_var()

                    # == torch.nn.functional / torch.Tensor == quick.fix
                    if l_name[0] == "torch" and len(l_name) == 2:
                        bool_found = False
                        for module_name in global_vars.list_python_modules:
                            try:
                                exec(f"{module_name}.{l_name[1]}")
                                fct_name = f"{module_name}.{l_name[1]}"
                                bool_found = True
                            except:
                                pass
                            if bool_found: break

                        if not bool_found:
                            raise Exception(
                                f"jit translate any torch function has: "\
                                f"torch.<function name>, for instance here:\n"\
                                f"torch.{l_name[1]}.\nSo we need to find the "\
                                f"submodule where the function belongs to, "\
                                f"we will tryed : {global_vars.list_python_modules}"
                            )
                    else:
                        fct_name = ".".join(l_name)

                    # == else ==
                    new_node = B_node(target=target, fct=fct_name)
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
                                        global_vars.get_torchscript_dtype(kw.value.value)
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
                    return B_var(ast.Name(target), node=new_node)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # isinstance(expr, ast.List or ast.Tuple)
        # constr = "list" or "tuple"
        # in this version, I'm *not* inserting them here, I will do it later.
        # -> because I need to precise the calling_node...
        def aux_handle_tuple_or_list(expr, target, constr):
            if target is None:
                target = get_fresh_var()
            new_node = B_node(target=target, fct=f"{constr} constructor")
            args_vars = [handle_expr(v) for v in expr.elts]
            args_ast = [v.get_value(calling_node=new_node) for v in args_vars]
            if constr == "list":
                c = ast.List(args_ast)
            else:
                c = ast.Tuple(args_ast)
            new_node.ast_code = c
            return B_var(ast.Name(target), node=new_node)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # -- handle any expr -- return type -> B_var
        # -> the main recursive fct to handle ast
        # if the expr is simple (e.g. constant or self's attr)
        # -> B_var.has_node == False
        # otherwise, a node (= a piece of code) is created.
        # The optional parameter  "target" imposes the name of the var created
        # /!\ TorchScript's global constant vars must have been removed
        def handle_expr(expr, target: str = None) -> B_var:
            if ast_add_on.is_constant(expr):
                return B_var(expr)
            elif isinstance(expr, ast.Name):
                assert expr.id in dict_vars
                return dict_vars[expr.id]
            elif (
                isinstance(expr, ast.Attribute)  # -> special constants
                and isinstance(expr.value, ast.Name)
                and expr.value.id == "CONSTANTS"
            ):
                s = get_constant(expr.attr)
                dict_constants[s] = memory[expr.attr]
                return B_var(ast.Name(s))
                #return B_var(ast_add_on.make_ast_constant(memory[expr.attr]))
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
                return B_var(expr)
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
    # === END OF MAIN RECURSIVE FUNCTION ===
    # ======================================
    
    main_str = "self"
    main_fct = "forward"
    main_g = open_sub_module(jit_result, main_str, main_fct, [], is_main=True)
    main_g.nodes = all_nodes
    main_g.dict_rand = dict_rand
    main_g.dict_constants = dict_constants
    main_g.node_unique_id_generator = node_unique_id_generator
    # -> reset global vars
    dict_rand = dict()
    node_unique_id_generator = Node_unique_id_generator()
    return main_g
