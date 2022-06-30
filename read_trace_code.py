# ------------------------------------
# Use AST functions to extract information from jit_trace.code
# Do some simplifications :
# -> Remove some useless getattr
# -> Decompose some operations (a = f(g(b)))
# -> Remove some TorchScript's operations (e.g. ops.prim.NumToTensor)
# ------------------------------------

import ast
import torch
from torch.jit import trace_module

list_rand_fct = ["torch.randn"]
dict_rand = {}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class B_node():
    def __init__(self,target="",code="",fct="",required=None,is_input=False):
        self.target = target
        self.make_code(code)
        self.fct = fct
        if required is None:
            self.req = []
        else:
            self.req = required
        self.is_input = is_input
        self.is_rand = None # unknown for the moment
        self.req_rand = []
    def make_code(self,code):
        self.code_without_target = code
        self.code = f"{self.target} = {code}"

class B_var():
    def __init__(self,val,node : B_node = None, is_attr_of_self = False, path_from_self = None):
        self.is_attr_of_self = is_attr_of_self
        self.path_from_self  = path_from_self
        self.val = val
        self.has_node = False # by default, e.g. has_node = False for const
        self.is_rand = False # by default
        if node is not None:
            if node.req==[] and not node.is_input:
                if node.fct in list_rand_fct:
                    dict_rand[node.target] = node.code
                    self.is_rand = True
                else:
                    self.val = node.code_without_target
            else:
                self.has_node = True
                self.node = node

    def get_value(self,calling_node):
        if self.is_rand:
            calling_node.is_rand = True
            calling_node.req_rand.append(self.val)
        elif self.has_node:
            calling_node.req.append(self.node)
        return self.val
    def inherits(self,parent,l_attr): # for a getattr
        if parent.has_node:
            self.has_node = True
            self.node = parent.node
        self.path_from_self = parent.path_from_self + l_attr

class B_graph():
    def __init__(self):
        self.nodes   = [] # tmp -> should not be trusted
        self.outputs = [] # list of B_var
        self.dict_rand = dict_rand
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ===== SMALL AUX FUNCTIONS =====
def open_attr_until_name(v):
    l_name = []
    while isinstance(v,ast.Attribute):
        l_name.append(v.attr)
        v = v.value
    l_name.append(v.id)
    l_name.reverse()
    return l_name
# =====

fresh_var = 0 # count the number of vars used over all the prgm

def open_sub_module(sub_mod,sub_mod_str,sub_fct,inputs_vars,is_main=False) -> B_graph:
    # ex : sub_mod     = jit_tr_GPT2.wpe 
    #      sub_mod_str = "self.wpe"
    #      sub_fct     = "forward"
    # inputs_vars : B_vars on which the sub_fct is applied
    if sub_fct=="forward": # quick fix
        code,memory = sub_mod.code_with_constants
    else:
        code,memory = getattr(sub_mod,sub_fct).code_with_constants
    if not isinstance(memory,dict):
        memory = memory.const_mapping
    a = (ast.parse(code)).body[0]

    dict_vars = {}
    dict_vars["self"] = B_var(sub_mod_str,is_attr_of_self=True,path_from_self=[])
    nodes = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # -- Inputs --
    inputs = []
    for arg in a.args.args:
        inputs.append(arg.arg)
    nb_i = len(inputs)
    if is_main:
        for i in range(1,nb_i):
            i_node = B_node(
                target=inputs[i],
                code=f"INPUT",
                fct="INPUT",
                required=[],
                is_input=True)
            nodes.append(i_node)
            dict_vars[inputs[i]]=B_var(inputs[i],node=i_node)
    else:
        assert(nb_i == len(inputs_vars)+1)
        for i in range(1,nb_i): #inputs[0]="self"
            dict_vars[inputs[i]]=inputs_vars[i-1]
            # Link local inputs' names with global vars
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # -> We must make variables' names are unique through all the program
    unique_str = sub_mod_str.replace('.','_')+"_"+sub_fct
    def make_unique(s):
        global fresh_var
        fresh_var += 1
        if concise:
            return f"__{fresh_var}_{s}"
        else:
            return f"__{unique_str}__{fresh_var}_{s}"
    # -> In case we add new lines :
    def get_fresh_var():
        global fresh_var
        fresh_var += 1
        if concise:
            return f"__{fresh_var}_fv"
        else:
            return f"__{unique_str}__{fresh_var}_fv"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ===== AUXILARY FUNCTIONS ===== 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # -- handle attribute -- -> explicit "getattr" or using "." (e.g. self.wpe)
    def aux_handle_attr(target,parent_var,format_fct,l_attr):
        if parent_var.is_attr_of_self:
            parent_val = parent_var.val
            new_val = format_fct(parent_val)
            new_var = B_var(new_val,is_attr_of_self=True)
            new_var.inherits(parent_var,l_attr)
        else:
            if target is None:
                new_id = get_fresh_var()
            else:
                new_id = make_unique(target)
            new_node = B_node(target=new_id,fct="getattr")
            setattr(new_node,"number",l_attr[0])
            nodes.append(new_node)
            parent_val = parent_var.get_value(calling_node=new_node)
            new_node.make_code(format_fct(parent_val))
            new_var = B_var(new_id,node=new_node)
        return new_var

    def handle_attr(expr : ast.Attribute,target : str):
        l_name = open_attr_until_name(expr)
        assert(l_name[0] in dict_vars) # -> else raise "Unknown variable, global ?"
        parent_var = dict_vars[l_name[0]]
        attr = '.'.join(l_name[1:])
        format_fct = lambda pv : pv + "." + attr
        return aux_handle_attr(target,parent_var,format_fct,l_name[1:])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # -- open list of targets e.g. tuple --
    # -> so that each node has only one target
    def init_targets(list_tg):
        if len(list_tg)==1:
            return make_unique(list_tg[0])
        else:
            return get_fresh_var()

    def handle_targets(list_tg,main_var): # str list of len > 1
        for i,tg in enumerate(list_tg):
            new_tg_id  = make_unique(tg)
            new_node   = B_node(target=new_tg_id,fct="getattr")
            nodes.append(new_node)
            main_val   = main_var.get_value(calling_node=new_node)
            # new_node.code = f"{new_tg_id} = getattr({main_val},\"{i}\")"
            new_node.make_code(f"{main_val}[{i}]")
            new_var    = B_var(new_tg_id,node=new_node)
            dict_vars[tg] = new_var
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # -- handle a function call -- (cross recursive with handle_expr)
    def handle_call(expr : ast.Call,target) -> B_var:
        l_name = open_attr_until_name(expr.func) # full name
        args = list(expr.args)

        # == explicit getattr ==
        if len(l_name)==1 and l_name[0]=='getattr':
            assert(len(args)==2)
            assert(isinstance(args[0],ast.Name))     # -> otherwise open_attr_until_name ?
            assert(isinstance(args[1],ast.Constant)) # -> otherwise handle_expr ?
            parent_id  = args[0].id
            parent_var = dict_vars[parent_id]
            attr = args[1].value
            if attr.isdigit(): format_fct = lambda pv : f"{pv}[{attr}]"
            else: format_fct = lambda pv : f"getattr({pv},\"{attr}\")"
            return aux_handle_attr(target,parent_var,format_fct,[attr]) # might create one node

        # == torchscript's functions == -> must be removed because some refer to TS global var
        elif l_name[0]=="ops":
            assert(len(args)==1)
            return handle_expr(args[0],target)
        elif l_name[0]=="int":
            return handle_expr(args[0],target)
        elif l_name[0]=="annotate":
            assert(len(args)==2)
            return handle_expr(args[1],target)

        else: # -> real function
            args_Bvar = [handle_expr(ar,target=None) for ar in args]
            # == sub module ==
            if l_name[0] in dict_vars:
                sub_var = dict_vars[l_name[0]]
                if show_debug_msg:
                    print(f"In {sub_mod_str}.{sub_fct} try to sub open {sub_var.val}.{l_name[1]}")
                assert(sub_var.is_attr_of_self)
                assert(len(l_name)==2)
                sub_sub_mod = sub_mod
                for at in sub_var.path_from_self:
                    sub_sub_mod = getattr(sub_sub_mod,at)
                sub_graph = open_sub_module(sub_sub_mod,sub_var.val,l_name[1],args_Bvar)
                nodes.extend(sub_graph.nodes)
                assert(len(sub_graph.outputs)==1) # else TODO
                return sub_graph.outputs[0] # which is a B_var !

            # == builtin functions ==
            else:
                if target is None:
                    target = get_fresh_var()

                # == torch.nn.functional / torch.Tensor == quick.fix
                if l_name[0]=="torch" and len(l_name)==2:
                    try: exec(f"torch.{l_name[1]}")
                    except:
                        try: exec(f"torch.nn.functional.{l_name[1]}")
                        except:
                            try: exec(f"torch.Tensor.{l_name[1]}")
                            except:
                                raise Exception(f"torch.{l_name[1]} neither found in torch, torch.Tensor and torch.nn.functional")
                            else: fct_name = f"torch.Tensor.{l_name[1]}"
                        else: fct_name = f"torch.nn.functional.{l_name[1]}"
                    else: fct_name = f"torch.{l_name[1]}"
                else:
                    fct_name = ".".join(l_name)

                # == else ==
                new_node = B_node(target=target,fct=fct_name)
                nodes.append(new_node)
                args_str = [v.get_value(calling_node=new_node) for v in args_Bvar]
                kwds_str = []
                for kw in expr.keywords:
                    if not (((kw.arg=="dtype" or kw.arg=="layout")
                        and isinstance(kw.value,ast.Constant)
                        and isinstance(kw.value.value,int))
                        or (kw.arg=="layout" and kw.value.value is None)):
                        kwds_str.append(f"{kw.arg} = {(handle_expr(kw.value)).get_value(new_node)}")
                all_args = ",".join(args_str + kwds_str)
                new_node.make_code(f"{fct_name}({all_args})")
                return B_var(target,node = new_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # isinstance(expr, ast.List or ast.Tuple)
    # constr = "list" or "tuple"
    def aux_handle_tuple_or_list(expr,target,constr): # tmp TODO to improve ++
        if target is None:
            target = get_fresh_var()
        new_node = B_node(target=target,fct=f"{constr} constructor")
        nodes.append(new_node)
        args_vars = [handle_expr(v) for v in expr.elts]
        args_str  = [v.get_value(new_node) for v in args_vars]
        join_s = ','.join(args_str)
        if constr=="list": c = f"[{join_s}]"
        else: c = f"({join_s})"
        new_node.make_code(c)
        return B_var(target,node=new_node)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # -- handle any expr -- return type -> B_var
    # if the expr is simple (e.g. constant or self attr) B_var.has_node == False
    # otherwise, a node (= a piece of code) is created. The optional parameter 
    # "target" imposes the name of the var created in this node
    # -> fct used in the main fct and to handle arguments given to a Call 
    # /!\ TorchScript's global constant vars must have been removed
    def handle_expr(expr,target : str = None) -> B_var :
        if isinstance(expr,ast.Constant): # never creates any node
            if isinstance(expr.value,str): return B_var(f"\"{expr.value}\"")
            else: return B_var(str(expr.value))
        elif isinstance(expr,ast.Name):
            assert(expr.id in dict_vars)
            return dict_vars[expr.id]
        elif (  isinstance(expr,ast.Attribute) # -> special constants
            and isinstance(expr.value,ast.Name)
            and expr.value.id == 'CONSTANTS' ):
            return B_var(str(memory[expr.attr]))
        elif isinstance(expr,ast.Attribute):
            return handle_attr(expr,target) # may creates one node
        elif isinstance(expr,ast.Call):
            return handle_call(expr,target) # may creates nodes for inputs + for output
        elif isinstance(expr,ast.List):
            return aux_handle_tuple_or_list(expr,target,"list")
        elif isinstance(expr,ast.Tuple):
            return aux_handle_tuple_or_list(expr,target,"tuple")
        elif isinstance(expr,ast.UnaryOp):
            assert(isinstance(expr.op,ast.USub)) # quick fix
            assert(isinstance(expr.operand,ast.Constant))
            return B_var(f"-{expr.operand.value}")
        else:
            raise Exception(f"{type(expr)} unknown")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # =========================

    # == MAIN ==
    for n in a.body:
        if isinstance(n,ast.Assign):
            # -- targets -- 
            list_tg = [] ; tg = n.targets[0]
            if isinstance(tg,ast.Name):
                list_tg = [tg.id]
                target_id = tg.id
            elif isinstance(tg,ast.Tuple) or isinstance(tg,ast.List):
                for e in tg.elts: list_tg.append(e.id)
                target_id = None
            else:
                raise Exception("error 2 : ast.Call's target neither name, tuple or list ?")

            # -- main --
            main_id  = init_targets(list_tg)
            main_var = handle_expr(n.value,main_id)
            if len(list_tg)>1:
                handle_targets(list_tg,main_var)

            if target_id is not None:
                dict_vars[target_id] = main_var

        else:
            assert(isinstance(n,ast.Return))
            ret_graph = B_graph()
            ret_graph.outputs = [handle_expr(n.value,target=None)]
            ret_graph.nodes = nodes
            return ret_graph

    raise Exception("error 4 : should have stoped with the ast.Return")

def main(nn_mod,ex_inputs,concise_name=True,show_debug=False):
    # main_mod must be a instance of torch.nn.Module
    # ex_inputs must be a tuple
    global concise, fresh_var, show_debug_msg, dict_rand
    dict_rand = {}
    concise = concise_name
    show_debug_msg = show_debug
    fresh_var = 0
    main_mod = trace_module(nn_mod, {'forward': ex_inputs}, check_trace=False)
    main_str = "self"
    main_fct = "forward"
    return open_sub_module(main_mod,main_str,main_fct,[],is_main=True)
