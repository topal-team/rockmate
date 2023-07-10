# ====================================
# = Useful functions for ast objects =
# ====================================
import inspect
from rkgb.utils.imports import *
from rkgb.utils.global_vars import default_forced_kwargs 
from rkgb.utils.small_fcts import remove_prefix,remove_suffix

# -- general --

def ast_to_str(ast_code):
    #return ast.unparse(ast.fix_missing_locations(ast_code))
    code = astunparse.unparse(ast_code)
    return remove_prefix(remove_suffix(code,"\n"),"\n")

def open_attr_until_name(v):
    l_name = []
    while isinstance(v,ast.Attribute):
        l_name.append(v.attr)
        v = v.value
    l_name.append(v.id)
    l_name.reverse()
    return l_name

def make_ast_constant(v):
    x = ast.Constant(v)
    setattr(x,"kind",None)
    return x
    #for astunparse compatibility with all versions of AST

def make_ast_module(l):
    try:    return ast.Module(l,[])
    except: return ast.Module(l)


# -- make_assign --
# optional parameter **force_special_kwargs** : 
#   by default its False, it means you don't do anything
#   otherwise it relies on a dict of type :
#   -> <fct_name -> (arg_name,arg_value) list to inforce>
#   if force_special_kwargs=True then you use :
#   -> global_vars.default_forced_kwargs
#   otherwise you can do force_special_kwargs=<your dict>

def make_ast_assign(
    c,prefix="",suffix="",force_special_kwargs=False
):
    tar,right_part = c
    assert(right_part is not None)
    if force_special_kwargs and isinstance(right_part,ast.Call):
        # nothing done inplace, so we don't impact the code
        dict_forced_kwargs = (
            default_forced_kwargs 
            if force_special_kwargs is True
            else force_special_kwargs)
        a = right_part
        fct_name = a.func.id
        if fct_name in dict_forced_kwargs:
            attrs_to_inforce = dict_forced_kwargs[fct_name]
            for arg_name,arg_pos,arg_value in attrs_to_inforce:
                arg_value_ast = make_ast_constant(arg_value)
                args_ast = list(a.args)
                kws_ast = list(a.keywords)
                found = False
                for i,kw_ast in enumerate(kws_ast):
                    if kw_ast.arg == arg_name:
                        kws_ast[i] = ast.keyword(arg_name,arg_value_ast)
                        found = True
                        break
                if not found:
                    fct = eval(fct_name)
                    try:
                        sign = inspect.signature(fct)
                        params = list(sign.parameters.items())
                        for i,(p_name,_) in enumerate(params):
                            if p_name == arg_name:
                                args_ast[i] = arg_value_ast
                                found = True
                                break
                        if not found:
                            kws_ast.append(
                                ast.keyword(arg_name,arg_value_ast)
                            )
                    except:
                        args_ast[arg_pos] = arg_value_ast
                a = ast.Call(a.func,args_ast,kws_ast)
        right_part = a
    a = ast.Assign([ast.Name(prefix+tar+suffix)],right_part)
    return a

def make_ast_list_assign(
    lc,prefix="",suffix="",force_special_kwargs=False
):
    la = [make_ast_assign(c,prefix="",suffix="",
        force_special_kwargs=force_special_kwargs) for c in lc]
    return make_ast_module(la)

def make_str_assign(
    c,prefix="",suffix="",force_special_kwargs=False
):
    if c is None or c[1] is None: return ""
    return ast_to_str(make_ast_assign(c,prefix,suffix,
        force_special_kwargs=force_special_kwargs))

def make_str_list_assign(
    lc,prefix="",suffix="",force_special_kwargs=False
):
    ls = [make_str_assign(c,prefix="",suffix="",
        force_special_kwargs=force_special_kwargs) for c in lc]
    return "\n".join(ls)


# -- is_constant --
# -> older version compatibility

def is_constant(v):
    if py_version >= 3.8:
        return isinstance(v,ast.Constant)
    else:
        rep = type(v) in [
            ast.Num,ast.Str,ast.Bytes,
            ast.NameConstant]
        if rep:
            if isinstance(v,ast.Num):
                setattr(v,"value",v.n)
            elif isinstance(v,ast.Str) or isinstance(v,ast.Bytes):
                setattr(v,"value",v.s)
        return rep

# ==========================

