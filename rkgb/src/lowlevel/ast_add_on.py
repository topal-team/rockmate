# ====================================
# = Useful functions for ast objects =
# ====================================

import sys
import inspect
import ast
import astunparse
from src.lowlevel import constants
from src.utils import utils

sys_info = sys.version_info
python_version = sys_info.major + sys_info.minor/10

# -- general --

def ast_to_str(code_ast):
    #return ast.unparse(ast.fix_missing_locations(code_ast))
    code = astunparse.unparse(code_ast)
    return utils.remove_prefix(utils.remove_suffix(code,"\n"),"\n")

def make_ast_constant(v):
    x = ast.Constant(v)
    setattr(x,"kind",None)
    return x
    #for astunparse compatibility with all versions of AST

def make_ast_module(l):
    try:    return ast.Module(l,[])
    except: return ast.Module(l)

def open_all_nested_attributes(value):
    list_attributes = []
    while isinstance(value,ast.Attribute):
        list_attributes.append(value.attr)
        value = value.value
    list_attributes.reverse()
    return value,list_attributes

def make_ast_attribute_from_list(parent_ast,list_attributes):
    new_ast = parent_ast
    for attr in list_attributes:
        if attr.isdigit():
            new_ast = ast.Subscript(new_ast,
                slice=make_ast_constant(int(attr)))
        else:
            new_ast = ast.Attribute(new_ast,attr)
    return new_ast



# -- make_assign --
# optional parameter **force_special_kwargs** : 
#   False (default): nothing is special
#   Otherwise, it force some specif kwargs for some specific 
#   functions, for instance torch.batch_norm shouldn't 
#   retain stats when a code is recomputed. 
#   It relies on a dict of type :
#   -> <fct_name -> (arg_name,arg_value) list to inforce>
#   if force_special_kwargs=True then you use :
#   -> constants.default_forced_kwargs
#   you can also do force_special_kwargs=<your dict>

def make_ast_assign(
    c,prefix="",suffix="",force_special_kwargs=False
):
    tar,right_part = c
    assert(right_part is not None)
    if force_special_kwargs and isinstance(right_part,ast.Call):
        # nothing done inplace, so we don't impact the code
        dict_forced_kwargs = (
            constants.default_forced_kwargs 
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
    if python_version >= 3.8:
        return isinstance(v,ast.Constant)
    else:
        answer = type(v) in [
            ast.Num,ast.Str,ast.Bytes,
            ast.NameConstant]
        if answer:
            if isinstance(v,ast.Num):
                setattr(v,"value",v.n)
            elif isinstance(v,ast.Str) or isinstance(v,ast.Bytes):
                setattr(v,"value",v.s)
        return answer

# ==========================

def substitute(main_code,sub_id,sub_code):
    r"""
    Substitute ast.Name(sub_id) in main_code by sub_code
    """
    if isinstance(main_code,ast.Name):
        if main_code.id == sub_id: return sub_code
        else: return main_code
    elif isinstance(main_code,ast.AST):
        for attr in main_code._fields:
            if attr != "ctx":
                old_val = getattr(main_code,attr)
                new_val = substitute(old_val,sub_id,sub_code)
                if not new_val is old_val:
                    setattr(main_code,attr,new_val)
    elif isinstance(main_code,list):
        return [
            substitute(old_val,sub_id,sub_code)
            for old_val in main_code]
    return main_code
