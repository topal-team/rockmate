# ====================================
# = Useful functions for ast objects =
# ====================================
from pgb.utils.imports import *
from pgb.utils.small_fcts import remove_prefix,remove_suffix

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

def make_ast_assign(c,prefix="",suffix=""):
    tar,right_part = c
    a = ast.Assign([ast.Name(prefix+tar+suffix)],right_part)
    return a
def make_ast_list_assign(lc,prefix="",suffix=""):
    la = [make_ast_assign(c,prefix="",suffix="") for c in lc]
    return make_ast_module(la)
def make_str_assign(c,prefix="",suffix=""):
    if c is None or c[1] is None: return ""
    return ast_to_str(make_ast_assign(c,prefix,suffix))
def make_str_list_assign(lc,prefix="",suffix=""):
    ls = [make_str_assign(c,prefix="",suffix="") for c in lc]
    return "\n".join(ls)

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

