# ==========================
# This file is the root of the pk file 
# hierarchy. It contains the global vars
# and auxiliary functions. But also all
# the imports actions, even those specific
# ==========================


# ==========================
# ======== IMPORTS =========
# ==========================

import ast
import astunparse
import torch
from torch import tensor
import graphviz

# for Btools -> get jit.code
from torch.jit import trace_module

# for Ktools -> inspection
try:
    from rotor.utils import *
except:
    from torch.hub import load_state_dict_from_url
    from rotor.utils import *
from rotor.timing import *
from rotor.memory import *
min_duration = 0

# for main -> get inputs
import inspect

# -> to support different versions of AST
import sys
svi = sys.version_info
py_version = svi.major + svi.minor/10

# ==========================



# ==========================
# === LISTS OF FUNCTIONS ===
# ==========================

list_rand_fct = ["torch.randn"]
# TODO : complete this list

list_cheap_fct = ["torch.add","torch.sub","torch.mul","torch.div"]
# TODO : complete this list
list_cheap_fct.extend(["list constructor","tuple constructor"])
# because I treat them in the same way

list_view_fct = [
    "torch.adjoint","torch.Tensor.adjoint",
    "torch.as_strided","torch.Tensor.as_strided",
    "torch.Tensor.detach",
    "torch.diagonal","torch.Tensor.diagonal",
    "torch.Tensor.expand","torch.Tensor.expand_as",
    "torch.movedim","torch.Tensor.movedim",
    "torch.narrow","torch.Tensor.narrow",
    "torch.permute","torch.Tensor.permute",
    "torch.select","torch.Tensor.select",
    "torch.squeeze","torch.Tensor.squeeze",
    "torch.transpose","torch.Tensor.transpose",
    "torch.view_as_real",
    "torch.Tensor.unflatten",
    "torch.Tensor.unfold",
    "torch.unsqueeze","torch.Tensor.unsqueeze",
    "torch.Tensor.view","torch.Tensor.view_as",
    "torch.unbind","torch.Tensor.unbind",
    "torch.split","torch.Tensor.split",
    "torch.hsplit","torch.Tensor.hsplit",
    "torch.vsplit","torch.Tensor.vsplit",
    "torch.tensor_split","torch.Tensor.tensor_split",
    "torch.split_with_sizes","torch.Tensor.split_with_sizes",
    "torch.swapaxes","torch.Tensor.swapaxes",
    "torch.swapdims","torch.Tensor.swapdims",
    "torch.chunk","torch.Tensor.chunk",
    "torch.Tensor.values","torch.Tensor.indices",
    ]
# list imported from https://pytorch.org/docs/stable/tensor_view.html

# ==========================



# ==========================
# = AUX FUNCTIONS FOR AST ==
# ==========================

def ast_to_str(ast_code):
    #return ast.unparse(ast.fix_missing_locations(ast_code))
    return astunparse.unparse(ast_code)

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



# ==========================
# == TO TOPO SORT GRAPHS ===
# ==========================

def sort_based_on_req(n):
    # n can be any type of node (B, D, S, K)
    # we just need attribut req
    dict_done = {}
    nodes = []
    def visit(n):
        if n not in dict_done:
            dict_done[n]=False
            for sub_n in n.req:
                visit(sub_n)
            dict_done[n]=True
            nodes.append(n)
        elif not dict_done[n]:
            raise Exception(
                "Cycle in the graph. How could this happened ??")
    visit(n)
    return nodes

# ==========================



# ==========================
# ======== FWD INFO ========
# ==========================

class FWD_info(): # everything needed to randomly regenerate a var
    def __init__(self):
        self.dtype = None
        self.ttype = None # target_type
        self.tsize = None # target_size
        self.sub_info = None # if ttype = list or tuple
        self.requires_grad = None # if Tensor or Size

def generate_val(info,device):
    tt = info.ttype
    if tt==torch.Size:
        return info.tsize
    elif tt==torch.Tensor:
        return torch.ones(info.tsize,
            dtype=info.dtype,
            requires_grad=info.requires_grad,
            device=device)
    else:
        assert(tt==list or tt==tuple)
        x = [generate_val(sub_info,device) for sub_info in info.sub_info]
        return tt(x)

# ==========================



# ==========================
# == SAFELY USE GRAPHVIZ ===
# ==========================

def graph_render(dot,open,graph_type):
    try:
      dot.render(directory="graphviz_dir",quiet=True,view=open)
    except:
      print("Warning : issue with graphviz to print {graph_type}_graph, "\
            "probably because Graphviz isn't installed on the computer "\
            "(the software, not the python module). Normally the .gv "\
            "has been generated, but not the .pdf",
            file = sys.stderr)

# ==========================
