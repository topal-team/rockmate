# ==========================
# This file is the root of the pgb file 
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
import numpy as np
from torch import tensor
#from .Btools import B_node 
#from .Stools import S_node 
#from .Ktools import K_node 
import graphviz

# == rotor == -> for inspection in Ktools.py
import rotor.timing # -> use .make_timer
import rotor.memory # -> use .MeasureMemory
from rotor.memory import MemSize
from rotor.inspection import tensorMsize
minus_mem = lambda m : MemSize(- m.v)

# for main.py -> get inputs
import inspect

# -> to support different versions of AST
import sys
svi = sys.version_info
py_version = svi.major + svi.minor/10

# ==========================



# ==========================
# ====== GLOBAL VARS =======
# ==========================

time_min_duration = 0
time_min_repeat = 5

# -> print debug messages
ref_verbose = [False]
def print_debug(*args, **kwargs):
    if ref_verbose[0]:
        print(*args, **kwargs)

# -> to raise exceptions with lambda functions
def raise_(s):
    raise Exception(s)

# -> device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device_and_check_all_same_device(
        model,dict_inputs,without_inp=False):
    d = None
    k = None
    print_err = lambda k1,d1,k2,d2 : raise_(
      f"Carelessness ! All inputs and parameters of the model\n"\
      f"must share the same device. Here {k1}'s device is {d1}\n"\
      f"and {k2}'s device is {d2}.")

    if not isinstance(dict_inputs,dict):
        dict_inputs = dict(enumerate(dict_inputs))

    for (key,inp) in dict_inputs.items():
        if isinstance(inp,torch.Tensor):
            if d is None: d = inp.device ; k = f"input {key}"
            else:
                if d != inp.device:
                    print_err(f"input {key}",inp.device,k,d)
    i = -1
    for p in model.parameters():
        i += 1
        if d is None: d = p.device ; k = f"{i}-th parameter"
        else:
            if d != p.device:
                print_err(f"{i}-th parameter",p.device,k,d)
    if d: return d
    elif without_inp: return get_device()
    else: raise Exception(
        "Sorry, at least one input or one parameter should be a tensor.")



# -> acceptance rate for two time measures to be declared equal
ref_reasonable_rate = [0.4]
def change_reasonable_rate(x):
    assert(0<=x)
    ref_reasonable_rate[0] = x

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
# === SMALL USEFULL FCT ====
# ==========================
def check_attr(o1,o2,list_attr,raise_exception=False):
    for s in list_attr:
        if getattr(o1,s) != getattr(o2,s):
            if raise_exception:
                raise Exception(f"attr diff {s}")
            return False
    return True
def vdir(c):
    return [s for s in dir(c) if not s.startswith("__")]
# ==========================



# ==========================
# = AUX FUNCTIONS FOR AST ==
# ==========================

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text
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
# ==== TOPO SORT GRAPHS ====
# ==========================

def get_tar_attr(n):
    return "target" if hasattr(n,"target") else "main_target"

def get_num(n): # can be used on B, D, S or K
    mt = getattr(n,get_tar_attr(n))
    try:
        return int(mt.split('_')[2])
    except:
        print(f"Exception pour mt : {mt}")
        return (-1)

def sort_based_on_req(origin_node): # used on B, S and K
    # /!\ origin_node is the root of .req relation 
    # /!\ => the last node to be computed !!

    # To be compatible with different names for attributes
    tar = get_tar_attr(origin_node)
    req = "req" if hasattr(origin_node,"req") else "req_real"

    # Compute incomming degree
    degree = {}
    def count_edges(n):
        for sub_n in getattr(n,req):
            if sub_n not in degree:
                d = 0
                count_edges(sub_n)
            else:
                d = degree[sub_n]
            degree[sub_n] = d+1
    count_edges(origin_node)

    # Explore nodes by increasing lexi-order of their n.target
    # BUT a node is explored iff all its used_by are explored => toposort
    sorted_list = []
    to_explore = set([origin_node])
    while to_explore: # not empty
        n = max(to_explore,key=lambda n : get_num(n))
        to_explore.discard(n)
        sorted_list.append(n)
        for sub_n in getattr(n,req):
            if sub_n in sorted_list:
                raise Exception("Cycle in the graph => no toposort")
            d = degree[sub_n]
            if d == 1:
                to_explore.add(sub_n)
            else:
                degree[sub_n] = d-1

    # return from first to last
    return sorted_list[::-1]

# ==========================



# ==========================
# ======= CUT GRAPHS =======
# ==========================

def cut_based_on_req(g): # used on D and S
    # returns the list of all 1-separator of the graph.
    to_be_visited = [g.output_node]
    seen = set([g.output_node])
    dict_nb_usages = dict([(m , len(m.used_by)) for m in g.nodes])
    separators = []
    while to_be_visited!=[]:
        n = to_be_visited.pop()
        seen.remove(n)
        if seen==set():
            separators.append(n)
        for req_n in n.req:
            seen.add(req_n)
            dict_nb_usages[req_n]-=1
            if dict_nb_usages[req_n]==0:
                to_be_visited.append(req_n)
    separators.reverse()
    return separators

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
        self.memsize = None # done much later
    def __eq__(self,i2):
        d = vdir(self)
        for s in d:
            if getattr(self,s) != getattr(i2,s): return False
        return True


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
      print(f"Warning : issue with graphviz to print {graph_type}_graph, "\
            f"probably because Graphviz isn't installed on the computer "\
            f"(the software, not the python module). Normally the .gv "\
            f"has been generated, but not the .pdf",
            file = sys.stderr)

# ==========================
