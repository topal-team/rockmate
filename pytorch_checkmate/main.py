from .root import *
from . import Btools
from . import Dtools
from . import Stools
from . import Ktools

def print_inputs(nn_mod):
    s = inspect.signature(nn_mod.forward)
    p = list(s.parameters.items())
    print(f"This module has {len(p)} parameters :")
    for c in p: print(c[1])

def check_inputs(nn_mod,dict_inputs):
    s = inspect.signature(nn_mod.forward)
    p = list(s.parameters.items())
    for (inp,u) in p:
        if ((u.default is inspect._empty)
            and (not (inp in dict_inputs))):
            raise Exception(
              f"input \"{inp}\" of type {u.annotation} is missing,\n"\
              f"you can use \"print_inputs\"(nn_mod) to help you.")

class all_graphs():
    def __init__(self,bg,dg,sg,kg,list_sg,list_kg):
        self.B_graph = bg
        self.D_graph = dg
        self.S_graph = sg
        self.K_graph = kg
        self.S_graph_list = list_sg
        self.K_graph_list = list_kg

def make_all_graphs(nn_mod,
    dict_inputs,
    show_debug=False,
    impose_device=False,
    D_device=None,
    K_device=None):
    """
    this function returns an objet with attributes :
     -> .B_graph, .D_graph, .S_graph and .K_graph -> the whole module
     -> .S_graph_list and .K_graph_list -> the sequentialized module
    |--> on which you can use :
      - pk.Dtools.print_D_graph
      - pk.Stools.print_S_graph
      - pk.Ktools.print_K_graph
      - pk.Stools.print_S_graph_list
      - pk.Ktools.print_K_graph_list
    """
    ref_print_debug[0] = show_debug
    check_inputs(nn_mod,dict_inputs)
    # -- the whole module --
    bg = Btools.make_B(nn_mod,dict_inputs,
                       impose_device=impose_device)
    dg = Dtools.B_to_D(bg,nn_mod,dict_inputs,D_device=D_device)
    sg = Stools.D_to_S(dg,keep_sequential=True)
    kg = Ktools.S_to_K(sg,nn_mod,K_device=K_device)
    # -- sequentialized --
    list_sg = Stools.cut(sg)
    list_kg = Ktools.S_list_to_K_list(list_sg,nn_mod)
    return all_graphs(bg,dg,sg,kg,list_sg,list_kg)

def print_all_graphs(a,name,open):
    Dtools.print_D_graph(a.D_graph,name=f"{name}_D_graph",open=open)
    Stools.print_S_graph(a.S_graph,name=f"{name}_S_graph",open=open)
    Ktools.print_K_graph(a.K_graph,name=f"{name}_K_graph",open=open)
    Stools.print_S_graph_list(a.S_graph_list,name=f"{name}_S_cut_graph",open=open)
    Ktools.print_K_graph_list(a.K_graph_list,name=f"{name}_K_cut_graph",open=open)

