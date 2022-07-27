from .utils import *
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

def print_all_graphs(a,name,open):
    Dtools.print_D_graph(a.D_graph,name=f"{name}_D_graph",open=open)
    Stools.print_S_graph(a.S_graph,name=f"{name}_S_graph",open=open)
    Ktools.print_K_graph(a.K_graph,name=f"{name}_K_graph",open=open)
    Stools.print_S_graph_list(a.S_graph_list,name=f"{name}_S_cut_graph",open=open)
    Ktools.print_K_graph_list(a.K_graph_list,name=f"{name}_K_cut_graph",open=open)


# ==========================
# ===== Main function ======
# ==========================

def make_all_graphs(nn_mod,
    dict_inputs,
    verbose=False,
    impose_device=True, D_device=None, K_device=None,
    bool_bg = True , bool_dg = True ,
    bool_sg = True , bool_kg = True ,
    bool_list_sg = True , bool_list_kg = True):
    r"""
    this function returns an objet with attributes :
     -> .B_graph, .D_graph, .S_graph and .K_graph -> the whole module
     -> .S_graph_list and .K_graph_list -> the sequentialized module
    |--> on which you can use :
      - pgb.Dtools.print_D_graph
      - pgb.Stools.print_S_graph
      - pgb.Ktools.print_K_graph
      - pgb.Stools.print_S_graph_list
      - pgb.Ktools.print_K_graph_list
    """
    bool_list_sg = bool_list_sg or bool_list_kg
    bool_sg = bool_sg or bool_kg or bool_list_sg
    bool_dg = bool_dg or bool_sg
    bool_bg = bool_bg or bool_dg

    ref_verbose[0] = verbose
    check_inputs(nn_mod,dict_inputs)
    # -- the whole module --
    if bool_bg:
        bg = Btools.make_B(nn_mod,dict_inputs,
                        impose_device=impose_device)
    else: bg = None
    if bool_dg: dg = Dtools.B_to_D(bg,nn_mod,dict_inputs,D_device=D_device)
    else: dg = None
    if bool_sg: sg = Stools.D_to_S(dg,keep_sequential=True)
    else: sg = None
    if bool_kg: kg = Ktools.S_to_K(sg,nn_mod,K_device=K_device)
    else: kg = None
    # -- sequentialized --
    if bool_list_sg:
        list_sg = Stools.cut(sg)
    else: list_sg = None
    if bool_list_kg:
        list_kg = Ktools.S_list_to_K_list(list_sg,nn_mod)
    else: list_kg = None

    return all_graphs(bg,dg,sg,kg,list_sg,list_kg)

# ==========================

