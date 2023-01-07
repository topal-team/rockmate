from .utils import *
from . import Btools
from . import Dtools
from . import Stools
from . import Ktools
from . import graph_translator

# ==========================
# ====== OUTPUT CLASS ======
# ==========================

class all_graphs():
    def __init__(self,bg,dg,sg,kg,list_sg,list_kg,cc,list_ano_S):
        self.B_graph = bg
        self.D_graph = dg
        self.S_graph = sg
        self.K_graph = kg
        self.S_graph_list = list_sg
        self.K_graph_list = list_kg
        self.equivalent_classes = cc
        self.list_ano_S = list_ano_S

# ==========================



# ==========================
# ===== AUX FUNCTIONS ======
# ==========================

# to check if the given dict_inputs is correct
def print_inputs(model):
    s = inspect.signature(model.forward)
    p = list(s.parameters.items())
    print(f"This module has {len(p)} parameters :")
    for c in p: print(c[1])

def check_inputs(model,dict_inputs):
    s = inspect.signature(model.forward)
    p = list(s.parameters.items())
    for (inp,u) in p:
        if ((u.default is inspect._empty)
            and (not (inp in dict_inputs))):
            raise Exception(
              f"input \"{inp}\" of type {u.annotation} is missing,\n"\
              f"you can use \"print_inputs\"(model) to help you.")

# to check is the device is cuda
def print_cuda_warning_msg(things_not_on_cuda):
    l = things_not_on_cuda
    if l == []: pass
    else:
      if len(l) == 1:
        main_line = f"{l[0]}'s device is not cuda !"
      else:
        s = " and ".join(l)
        main_line = f"{s}'s devices are not cuda !"
      print(
        f"/!\\/!\\=======================================/!\\/!\\\n"\
        f"/!\\/!\\= WARNING : {main_line}\n"\
        f"/!\\/!\\=======================================/!\\/!\\\n\n"\
        f"/!\\You ask PGB to measure the time and memory used by all\n"\
        f"/!\\the computation nodes. But measuring memory can only\n"\
        f"/!\\be done with cuda, therefore model and inputs' devices\n"\
        f"/!\\should be cuda to get relevant results. You can use the \n"\
        f"/!\\parameter \"check_device_is_gpu\" to avoid this warning.\n")


# ==========================
# ===== Main function ======
# ==========================

def make_all_graphs(model,
    dict_inputs,
    verbose=False,
    impose_device=True,
    bool_bg = True , bool_dg = True ,
    bool_sg = True , bool_kg = True ,
    bool_list_sg = True , bool_list_kg = True,
    check_device_is_gpu = True):
    r"""
    this function returns an objet with attributes :
     -> .B_graph, .D_graph, .S_graph and .K_graph -> the whole module
     -> .S_graph_list and .K_graph_list -> the sequentialized module
    on which you can use :
    pgb.print_graph and pgb.print_graph_list or pgb.print_all_graphs
    """
    bool_list_sg = bool_list_sg or bool_list_kg
    bool_sg = bool_sg or bool_kg or bool_list_sg
    bool_dg = bool_dg or bool_sg
    bool_bg = bool_bg or bool_dg

    # check inputs
    ref_verbose[0] = verbose
    check_inputs(model,dict_inputs)

    # check device
    things_not_on_cuda = []
    if bool_kg and check_device_is_gpu:
        for (key,inp) in dict_inputs.items():
            if not isinstance(inp,torch.Tensor):
                raise Exception(
                    f"Sorry, all inputs should have type torch.Tensor\n"\
                    f"{key} has type {type(inp)}")
            if not inp.is_cuda:
                things_not_on_cuda.append(key)
        b = False
        for p in model.parameters():
            if not p.is_cuda: b=True
        if b: things_not_on_cuda.append("the model")
    print_cuda_warning_msg(things_not_on_cuda)
    device = get_device_and_check_all_same_device(model,dict_inputs)


    # -- the whole module --
    if bool_bg:
        bg = Btools.make_B(model,dict_inputs,
            impose_device=impose_device,device=device)
    else: bg = None
    if bool_dg: dg = Dtools.B_to_D(bg,model,dict_inputs,device=device)
    else: dg = None
    if bool_sg: sg = Stools.D_to_S(
        dg,keep_sequential=True,model=model,device=device)
    else: sg = None
    if bool_kg: kg = Ktools.S_to_K(sg,model,device=device)
    else: kg = None
    # -- sequentialized --
    if bool_list_sg:
        list_sg = Stools.cut(sg)
    else: list_sg = None
    if bool_list_kg:
        cc,list_kg,list_ano_S = graph_translator.S_list_to_K_list_eco(
            list_sg,model,device=device)
    else: list_kg = None ; cc = None ; list_ano_S = None

    return all_graphs(bg,dg,sg,kg,list_sg,list_kg,cc,list_ano_S)

# ==========================



# ==========================
# === printing functions ===
# ==========================

def print_graph(g,name=None,open=True,render_format="svg"):
    r"""To visualize D, S or K graph.
    This function creates a .gv file, and using
    graphviz's dot function builds a .pdf file.
    They are stored in "graphviz_dir" sub-directory.
    inputs:
    name (string):
        To name .gv and .pdf files.
        By default named after the type of the graph.
    open (boolean):
        To automatically open the .pdf with the default reader.
    """
    if g is None: pass
    elif isinstance(g,Dtools.D_graph):
        Dtools.print_D_graph(g,name,open,render_format)
    elif isinstance(g,Stools.S_graph):
        Stools.print_S_graph(g,name,open,render_format)
    elif isinstance(g,Ktools.K_graph):
        Ktools.print_K_graph(g,name,open,render_format)
    else: raise Exception(
        "g is neither of type D_graph, S_graph or K_graph")

def print_graph_list(gl,name=None,open=True,render_format="svg"):
    r"""The equivalent of pgb.print_graph for a list of graph.
    Generates all graphs next to each other in a single pdf.
    Note:
         Originally intented to visualize a sequentialized graph :
         i.e. one graph cut by PGB in blocks
         i.e. S_graph_list of K_graph_list
    """
    if gl is None: pass
    elif len(gl) == 0: print("Empty list, no graph to visualize")
    else:
        t = type(gl[0])
        for i in range(1,len(gl)):
            if type(gl[i]) != t: raise Exception(
              f"All graphs in the list must share the same type"\
              f"type(gl[{i}])={type(gl[i])} and type(gl[0])={t}")
        if t == Stools.S_graph:
            Stools.print_S_graph_list(gl,name,open,render_format)
        elif t == Ktools.K_graph:
            Ktools.print_K_graph_list(gl,name,open,render_format)
        else: raise Exception(
            "gl is neither a S_graph list or K_graph list")

def print_all_graphs(a,name="",open=True,render_format="svg"):
    print_graph(a.D_graph,f"{name}_D_graph",open,render_format)
    print_graph(a.S_graph,f"{name}_S_graph",open,render_format)
    print_graph(a.K_graph,f"{name}_K_graph",open,render_format)
    print_graph_list(a.S_graph_list,f"{name}_S_cut_graph",
        open,render_format)
    print_graph_list(a.K_graph_list,f"{name}_K_cut_graph",
        open,render_format)

# ==========================

