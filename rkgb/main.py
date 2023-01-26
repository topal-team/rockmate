from rkgb.utils import *
from rkgb import Btools
from rkgb import Dtools
from rkgb import Stools
from rkgb import Ktools
from rkgb import Atools
import inspect


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


def make_inputs(model,model_inputs,model_kwargs):
    # 1) Build dict_inputs
    # -- load params list --
    sign = inspect.signature(model.forward)
    params = list(sign.parameters.items())
    # -- build model_kwargs --
    if model_kwargs is None: model_kwargs = dict()
    elif not isinstance(model_kwargs,dict): raise Exception(
        f"model_kwargs must be a dict not {type(model_kwargs)}")
    # -- positional params --
    not_kw_params = [
        p[0] for p in params
        if p[0] not in model_kwargs]
    pos_params = [
        p[0] for p in params
        if (p[1].default is inspect._empty
        and p[0] not in model_kwargs)]
    # -- build positional inputs --
    if isinstance(model_inputs,dict):
        dict_inputs = model_inputs.copy()
        st_given = set(dict_inputs.keys())
        st_asked = set(pos_params)
        st_missing = st_asked - st_given
        nb_missing = len(st_missing)
        if nb_missing>0: raise Exception(
            f"Missing {nb_missing} inputs for the model: {st_missing}")
    else:
        if (isinstance(model_inputs,set)
        or  isinstance(model_inputs,list)
        or  isinstance(model_inputs,tuple)):
            inputs = list(model_inputs)
        else:
            inputs = [model_inputs]
        nb_given = len(inputs)
        nb_asked_pos = len(pos_params)
        nb_asked_tot = len(not_kw_params)
        if nb_given < nb_asked_pos: raise Exception(
            f"To few values given in model_inputs "\
            f"({nb_asked_pos - nb_given} missing).\n"\
            f"You can use \"rkgb.print_inputs(<model>)\" to help you.")
        if nb_given > nb_asked_tot: raise Exception(
            f"To much values given in model_inputs "\
            f"({nb_given - nb_asked_tot} too many, including kwargs).\n"\
            f"You can use \"rkgb.print_inputs(<model>)\" to help you.")
        dict_inputs = dict(zip(not_kw_params,inputs))

    dict_inputs.update(model_kwargs)

    # 2) check types
    """ # -> might fail
    for (name,value) in dict_inputs.items():
        info = sign.parameters[name]
        if not info.annotation is inspect._empty:
            if not isinstance(value,info.annotation): raise Exception(
                f"According to model's signature, {name} argument "\
                f"is supposed to be of type {info.annotation}, but "\
                f"the given value has type {type(value)}")
    """
    return dict_inputs


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
        f"/!\\You ask rk-GB to measure the time and memory used by all\n"\
        f"/!\\the computation nodes. But measuring memory can only\n"\
        f"/!\\be done with cuda, therefore model and inputs' devices\n"\
        f"/!\\should be cuda to get relevant results. You can use the \n"\
        f"/!\\parameter \"check_device_is_gpu\" to avoid this warning.\n")


# ==========================
# ===== Main function ======
# ==========================

def make_all_graphs(model,
    model_inputs,
    model_kwargs=None,
    verbose=False,
    impose_device=True,
    bool_bg = True , bool_dg = True ,
    bool_sg = True , bool_kg = True ,
    bool_list_sg = True , bool_list_kg = True,
    check_device_is_gpu = True):
    r"""
    ***** this function returns an objet with attributes *****
     -> .B_graph, .D_graph, .S_graph and .K_graph of the whole module
     -> .S_graph_list and .K_graph_list of the sequentialized module
    on which you can use :
    rkgb.print_graph and rkgb.print_graph_list or rkgb.print_all_graphs

    ***** args *****
     -> model must be a torch.nn.Module
    /!\ Most of the time errors occur because of jit.trace /!\
    /!\ so 'model' must be compatible with jit.trace       /!\
    -> model_inputs :
        args of 'model', it can either be a simple
        variable or an iterable of variables.
    -> model_kwargs :
        optional dictionnary in case you want to
        call 'model' with kwargs
    """
    bool_list_sg = bool_list_sg or bool_list_kg
    bool_sg = bool_sg or bool_kg or bool_list_sg
    bool_dg = bool_dg or bool_sg
    bool_bg = bool_bg or bool_dg

    # check inputs
    global_vars.ref_verbose[0] = verbose
    dict_inputs = make_inputs(model,model_inputs,model_kwargs)

    # check device
    things_not_on_cuda = []
    if bool_kg and check_device_is_gpu:
        for (key,inp) in dict_inputs.items():
            if not isinstance(inp,torch.Tensor):
                print(f"Warning : {key} has type {type(inp)}")
            elif not inp.is_cuda:
                things_not_on_cuda.append(key)
        b = False
        for p in model.parameters():
            if not p.is_cuda: b=True
        if b: things_not_on_cuda.append("the model")
    print_cuda_warning_msg(things_not_on_cuda)
    device = small_fcts.get_device_and_check_all_same_device(
        model,dict_inputs)

    # -- protect original module from impact on eval mode --
    # -> save running stats
    saved_running_stats = dict()
    for m in model.modules():
        for batch_fct in global_vars.list_batch_fct:
            if isinstance(m,batch_fct):
                r_mean = m.running_mean
                r_var  = m.running_var
                saved_running_stats[m] = (
                    r_mean.clone() if r_mean is not None else None,
                    r_var.clone() if r_var is not None else None,
                )

    # -- the whole module --
    if bool_bg:
        bg = Btools.make_B(model,dict_inputs,
            impose_device=impose_device,device=device)
    else: bg = None
    if bool_dg: dg = Dtools.B_to_D(bg,model,dict_inputs,device=device)
    else: dg = None
    if bool_sg: sg = Stools.D_to_S(
        dg,model=model,device=device)
    else: sg = None
    if bool_kg: kg = Ktools.S_to_K(sg,model,device=device)
    else: kg = None
    # -- sequentialized --
    if bool_list_sg:
        list_sg = Stools.cut(sg)
    else: list_sg = None
    if bool_list_kg:
        cc,list_kg,list_ano_S = Atools.S_list_to_K_list_eco(
            list_sg,model,device=device)
    else: list_kg = None ; cc = None ; list_ano_S = None

    # -- restore running_stats --
    for (m,(r_mean,r_var)) in saved_running_stats.items():
        m.running_mean = r_mean
        m.running_var  = r_var


    return all_graphs(bg,dg,sg,kg,list_sg,list_kg,cc,list_ano_S)

# ==========================



# ==========================
# === printing functions ===
# ==========================

def print_graph(g,name=None,open=True,render_format="svg"):
    r"""To visualize D, S or K graph.
    This function creates a .gv file, and using
    Graphviz's dot function builds a .pdf file.
    They are stored in "graphviz_dir" sub-directory.
    inputs:
    name (string):
        To name .gv and .pdf files.
        By default named after the type of the graph.
    render_format (string):
        Render format wanted for the output file
    open (boolean):
        To automatically open the file with the default reader.
    """
    if g is None: pass
    elif isinstance(g,Dtools.D_graph):
        Dtools.print_D_graph(g,name,open,render_format)
    elif isinstance(g,Stools.S_graph):
        Stools.print_S_graph(g,name,open,render_format)
    elif isinstance(g,Ktools.K_graph):
        Ktools.print_K_graph(g,name,open,render_format)
    else: raise Exception(
        "The graph given is neither of type D_graph, S_graph nor K_graph")

def print_graph_list(gl,name=None,open=True,render_format="svg"):
    r"""The equivalent of rkgb.print_graph for a list of graph.
    Generates all graphs next to each other in a single pdf.
    Note:
         Originally intented to visualize a sequentialized graph :
         i.e. one graph cut by rkgb in blocks
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
            "The list given is neither a S_graph list nor K_graph list")

def print_all_graphs(a,name="",open=True,render_format="svg"):
    print_graph(a.D_graph,f"{name}_D_graph",open,render_format)
    print_graph(a.S_graph,f"{name}_S_graph",open,render_format)
    print_graph(a.K_graph,f"{name}_K_graph",open,render_format)
    print_graph_list(a.S_graph_list,f"{name}_seq_S_graph",
        open,render_format)
    print_graph_list(a.K_graph_list,f"{name}_seq_K_graph",
        open,render_format)

# ==========================

# ===================
# == TO TEST rk-GB ==
# ===================

def test_rkgb(module, model_inputs, **kwargs):
    rkgb_res = make_all_graphs(module, model_inputs, **kwargs)
    list_kg = rkgb_res.K_graph_list
    kg = rkgb_res.K_graph
    print("Generated all the graphs !\n")
    print(f"Equiv classes are : {rkgb_res.equivalent_classes}")
    print(
        f"So we have only {len(rkgb_res.equivalent_classes)} "
        f"blocks to solve ILP on, instead of {len(list_kg)}\n"
    )
    print("CONCERNING K_graph_list :")
    list_nb_kcn = [len(kg.list_kcn) for kg in list_kg]
    list_nb_kdn = [len(kg.list_kdn) for kg in list_kg]
    tot_nb_kcn = sum(list_nb_kcn)
    tot_nb_kdn = sum(list_nb_kdn)
    str_list_nb_kcn = "+".join(str(i) for i in list_nb_kcn)
    str_list_nb_kdn = "+".join(str(i) for i in list_nb_kdn)
    print(
        f"{len(list_kg)} K_graphs in seq, with :\n"
        f"{str_list_nb_kcn} = {tot_nb_kcn} Comp nodes\n"
        f"{str_list_nb_kdn} = {tot_nb_kdn} Data nodes\n"
        f"=> total of {tot_nb_kcn + tot_nb_kdn} nodes\n"
    )
    print("CONCERNING phantoms impossible to restore :")
    nb_ips = 0
    for kcn in kg.list_kcn:
        deps_ips = kcn.deps_impossible_to_restore
        if len(deps_ips) != 0:
            nb_ips += 1
            print(
                f"{kcn.main_target}'s phantoms must be "
                f"protected, because deps_impossible_to_restore :"
            )
            for kdn, ph_name in deps_ips:
                print(f"deps on {kdn} through {ph_name}")
    print(f"Total nb of special phantoms :  {nb_ips}")
    return rkgb_res