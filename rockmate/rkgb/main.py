from .utils import *
from . import Btools
from . import Dtools
from . import Stools
from . import Ktools
from . import Atools_for_S_and_K
from . import Ptools
from . import Htools
import inspect
import time

# ==========================
# ====== OUTPUT CLASS ======
# ==========================

class rkGB_res():
    def __init__(self,bg,dg,sg,kg,ps,hc,list_sg,list_kg,cc,list_ano_S):
        self.B_graph = bg
        self.D_graph = dg
        self.S_graph = sg
        self.K_graph = kg
        self.P_structure = ps
        self.H_cluster = hc
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
    sign = inspect.signature(model.forward)
    params = list(sign.parameters.items())

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
    wanted_graphs = {"B","D","S","K","P","H","Sl","Kl"},
    partitioners = [
        Ptools.Partitioner(),
        Ptools.Partitioner_bottom_to_top_1(),
        Ptools.Partitioner_bottom_to_top_2(),
        Ptools.Partitioner_seq()
    ],
    verbose=False,
    impose_device=True,
    check_device_is_gpu = True,
    print_time_rkgb=False):
    r"""
    ***** this function returns an objet with attributes *****
     -> .B_graph, .D_graph, .S_graph and .K_graph of the whole module
     -> .S_graph_list and .K_graph_list of the sequentialized module
     -> .P_graph, .H_graph
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
        optional dictionary in case you want to
        call 'model' with kwargs
    """
    bool_list_kg = "Kl" in wanted_graphs
    bool_list_sg = ("Sl" in wanted_graphs) or bool_list_kg
    bool_hg = "H" in wanted_graphs
    bool_pg = ("P" in wanted_graphs) or bool_hg
    bool_kg = ("K" in wanted_graphs) or bool_hg
    bool_sg = ("S" in wanted_graphs) or bool_kg or bool_list_sg or bool_pg
    bool_dg = ("D" in wanted_graphs) or bool_sg
    bool_bg = ("B" in wanted_graphs) or bool_dg

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

    # -- measure time in each part --
    last_time = time.time()
    def print_time(where):
        nonlocal last_time
        if print_time_rkgb:
            print(f"Time passed in {where} : {time.time()-last_time}")
            last_time = time.time()


    # ============
    # === CORE ===
    # -- whole module --
    bg = Btools.make_B(model,dict_inputs,impose_device=impose_device,device=device) if bool_bg else None
    print_time("make_B")
    dg = Dtools.B_to_D(bg,model,dict_inputs,device=device) if bool_dg else None
    print_time("B_to_D")
    sg = Stools.D_to_S(dg,model=model,device=device) if bool_sg else None
    print_time("D_to_S")
    kg = Ktools.S_to_K(sg,model,device=device) if bool_kg else None
    print_time("S_to_K")
    # -- sequentialized --
    list_sg = Stools.cut(sg) if bool_list_sg else None
    print_time("S_cut")
    if bool_list_kg:
        cc,list_kg,list_ano_S = Atools_for_S_and_K.S_list_to_K_list_eco(
            list_sg,model,device=device)
    else: list_kg = None ; cc = None ; list_ano_S = None
    print_time("S_list_to_K_list via Atools")
    # -- hierarchical --
    ps = Ptools.S_to_P(sg,model,partitioners) if bool_pg else None
    print_time("S_to_P")
    hc = Htools.P_and_K_to_H(ps,kg) if bool_hg else None
    print_time("P_and_K_to_H")

    # -- restore running_stats --
    for (m,(r_mean,r_var)) in saved_running_stats.items():
        m.running_mean = r_mean
        m.running_var  = r_var

    return rkGB_res(bg,dg,sg,kg,ps,hc,list_sg,list_kg,cc,list_ano_S)

# ==========================



# ==========================
# === printing functions ===
# ==========================

def RK_print(*args,
        name=None,
        open=True,
        render_format="svg",
        **kwargs):
    r"""Overwrite python default print function,
    Render rk-GB graphs using Graphviz.
    - Given a rk-GB graph, this function creates a .gv file, 
      then external Graphviz's dot tool renders it, as a .pdf or .svg file.
      The result is stored in "graphviz_dir" sub-directory.
    - Given a list of rk-GB graphs, render all graphs next to each other in a single file.
    - Given a `rkGB_res`, render all the graphs in separate files.
    - Given a cluster, render all the possible partitioning in separate files.
    - Given a P_structure, render main_cluster.
    - For any other object, call python default print.

    Note: /!\ You need external Graphviz tool to generate the pdf/svg /!\
    -> On Ubuntu : sudo apt-get install graphviz

    kwargs:
        - name : str | list[str] | tuple[str] = None:
            To name .gv and .pdf file(s).
            By default named after the type of the graph.
        - render_format : str = "svg":
            Render format wanted for the generated file
        - open : bool = True:
            To automatically open the file with the default reader.
    """

    # === Names ===
    except_msg = (
        "Unsupported type for kwarg `name`.\n"\
        "Can be None, a string, a list or tuple of strings")
    if name is None:
        names = []
    elif isinstance(name,str):
        names = [name]
    elif isinstance(name,list) or isinstance(name,tuple):
        names = list(name)
        for s in name:
            if not isinstance(s,str):
                raise Exception(except_msg)
    else:
        raise Exception(except_msg)
    def get_name(name):
        if name is None:
            if names == []:
                return None
            else:
                n = names.pop(0)
                return n
                # return names.pop(0)
        else: return name
    # ===============

    graphs_to_render = []
    filtered_args = []
    def process_arg(arg,to_render,indent=0,pre_msg="",post_msg="",name=None):
        msg = pre_msg + " "*indent
        if isinstance(arg,Btools.B_graph):
            msg += f"B_graph cannot be rendered, just raw edges"
        elif isinstance(arg,Dtools.D_graph):
            msg += Dtools.aux_print_D_graph_message(arg)
            name = Dtools.aux_print_D_graph_name(arg,get_name(name))
            to_render.append((name,arg,Dtools.print_D_graph))
        elif isinstance(arg,Stools.S_graph):
            msg += Stools.aux_print_S_graph_message(arg)
            name = Stools.aux_print_S_graph_name(arg,get_name(name))
            to_render.append((name,arg,Stools.print_S_graph))
        elif isinstance(arg,Ktools.K_graph):
            msg += Ktools.aux_print_K_graph_message(arg)
            name = Ktools.aux_print_K_graph_name(arg,get_name(name))
            to_render.append((name,arg,Ktools.print_K_graph))
        elif isinstance(arg,Ptools.P_graph):
            msg += Ptools.aux_print_P_graph_message(arg)
            name = Ptools.aux_print_P_graph_name(arg,get_name(name))
            to_render.append((name,arg,Ptools.print_P_graph))
        elif isinstance(arg,Htools.H_graph):
            msg += Htools.aux_print_H_graph_message(arg)
            name = Htools.aux_print_H_graph_name(arg,get_name(name))
            to_render.append((name,arg,Htools.print_H_graph))
        elif isinstance(arg,Stools.S_graph_list):
            msg += Stools.aux_print_S_graph_list_message(arg)
            name = Stools.aux_print_S_graph_list_name(arg,get_name(name))
            to_render.append((name,arg,Stools.print_S_graph_list))
        elif isinstance(arg,Ktools.K_graph_list):
            msg += Ktools.aux_print_K_graph_list_message(arg)
            name = Ktools.aux_print_K_graph_list_name(arg,get_name(name))
            to_render.append((name,arg,Ktools.print_K_graph_list))
        elif ((isinstance(arg,list) or isinstance(arg,tuple))
                and len(arg) != 1
                and all(isinstance(a,RK_graph) for a in arg)):
            msg += f"List of {len(arg)} RK graphs:\n"
            sub_msgs = []
            list_sub = []
            for a in arg:
                sub_msgs.append(
                    process_arg(a,list_sub,
                        indent=2,pre_msg="",post_msg="",name="Empty")
                )
            name = get_name(name)
            name = name if name is not None else f"List_of_{len(arg)}_RK_graphs"
            to_render.append((name,[c[1] for c in list_sub],[c[2] for c in list_sub]))
            msg += "\n".join(sub_msgs)
        elif isinstance(arg,Ptools.P_cluster):
            msg += Ptools.aux_print_P_cluster_message(arg)+"\n"
            names[:0] = Ptools.aux_print_P_cluster_names(arg,get_name(name))
            sub_msgs = []
            for pg in arg.representee_cluster.possible_partitioning:
                sub_msgs.append(process_arg(pg,to_render,2))
            msg += "\n".join(sub_msgs)
        elif isinstance(arg,Htools.H_cluster):
            msg += Htools.aux_print_H_cluster_message(arg)+"\n"
            names[:0] = Htools.aux_print_H_cluster_names(arg,get_name(name))
            sub_msgs = []
            for pg in arg.representee_cluster.possible_hg:
                sub_msgs.append(process_arg(pg,to_render,2))
            msg += "\n".join(sub_msgs)
        elif isinstance(arg,Ptools.P_structure):
            msg += "P_structure's main cluster :\n"
            msg += process_arg(arg.main_cluster,to_render)
        elif isinstance(arg,rkGB_res):
            msg += "rkGB_res : all graphs\n"
            sub_msgs = []
            for at in [
                "B_graph",
                "D_graph",
                "S_graph",
                "K_graph",
                "S_graph_list",
                "K_graph_list",
                "P_structure",
                "H_cluster"]:
                if getattr(arg,at) is None: continue
                sub_msgs.append(
                    process_arg(getattr(arg,at),to_render,0,"="*3+"\n","\n")
                )
            msg += "\n".join(sub_msgs)
        else:
            return arg
        return msg + post_msg

    for arg in args:
        msg = process_arg(arg,graphs_to_render,0,"="*10+"\n","\n"+"="*10)
        filtered_args.append(msg)

    print(*filtered_args, **kwargs)

    if len(graphs_to_render) != 0:
        print("*** START TO RENDER ***")
    for name,obj,print_fct in graphs_to_render:
        if not isinstance(print_fct,list):
            print_fct(obj,name=name,open=open,render_format=render_format)
        else:
            dot = graphviz.Digraph(name,comment=name)
            for i,fct in enumerate(print_fct):
                fct(obj[i],dot=dot,uniq_num=i)
            small_fcts.graph_render(dot,open,"various",render_format)


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