import torch
from Stools import *
try:
    from rotor.utils import *
except:
    from torch.hub import load_state_dict_from_url
    from rotor.utils import *
from rotor.timing import *
from rotor.memory import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

min_duration = 0

# ==========================
# = Move from S to K graph =
# ==========================

class K_node():
    def __init__(self,is_fwd,req,
            is_artefact=False,
            target="/!\\ No target /!\\",
            tensor_targets=None,
            full_code=None):
        self.is_fwd = is_fwd
        self.main_target = target
        if tensor_targets is None: tensor_targets = [target]
        self.tensor_targets = tensor_targets
        if is_fwd: self.name = "fwd_"+target
        else:      self.name = "bwd_"+target
        self.is_artefact = is_artefact
        self.run_mem  = None
        self.fgt_mem  = None
        self.overhead = None
        self.time = None
        self.full_code = full_code
        self.req = req
        self.used_by = set()
    def get_code(self):
        return ast_to_str(self.full_code)

class K_graph():
    def __init__(self,sg : S_graph):
        self.dict_nodes = dict()
        self.output = sg.output
        self.init_code = ast.Module(sg.init_node.body_code)
        self.dict_rand = sg.dict_rand
    def make_used_by(self):
        for n in self.dict_nodes.values():
            for req_n in n.req:
                req_n.used_by.add(n)


def generate_tmp_local(n : S_node,g : S_graph,our_global):
    tmp_local = {}
    exec(g.init_node.get_code(),our_global,tmp_local)
    for req_n in n.req:
        if not (req_n is g.init_node):
            # we create the main_target value, and we run the body_code
            # but the body_code may require some artefacts
            req_tar = req_n.main_target
            main_info = g.dict_info[req_tar]
            tmp_local[req_tar] = generate_val(main_info) # from Dtools
            for req_req_n in req_n.req:
                if not (req_req_n is g.init_node):
                    #if req_req_n.is_artefact:
                    for req_req_tar in req_req_n.all_targets:
                        req_req_info = g.dict_info[req_req_tar]
                        tmp_local[req_req_tar] = generate_val(req_req_info)
            for c in req_n.body_code:
                try:
                    exec(ast_to_str(c),our_global,tmp_local)
                except:
                    raise Exception(
                        f"pb to gene {req_n.main_target} for {n.main_target} "\
                        f"\n {ast_to_str(c)} impossible to exec")
    """ TODO
    if n.is_rand:
        for sub_r in n.req_rand:
            exec(g.dict_rand[sub_r],our_global,tmp_local)
    """
    return tmp_local


def inspection(n : S_node,g : S_graph,our_global):
    mt = n.main_target
    info = g.dict_info[mt]
    timer = make_timer(device)
    memUsage = MeasureMemory(device)
    tmp_local = generate_tmp_local(n,g,our_global)
    ret = {}

    # -- aux --
    def measure_time(fct, inter_fct=None):
        duration = timer.measure_median(fct,samples=1)
        if duration < min_duration:
            number_repetitions = 1 + int(min_duration // duration)
            #duration = timer.measure_median(fct, iterations = number_repetitions)
            for _ in range(1,number_repetitions):
                if inter_fct:
                    inter_fct()
                duration += timer.measure_median(fct,samples=1)
        else: number_repetitions = 1
        return duration/number_repetitions
    # ---------

    # === FORWARD ===
    def fct_run_fwd():
        exec(n.get_code(), our_global, tmp_local)
    def fct_fgt_fwd():
        for tar in n.tensor_targets:
            #tar_info = g.dict_info[tar]
            #assert(tar_info.ttype == torch.Tensor)
            tmp_local[tar].data = torch.zeros(0,device=device)

    # -- measure forward --
    _ , mem_run_fwd , peak_fwd = memUsage.measure(fct_run_fwd)
    overhead_fwd = peak_fwd - mem_run_fwd
    time_run_fwd = measure_time(fct_run_fwd)
    _ , mem_fgt_fwd , _ = memUsage.measure(fct_fgt_fwd)
    ret["overhead_fwd"] = overhead_fwd
    ret["mem_run_fwd"] = mem_run_fwd
    ret["mem_fgt_fwd"] = mem_fgt_fwd
    ret["time_run_fwd"] = time_run_fwd
    # ===============

    # === BACKWARD ===
    if info.requires_grad:
        tmp_local[mt].data = generate_val(info)
        tmp_local[mt].grad = generate_val(info)
        #tmp_local[mt].data = torch.zeros(0, device=device)

        def fct_run_bwd():
            exec(f"{mt}.backward({mt}.grad)", our_global, tmp_local)
        def fct_fgt_bwd():
            for req_n in n.req:
                #if not (req_n is g.init_node):
                if not req_n.is_artefact:
                    for tar in req_n.tensor_targets:
                        #tar_info = g.dict_info[tar]
                        #if tar_info.ttype != torch.Size:
                        tmp_local[tar].grad = None

        # measure
        _ , mem_run_bwd , peak_bwd = memUsage.measure(fct_run_bwd)
        overhead_bwd = peak_bwd - mem_run_bwd
        _ , mem_fgt_bwd , _ = memUsage.measure(fct_fgt_bwd)
        fct_run_fwd()
        timer.measure_median(fct_run_fwd)
        tmp_local[n.main_target].grad = generate_val(info)
        time_run_bwd = measure_time(fct_run_bwd, fct_run_fwd)
        # overhead_bwd contains n.target.data now /!\

        ret["overhead_bwd"] = overhead_bwd
        ret["mem_run_bwd"]  = mem_run_bwd
        ret["mem_fgt_bwd"]  = mem_fgt_bwd
        ret["time_run_bwd"] = time_run_bwd
    # ===============
    return ret

def S_to_K(sg : S_graph,nn_mod,dict_inputs,show_debug=False):
    # returns a list of K_nodes
    dict_Kbwd = dict() # dict : D_node.target -> K_node(bwd)
    dict_Kfwd = dict() # dict : D_node.target -> K_node(fwd)
    our_global = globals().copy() | dict_inputs
    our_global["self"] = nn_mod
    our_global["device"] = device
    our_global
    kg = K_graph(sg)

    # ------------
    def handle_node(n : S_node):
        mt = n.main_target
        if show_debug: print(mt)
        # -- build Kfwd --
        n_req = set(n.req)
        n_req.discard(sg.init_node)
        Kreq = set(dict_Kfwd[sub_n.main_target] for sub_n in n_req)
        #non_size_targets = []
        #for tar in n.all_targets:
        #    tar_info = sg.dict_info[tar]
        #    if tar_info != torch.Size:
        #        non_size_targets.append(tar)
        Kfwd = K_node(
                is_artefact = n.is_artefact,
                is_fwd     = True,
                req        = Kreq,
                target     = mt,
                tensor_targets = n.tensor_targets,
                #all_targets= non_size_targets,
                full_code  = n.full_code())
        dict_Kfwd[mt] = Kfwd

        # -- build Kbwd --
        info = sg.dict_info[mt]
        if info.requires_grad:
            if show_debug: print(f"{mt} req bwd")
            Kbwd = K_node(is_fwd=False, req=set(Kreq), target=mt)
            dict_Kbwd[mt] = Kbwd
            for sub_n in n.req:
                sub_tar = sub_n.main_target
                if sub_tar in dict_Kbwd: # requires_grad
                    dict_Kbwd[sub_tar].req.add(Kbwd)

        # -- inspection --
        if info.ttype == torch.Size:
            Kfwd.run_mem  = MemSize(0)
            Kfwd.fgt_mem  = MemSize(0)
            Kfwd.overhead = MemSize(0)
            Kfwd.time     = 0
        else:
            res = inspection(n,sg,our_global)
            Kfwd.run_mem  = res["mem_run_fwd"]
            Kfwd.fgt_mem  = res["mem_fgt_fwd"]
            Kfwd.overhead = res["overhead_fwd"]
            Kfwd.time     = res["time_run_fwd"]
            if info.requires_grad:
                Kbwd.run_mem  = res["mem_run_bwd"]
                Kbwd.fgt_mem  = res["mem_fgt_bwd"]
                Kbwd.overhead = res["overhead_bwd"]
                Kbwd.time     = res["time_run_bwd"]
    # ------------
    for n in sg.nodes:
        handle_node(n)
    dict_nodes = kg.dict_nodes
    for Kfwd in dict_Kfwd.values():
        dict_nodes[Kfwd.name]=Kfwd
    for Kbwd in dict_Kbwd.values():
        dict_nodes[Kbwd.name]=Kbwd

    # -- loss node --
    loss_node = K_node(
        is_fwd = True,
        target = "loss",
        req = {dict_Kfwd[sg.output]},
        full_code = ast.Assign([ast.Name(f"{sg.output}.grad")],ast.Name("loss")))
    loss_node.run_mem  = MemSize(0)
    loss_node.fgt_mem  = MemSize(0)
    loss_node.overhead = MemSize(0)
    loss_node.time     = 0
    dict_Kbwd[sg.output].req.add(loss_node)
    dict_nodes["fwd_loss"] = loss_node
    # ------------

    kg.make_used_by()
    return kg



# ==========================



# ==========================
# === printing functions ===
# ==========================

import graphviz

def print_K_nodes(g,name=None,open=True):
    print(len(g.dict_nodes))
    if name is None:
        name = "K-nodes graph"
    dot = graphviz.Digraph(name,comment="Forward + Backward graph")
    def print_node(n):
        if n.main_target == "loss":
            dot.node(n.name,"loss placeholder",color="green")
        elif n.is_fwd:
            dot.node(n.name,n.get_code(),color="blue")
        else:
            dot.node(n.name,f"backward of {n.main_target}",color="red")
    nodes = g.dict_nodes.values()
    for n in nodes:
        print_node(n)
    for n in nodes:
        for sub_n in n.req:
            dot.edge(sub_n.name,n.name)
    dot.render(directory="graphviz_dir",view=open)

# ==========================

