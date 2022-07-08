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
    def __init__(self,is_fwd,target="/!\\ No target /!\\",req,all_targets=None,full_code=None):
        self.is_fwd = is_fwd
        self.main_target = target
        if all_targets is None: all_targets = [target]
        self.all_targets = all_targets
        if is_fwd: self.name = "fwd_"+target
        else:      self.name = "bwd_"+target
        self.run_mem  = None
        self.fgt_mem  = None
        self.overhead = None
        self.time = None
        self.full_code = full_code
        self.req = req
        self.used_by = set()

class K_graph():
    def __init__(self,sg : S_graph):
        self.dict_nodes = set()
        self.output = sg.output
        self.init_code = ast.Module(sg.init_code.body_code)
        self.dict_rand = sg.dict_rand
    def make_used_by(self):
        for n in self.dict_nodes.values():
            for req_n in n.req:
                req_n.used_by.add(n)

def generate_bwd_code(n : D_node,info,dict_info):
    tt = info.target_type
    assert(tt!=torch.Size)
    if n.is_input: input_str = None
    else:
        req = []
        for inp in n.req:
            if dict_info[inp.target].requires_grad:
                req.append(inp.target)
        if req==[]:
            inputs_str='None'
        else:
            inputs_str = "["+ ','.join(req) +"]"
    code='{o}.backward({o}.grad, inputs={i})'.format(o=n.target,i=inputs_str)
    targ = n.target
    bwd_code = f"if {targ}.data.shape == torch.Size([0]):\n"
    bwd_code += f"\t{targ}.data = torch.randn_like({targ}.grad,device=device)\n"
    bwd_code += f"\t{code}\n"
    bwd_code += f"\t{targ}.data = torch.randn(0,device=device)\n"
    bwd_code += f"else:\n\t{code}\n"
    return bwd_code

def generate_tmp_local(n : S_node,g : S_graph):
    tmp_local = {}
    for req_n in n.req:
        # we create the main_target value, and we run the body_code
        # but the body_code may require some artefacts
        req_tar = req_n.main_target
        main_info = g.dict_info[req_tar]
        main_x = generate_val(main_info) # from Dtools
        for req_req_n in req_n.req:
            if req_req_n.is_artefact:

def generate_val(info):
    tt = info.target_type
    if tt==torch.Size:
        return info.target_size
    else:
        assert(tt==torch.Tensor)
        return torch.ones(info.target_size,
            dtype=info.dtype,
            requires_grad=info.requires_grad,
            device=device)


# -- generate random inputs --
def generate_tmp_local(g,dict_info,n):
    tmp_local = {}
    for sub_n in n.req:
        sub_info = dict_info[sub_n.target]
        sub_x = generate_val(sub_info)
        tmp_local[sub_n.target] = sub_x
    if n.is_rand:
        for sub_r in n.req_rand:
            exec(g.dict_rand[sub_r],our_global,tmp_local)
    return tmp_local

def inspection(n,fwd_code,g,our_global):
    info = g.dict_info[n.target]
    timer = make_timer(device)
    memUsage = MeasureMemory(device)
    tmp_local = generate_tmp_local(g,dict_info,n)
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
    fct_run_fwd = lambda : exec(fwd_code, our_global, tmp_local)
    def fct_fgt_fwd():
        tmp_local[n.target].data = torch.zeros(0,device=device)

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
        tmp_local[n.target].data = generate_val(info)
        tmp_local[n.target].grad = generate_val(info)
        tmp_local[n.target].data = torch.randn(0, device=device)
        
        # def of run bwd
        fct_run_bwd = lambda : exec(bwd_code, our_global, tmp_local)
        # def of forget bwd
        def fct_fgt_bwd():
            for sub_n in n.req:
                if dict_info[sub_n.target].requires_grad:
                    tmp_local[sub_n.target].grad = None

        # measure
        _ , mem_run_bwd , peak_bwd = memUsage.measure(fct_run_bwd)
        overhead_bwd = peak_bwd - mem_run_bwd
        _ , mem_fgt_bwd , _ = memUsage.measure(fct_fgt_bwd)
        fct_run_fwd()
        timer.measure_median(fct_run_fwd)
        tmp_local[n.target].grad = generate_val(info)
        time_run_bwd = measure_time(fct_run_bwd, fct_run_fwd)
        # overhead_bwd contains n.target.data now /!\

        ret["overhead_bwd"] = overhead_bwd
        ret["mem_run_bwd"] = mem_run_bwd
        ret["mem_fgt_bwd"] = mem_fgt_bwd
        ret["time_run_bwd"] = time_run_bwd
    # ===============
    return ret

def S_to_K(sg : S_graph,nn_mod,dict_inputs : dict,show_debug=False):
    # returns a list of K_nodes
    dict_Kbwd = {} # dict : D_node.target -> K_node(bwd)
    dict_Kfwd = {} # dict : D_node.target -> K_node(fwd)
    our_global = globals().copy()
    our_global["self"] = nn_mod
    our_global["device"] = device
    kg = K_graph(sg)

    # ------------
    def handle_node(n : S_node):
        tar = n.main_target
        if show_debug: print(tar)
        # -- build Kfwd --
        Kreq = set(dict_Kfwd[sub_n.main_target] for sub_n in n.req)
        Kfwd = K_node(
                is_fwd     = True,
                req        = Kreq,
                target     = tar,
                all_targets= n.all_targets,
                full_code  = n.full_code())
        dict_Kfwd[tar] = Kfwd
        fwd_code = Kfwd.full_code

        # -- build Kbwd --
        info = sg.dict_info[tar]
        if info.requires_grad:
            if show_debug: print(f"{tar} req bwd")
            Kbwd = K_node(is_fwd=False, req=set(Kreq), target=tar)
            dict_Kbwd[tar] = Kbwd
            for sub_n in n.req:
                sub_tar = sub_n.main_target
                if sub_tar in dict_Kbwd: # requires_grad
                    dict_Kbwd[sub_tar].req.add(Kbwd)

        # -- inspection --
        if info.target_type == torch.Size:
            Kfwd.run_mem  = 0
            Kfwd.fgt_mem  = 0
            Kfwd.overhead = 0
            Kfwd.time     = 0
        else:
            res = inspection(n,fwd_code,sg,our_global)
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

    dict_nodes = kg.dict_nodes
    for Kfwd in dict_Kfwd.values():
        dict_nodes[Kfwd.name]=Kfwd
    for Kbwd in dict_Kbwd.values():
        dict_nodes[Kbwd.name]=Kbwd
    kg.make_used_by()
    return kg



# ==========================



# ==========================
# === printing functions ===
# ==========================

import graphviz

def print_K_nodes(dict_Kfwd,dict_Kbwd,name=None):
    if name is None:
        name = "K-nodes graph"
    dot = graphviz.Digraph(name,comment="K_nodes")
    def print_node(n,default_color):
        if n.code=="INPUT":
            dot.node(n.name,f"{n.name} = INPUT",color="green")
        else:
            dot.node(n.name,n.code,color=default_color)
    fwd_nodes = list(dict_Kfwd.values())
    bwd_nodes = list(dict_Kbwd.values())
    all_nodes = fwd_nodes + bwd_nodes
    for n in fwd_nodes:
        print_node(n,"blue")
    for n in bwd_nodes:
        print_node(n,"red")
    for n in all_nodes:
        for sub_n in n.req:
            dot.edge(sub_n.name,n.name)
    dot.render(directory="graphviz_dir",view=True)

# ==========================

