import torch
from Btools import *
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

class Node_info():
    def __init__(self):
        self.dtype = None
        self.target_type = None # torch.Tensor or torch.size 
        self.target_size = None # depends on type
        self.requires_grad = None

class K_node():
    def __init__(self,name=None,code=None,req=None):
        self.name = name
        self.run_mem  = None
        self.fgt_mem  = None
        self.overhead = None
        self.time = None
        self.code = code
        if req is None:
            self.req = []
        else:
            self.req = req # list of K_node

def get_info(x) -> Node_info:
    info = Node_info()
    tt = type(x)
    info.target_type = tt
    if tt==torch.Size:
        info.target_size = x
        info.requires_grad = False
    elif tt==torch.Tensor:
        info.target_size = x.shape
        info.dtype = x.dtype
        info.requires_grad = x.requires_grad
    else:
        raise Exception(f"normally there should only be tensor or size at this point, but {tt} found")
    return info

def generate_val(info):
    tt = info.target_type
    if tt==torch.Size:
        return info.target_size
    else:
        assert(tt==torch.Tensor)
        return torch.randn(info.target_size,
            dtype=info.dtype,
            requires_grad=info.requires_grad,
            device=device)

def detach_code(n): # TODO TO IMPROVE
    code = (n.code).replace(n.target,"_"+n.target)
    return f"{code} ; {n.target} = _{n.target}.detach()"

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
    code='_{o}.backward({o}.grad, inputs={i})'.format(o=n.target,i=inputs_str)
    targ = n.target
    bwd_code = f"if _{targ}.data.shape == torch.Size([0]):\n"
    bwd_code += f"\t_{targ}.data = torch.randn_like({targ}.grad,device=device)\n"
    bwd_code += f"\t{code}\n"
    bwd_code += f"\t_{targ}.data = torch.randn(0,device=device)\n"
    bwd_code += f"else:\n\t{code}\n"
    return bwd_code

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

min_duration = 0

def inspection(n,fwd_code,bwd_code,dict_info,g,our_global):
    info = dict_info[n.target]
    assert(info.target_type == torch.Tensor)
    timer = make_timer(device)
    memUsage = MeasureMemory(device)
    tmp_local = generate_tmp_local(g,dict_info,n)
    ret = {}

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

    # === FORWARD ===
    # def of run fwd
    fct_run_fwd = lambda : exec(fwd_code, our_global, tmp_local)
    # def of forget fwd
    if info.requires_grad:
        def fct_fgt_fwd():
            tmp_local["_"+n.target].data = torch.randn(0,device=device)
            tmp_local[n.target].data = torch.randn(0,device=device)
    else:
        def fct_fgt_fwd():
            tmp_local[n.target].data = torch.randn(0,device=device)

    # measure
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

def make_k_nodes(g : D_graph,nn_mod,dict_inputs : dict,show_debug=False):
    # returns a list of K_nodes
    dict_info = {} # dict : D_node.target -> node_info
    dict_Kbwd = {} # dict : D_node.target -> K_node(bwd)
    dict_Kfwd = {} # dict : D_node.target -> K_node(fwd)
    our_global = globals().copy()
    our_global["self"] = nn_mod
    our_global["device"] = device

    const_mem = 0

    # -- inputs --  
    for inp in g.inputs:
        dict_info[inp] = get_info(dict_inputs[inp])
        dict_Kfwd[inp] = K_node(name="fwd_"+inp, code="input")
        
        dict_Kbwd[inp] = K_node(name="bwd_"+inp, code="input")
    # ------------
    def handle_node(n : D_node):
        if show_debug:
            print(n.target)
        if n.is_input:
            pass # info already known
        else:
            # -- generate random inputs --
            tmp_local = generate_tmp_local(g,dict_info,n)

            # -- get info --
            try:
                exec(n.code , our_global , tmp_local)
            except:
                print('Fail to execute:', n.target, n.code)
            x = tmp_local[n.target]
            if show_debug: print(x.requires_grad)
            info = get_info(x)
            dict_info[n.target] = info
            # -- build Kfwd --
            Kreq = [dict_Kfwd[sub_n.target] for sub_n in n.req]
            fwd_code = n.code
            if info.requires_grad:
                fwd_code = detach_code(n)
            Kfwd = K_node(name="fwd_"+n.target,code=fwd_code,req = Kreq)
            dict_Kfwd[n.target] = Kfwd

            # -- build Kbwd --
            if info.requires_grad:
                assert(info.target_type == torch.Tensor)
                bwd_code = generate_bwd_code(n,info,dict_info)
                Kbwd = K_node(name="bwd_"+n.target, code=bwd_code, req=Kreq)
                dict_Kbwd[n.target] = Kbwd
                for sub_n in n.req:
                    if sub_n.target in dict_Kbwd:
                        dict_Kbwd[sub_n.target].req.append(Kbwd)
            else: bwd_code=None

            # -- inspection --
            if info.target_type == torch.Size:
                Kfwd.run_mem = 0
                Kfwd.fgt_mem = 0
                Kfwd.overhead = 0
                Kfwd.time = 0
            else:
                inspection_ret = inspection(n,fwd_code,bwd_code,dict_info,g,our_global)
                Kfwd.run_mem = inspection_ret["mem_run_fwd"]
                Kfwd.fgt_mem = inspection_ret["mem_fgt_fwd"]
                Kfwd.overhead = inspection_ret["overhead_fwd"]
                Kfwd.time = inspection_ret["time_run_fwd"]
                if info.requires_grad:
                    Kbwd.run_mem = inspection_ret["mem_run_bwd"]
                    Kbwd.fgt_mem = inspection_ret["mem_fgt_bwd"]
                    Kbwd.overhead = inspection_ret["overhead_bwd"]
                    Kbwd.time = inspection_ret["time_run_bwd"]
    # ------------

    for n in g.nodes:
        handle_node(n)
    return dict_Kfwd , dict_Kbwd




