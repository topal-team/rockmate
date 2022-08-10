from .utils import *
from .Stools import S_node,S_graph

# ==========================
# ====== K structure =======
# ==========================

class K_node():
    def __init__(self,is_fwd,req,
            is_artefact=False,
            target="/!\\ No target /!\\",
            all_targets=None,
            tensor_targets=None,
            main_code=None,
            body_code=None,
            info=None):
        self.is_fwd = is_fwd
        self.main_target = target
        if tensor_targets is None: tensor_targets = [target]
        self.tensor_targets = tensor_targets
        if all_targets is None: all_targets = [target]
        self.all_targets = all_targets
        if is_fwd: self.name = "fwd_"+target
        else:      self.name = "bwd_"+target
        self.is_artefact = is_artefact
        self.run_mem  = None
        self.fgt_mem  = None
        self.overhead = None
        self.time = None
        self.main_code = main_code
        self.body_code = body_code
        self.req = req
        self.used_by = set()
        self.info = info

    def full_code(self):
        if self.main_code is None: mc = []
        else: mc = [self.main_code]
        return make_ast_module(mc + self.body_code)
    def get_main_code(self):
        if self.main_code is None: mc = []
        else: mc = [self.main_code]
        return ast_to_str(make_ast_module(mc))
    def get_code(self):
        return ast_to_str(self.full_code())

class K_graph():
    def __init__(self,sg : S_graph):
        self.dict_nodes = dict()
        self.hidden_inputs  = sg.hidden_inputs
        self.direct_inputs  = sg.direct_inputs
        self.hidden_output  = sg.hidden_output
        self.direct_outputs = sg.direct_outputs
        self.init_code = make_ast_module(sg.init_node.body_code)
        self.dict_info = sg.dict_info
        self.dict_rand = sg.dict_rand
        self.loss_node = None
    def make_used_by(self):
        for n in self.dict_nodes.values():
            for req_n in n.req:
                req_n.used_by.add(n)

# ==========================



# ==========================
# = Move from S to K graph =
# ==========================

def generate_tmp_local(n : S_node,g : S_graph,our_global):
    tmp_local = {}
    exec(g.init_node.get_code(),our_global,tmp_local)
    for req_n in n.req:
        if not (req_n is g.init_node):
            # we create the main_target value, and we run the body_code
            # but the body_code may requires some artefacts
            req_tar = req_n.main_target
            main_info = g.dict_info[req_tar]
            tmp_local[req_tar] = generate_val(main_info,device) #utils.py
            for req_req_n in req_n.req:
                if not (req_req_n is g.init_node):
                    for req_req_tar in req_req_n.all_targets:
                        req_req_info = g.dict_info[req_req_tar]
                        tmp_local[req_req_tar] = (
                            generate_val(req_req_info,device))
            for c in req_n.body_code:
                try:
                    exec(ast_to_str(c),our_global,tmp_local)
                except:
                    raise Exception(
                      f"pb to generate {req_tar} for {n.main_target} "\
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
    timer = rotor.timing.make_timer(device)
    memUsage = rotor.memory.MeasureMemory(device)
    tmp_local = generate_tmp_local(n,g,our_global)
    ret = {}

    # -- aux --
    def measure_time(fct, inter_fct=None):
        duration = timer.measure_median(fct,samples=1)
        if duration < min_duration:
            number_repetitions = 1 + int(min_duration // duration)
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
    ret["mem_fgt_fwd"] = minus_mem(mem_fgt_fwd)
    ret["time_run_fwd"] = time_run_fwd
    # ===============

    # === BACKWARD ===
    if info.requires_grad:
        tmp_local[mt].data = generate_val(info,device)
        tmp_local[mt].grad = generate_val(info,device)

        def fct_run_bwd():
            exec(f"{mt}.backward({mt}.grad)", our_global, tmp_local)
        def fct_fgt_bwd():
            for req_n in n.req:
                #if not (req_n is g.init_node):
                if not req_n.is_artefact:
                    for tar in req_n.tensor_targets:
                        tmp_local[tar].grad = None

        # measure
        _ , mem_run_bwd , peak_bwd = memUsage.measure(fct_run_bwd)
        overhead_bwd = peak_bwd - mem_run_bwd
        _ , mem_fgt_bwd , _ = memUsage.measure(fct_fgt_bwd)
        fct_run_fwd()
        timer.measure_median(fct_run_fwd)
        tmp_local[n.main_target].grad = generate_val(info,device)
        time_run_bwd = measure_time(fct_run_bwd, fct_run_fwd)
        # overhead_bwd contains n.target.data now /!\

        ret["overhead_bwd"] = overhead_bwd
        ret["mem_run_bwd"]  = mem_run_bwd
        ret["mem_fgt_bwd"]  = minus_mem(mem_fgt_bwd)
        ret["time_run_bwd"] = time_run_bwd
    # ===============
    return ret

# aux function to handle verbose and device
def aux_init_S_to_K(nn_mod,verbose,K_device):
    if not (verbose is None): ref_verbose[0] = verbose
    global device
    if K_device is None: device = get_device()
    else: device = K_device
    nn_mod.to(device)


# the function that does it all
def aux_build_S_to_K(sg : S_graph,nn_mod):
    # -- init --
    dict_Kbwd = dict() # dict : target -> K_node(bwd)
    dict_Kfwd = dict() # dict : target -> K_node(fwd)
    our_global = globals().copy()
    our_global["self"] = nn_mod
    our_global["device"] = device
    our_global
    kg = K_graph(sg)
    # -> rebuilt dict_inputs and make memsize of inputs
    for inp in sg.direct_inputs:
        info = sg.dict_info[inp]
        x = generate_val(info,device)
        our_global[inp]=x
        if inp in sg.hidden_inputs:
            info.memsize = MemSize(int(tensorMsize(x)))
        # our_global[inp] = generate_val(sg.dict_info[inp],device) #utils

    # ------------
    def handle_node(n : S_node):
        mt = n.main_target
        print_debug(mt)
        # -- build Kfwd --
        n_req = set(n.req)
        n_req.discard(sg.init_node)
        Kreq = set(dict_Kfwd[sub_n.main_target] for sub_n in n_req)
        info = sg.dict_info[mt]
        Kfwd = K_node(
                is_artefact = n.is_artefact,
                is_fwd     = True,
                req        = Kreq,
                target     = mt,
                all_targets    = n.all_targets,
                tensor_targets = n.tensor_targets,
                main_code  = n.main_code,
                body_code  = n.body_code,
                info = info)
        dict_Kfwd[mt] = Kfwd

        # -- build Kbwd --
        info = sg.dict_info[mt]
        if info.requires_grad:
            print_debug(f"{mt} req bwd")
            Kbwd = K_node(is_fwd=False, req=set(Kreq), target=mt, info=info)
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
            info.memsize  = res["mem_fgt_fwd"]
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
        req = {dict_Kfwd[sg.hidden_output]},
        main_code = make_ast_constant("LOSS"),
        #main_code = (
        #  ast.Assign(
        #    [ast.Name(f"{sg.hidden_output}.grad")],
        #    ast.Name("loss"))),
        body_code = [])
    loss_node.run_mem  = MemSize(0)
    loss_node.fgt_mem  = MemSize(0)
    loss_node.overhead = MemSize(0)
    loss_node.time     = 0
    kg.loss_node = loss_node
    dict_Kbwd[sg.hidden_output].req.add(loss_node)
    dict_nodes["fwd_loss"] = loss_node
    # ------------

    kg.make_used_by()
    return kg


def S_to_K(sg : S_graph,nn_mod,verbose=None,K_device=None):
    aux_init_S_to_K(nn_mod,verbose,K_device)
    return aux_build_S_to_K(sg,nn_mod)


def S_list_to_K_list(list_sg,nn_mod,verbose=None,K_device=None):
    aux_init_S_to_K(nn_mod,verbose,K_device)
    return [aux_build_S_to_K(sg,nn_mod) for sg in list_sg]

# ==========================



# ==========================
# === printing functions ===
# ==========================

def aux_print_graph(dot,g,uniq_num):
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2): dot.edge(uni(i1),uni(i2))
    def print_node(n):
        if n.main_target == "loss":
            node(n.name,
                f"LOSS\ncode: {n.get_code()}",
                color="green")
        elif n.is_fwd:
            node(n.name,n.get_code(),color="blue")
        else:
            node(n.name,f"backward of {n.main_target}",color="red")
    nodes = g.dict_nodes.values()
    for n in nodes:
        print_node(n)
    for n in nodes:
        for sub_n in n.req:
            edge(sub_n.name,n.name)

    # -- io --
    str_inp = "\n".join(g.direct_inputs)
    node("input_ph",
        f"INPUTS : {str_inp}",
        color="green",style="dashed")
    str_out = "\n".join(g.hidden_inputs)
    node("output_ph",
        f"OUTPUTS : inputs' grad\n{str_out}",
        color="green",style="dashed")
    for n in nodes:
        if n.req == set(): # src nodes
            edge("input_ph",n.name)
        if n.used_by == set(): # leaves
            edge(n.name,"output_ph")


def print_K_graph(g : K_graph,name=None,open=True):
    print(f"Forward + Backward graph : {len(g.dict_nodes)} nodes")
    if name is None: name = "backward_K-graph"
    dot = graphviz.Digraph(name,
        comment="K_graph = Forward + Backward graph")
    aux_print_graph(dot,g,0)
    graph_render(dot,open,"K") # from utils.py


def print_K_graph_list(list_g,name=None,open=True):
    s = "+".join([str(len(g.dict_nodes)) for g in list_g])
    print(
        f"{len(list_g)} blocs of K_graph, with {s} = "\
        f"{sum([len(g.dict_nodes) for g in list_g])} nodes")
    if name is None: name = "all_K-graphs"
    dot = graphviz.Digraph(
        name,
        comment="K_graph list : cut forward+backward graph")
    for i in range(len(list_g)):
        aux_print_graph(dot,list_g[i],i)
    graph_render(dot,open,"K") # from utils.py

# ==========================

