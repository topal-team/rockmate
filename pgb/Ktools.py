from .utils import *
from .Stools import S_node,S_graph
import copy
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
        self.del_mem  = MemSize(0)#None
        self.overhead = None
        self.time = None
        self.main_code = main_code
        self.body_code = body_code
        self.req = req
        self.used_by = set()
        self.info = info
        self.abar = None

        self.phantoms = []

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
    def get_fgt_code(self):
        code = ""
        for tar in self.phantoms:
            code += ast.Assign(
                [ast.Attribute(tar,"data")],
                ast.Call(
                    ast.Attribute(ast.Name("torch"),"zeros"),
                    [make_ast_constant(0)],
                    []
                    )
                )
        return code

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
        self.sg = sg
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
    #print("gene", dict_Kfwd.keys())
    # print(g.direct_inputs)
    # for inp in g.direct_inputs:
    #     # print(inp)
    #     # info = sg.dict_info[inp]
    #     # x = generate_val(info,device)
    #     tmp_local[inp] = copy.deepcopy(our_global[inp])
        # if inp in sg.hidden_inputs:
        #     info.memsize = MemSize(int(tensorMsize(x)))
    exec(g.init_node.get_code(),our_global,tmp_local)
    for req_n in n.req:
        if not (req_n is g.init_node):
            # we create the main_target value, and we run the body_code
            # but the body_code may requires some artefacts
            req_tar = req_n.main_target
            main_info = g.dict_info[req_tar]
            tmp_local[req_tar] = generate_val(main_info,device)#.detach()
            # if g.dict_info[req_tar].requires_grad: tmp_local[req_tar].requires_grad_()#utils.py
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


def get_dependencies_and_phantoms(n : S_node,g : S_graph,our_global):
    params = dict(our_global['self'].named_parameters())#TODO:self->general name?
    print_debug(f"Try to open {n.main_target}'s grad_fn")
    # == INIT ==
    tmp_local = generate_tmp_local(n,g,our_global)
    exec(n.get_code(), our_global, tmp_local)
    mt = n.main_target
    fn = tmp_local[mt].grad_fn
    used_vars = set()    # set of torch.Tensor
    phantom_vars = [] # set of "path" from fn

    # == SEARCH THROUGH GRAD_FN == 
    def aux(f,path):
        if hasattr(f,"variable"):
            used_vars.add(f.variable)
        for attr in dir(f):
            if (isinstance(getattr(f,attr),torch.Tensor)
                and attr != "variable"):
                is_para = False
                for k,p in params.items():
                    if p is getattr(f,attr): is_para = True
                is_hidden = True
                for k,t in tmp_local.items():
                    if t is getattr(f,attr): is_hidden = False
                is_input = False
                for k,t in our_global.items():
                    if t is getattr(f,attr): is_input = True
                if is_hidden and not is_para and not is_input: phantom_vars.append((attr,path))
        if hasattr(f,"next_functions"):
            for k,t in enumerate(f.next_functions):
                aux(t[0],path+[k])
    aux(fn,[])

    # == get dependencies -> recognize required_nodes' tensor ==
    dependencies = set()
    for used_var in used_vars:
        # we are looking for used_var
        is_input = False
        keys = [k for (k,v) in tmp_local.items() if v is used_var]
        if len(keys) == 0: continue#pass
        for k,v in our_global.items():
            if keys[0] in g.direct_inputs: is_input=True
            # -> a parameter ?
        # if keys[0] == '__13_input0': print(is_input)
        else:
            if len(keys) > 1:
                print(f"warning : used_var matchs several variables"\
                      f"{keys} \n one has been chosen arbitrarily")
            elif not is_input: dependencies.add(keys[0])

    # == GENERATE AST NAME FOR PHANTOM VARS ==
    """
    def make_ast_fgt(tensor):
        return ast.Assign(
            [ast.Attribute(tensor,"data")],
            ast.Call(
                ast.Attribute(ast.Name("torch"),"zeros"),
                [make_ast_constant(0)],
                []
                )
            )
    """
    def make_ast_path(attr,path):
        ast_code = ast.Attribute(ast.Name(mt),"grad_fn")
        for k in path:
            ast_code = ast.Attribute(ast_code,"next_functions")
            ast_code = ast.Subscript(ast_code,make_ast_constant(k))
            ast_code = ast.Subscript(ast_code,make_ast_constant(0))
        return ast.Attribute(ast_code,attr)

    phantoms = [
        make_ast_path(attr,path)
        for (attr,path) in phantom_vars
        ]

    return (
        dependencies , # str set
        phantoms) # ast.Module



def inspection(n : S_node,g : S_graph,our_global, phantoms=[]):
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
            # del tmp_local[tar]
        
    #    for tar in n.all_targets:
            #tar_info = g.dict_info[tar]
            #assert(tar_info.ttype == torch.Tensor)
            # tmp_local[tar].data = torch.zeros(0,device=device)
    #        del tmp_local[tar]
    def fct_del_fwd():
        for tar in n.tensor_targets:
            #tar_info = g.dict_info[tar]
            #assert(tar_info.ttype == torch.Tensor)
            tmp_local[tar].data = torch.zeros(0,device=device)
        code = ""
        for tar in phantoms:
            # if ast_to_str(tar).split('.')[0] not in our_global.keys():
            code += ast_to_str(ast.Assign(
                [ast.Attribute(tar,"data")],
                ast.Call(
                    ast.Attribute(ast.Name("torch"),"zeros"),
                    [make_ast_constant(0)],
                    [ast.alias("device=device")]
                    )
                ))+'\n'
        exec(code, our_global, tmp_local)
    # -- measure forward --
    _ , mem_run_fwd , peak_fwd = memUsage.measure(fct_run_fwd)
    #if mem_run_fwd.v<0: print(n.get_code())
    overhead_fwd = peak_fwd - mem_run_fwd
    _ , mem_del_fwd , _ = memUsage.measure(fct_del_fwd)
    ret["mem_del_fwd"] = minus_mem(mem_del_fwd)
    _ , _ , _ = memUsage.measure(fct_run_fwd)

    _ , mem_fgt_fwd , _ = memUsage.measure(fct_fgt_fwd)
    time_run_fwd = measure_time(fct_run_fwd)
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
    # # ===============
    return ret

# def inspection_del(n : S_node,g : S_graph,our_global, phantoms=[]):
#     mt = n.main_target
#     info = g.dict_info[mt]
#     timer = rotor.timing.make_timer(device)
#     memUsage = rotor.memory.MeasureMemory(device)
#     tmp_local = generate_tmp_local(n,g,our_global)
#     ret = {}

#     # -- aux --
#     def measure_time(fct, inter_fct=None):
#         duration = timer.measure_median(fct,samples=1)
#         if duration < min_duration:
#             number_repetitions = 1 + int(min_duration // duration)
#             for _ in range(1,number_repetitions):
#                 if inter_fct:
#                     inter_fct()
#                 duration += timer.measure_median(fct,samples=1)
#         else: number_repetitions = 1
#         return duration/number_repetitions
#     # ---------

#     # === FORWARD ===
#     def fct_run_fwd():
#         exec(n.get_code(), our_global, tmp_local)
#     def fct_fgt_fwd():
#         for tar in n.tensor_targets:
#             #tar_info = g.dict_info[tar]
#             #assert(tar_info.ttype == torch.Tensor)
#             tmp_local[tar].data = torch.zeros(0,device=device)
#             # del tmp_local[tar]
            
#     def fct_del_fwd():
#         for tar in n.tensor_targets:
#             #tar_info = g.dict_info[tar]
#             #assert(tar_info.ttype == torch.Tensor)
#             tmp_local[tar].data = torch.zeros(0,device=device)
#         code = ""
#         for tar in phantoms:
#             # if ast_to_str(tar).split('.')[0] not in our_global.keys():
#             code += ast_to_str(ast.Assign(
#                 [ast.Attribute(tar,"data")],
#                 ast.Call(
#                     ast.Attribute(ast.Name("torch"),"zeros"),
#                     [make_ast_constant(0)],
#                     [ast.alias("device=device")]
#                     )
#                 ))+'\n'
#         if "src" not in n.get_code():#TODO: find a better way to deal with src
#             exec(code, our_global, tmp_local)
    # -- measure forward --
#     _ , mem_run_fwd , peak_fwd = memUsage.measure(fct_run_fwd)
#     overhead_fwd = peak_fwd - mem_run_fwd
#     _ , mem_fgt_fwd , _ = memUsage.measure(fct_fgt_fwd)
#     time_run_fwd = measure_time(fct_run_fwd)
#     ret["overhead_fwd"] = overhead_fwd
#     ret["mem_run_fwd"] = mem_run_fwd
#     ret["mem_fgt_fwd"] = minus_mem(mem_fgt_fwd)
#     ret["time_run_fwd"] = time_run_fwd
#     # ===============

#     # === BACKWARD ===
#     if info.requires_grad:
#         tmp_local[mt].data = generate_val(info,device)
#         tmp_local[mt].grad = generate_val(info,device)

#         def fct_run_bwd():
#             exec(f"{mt}.backward({mt}.grad)", our_global, tmp_local)
#         def fct_fgt_bwd():
#             for req_n in n.req:
#                 #if not (req_n is g.init_node):
#                 if not req_n.is_artefact:
#                     for tar in req_n.tensor_targets:
#                         tmp_local[tar].grad = None

#         # measure
#         _ , mem_run_bwd , peak_bwd = memUsage.measure(fct_run_bwd)
#         overhead_bwd = peak_bwd - mem_run_bwd
#         _ , mem_fgt_bwd , _ = memUsage.measure(fct_fgt_bwd)
#         fct_run_fwd()
#         timer.measure_median(fct_run_fwd)
#         tmp_local[n.main_target].grad = generate_val(info,device)
#         time_run_bwd = measure_time(fct_run_bwd, fct_run_fwd)
#         # overhead_bwd contains n.target.data now /!\

#         ret["overhead_bwd"] = overhead_bwd
#         ret["mem_run_bwd"]  = mem_run_bwd
#         ret["mem_fgt_bwd"]  = minus_mem(mem_fgt_bwd)
#         ret["time_run_bwd"] = time_run_bwd
    # ===============
    # _ , _ , _ = memUsage.measure(fct_run_fwd)
    # _ , mem_del_fwd , _ = memUsage.measure(fct_del_fwd)
    # ret["mem_del_fwd"] = minus_mem(mem_del_fwd)
    # tmp_local = generate_tmp_local(n,g,our_global)
    # try:
    #     _ , _ , _ = memUsage.measure(fct_run_fwd)
    # except:
    #     print(n.get_code())
    #     print(our_global['self'].wte.weight)
    #     print(g.init_node.get_code())
    #     print(our_global['src'])
    #     print(tmp_local['src'])
    # return ret

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
    #print(sg.direct_inputs)

    for inp in sg.direct_inputs:
        # print('input:', inp)
        info = sg.dict_info[inp]
        x = generate_val(info,device)
        our_global[inp]=x
        if inp in sg.hidden_inputs:
            info.memsize = MemSize(int(tensorMsize(x)))
        # our_global[inp] = generate_val(sg.dict_info[inp],device) #utils

    # ------------
    def handle_node(n : S_node):
        #print('weight.shape', our_global['self'].h[0].attn.c_attn.weight.shape)

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

        # print(dict_Kfwd.keys())
        # -- build Kbwd --
        info = sg.dict_info[mt]
        if info.requires_grad:
            print_debug(f"{mt} req bwd")
            #if '__116_input0' in our_global.keys(): print(our_global['__116_input0'].shape, n.main_target)

            # -- extract "real" dependencies through grad_fn --
            try:
                dep , phantoms = get_dependencies_and_phantoms(
                    n,sg,our_global)
            except: 
                print(n.get_code(), our_global['__13_input0'].shape)
                # print(our_global['self'].h[0].ln_1.weight.shape)
            #     if '__116_input0' in our_global.keys(): print(our_global['__116_input0'].shape, n.main_target)

            try:Kbwd_req = set(dict_Kfwd[d] for d in dep)
            except: print(dict_Kfwd, dep, n.get_code())
            Kfwd.phantoms = phantoms

            Kbwd = K_node(
                is_fwd=False, req=Kbwd_req, target=mt, info=info)
            dict_Kbwd[mt] = Kbwd
            for sub_n in n.req:
                sub_tar = sub_n.main_target
                if sub_tar in dict_Kbwd: # requires_grad
                    dict_Kbwd[sub_tar].req.add(Kbwd)
            
        # print(dict_Kfwd.keys())
        
        # -- inspection --
        if info.ttype == torch.Size:
            Kfwd.run_mem  = MemSize(0)
            Kfwd.fgt_mem  = MemSize(0)
            Kfwd.del_mem  = MemSize(0)
            Kfwd.overhead = MemSize(0)
            Kfwd.time     = 0
        else:
            res = inspection(n,sg,our_global,Kfwd.phantoms)
            Kfwd.run_mem  = res["mem_run_fwd"]
            Kfwd.fgt_mem  = res["mem_fgt_fwd"]
            Kfwd.del_mem  = res["mem_del_fwd"]
            # del_global = our_global.copy()
            # Kfwd.del_mem  = inspection_del(n,sg,our_global,Kfwd.phantoms)["mem_del_fwd"]
            # if our_global['self'].h[0].attn.c_attn.weight.shape[0]<1:print([ast_to_str(p) for p in Kfwd.phantoms])

            #if '__116_input0' in our_global.keys(): print(our_global['__116_input0'].shape, n.main_target)
            Kfwd.overhead = res["overhead_fwd"]
            Kfwd.time     = res["time_run_fwd"]
            info.memsize  = res["mem_fgt_fwd"]
            if info.requires_grad:
                Kbwd.run_mem  = res["mem_run_bwd"]
                Kbwd.fgt_mem  = res["mem_fgt_bwd"]
                Kbwd.overhead = res["overhead_bwd"]
                Kbwd.time     = res["time_run_bwd"]
                
            if Kfwd.run_mem != Kfwd.fgt_mem:#a_bar case
                Kfwd.abar = True
                if info.requires_grad:
                    Kbwd.abar = True
                    Kbwd.req.add(Kfwd)
        # print(dict_Kfwd.keys())
    # ------------
    for n in sg.nodes:
        # print(n.main_target, dict_Kfwd)
        handle_node(n)
        # print(n.main_target, dict_Kfwd)

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
    loss_node.del_mem  = MemSize(0)
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

