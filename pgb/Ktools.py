from .utils import *
from .Stools import S_node,S_graph
import copy
import gc
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
        self.inspector = None
        # self.phantoms = []
    def __eq__(self,n2):
        # agressive tests, if diff then raise Error
        n1 = self
        b = check_attr(n1,n2,[
            "is_fwd","main_target",
            "all_targets","tensor_targets",
            "name","is_artefact","info","abar",
            "run_mem","fgt_mem","del_mem","overhead"],
            raise_exception=False)
        mkstr = lambda nl : [rn.main_target for rn in sort_targets(nl)]
        b = (b
            and (mkstr(n1.req) == mkstr (n2.req))
            and (mkstr(n1.used_by) == mkstr (n2.used_by))
            and (n1.get_code() == n2.get_code()))
        # TIME AND MEMORY : if not equal then raise Exception
        if not b: return False#raise Exception("not equal on req,used_by or code")
        t1 = n1.time ; t2 = n2.time ; r = ref_reasonable_rate[0]
        if not (((t1 == t2)
            or (isinstance(t1,float) and isinstance(t2,float)
            and (abs(t1 - t2) < (r * max(t1,t2)))))):return False
            #raise Exception(
            #    f"time diff - t1: {t1} - t2: {t2} on {n1.main_target}")
        return True
    def __hash__(self):
        return self.main_target.__hash__()
        #return id(self) # __eq__ => need __hash__

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
    #def get_fgt_code(self):
    #    code = ""
    #    for tar in self.phantoms:
    #        code += ast.Assign(
    #            [ast.Attribute(tar,"data")],
    #            ast.Call(
    #                ast.Attribute(ast.Name("torch"),"zeros"),
    #                [make_ast_constant(0)],
    #                []
    #                )
    #            )
    #    return code

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
    def __eq__(self,g2):
        if ast_to_str(self.init_code) != ast_to_str(g2.init_code):
            raise Exception("diff init_code")
        return check_attr(self,g2,["sg",
            "direct_inputs","hidden_inputs",
            "direct_outputs","hidden_output",
            "dict_info","dict_nodes","loss_node"],raise_exception=True)
    def __hash__(self):
        return id(self)

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

class inspector():
    def __init__(self, n : S_node,g : S_graph,
                 our_global, phantoms=[]):
        self.n = n
        self.mt = self.n.main_target
        self.info = g.dict_info[self.mt]
        self.timer = rotor.timing.make_timer(device)
        self.memUsage = rotor.memory.MeasureMemory(device)
        self.our_global = our_global
        #with torch.no_grad():
        self.tmp_local = generate_tmp_local(self.n,g,our_global)
        self.phantoms = phantoms
        self.ret = {}
    
    # -- aux --
    def measure_time(self, fct, inter_fct=None):
        t = self.timer.measure_median(fct,samples=1)
        nb_repeat = 1
        measures = [t] ; tot = t
        while (tot < time_min_duration or nb_repeat < time_min_repeat):
            if inter_fct: inter_fct()
            t = self.timer.measure_median(fct,samples=1)
            measures.append(t)
            tot += t ; nb_repeat += 1
        if len(measures)>2:
            return (sum(measures)-max(measures)-min(measures))/(len(measures)-2)
        else:np.median(measures)
    # ---------

    # === FORWARD ===
    # -- measure forward --
    def measure_fwd(self,only_run=False):
        def fct_run_fwd():
            self.code_run_fwd = self.n.get_code() 
            exec(self.code_run_fwd, self.our_global, self.tmp_local)

        def fct_fgt_fwd():
            for tar in self.n.tensor_targets:
                self.tmp_local[tar].data = torch.zeros(0,device=device)
            
        def fct_del_fwd():
            code = ""
            for tar in self.n.tensor_targets:
                code += f"del {tar};"
            self.code_del_fwd = code#Only include the phantom part 
            exec(self.code_del_fwd, self.our_global, self.tmp_local)
        #gc.disable()
        _ , mem_run_fwd , peak_fwd = self.memUsage.measure(fct_run_fwd)
        overhead_fwd = peak_fwd - mem_run_fwd
        self.ret["overhead_fwd"] = overhead_fwd
        self.ret["mem_run_fwd"] = mem_run_fwd
        if not only_run:
            _ , mem_del_fwd , _ = self.memUsage.measure(fct_del_fwd)
            self.ret["mem_del_fwd"] = minus_mem(mem_del_fwd)
            _ , _ , _ = self.memUsage.measure(fct_run_fwd)

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret["mem_fgt_fwd"] = minus_mem(mem_fgt_fwd)
            self.ret["time_run_fwd"] = time_run_fwd
        #gc.enable()
    # ===============

    # === BACKWARD ===

    def fct_run_bwd(self):
        rec_list = []
        for sub_n in self.n.used_by:
            rec_list += sub_n.tensor_targets
        inputs = ",".join(rec_list)
        if inputs:
            exec(f"{self.mt}.backward({self.mt}.grad, inputs=[{inputs}])", self.our_global, self.tmp_local)
        else:
            exec(f"{self.mt}.backward({self.mt}.grad)", self.our_global, self.tmp_local)

    def fct_fgt_bwd(self):
        for req_n in self.n.req:
            #if not (req_n is g.init_node):
            if not req_n.is_artefact:
                for tar in req_n.tensor_targets:
                    self.tmp_local[tar].grad = None
    def fct_prepare_bwd(self):
        self.code_run_fwd = self.n.get_code()
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.n.main_target].grad = generate_val(self.info,device)

    # measure
    def measure_bwd(self):
        #def fct_run_fwd():
        #    self.code_run_fwd = self.n.get_code() 
        #    exec(self.code_run_fwd, self.our_global, self.tmp_local)
        if self.info.requires_grad:
            #self.tmp_local[self.mt].data = generate_val(self.info,device)
            #self.tmp_local[self.mt].grad = generate_val(self.info,device)
            #gc.disable()
            self.fct_prepare_bwd()
            _ , mem_run_bwd , peak_bwd = self.memUsage.measure(self.fct_run_bwd)
            overhead_bwd = peak_bwd - mem_run_bwd
            _ , mem_fgt_bwd , _ = self.memUsage.measure(self.fct_fgt_bwd)
            #fct_run_fwd()
            #self.timer.measure_median(fct_run_fwd)

            #self.tmp_local[self.n.main_target].grad = generate_val(self.info,device)
            self.fct_prepare_bwd()
            time_run_bwd = self.measure_time(self.fct_run_bwd, self.fct_prepare_bwd)
            # overhead_bwd contains n.target.data now /!\
            #gc.enable()
            self.ret["overhead_bwd"] = overhead_bwd
            self.ret["mem_run_bwd"]  = mem_run_bwd
            self.ret["mem_fgt_bwd"]  = minus_mem(mem_fgt_bwd)
            self.ret["time_run_bwd"] = time_run_bwd
    # # ===============

# aux function to handle verbose and device
def aux_init_S_to_K(model,verbose,d):
    global device
    device = d if d else (
        get_device_and_check_all_same_device(model,dict(),True))
    if not (verbose is None): ref_verbose[0] = verbose


# the function that does it all
def aux_build_S_to_K(sg : S_graph,model):
    # -- init --
    dict_Kbwd = dict() # dict : target -> K_node(bwd)
    dict_Kfwd = dict() # dict : target -> K_node(fwd)
    our_global = {}#globals().copy()
    our_global["self"] = model
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
        our_global = globals().copy()
        our_global["self"] = model
        our_global["device"] = device
        for inp in sg.direct_inputs:
            # print('input:', inp)
            info = sg.dict_info[inp]
            x = generate_val(info,device)
            our_global[inp]=x
            if inp in sg.hidden_inputs:
                info.memsize = MemSize(int(tensorMsize(x)))

        # print(dict_Kfwd.keys())
        # -- build Kbwd --
        info = sg.dict_info[mt]
        if info.requires_grad:
            print_debug(f"{mt} req bwd")
            #try:
            #    dep , phantoms = get_dependencies_and_phantoms(
            #        n,sg,our_global)
            #except: 
            #    print(n.get_code(), our_global['__13_input0'].shape)

            #try:Kbwd_req = set(dict_Kfwd[d] for d in dep)
            #except: print(dict_Kfwd, dep, n.get_code())
            #Kfwd.phantoms = phantoms

            Kbwd = K_node(
                is_fwd=False, req=set(Kreq), target=mt, info=info,body_code=[])
            dict_Kbwd[mt] = Kbwd
            for sub_n in n.req:
                sub_tar = sub_n.main_target
                if sub_tar in dict_Kbwd: # requires_grad
                    dict_Kbwd[sub_tar].req.add(Kbwd)
            
        
        # -- inspection --
        if info.ttype == torch.Size:
            Kfwd.run_mem  = MemSize(0)
            Kfwd.fgt_mem  = MemSize(0)
            Kfwd.del_mem  = MemSize(0)
            Kfwd.overhead = MemSize(0)
            Kfwd.time     = 0
        else:
            #res = inspection(n,sg,our_global,Kfwd.phantoms)
            ins = inspector(Kfwd,sg,our_global)
            ins.measure_fwd()
            ins.measure_bwd()
            res = ins.ret
            Kfwd.run_mem  = res["mem_run_fwd"]
            Kfwd.fgt_mem  = res["mem_fgt_fwd"]
            Kfwd.del_mem  = res["mem_del_fwd"]
            Kfwd.inspector = ins
            Kfwd.overhead = res["overhead_fwd"]
            Kfwd.time     = res["time_run_fwd"]
            info.memsize  = res["mem_fgt_fwd"]
            if info.requires_grad:
                Kbwd.run_mem  = res["mem_run_bwd"]
                Kbwd.fgt_mem  = res["mem_fgt_bwd"]
                Kbwd.overhead = res["overhead_bwd"]
                Kbwd.time     = res["time_run_bwd"]
                Kbwd.inspector = ins
                
            if Kfwd.run_mem.v != Kfwd.fgt_mem.v:#a_bar case
                Kfwd.abar = True
                if info.requires_grad:
                    Kbwd.abar = True
                    Kbwd.req.add(Kfwd)
            else: Kfwd.del_mem = Kfwd.fgt_mem
        k_list = list(ins.tmp_local.keys())
        for k in k_list: del ins.tmp_local[k]
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


def S_to_K(sg : S_graph,model,verbose=None,device=None):
    aux_init_S_to_K(model,verbose,device)
    return aux_build_S_to_K(sg,model)


def S_list_to_K_list(list_sg,model,verbose=None,device=None):
    aux_init_S_to_K(model,verbose,device)
    return [aux_build_S_to_K(sg,model) for sg in list_sg]

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

