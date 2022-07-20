from .root import *
from .Dtools import D_node,D_graph

# ==========================
# ====== S structure =======
# ==========================

class S_node():
    def __init__(self,code=None,protected=False,fct="",target="No target"):
        self.is_artefact = False
        self.main_code = code
        self.main_fct = fct
        self.body_code = [] # list of ast.Assign
        self.main_target = target # str
        self.all_targets = [target]
        self.tensor_targets = [] # later
        self.req = set()
        self.used_by = set()
        self.protected = protected

    def full_code(self):
        if self.main_code is None: mc = []
        else: mc = [self.main_code]
        return make_ast_module(mc + self.body_code)
    def get_code(self):
        return ast_to_str(self.full_code())

    def insert(self,sub_n,strong,g):
        # this is the fct to merge nodes : we insert "sub_n" in "self"
        # if strong: delete sub_n else: sub_node <- artefact
        # in any case cut as many edges as possible

        merged_req = (self.req | sub_n.req) - {self}

        # -- disconnect sub_n with its children (if possible) --
        if strong: # e.g. for "view"
            for sub_sub_n in sub_n.used_by:
                sub_sub_n.req.discard(sub_n)
            merged_used_by = (self.used_by | sub_n.used_by) - {sub_n}
            sub_n.used_by = set()
        else: # e.g. for "size"
            for sub_sub_n in self.used_by:
                sub_sub_n.req.discard(sub_n)
                sub_n.used_by.discard(sub_sub_n)
            merged_used_by = self.used_by
        # -- if sub_n is deleted, remove it from parents' used_by --
        if sub_n.used_by == set():
            for req_n in sub_n.req:
                req_n.used_by.discard(sub_n)
            sub_n.req = set()
            # -> sub_n has been fully unpluged
        else:
            sub_n.is_artefact = True
            # -> artefact
        # -- insert sub_n code --
        assert(sub_n.main_code is not None)
        self.body_code.append(sub_n.main_code)
        self.body_code.extend(sub_n.body_code)
        self.all_targets.extend(sub_n.all_targets)
        self.req = merged_req
        self.used_by = merged_used_by
        for req_n in merged_req:
            req_n.used_by.add(self)
        for sub_sub_n in merged_used_by:
            sub_sub_n.req.add(self)

        # -- special case if sub_n is the output --
        if sub_n is g.output_node:
            g.output_node = self

    def clear_children_artefact(self):
        # clean useless artefact children of self
        children = set(self.used_by)
        for sub_n in children:
            if sub_n.is_artefact:
                if sub_n.req != {self}:
                    s = ",".join(
                        [aux_n.main_target for aux_n in sub_n.req])
                    raise Exception(
                      f"{self.main_target} should be the only parent of "\
                      f"{sub_n.main_target} : {len(sub_n.req)}\n{s}")
                for aux_n in self.used_by:
                    sub_n.used_by.discard(aux_n)
                    aux_n.req.discard(sub_n)
                if sub_n.used_by == set():
                    for aux_n in sub_n.req:
                        aux_n.used_by.remove(sub_n)
                    sub_n.req = set()

    def clear_siblings_artefact(self):
        real_req = set()
        for req_n in self.req:
            if not req_n.is_artefact:
                real_req.add(req_n)
        for req_n in real_req:
            req_n.clear_children_artefact()

class S_graph():
    def __init__(self,dg : D_graph = None):
        self.nodes          = []
        self.init_node      = None
        self.output_node    = None
        self.hidden_inputs  = [] # str list
        self.direct_inputs  = [] # str list
        self.hidden_output  = "" # str
        self.direct_outputs = [] # str list
        self.dict_info      = {}
        self.dict_rand      = {}
        if dg:
            self.hidden_inputs  = dg.inputs
            self.direct_outputs = [dg.output]
            self.dict_info      = dg.dict_info
            self.dict_rand      = dg.dict_rand

    def make_io(self):
        # assert : hidden_inputs & direct_outputs exist
        # assert : init_node & output_node exist
        # make direct_inputs & hidden_ouput
        self.hidden_output = self.output_node.main_target
        self.direct_inputs = (
            self.hidden_inputs + self.init_node.all_targets
        )


    def check_artefact(self):
        for n in self.nodes:
            if n.is_artefact:# and not (n is self.init_node):
                if len(n.req)!=1:
                    raise Exception(
                      f"{n.main_target} is_artefact, but with "\
                      f"len(req)={len(n.req)}")
                req_n = list(n.req)[0]
                if n.used_by <= (req_n.used_by | set([n])):
                    print(f"{n.main_target} is a useless "\
                          f"artefact of {req_n.main_target}")

    def check_relations(self):
        # n1 in n2.used_by iff n2 in n1.req
        for n in self.nodes:
            for req_n in n.req:
                if not (n in req_n.used_by):
                    raise Exception(
                      f"{req_n.main_target} in {n.main_target}.req "\
                      f"but one sided relation...")
            for sub_n in n.used_by:
                if not (n in sub_n.req):
                    raise Exception(
                      f"{sub_n.main_target} in {n.main_target}.used_by "\
                      f"but one sided relation...")

    def clear(self):
        # -- re-sorting nodes -- 
        # due to merging, the topo order may not be correct anymore
        # by the way, remove unpluged nodes
        self.nodes = sort_based_on_req(self.output_node)
        self.nodes.remove(self.init_node)
        self.check_artefact()
        self.check_relations()
        self.make_io()

    def make_tensor_targets(self):
        for n in self.nodes:
            if not n.is_artefact:
                l = []
                for tar in n.all_targets:
                    if self.dict_info[tar].ttype==torch.Tensor:
                        l.append(tar)
                n.tensor_targets = l


    def assert_ready(self):
        # check if ready to be given to S_to_K
        # ie main_targets are tensors, except if artefact -> sizes
        for n in self.nodes:
            if not (n.main_target in self.dict_info):
                raise Exception(
                  f"{n.main_target} not in dict_info ??")
            info = self.dict_info[n.main_target]
            if not (info.ttype in [torch.Tensor,torch.Size]):
                raise Exception(
                  f"After simplifications there should "\
                  f"only be tensors or sizes, but {info.ttype} "\
                  f"found for {n.main_target}.")
            if info.ttype==torch.Size and not n.is_artefact:
                raise Exception(
                  f"After simplifications, all remaining "\
                  f"\"size\" should be \"artefacts\", but "\
                  f"{n.main_target} isn't an artefact")

# ==========================


# ==========================
# = Init move from D to S  =
# ==========================

def D_to_S_init(dg : D_graph,keep_sequential=False) -> S_graph:
    global ref_keep_seq ; ref_keep_seq = keep_sequential
    sg = S_graph(dg)
    init_node = S_node(target="-- inputs --")
    init_node.all_targets=[]
    s_nodes = sg.nodes
    dict_s_nodes = {} # to translate D to S
    for n in dg.nodes:
        sn = S_node(code=n.ast_code,
                protected=n.protected,
                fct=n.fct,
                target=n.target)
        s_nodes.append(sn)
        dict_s_nodes[n.target] = sn
        for req_n in n.req:
            sreq_n = dict_s_nodes[req_n.target]
            sn.req.add(sreq_n)
            sreq_n.used_by.add(sn)
    # -- merge all the inputs in the special "init_node" --
    for inp in dg.inputs:
        init_node.insert(dict_s_nodes[inp],strong=True,g=sg)
    init_node.body_code = []
    sg.init_node = init_node
    sg.output_node = dict_s_nodes[dg.output]
    sg.clear()
    return sg

# ==========================



# ==========================
# ==== Simplification 1 ====
# === remove cheap nodes ===
# ==========================

def insert_ast_code(main_n,mc,target : str,sc):
    # mc : main_code , sc : sub_code
    assert(isinstance(mc,ast.Assign))
    assert(isinstance(sc,ast.Assign))
    assert(sc.targets[0].id == target)
    # if not ast.Assign -> simplifications haven't been done in
    # the right order ! (cheap -> size > view)
    # assert main_code is a one layer Call (no sub calls)
    scv = sc.value
    mcv = mc.value
    if isinstance(mcv,ast.Call):
        args = []
        kwds = []
        for s in mcv.args:
            if isinstance(s,ast.Name) and s.id == target:
                args.append(scv)
            else: args.append(s)
        for k in mcv.keywords:
            if isinstance(s,ast.Name) and s.id == target:
                kwds.append(scv)
            else: kwds.append(s)
        ret = ast.Call(mcv.func,args,kwds)
        main_n.main_code = ast.Assign(mc.targets,ret)
    elif (isinstance(mcv,ast.Tuple)
        or isinstance(mcv,ast.List)):
        l = []
        for s in mcv.elts:
            if isinstance(s,ast.Name) and s.id == target:
                l.append(scv)
            else: l.append(s)
        ret = type(mcv)(l)
        main_n.main_code = ast.Assign(mc.targets,ret)
    elif isinstance(mcv,ast.Subscript):
        assert(isinstance(scv,ast.List)
            or isinstance(scv,ast.Tuple))
        assert(len(mc.targets)==1)
        # mcv = scv.elts[mcv.slice.value]
        main_n.main_code = ast.Assign(mc.targets,scv.elts[mcv.slice.value])
        simplify_node(main_n)
    else:
        print(ast.dump(mc,indent=4))
        raise Exception(
            f"unknown type of code where we should "\
            f"insert things: {type(mc.value)}")

def simplify_node(n):
    # aux fct, insert n.ast_code in children's code, and unplug it
    for sub_n in n.used_by:
        # -- unplug n --
        sub_n.req.update(n.req)
        sub_n.req.discard(n)
        req = set(n.req)
        for req_n in req:
            req_n.used_by.add(sub_n)
            req_n.used_by.discard(n)
        # -- insert the code --
        insert_ast_code(sub_n,sub_n.main_code,n.main_target,n.main_code)
    n.req     = set()
    n.used_by = set()

def simplify_cheap(g : S_graph):
    # from root to leaves
    for n in g.nodes:
        if ( (not n is g.output_node)
         and n.main_fct in list_cheap_fct
         and (not ref_keep_seq or not n.protected)):
            simplify_node(n)
    g.clear()

# ==========================



# ==========================
# ==== Simplification 2 ====
# === insert size nodes ====
# ==========================

# 1) merge the size nodes which have the same parent
# 2) insert the size nodes in the body code of the
#    parent, and keep them only if needed -> artefact

def size_children(g,n):
    # give the list of child nodes of n which are size
    ret = []
    for sub_n in n.used_by:
        if g.dict_info[sub_n.main_target].ttype == torch.Size:
            ret.append(sub_n)
    return ret


def simplify_size(g : S_graph):
    # from leaves to root
    nodes = [g.init_node] + list(g.nodes) ; nodes.reverse()
    for n in nodes:
        if not n is g.output_node:
            list_size = size_children(g,n)
            if list_size != []:
                # -- merge into one node --
                size_n = list_size[0]
                for other_n in list_size[1:]:
                    size_n.insert(other_n,strong=True,g=g)
                # -- insert their code --
                if n is g.init_node:
                    n.insert(size_n,strong=True,g=g)
                else: n.insert(size_n,strong=False,g=g)
    g.clear()

# ==========================



# ==========================
# ==== Simplification 3 ====
# === remove view nodes ====
# ==========================

def simplify_view(g):
    # from root to leaves
    g.init_node.is_artefact = True
    for n in g.nodes:
        #if ( n.main_target != g.output
        #    and (not ref_keep_seq or not n.protected)
         if n.main_fct in list_view_fct or n.main_fct == "getattr":
            # /!\ ASSERTION remaining getattr are related to views !! 
            real_req = []
            for req_n in n.req:
                if not req_n.is_artefact:
                    real_req.append(req_n)
            if len(real_req)==1:
                req_n = real_req[0]
                req_n.insert(n,strong=True,g=g)
                req_n.clear_siblings_artefact()
            elif len(real_req)==0 and len(n.req)>0:
                # experimental : I assume that views which don't 
                # require any real tensor are views over parameters
                # so mem=0 and no bwd K_node, so I can insert them
                # in their parents even if they are artefacts.
                # But artefact nodes aren't safe, they might disappear
                # if self.used_by sub set of self.parent.used_by
                # so I must share the code with artifacts' parents
                # I can insert the code in many different nodes
                # because views operations are cheap.
                # But I must avoid creating cycle dependancies, so
                # for the moment I assert len(n.req)==1
                if len(n.req)>1: print(
                    f"Warning : {n.main_target} is a view op, but without"\
                    f" a real parent, and several artifact dependancies",
                    file = sys.stderr)
                else:
                    art_req = list(n.req)[0]
                    assert(len(art_req.req)==1)
                    real_req = list(art_req.req)[0]
                    for aux_n in [art_req,real_req]:
                        aux_n.body_code.append(n.main_code)
                        aux_n.body_code.extend(n.body_code)
                    art_req.used_by.update(n.used_by)
                    for sub_n in n.used_by:
                        sub_n.req.add(art_req)
                        sub_n.req.remove(n)
                    art_req.used_by.remove(n)
                    n.req = set()
                    n.used_by = set()
                    real_req.clear_children_artefact()

    g.clear()

# ==========================



# ==========================
# = Move from D to S graph =
# ==========================

def D_to_S(dg,keep_sequential=False):
    sg = D_to_S_init(dg,keep_sequential)
    simplify_cheap(sg)
    simplify_size(sg)
    simplify_view(sg)
    sg.check_relations()
    sg.make_tensor_targets()
    sg.assert_ready()
    return sg

# ==========================



# ==========================
# ==== Cut the graph in ====
# ==== sequential parts ====
# ==========================

def copy_node(n : S_node): # aux for copy_graph
    new_n = S_node()
    new_n.is_artefact    = n.is_artefact
    new_n.main_code      = n.main_code
    new_n.main_fct       = n.main_fct
    new_n.body_code      = list(n.body_code)
    new_n.main_target    = n.main_target
    new_n.all_targets    = list(n.all_targets)
    new_n.tensor_targets = list(n.tensor_targets)
    new_n.req            = set() # /!\
    new_n.used_by        = set() # /!\
    new_n.protected      = n.protected
    return new_n

def copy_graph(g : S_graph):
    # -> a copy of g with fresh nodes
    new_g = S_graph()
    new_g.hidden_inputs  = list(g.hidden_inputs)
    new_g.direct_inputs  = list(g.direct_inputs)
    new_g.hidden_output  = g.hidden_output
    new_g.direct_outputs = list(g.direct_outputs)
    new_g.dict_info      = g.dict_info
    new_g.dict_rand      = g.dict_rand
    dict_nodes = {}
    new_init = copy_node(g.init_node)
    new_nodes = []
    dict_nodes[new_init.main_target] = new_init
    for n in g.nodes:
        new_n = copy_node(n)
        new_nodes.append(new_n)
        dict_nodes[n.main_target] = new_n
        for req_n in n.req:
            new_req = dict_nodes[req_n.main_target]
            new_req.used_by.add(new_n)
            new_n.req.add(new_req)
    new_g.init_node     = new_init
    new_g.output_node   = dict_nodes[g.hidden_output]
    new_g.nodes         = new_nodes
    return new_g


def cut(g : S_graph): # -> list of S_graph
    main_g = copy_graph(g) # to protect from side effects
    main_g.nodes.insert(0,main_g.init_node)
    seps = cut_based_on_req(main_g)
    print_debug(f"S separators : {[sep.main_target for sep in seps]}")
    list_g = []
    for i in range(1,len(seps)):
        new_g = S_graph()
        list_g.append(new_g)
        # -- get nodes --
        inp_node = seps[i-1]
        out_node = seps[i]
        inp_i = main_g.nodes.index(inp_node)
        out_i = main_g.nodes.index(out_node)
        nodes = main_g.nodes[inp_i+1:out_i+1]
        new_g.nodes = nodes
        print_debug(f"size of bloc {i} : {out_i}-{inp_i}")
        # -- input --
        if i==1:
            new_g.init_node = main_g.init_node
            new_g.hidden_inputs = main_g.hidden_inputs
            new_g.direct_inputs = main_g.direct_inputs
        else:
            ino = S_node(
                target=f"init_node of bloc {i}>1, should NEVER be used")
            new_g.hidden_inputs = [inp_node.main_target]
            new_g.direct_inputs = inp_node.all_targets
            new_g.init_node = ino
            for sub_n in inp_node.used_by:
                sub_n.req.remove(inp_node)
                sub_n.req.add(ino)
                ino.used_by.add(sub_n)
            inp_node.used_by = set() # previous bloc's output node
        # -- output --
        new_g.output_node    = out_node
        new_g.hidden_output  = out_node.main_target
        new_g.direct_outputs = out_node.all_targets
        # --
        new_g.dict_info = main_g.dict_info
        new_g.dict_rand = main_g.dict_rand
    return list_g

# ==========================



# ==========================
# === printing functions ===
# ==========================

def aux_print_graph(dot,g,uniq_num):
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2): dot.edge(uni(i1),uni(i2))
    str_ino = g.init_node.main_target
    node(str_ino,g.init_node.get_code(),style="dashed")
    for n in g.nodes:
        if n.is_artefact:
            node(n.main_target,n.get_code(),style="dashed")
        else: node(n.main_target,n.get_code())
    for n in g.nodes:
        for sub_n in n.req:
            edge(sub_n.main_target,n.main_target)

    # -- io --
    str_inp = "\n".join(g.direct_inputs)
    node("input",
        f"INPUT : {str_inp}",
        color="green",style="dashed")
    str_out = "\n".join(g.direct_outputs)
    node("output",
        f"OUTPUT : {str_out}\nhidden : {g.hidden_output}",
        color="green",style="dashed")
    edge("input",g.init_node.main_target)
    edge(g.hidden_output,"output")


def print_S_graph(g : S_graph,name=None,open=True):
    print(f"Simplified forward graph : {len(g.nodes)} nodes")
    if name is None: name = "forward S-graph"
    dot = graphviz.Digraph(
        name,
        comment="S_graph = Simplified forward graph")
    aux_print_graph(dot,g,0)
    graph_render(dot,open,"S") # from root.py


def print_S_graph_list(list_g,name=None,open=True):
    s = "+".join([str(len(g.nodes)) for g in list_g])
    print(
        f"{len(list_g)} blocs of S_graph, with {s} = "\
        f"{sum([len(g.nodes) for g in list_g])} nodes")
    if name is None: name = "cut forward S-graph"
    dot = graphviz.Digraph(
        name,
        comment="S_graph list : cut simplified forward graph")
    for i in range(len(list_g)):
        aux_print_graph(dot,list_g[i],i)
    graph_render(dot,open,"S") # from root.py

# ==========================
