from .utils import *
from .Dtools import D_node,D_graph

# ==========================
# ====== S structure =======
# ==========================

class S_node():
    def __init__(self,
            target="No target",fct="",
            code=None,protected=False,
            unique_id_generator = None):
        """
        A S_node is composed by one "real" computation, defining the
        "main_target", and followed by size / view operations over it.
        Attributes :
        .main_target : str
        .all_targets : str list
            -> names of all the vars defined
            -> (including .main_target)
        .tensor_targets : str list
            -> all_targets which are tensors
            -> (done by s_graph.make_tensor_targets)
        .main_code  : tar*AST :
            -> .main_target * AST right part of the assigning code of it
        .body_code  : tar*AST list
            -> for every tar except main_target:
            -> target name * AST value of the assign
        .main_fct   : str  : fct used in .main_code
        .protected  : bool : see Doc (1-separator of the graph)
        .is_artefact: bool : see Doc (useful size node)
        .deps       : (S_node,str set) dict = dict_edges
            -> required nodes with the vars needed per node.
        .users      : dict_edges : reciprocal of .deps
        <TODO> : is_rand and deps_rand ?
        """
        self.is_artefact = False
        self.main_code = (target,code)
        self.main_fct = fct
        self.body_code = [] # list of tar * AST
        self.main_target = target # str
        self.all_targets = [target]
        self.tensor_targets = [] # later
        self.deps = dict()
        self.users = dict()
        self.protected = protected
        if unique_id_generator is None: self.unique_id = id(self)
        else:
            u = unique_id_generator[0]
            self.unique_id = u
            unique_id_generator[0] = u+1
    def __eq__(self,sn2):
        sn1 = self
        b = check_attr(sn1,sn2,[
            "is_artefact","main_fct",
            "main_target","all_targets",
            "tensor_targets","protected"])
        b = (b
            and dict_edges_eq(sn1.deps,sn2.deps)
            and dict_edges_eq(sn1.users,sn2.users)
            and (sn1.get_code() == sn2.get_code()))
        return b
    def __hash__(self):
        return self.unique_id
    # -> /!\ /!\ doing set/dict of S_nodes is dangereous /!\ /!\ 
    # but I'm doing this to avoid undeterminism

    def get_code(self):
        mc = make_str_assign(self.main_code)
        mc = "" if mc == "" else mc+"\n"
        bc = make_str_list_assign(self.body_code)
        return mc+bc

    def insert(self,aux_sn,strong,sg):
        # this is the fct to merge nodes : we insert "aux_sn" in "self"
        # if strong: delete aux_sn else aux_sn becomes an artefact
        # in any case cut as many edges as possible

        merged_deps = dict_edges_merge(self.deps,aux_sn.deps)
        dict_edges_discard_inplace(merged_deps,self)

        # -- disconnect aux_n with its children (if possible) --
        if strong: # e.g. for "view"
            dict_edges_discard_sn_from_deps_of_its_users(aux_sn)
            merged_users = dict_edges_merge(self.users,aux_sn.users)
            dict_edges_discard_inplace(merged_users,aux_sn)
            aux_sn.users = dict()
        else: # e.g. for "size"
            for user_sn in self.users.keys():
                dict_edges_discard_inplace(user_sn.deps,aux_sn)
                dict_edges_discard_inplace(aux_sn.users,user_sn)
            merged_users = self.users
        # -- if aux_sn is deleted, remove it from parents' users --
        if len(aux_sn.users) == 0:
            dict_edges_discard_sn_from_users_of_its_deps(aux_sn)
            aux_sn.deps = dict()
            # -> aux_sn has been fully unpluged
        else:
            aux_sn.is_artefact = True
            # -> artefact

        # -- insert aux_sn code --
        assert(aux_sn.main_code is not None)
        self.body_code.append(aux_sn.main_code)
        self.body_code.extend(aux_sn.body_code)
        self.all_targets.extend(aux_sn.all_targets)
        self.deps = merged_deps
        self.users = merged_users
        dict_edges_make_users_using_deps(self)
        dict_edges_make_deps_using_users(self)

        # -- special case if aux_n is the output --
        if aux_sn is sg.output_node:
            sg.output_node = self

    def clear_children_artefact(self):
        # clean useless artefact children of self
        children = dict(self.users)
        for art_user_sn in children.keys():
            if art_user_sn.is_artefact:
                if set(art_user_sn.deps.keys()) != {self}:
                    s = ",".join(
                      [aux_sn.main_target for aux_sn in art_user_sn.deps])
                    raise Exception(
                      f"{self.main_target} should be the only "\
                      f"parent of {art_user_sn.main_target} : "\
                      f"{len(art_user_sn.deps)}\n{s}")
                for other_user_sn in self.users.keys():
                    if art_user_sn in other_user_sn.deps.keys():
                        dict_edges_add_edge_inplace(
                            self,other_user_sn,
                            other_user_sn.deps[art_user_sn])
                        # We add the names of the variables generated 
                        # by the code of art_user_sn (present in self)
                        # which are used in other_user_sn. Because
                        # to get those vars we will no longer use
                        # art_user_sn since we already need self for
                        # something else.
                    dict_edges_discard_edge_inplace(
                        art_user_sn,other_user_sn)
                if art_user_sn.users == dict():
                    dict_edges_discard_sn_from_users_of_its_deps(
                        art_user_sn)
                    art_user_sn.deps = dict()

    def clear_siblings_artefact(self):
        real_deps = set()
        for req_sn in self.deps.keys():
            if not req_sn.is_artefact:
                real_deps.add(req_sn)
        for req_sn in real_deps:
            req_sn.clear_children_artefact()

class S_graph():
    def __init__(self,dg : D_graph = None,unique_id_generator=None):
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
        self.unique_id_generator = unique_id_generator
        # -> to generate S_node.__hash__
    def __eq__(self,sg2):
        return check_attr(self,sg2,[
            "direct_inputs","hidden_inputs",
            "direct_outputs","hidden_output",
            "dict_info","nodes"])
    def __hash__(self):
        return id(self)

    def make_io(self):
        # assert : hidden_inputs & direct_outputs exist
        # assert : init_node & output_node exist
        # make direct_inputs & hidden_ouput
        self.hidden_output = self.output_node.main_target
        self.direct_inputs = (
            self.hidden_inputs + self.init_node.all_targets
        )


    def check_artefact(self):
        for sn in self.nodes:
            if sn.is_artefact:# and not (sn is self.init_node):
                if len(sn.deps)!=1:
                    raise Exception(
                      f"{sn.main_target} is_artefact, but with "\
                      f"len(deps)={len(sn.deps)} (should be 1)")
                req_sn = list(sn.deps.keys())[0]
                if dict_edges_is_subset(sn.users,req_sn.users):
                    # if sn.users <= (req_sn.users | set([sn])): TO REMOVE
                    print(f"{sn.main_target} is a useless "\
                          f"artefact of {req_sn.main_target}")

    def check_relations(self):
        for sn in self.nodes:
            for (req_sn,str_set) in sn.deps.items():
                if (sn not in req_sn.users) or str_set != req_sn.users[sn]:
                    raise Exception(
                      f"{req_sn.main_target} in {sn.main_target}.deps "\
                      f"but one sided relation...")
            for (user_sn,str_set) in sn.users.items():
                if (sn not in user_sn.deps) or str_set != user_sn.deps[sn]:
                    raise Exception(
                      f"{user_sn.main_target} in {sn.main_target}.users "\
                      f"but one sided relation...")

    def clear(self):
        # -- re-sorting nodes -- 
        # due to merging, the topo order may not be correct anymore
        # by the way, remove unpluged nodes
        self.nodes = sort_based_on_deps(self.output_node)
        self.nodes.remove(self.init_node)
        self.check_artefact()
        self.check_relations()
        self.make_io()

    def make_tensor_targets(self):
        for sn in self.nodes:
            if not sn.is_artefact:
                l = []
                for tar in sn.all_targets:
                    if self.dict_info[tar].ttype==torch.Tensor:
                        l.append(tar)
                sn.tensor_targets = l


    def assert_ready(self):
        # check if ready to be given to S_to_K
        # ie main_targets are tensors, except if artefact -> sizes
        for sn in self.nodes:
            if not (sn.main_target in self.dict_info):
                raise Exception(
                  f"{sn.main_target} not in dict_info ??")
            info = self.dict_info[sn.main_target]
            if not (info.ttype in [torch.Tensor,torch.Size]):
                raise Exception(
                  f"After simplifications there should "\
                  f"only be tensors or sizes, but {info.ttype} "\
                  f"found for {sn.main_target}.")
            if info.ttype==torch.Size and not sn.is_artefact:
                raise Exception(
                  f"After simplifications, all remaining "\
                  f"\"size\" should be \"artefacts\", but "\
                  f"{sn.main_target} isn't an artefact")


# ==========================


# ==========================
# = Init move from D to S  =
# ==========================

def D_to_S_init(dg : D_graph,keep_sequential=False) -> S_graph:
    global ref_keep_seq ; ref_keep_seq = keep_sequential
    unique_id_generator = [0]
    sg = S_graph(dg,unique_id_generator)
    init_node = S_node(target="-- inputs --",
        unique_id_generator = unique_id_generator)
    init_node.all_targets=[]
    s_nodes = sg.nodes
    dict_s_nodes = {} # to translate D to S
    for dn in dg.nodes:
        sn = S_node(code=dn.ast_code,
                protected=dn.protected,
                fct=dn.fct,
                target=dn.target,
                unique_id_generator = unique_id_generator)
        s_nodes.append(sn)
        dict_s_nodes[dn.target] = sn
        for req_dn in dn.deps:
            req_sn = dict_s_nodes[req_dn.target]
            sn.deps[req_sn] = set((req_dn.target,))
            req_sn.users[sn] = set((req_dn.target,))
    # -- merge all the inputs in the special "init_node" --
    for inp in dg.inputs:
        init_node.insert(dict_s_nodes[inp],strong=True,sg=sg)
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

def insert_ast_code(main_sn,mc,st : str,sc):
    # st : sub target, sc : sub code
    # mc : main_sn.main_code
    # assert main_code is has depth=1 (no sub calls)
    if isinstance(mc,ast.Call):
        args = []
        kwds = []
        for s in mc.args:
            if isinstance(s,ast.Name) and s.id == st:
                args.append(sc)
            else: args.append(s)
        for k in mc.keywords:
            if isinstance(s,ast.Name) and s.id == st:
                kwds.append(sc)
            else: kwds.append(s)
        ret = ast.Call(mc.func,args,kwds)
        main_sn.main_code = (main_sn.main_target,ret)
    elif (isinstance(mc,ast.Tuple)
        or isinstance(mc,ast.List)):
        l = []
        for s in mc.elts:
            if isinstance(s,ast.Name) and s.id == st:
                l.append(sc)
            else: l.append(s)
        ret = type(mc)(l) # ast.Tuple/List(...)
        main_sn.main_code = (main_sn.main_target,ret)
    elif isinstance(mc,ast.Subscript):
        assert(isinstance(sc,ast.List)
            or isinstance(sc,ast.Tuple))
        ret = sc.elts[mc.slice.value]
        main_sn.main_code = (main_sn.main_target,ret)
        simplify_node(main_sn)
    else:
        print(ast.dump(mc,indent=4))
        raise Exception(
            f"unknown type of code where we should "\
            f"insert things: {type(mc.value)}")

def simplify_node(sn):
    # aux fct, insert n.ast_code in children's code, and unplug it
    for user_sn in sn.users.keys():
        # -- plug user_sn directly to deps of sn --
        dict_edges_merge_inplace(user_sn.deps,sn.deps)
        dict_edges_discard_inplace(user_sn.deps,sn)
        for (req_sn,str_set) in sn.deps.items():
            dict_edges_discard_inplace(req_sn.users,sn)
            dict_edges_add_inplace(req_sn.users,user_sn,str_set)
        # -- insert the code --
        insert_ast_code(
            user_sn,user_sn.main_code[1],
            sn.main_target,sn.main_code[1])
    sn.deps  = dict()
    sn.users = dict()

def simplify_cheap(sg : S_graph):
    # from root to leaves
    for sn in sg.nodes:
        if ( not (sn is sg.output_node)
         and sn.main_fct in list_cheap_fct
         and (not ref_keep_seq or not sn.protected)):
            simplify_node(sn)
    sg.clear()

# ==========================



# ==========================
# ==== Simplification 2 ====
# === insert size nodes ====
# ==========================

# 1) merge the size nodes which have the same parent
# 2) insert the size nodes in the body code of the
#    parent, and keep them only if needed -> artefact

def size_children(sg,sn):
    # give the list of child nodes of n which are size
    ret = []
    for user_sn in sn.users.keys():
        if sg.dict_info[user_sn.main_target].ttype == torch.Size:
            ret.append(user_sn)
    return ret


def simplify_size(sg : S_graph):
    # from leaves to root
    nodes = [sg.init_node] + list(sg.nodes) ; nodes.reverse()
    for sn in nodes:
        if not sn is sg.output_node:
            list_size = size_children(sg,sn)
            if list_size != []:
                # -- merge into one node --
                size_sn = list_size[0]
                for other_sn in list_size[1:]:
                    size_sn.insert(other_sn,strong=True,sg=sg)
                # -- insert their code --
                if (sn is sg.init_node
                or sg.dict_info[sn.main_target].ttype == torch.Size):
                    sn.insert(size_sn,strong=True,sg=sg)
                else: sn.insert(size_sn,strong=False,sg=sg)
    sg.clear()

# ==========================



# ==========================
# ==== Simplification 3 ====
# === remove view nodes ====
# ==========================

def get_all_real_deps(sn):
    candidates = set(sn.deps.keys())
    deps = set()
    while len(candidates) != 0:
        req_sn = candidates.pop()
        if req_sn not in deps:
            if not req_sn.is_artefact:
                deps.add(req_sn)
            else:
                candidates.update(req_sn.deps.keys())
    return deps

def get_direct_real_deps(sn):
    deps = get_all_real_deps(sn)
    for req_sn in deps:
        if get_all_real_deps(req_sn) ==  deps-set([req_sn]):
            return set([req_sn])
    return deps

def simplify_view(sg):
    # from root to leaves
    sg.init_node.is_artefact = True
    for sn in sg.nodes:
        #if ( sn.main_target != sg.output
        #    and (not ref_keep_seq or not sn.protected)
         if sn.main_fct in list_view_fct or sn.main_fct == "getattr":
            # /!\ ASSERTION remaining getattr are related to views !! 
            real_deps = get_direct_real_deps(sn)
            if len(real_deps)==1:
                req_sn = real_deps.pop()
                req_sn.insert(sn,strong=True,sg=sg)
                req_sn.clear_siblings_artefact()
            elif len(real_deps) > 1: print(
                f"Warning : {sn.main_target} is a view op, with "\
                f"several tensor deps, thus it's impossible to "\
                f"to simplify it, very dangerous...",
                file = sys.stderr)
            elif len(real_deps)==0 and len(sn.deps)>0:
                # experimental : I assume that views which don't 
                # require any real tensor are views over parameters
                # so mem=0 and no bwd K_node, so I can insert them
                # in their parents even if they are artefacts.
                # But artefact nodes aren't safe, they might disappear
                # if self.users sub set of self.parent.users
                # so I must share the code with artifacts' parent
                # It's not a problem to insert the code in different 
                # nodes because view operations are cheap.
                # But I must avoid creating cycle dependancies, so
                # for the moment I assert len(sn.deps)==1
                if len(sn.deps)>1: print(
                    f"Warning : {sn.main_target} is a view op, without "\
                    f"a real parent, and with several artifact deps",
                    file = sys.stderr)
                else:
                    art_req = list(sn.deps.keys())[0]
                    assert(len(art_req.deps)==1) # as an artefact
                    real_req = list(art_req.deps.keys())[0]
                    # - Insert sn's code both in art_req and real_req -
                    for aux_sn in [art_req,real_req]:
                        aux_sn.body_code.append(sn.main_code)
                        aux_sn.body_code.extend(sn.body_code)
                    # - plug art_req to sn's users -
                    dict_edges_merge_inplace(art_req.users,sn.users)
                    for (user_sn,str_set) in sn.users.items():
                        dict_edges_add_inplace(user_sn.deps,art_req,str_set)
                    # - unplug sn -
                    dict_edges_discard_inplace(art_req.users,sn)
                    dict_edges_discard_sn_from_deps_of_its_users(sn)
                    sn.deps = dict()
                    sn.users = dict()
                    real_req.clear_children_artefact()

    sg.clear()

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

def copy_S_node(sn : S_node): # aux for copy_S_graph
    new_sn = S_node()
    new_sn.is_artefact    = sn.is_artefact
    new_sn.main_code      = tuple(sn.main_code)
    new_sn.main_fct       = sn.main_fct
    new_sn.body_code      = [tuple(c) for c in sn.body_code]
    new_sn.main_target    = sn.main_target
    new_sn.all_targets    = list(sn.all_targets)
    new_sn.tensor_targets = list(sn.tensor_targets)
    new_sn.deps           = dict() # /!\
    new_sn.users          = dict() # /!\
    new_sn.protected      = sn.protected
    new_sn.unique_id      = sn.unique_id
    return new_sn

def copy_S_graph(sg : S_graph):
    # -> a copy of sg with fresh nodes
    new_sg = S_graph()
    new_sg.hidden_inputs  = list(sg.hidden_inputs)
    new_sg.direct_inputs  = list(sg.direct_inputs)
    new_sg.hidden_output  = sg.hidden_output
    new_sg.direct_outputs = list(sg.direct_outputs)
    new_sg.dict_info      = dict(sg.dict_info)
    new_sg.dict_rand      = dict(sg.dict_rand)
    new_sg.unique_id_generator = copy_generator(sg.unique_id_generator)
    id_gen = sg.unique_id_generator
    if id_gen: new_sg.unique_id_generator = [id_gen[0]]
    dict_nodes = {}
    new_init = copy_S_node(sg.init_node)
    new_nodes = []
    dict_nodes[new_init.main_target] = new_init
    for sn in sg.nodes:
        new_sn = copy_S_node(sn)
        new_nodes.append(new_sn)
        dict_nodes[sn.main_target] = new_sn
        for (req_sn,set_str) in sn.deps.items():
            new_req_sn = dict_nodes[req_sn.main_target]
            dict_edges_add_inplace(new_req_sn.users,new_sn,set_str)
            dict_edges_add_inplace(new_sn.deps,new_req_sn,set_str)
    new_sg.init_node     = new_init
    new_sg.output_node   = dict_nodes[sg.hidden_output]
    new_sg.nodes         = new_nodes
    return new_sg


def cut(sg : S_graph): # -> list of S_graph
    main_sg = copy_S_graph(sg) # to protect from side effects
    main_sg.nodes.insert(0,main_sg.init_node)
    seps = cut_based_on_deps(main_sg)
    print_debug(f"S separators : {[sep.main_target for sep in seps]}")
    list_sg = []
    for i in range(1,len(seps)):
        unique_id_generator = [0]
        new_sg = S_graph(
            unique_id_generator=copy_generator(sg.unique_id_generator))
        list_sg.append(new_sg)
        # -- get nodes --
        inp_node = seps[i-1]
        out_node = seps[i]
        inp_i = main_sg.nodes.index(inp_node)
        out_i = main_sg.nodes.index(out_node)
        nodes = main_sg.nodes[inp_i+1:out_i+1]
        new_sg.nodes = nodes
        print_debug(f"size of bloc {i} : {out_i}-{inp_i}")
        # -- input --
        if i==1:
            new_sg.init_node = main_sg.init_node
            new_sg.hidden_inputs = main_sg.hidden_inputs
            new_sg.direct_inputs = main_sg.direct_inputs
        else:
            ino = S_node(
                target=f"init_node of bloc {i}>1, should NEVER be used",
                unique_id_generator = new_sg.unique_id_generator)
            new_sg.hidden_inputs = [inp_node.main_target]
            new_sg.direct_inputs = inp_node.all_targets
            new_sg.init_node = ino
            for (user_sn,str_set) in inp_node.users.items():
                dict_edges_discard_inplace(user_sn.deps,inp_node)
                dict_edges_add_inplace(user_sn.deps,ino,str_set)
                dict_edges_add_inplace(ino.users,user_sn,str_set)
            inp_node.users = dict() # previous bloc's output node
        # -- output --
        new_sg.output_node    = out_node
        new_sg.hidden_output  = out_node.main_target
        new_sg.direct_outputs = out_node.all_targets
        # --
        new_sg.dict_info = main_sg.dict_info
        new_sg.dict_rand = main_sg.dict_rand
    return list_sg

# ==========================



# ==========================
# === printing functions ===
# ==========================

def aux_print_graph(dot,sg,uniq_num):
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,str_set):
        dot.edge(uni(i1),uni(i2),label="\n".join(str_set))
    str_ino = sg.init_node.main_target
    node(str_ino,sg.init_node.get_code(),style="dashed")
    for sn in sg.nodes:
        if sn.is_artefact:
            node(sn.main_target,sn.get_code(),style="dashed")
        else: node(sn.main_target,sn.get_code())
    for sn in sg.nodes:
        for (req_sn,str_set) in sn.deps.items():
            edge(req_sn.main_target,sn.main_target,str_set)

    # -- io --
    node("input",f"INPUT",color="green",style="dashed")
    node("output",f"OUTPUT",color="green",style="dashed")
    edge("input",sg.init_node.main_target,sg.hidden_inputs)
    edge(sg.hidden_output,"output",sg.direct_outputs)


def print_S_graph(sg : S_graph,name=None,open=True):
    print(f"Simplified forward graph : {len(sg.nodes)} nodes")
    if name is None: name = "forward S-graph"
    dot = graphviz.Digraph(
        name,
        comment="S_graph = Simplified forward graph")
    aux_print_graph(dot,sg,0)
    graph_render(dot,open,"S") # from utils.py


def print_S_graph_list(list_sg,name=None,open=True):
    s = "+".join([str(len(sg.nodes)) for sg in list_sg])
    print(
        f"{len(list_sg)} blocs of S_graph, with {s} = "\
        f"{sum([len(sg.nodes) for sg in list_sg])} nodes")
    if name is None: name = "cut forward S-graph"
    dot = graphviz.Digraph(
        name,
        comment="S_graph list : cut simplified forward graph")
    for i in range(len(list_sg)):
        aux_print_graph(dot,list_sg[i],i)
    graph_render(dot,open,"S") # from utils.py

# ==========================

