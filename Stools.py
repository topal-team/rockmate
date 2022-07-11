from Dtools import *
import ast

list_cheap_fct = ["torch.add","torch.mul"]
list_cheap_fct.extend(["list constructor","tuple constructor"])
# because I treat them in the same way

# ==========================
# = Init move from D to S  =
# ==========================

class S_node():
    def __init__(self,code=None,fct="",target="No target"):
        self.is_artefact = False
        self.main_code = code
        self.main_fct = fct
        self.body_code = [] # list of ast.Assign
        self.main_target = target # str
        self.all_targets = [target]
        self.req = set()
        self.used_by = set()
    def full_code(self):
        if self.main_code is None: mc = []
        else: mc = [self.main_code]
        return ast.Module(mc + self.body_code,[])
    def get_code(self):
        return ast_to_str(self.full_code())
    def insert(self,sub_n,strong):
        # self is the main node ; sub_n is the node inserted
        # if strong: delete sub_n else: sub_node <- artefact
        # in any case cut as many edges as possible
        # -- disconnect sub_n and self --
        sub_n.req.discard(self)
        self.used_by.discard(sub_n)
        merged_req = self.req | sub_n.req
        merged_used_by = self.used_by | sub_n.used_by
        # -- disconnect sub_n with its children (if possible) --
        if strong: # for "view"
            for sub_sub_n in sub_n.used_by:
                sub_sub_n.req.discard(sub_n)
            sub_n.used_by = set()
        else: # for "size"
            for sub_sub_n in self.used_by:
                sub_sub_n.req.discard(sub_n)
                sub_n.used_by.discard(sub_sub_n)
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

class S_graph():
    def __init__(self,dg : D_graph = None):
        self.nodes = []
        self.init_node = None
        if dg:
            self.output    = dg.output
            self.dict_info = dg.dict_info
            self.dict_rand = dg.dict_rand
        else:
            self.output    = None
            self.dict_info = {}
            self.dict_rand = {}

    def clear(self): # remove unpluged nodes
        l = []
        for n in self.nodes:
            if n.req != set() or n.used_by != set():
                l.append(n)
        self.nodes = l

    def assert_ready(self):
        # check if ready to be given to S_to_K
        # -> main_targets are tensors, except if artefact -> sizes
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


def D_to_S_init(dg : D_graph) -> S_graph:
    sg = S_graph(dg)
    init_node = S_node(target="-- inputs --")
    s_nodes = sg.nodes
    dict_s_nodes = {} # to translate D to S
    for n in dg.nodes:
        sn = S_node(code=n.ast_code,fct=n.fct,target=n.target)
        s_nodes.append(sn)
        dict_s_nodes[n.target] = sn
        for req_n in n.req:
            sreq_n = dict_s_nodes[req_n.target]
            sn.req.add(sreq_n)
            sreq_n.used_by.add(sn)
    # -- merge all the inputs in the special "init_node" --
    for inp in dg.inputs:
        init_node.insert(dict_s_nodes[inp],strong=True)
    init_node.body_code = []
    sg.init_node = init_node
    sg.clear()
    return sg

# ==========================



# ==========================
# ==== Simplification 1 ====
# === remove cheap nodes ===
# ==========================

def insert_ast_code(main_n,mc,target : str,sc): # mc : main_code , sc : sub_code
    assert(isinstance(mc,ast.Assign))
    assert(isinstance(sc,ast.Assign))
    if not isinstance(sc.targets[0],ast.Name):
        print(ast.dump(sc,indent=4))
    assert(sc.targets[0].id == target)
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
        if n.main_target != g.output and n.main_fct in list_cheap_fct:
            simplify_node(n)
    g.clear()


# ==========================



# ==========================
# = Move from D to S graph =
# ==========================

def D_to_S(dg):
    sg = D_to_S_init(dg)
    simplify_cheap(sg)
    sg.assert_ready()
    return sg

# ==========================



# ==========================
# === printing functions ===
# ==========================

def print_graph(g : D_graph,name=None):
    print(len(g.nodes))
    if name is None:
        name = "forward S-graph"
    dot = graphviz.Digraph(name,comment="S_graph = Simplified forward graph")
    dot.node(g.init_node.main_target,g.init_node.get_code(),color="blue")
    for n in g.nodes:
        if n.main_target == g.output:
            dot.node(n.main_target,n.get_code(),color="red")
        elif n.is_artefact:
            dot.node(n.main_target,n.get_code(),style="dashed")
        else: dot.node(n.main_target,n.get_code())
    for n in g.nodes:
        for sub_n in n.req:
            dot.edge(sub_n.main_target,n.main_target)
    dot.render(directory="graphviz_dir",view=True)

# ==========================


