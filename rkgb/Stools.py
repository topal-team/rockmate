from rkgb.utils import *
from rkgb.utils.complement_for_Stools import *
from rkgb.Dtools import D_node,D_graph

# ==========================
# ====== S structure =======
# ==========================

class S_node():
    def __init__(self,
            target="No target",fct="",
            code=None,protected=False,
            is_rand=False,deps_rand=None,
            unique_id_generator = None):
        """
        A S_node is composed by one "real" computation, defining the
        "main_target", and followed by size / view operations over it.
        Attributes :
        .main_target : str
        .all_targets : str list
            -> names of all the vars defined
            -> (including .main_target)
        .tensor_targets / .inplace_targets / .container_targets : str list
            -> part of all_targets
            -> (done by s_graph.make_targets_attrs)
        .main_code  : tar*AST :
            -> .main_target * AST right part of the assigning code of it
        .inplace_code : tar*AST list
            -> every assigns needed before the last inplace op
        .body_code  : tar*AST list
            -> for every tar except main_target:
            -> in case of inplace op: "a = b.relu_" -> "a = b" in body_code
        .main_fct   : str  : fct used in .main_code
        .protected  : bool : see Doc (1-separator of the graph)
        .is_artefact: bool : see Doc (useful size node)
        .deps       : (S_node,str set) dict = dict_edges
            -> required nodes with the list of vars needed per node.
        .users      : dict_edges : reciprocal of .deps
        .is_rand    : bool
        .deps_rand  : str set : because we don't want random src nodes here
        """
        self.is_artefact = False
        self.main_code = (target,code)
        self.main_fct = fct
        self.inplace_code = [] # list of tar * AST
        self.body_code    = []
        self.main_target = target # str
        self.all_targets = [target]
        self.tensor_targets = [] # later
        self.inplace_targets = [] # later
        self.container_targets = [] # later
        self.deps = dict()
        self.users = dict()
        self.protected = protected
        self.is_rand = is_rand
        self.deps_rand = deps_rand if deps_rand else set()
        if unique_id_generator is None: self.unique_id = id(self)
        else:
            u = unique_id_generator[0]
            self.unique_id = u
            unique_id_generator[0] = u+1
    def __eq__(self,sn2,force_order=False,raise_exception=False):
        sn1 = self
        try:
            b = small_fcts.check_attr(sn1,sn2,[
                "is_artefact","main_fct",
                "is_rand","deps_rand",
                "main_target","all_targets",
                "inplace_targets","container_targets",
                "tensor_targets","protected"],
                raise_exception=raise_exception)
            b = (b
                and dict_edges_eq(sn1.deps,sn2.deps,
                    raise_exception=raise_exception)
                and dict_edges_eq(sn1.users,sn2.users,
                    raise_exception=raise_exception)
                and (sn1.full_code() == sn2.full_code()))
            if not b and raise_exception: raise Exception(
                f"Node diff : code diff : \n {sn1.full_code()}\n"\
                f"==== DIFFERENT ==== \n {sn2.full_code()}")
            return b
        except AttributeError as a: return sn1.__hash__() == sn2.__hash__()
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self)
    # -> /!\ /!\ doing set/dict of S_nodes is dangereous /!\ /!\ 
    # but I'm doing this to avoid undeterminism

    def get_code(self,*args, **kwargs):
        return shared_methods.get_code(self,*args, **kwargs)
    def full_code(self,*args, **kwargs):
        return shared_methods.full_code(self,*args, **kwargs)

    # -----
    def insert_code(self,aux_sn,sg):
        # -> aux function of .insert method
        # -> but used directly sometimes
        dict_info = sg.dict_info
        aux_mt = aux_sn.main_target
        aux_info = dict_info[aux_mt]
        if not aux_info.is_inplace:
            if aux_sn.main_code is None: print(
                f"Warning : we tried to insert {aux_mt}'s "\
                f"node in {self.main_target}'s node, but aux_sn's "\
                f"main_code is empty ? How could this be possible ?!",
                file = sys.stderr)
            else:
                self.body_code.append(aux_sn.main_code)
        else:
            # -> we need to push inplace code (and its deps)
            data_parents = []
            data_owner = aux_info.data_owner_name
            p_info = aux_info
            p_name = p_info.data_direct_parent_name
            while p_name != data_owner:
                data_parents.append(p_name)
                p_info = dict_info[p_name]
                p_name = p_info.data_direct_parent_name
            ic = self.inplace_code
            bc = self.body_code
            already_in_ic = set(c[0] for c in ic)
            for code in bc:
                if (code[0] in data_parents
                and code[0] not in already_in_ic):
                    ic.append(code)
            ic.append(aux_sn.main_code)
            bc.append((aux_mt,ast.Name(aux_info.data_direct_parent_name)))
        # anyway :
        self.inplace_code.extend(aux_sn.inplace_code)
        self.body_code.extend(aux_sn.body_code)

        self.all_targets.extend(aux_sn.all_targets)
        self.is_rand = self.is_rand or aux_sn.is_rand
        self.deps_rand.update(aux_sn.deps_rand)
    # -----


    # -----
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
        self.insert_code(aux_sn,sg)
        # -- edges --
        self.deps = merged_deps
        self.users = merged_users
        dict_edges_make_users_using_deps(self)
        dict_edges_make_deps_using_users(self)
        # -- special case if aux_n is the output --
        if aux_sn is sg.output_node:
            sg.output_node = self
    # -----

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
        self.direct_inputs  = [] # str list
        self.hidden_output  = "" # str
        if dg:
            self.hidden_inputs  = dg.inputs
            self.direct_outputs = [dg.output]
            self.dict_info      = dg.dict_info
            self.dict_rand      = dg.dict_rand
            self.dict_constants = dg.dict_constants
        else:
            self.hidden_inputs  = []
            self.direct_outputs = []
            self.dict_info      = dict()
            self.dict_rand      = dict()
            self.dict_constants = dict()
        self.unique_id_generator = unique_id_generator
        # -> to generate S_node.__hash__
    def __eq__(self,sg2,force_order=False,raise_exception=False):
        sg1 = self
        return small_fcts.check_attr(sg1,sg2,[
            # "direct_inputs","hidden_inputs",
            # sg1 and sg2 equality shouldn't depend on their 
            # previous block equality
            "direct_outputs","hidden_output","dict_info",
            "nodes","dict_constants"],
            raise_exception=raise_exception)
        """ # TO REMOVE
        mt = lambda l : [sn.main_target for sn in l]
        b *= (mt(sg1.nodes) == mt(sg2.nodes))
        if raise_exception and not b:
            raise Exception("S_graphs' differ on nodes order or length")
        if b:
            for sn1,sn2 in zip(sg1.nodes,sg2.nodes):
                b *= sn1.__eq__(sn2,force_order,raise_exception)
        return b
        """
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
        self.nodes = shared_methods.sort_based_on_deps(self.output_node)
        self.nodes.remove(self.init_node)
        self.check_artefact()
        self.check_relations()
        self.make_io()

    def make_targets_attrs(self):
        # -> tensor_targets ; inplace_targets ; container_targets
        dict_info = self.dict_info
        for sn in self.nodes:
            if not sn.is_artefact:
                tensors = []
                containers = []
                for tar in sn.all_targets:
                    ttype = dict_info[tar].ttype
                    if ttype == torch.Tensor:
                        tensors.append(tar)
                    elif ttype == tuple or ttype == list:
                        containers.append(tar)
                sn.tensor_targets = tensors
                sn.container_targets = containers
                sn.inplace_targets = [c[0] for c in sn.inplace_code]

    def refresh_info_data_name(self):
        dict_info = self.dict_info
        # First we need to know where each var is :
        dict_nodes = dict() # any target -> main_target
        for sn in self.nodes:
            for tar in sn.all_targets:
                dict_nodes[tar] = sn
        for name,info in dict_info.items():
            if name in dict_nodes:
                owner_sn = dict_nodes[info.data_owner_name]
                if owner_sn.is_artefact:
                    info.data_owner_name = "PARAM"
                else:
                    info.data_owner_name = owner_sn.main_target


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

def D_to_S_init(dg : D_graph) -> S_graph:
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
                is_rand=dn.is_rand,
                deps_rand= set(dn.deps_rand),
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
    init_node.body_code = [] # because all input nodes have dummy code
    sg.init_node = init_node
    sg.output_node = dict_s_nodes[dg.output]
    sg.clear()
    return sg

# ==========================



# ==========================
# ==== Simplification 1 ====
# === remove cheap nodes ===
# ==========================

def insert_ast_code(main_sn,sub_sn):
    mc = main_sn.main_code[1]
    st = sub_sn.main_target
    sc = sub_sn.main_code[1]
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
            if isinstance(k.value,ast.Name) and k.value.id == st:
                kwds.append(ast.Keyword(k.arg,sc))
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
        insert_ast_code(user_sn,sn)
        # -- handle randomness --
        user_sn.is_rand = user_sn.is_rand or sn.is_rand
        user_sn.deps_rand.update(sn.deps_rand)
    sn.deps  = dict()
    sn.users = dict()

def simplify_cheap(sg : S_graph):
    # from root to leaves
    for sn in sg.nodes:
        if ( not (sn is sg.output_node)
         and sn.main_fct in global_vars.list_cheap_fct
         and not sn.protected):
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

"""
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
"""
def get_all_real_deps(sn):
    return set(
        req_sn for req_sn in sn.deps.keys() 
        if not req_sn.is_artefact)


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
        sn_info = sg.dict_info[sn.main_target]
        if (sn_info.is_view
        or  sn.main_fct in global_vars.list_view_fct
        # -> in case of op over params
        or  sn.main_fct == "getattr"
        or  sn_info.is_inplace):
            # ASSERTION remaining getattr are related to views !! 
            # we also consider inplace ops as views
            real_deps = get_direct_real_deps(sn)
            if len(real_deps)==1:
                req_sn = real_deps.pop()
                req_sn.insert(sn,strong=True,sg=sg)
                req_sn.clear_children_artefact()
                req_sn.clear_siblings_artefact()
            elif len(real_deps) > 1:
                if not sn_info.is_inplace: print(
                    f"Warning : {sn.main_target} is a view op (not "\
                    f"inplace), with several tensor deps, thus it's "\
                    f"impossible to simplify it, very dangerous...\n"\
                    f"deps are : {[req_sn.main_target for req_sn in real_deps]}",
                    file = sys.stderr)
                else:
                    inplace_real_node = None
                    for req_sn in real_deps:
                        if req_sn.main_target == sn_info.data_owner_name:
                            inplace_real_node = req_sn
                            break
                    if inplace_real_node is None: print(
                        f"Warning : {sn.main_target} comes from an "\
                        f"inplace operations, but it's main tensor "\
                        f"isn't in {sn.main_target}'s node deps",
                        file = sys.stderr)
                    else:
                        inplace_real_node.insert(sn,strong=True,sg=sg)
                        inplace_real_node.clear_siblings_artefact()
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
                if sn_info.is_inplace: raise Exception(
                    f"Sorry we do not support inplace operations over "\
                    f"parameters (or anything that isn't a Tensor). \n"\
                    f"Here {sn.main_target} only deps on artefacts, but"\
                    f"sn_info.is_inplace=True :/")
                for art_req in sn.deps.keys():
                    if len(art_req.deps)==0:
                        assert(art_req is sg.init_node)
                        real_req = None
                    else:
                        assert(len(art_req.deps)==1) # as an artefact
                        real_req = list(art_req.deps.keys())[0]
                        real_req.insert_code(sn,sg)
                    art_req.insert_code(sn,sg)
                    # -> Insert sn's code BOTH in art_req and real_req

                    # - plug art_req to sn's users -
                    dict_edges_merge_inplace(art_req.users,sn.users)
                    for (user_sn,str_set) in sn.users.items():
                        dict_edges_add_inplace(user_sn.deps,art_req,str_set)
                    # - unplug sn -
                    dict_edges_discard_inplace(art_req.users,sn)
                    dict_edges_discard_sn_from_deps_of_its_users(sn)
                    sn.deps = dict()
                    sn.users = dict()
                    if real_req: real_req.clear_children_artefact()

    sg.clear()

# ==========================



# ==========================
#  Insert random operations 
#  which were in dict_rand
# ==========================
# -> Now that we simplified everything,
# -> we can insert random nodes from dict_rand

def create_random_snodes_from_dict_rand(sg,model,device):
    dict_random_sn = dict() # str -> S_node
    dict_info = sg.dict_info
    for name,ast_code in sg.dict_rand.items():
        dict_random_sn[name] = S_node(
            target    = name,
            fct       = "--Random function--",
            code      = ast_code,
            protected = True,
            is_rand   = True,
            unique_id_generator = sg.unique_id_generator)
        # -> We need to generate ".info" from def_info.py
        # -> to do so we first need to generate the variable <name>
        our_global = globals().copy()
        our_global.update(sg.dict_constants)
        if model: our_global["self"] = model
        if device: our_global["device"] = device
        dict_info[name] = def_info.Var_info(
            eval(ast_add_on.ast_to_str(ast_code),our_global)
        )
    for sn in sg.nodes:
        for req_rd in sn.deps_rand:
            req_sn_rd = dict_random_sn[req_rd]
            dict_edges_add_inplace(sn.deps,req_sn_rd,set([req_rd]))
            dict_edges_add_inplace(req_sn_rd.users,sn,set([req_rd]))
    sg.nodes = list(dict_random_sn.values()) + sg.nodes

# ==========================



# ==========================
# = Move from D to S graph =
# ==========================

def D_to_S(dg,model=None,device=None):
    sg = D_to_S_init(dg)
    simplify_cheap(sg)
    simplify_size(sg)
    simplify_view(sg)
    create_random_snodes_from_dict_rand(sg,model,device)
    sg.check_relations()
    sg.make_targets_attrs()
    sg.refresh_info_data_name()
    sg.assert_ready()
    return sg

# ==========================



# ==========================
# ==== Cut the graph in ====
# ==== sequential parts ====
# ==========================

def copy_S_node(sn : S_node): # aux for copy_S_graph
    new_sn = S_node()
    new_sn.is_artefact       = sn.is_artefact
    new_sn.main_code         = tuple(sn.main_code)
    new_sn.main_fct          = sn.main_fct
    new_sn.inplace_code      = [tuple(c) for c in sn.inplace_code]
    new_sn.body_code         = [tuple(c) for c in sn.body_code]
    new_sn.main_target       = sn.main_target
    new_sn.all_targets       = list(sn.all_targets)
    new_sn.tensor_targets    = list(sn.tensor_targets)
    new_sn.inplace_targets   = list(sn.inplace_targets)
    new_sn.container_targets = list(sn.container_targets)
    new_sn.is_rand           = sn.is_rand
    new_sn.deps_rand         = set(sn.deps_rand)
    new_sn.deps              = dict() # /!\
    new_sn.users             = dict() # /!\
    new_sn.protected         = sn.protected
    new_sn.unique_id         = sn.unique_id
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
    new_sg.dict_constants = dict(sg.dict_constants)
    new_sg.unique_id_generator = small_fcts.copy_generator(
            sg.unique_id_generator)
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
    seps = [main_sg.init_node]+shared_methods.cut_based_on_deps(main_sg)
    print_debug(f"S separators : {[sep.main_target for sep in seps]}")
    list_sg = []
    for i in range(1,len(seps)):
        unique_id_generator = [0]
        new_sg = S_graph(
            unique_id_generator=small_fcts.copy_generator(
                sg.unique_id_generator))
        list_sg.append(new_sg)
        # -- get nodes --
        inp_node = seps[i-1]
        out_node = seps[i]
        inp_i = main_sg.nodes.index(inp_node)
        out_i = main_sg.nodes.index(out_node)
        nodes = main_sg.nodes[inp_i+1:out_i+1] # seperator is included
        new_sg.nodes = nodes
        print_debug(f"size of bloc {i} : {out_i}-{inp_i}")
        # -- input --
        if i==1:
            new_sg.init_node = main_sg.init_node
            new_sg.hidden_inputs = main_sg.hidden_inputs
            new_sg.direct_inputs = main_sg.direct_inputs
        else:
            ino = S_node(
                target=f"init_node of bloc, should NEVER be used",
                unique_id_generator = new_sg.unique_id_generator)
            new_sg.hidden_inputs = [inp_node.main_target]
            new_sg.direct_inputs = inp_node.all_targets
            new_sg.init_node = ino
            for (user_sn,str_set) in inp_node.users.items():
                dict_edges_discard_inplace(user_sn.deps,inp_node)
                dict_edges_add_inplace(user_sn.deps,ino,str_set)
                dict_edges_add_inplace(ino.users,user_sn,str_set)
                if user_sn.is_artefact:
                    ino.insert(user_sn,strong=True,sg=main_sg)
            inp_node.users = dict() # previous bloc's output node
        # -- output --
        new_sg.output_node    = out_node
        new_sg.hidden_output  = out_node.main_target
        new_sg.direct_outputs = out_node.all_targets
        # --
        new_sg.dict_info = main_sg.dict_info
        new_sg.dict_rand = main_sg.dict_rand
        new_sg.dict_constants = main_sg.dict_constants
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


def print_S_graph(sg : S_graph,name=None,open=True,render_format="svg"):
    print(f"Simplified forward graph : {len(sg.nodes)} nodes")
    if name is None: name = "Simplified_forward_graph"
    dot = graphviz.Digraph(
        name,
        comment="S_graph = Simplified forward graph")
    aux_print_graph(dot,sg,0)
    small_fcts.graph_render(dot,open,"S",render_format)


def print_S_graph_list(list_sg,name=None,open=True,render_format="svg"):
    s = "+".join([str(len(sg.nodes)) for sg in list_sg])
    print(
        f"{len(list_sg)} blocs of S_graph, with {s} = "\
        f"{sum([len(sg.nodes) for sg in list_sg])} nodes")
    if name is None: name = "Sequentialized_Simplified_Forward_graph"
    dot = graphviz.Digraph(
        name,
        comment="S_graph_list: sequentialized simplified forward graph")
    for i in range(len(list_sg)):
        aux_print_graph(dot,list_sg[i],i)
    small_fcts.graph_render(dot,open,"S",render_format)

# ==========================

