# ==========================
# ====== S structure =======
# ==========================

from .utils import *
from .Dtools import D_node,D_graph

# ***********
# * S_edges *
# ***********

# S edges (ie S_nodes.deps/users) are dict : (S_node -> str set)
# Note that .deps/users won't be instances of S_edges class,
# this class has only static methods. Since keeping standard dict
# have advantage and some of the following functions aren't taking
# dict_edges as inputs. 
# /!\ By default the following operations are NOT inplace
# inplace operations names end with "_inplace". 

class S_edges():
    @staticmethod
    def merge_inplace(de_main,de_sub):
        for k in de_sub.keys():
            s = de_main[k] if k in de_main else set()
            de_main[k] = s.union(de_sub[k])
    @staticmethod
    def merge(de1,de2):
        d = dict(de1)
        S_edges.merge_inplace(d,de2)
        return d

    @staticmethod
    def discard(de,sn):
        return dict((n,s) for (n,s) in de.items() if n != sn)
    @staticmethod
    def discard_inplace(de,sn):
        if sn in de: del de[sn]

    @staticmethod
    def add_inplace(de,sn,str_set):
        s = de[sn] if sn in de else set()
        de[sn] = s.union(str_set)
    @staticmethod
    def add(de,sn,str_set):
        d = dict(de) # not inplace
        S_edges.add_inplace(d,sn,str_set)
        return d

    @staticmethod
    def eq(de1,de2,raise_exception=False):
        ds1 = dict((n.main_target,s) for (n,s) in de1.items())
        ds2 = dict((n.main_target,s) for (n,s) in de2.items())
        if not raise_exception:
            return ds1 == ds2
        else:
            keys1 = base.Node.sort_targets(ds1.keys())
            keys2 = base.Node.sort_targets(ds2.keys())
            if len(keys1) != len(keys2): raise Exception(
                "Difference of dict_edges: "\
                "number of edges (keys) diff")
            for i,(k1,k2) in enumerate(zip(keys1,keys2)):
                if k1 != k2: raise Exception(
                    f"Difference of dict_edges: "\
                    f"{i}-th edge key diff : {k1} != {k2}")
                if ds1[k1] != ds2[k2]: raise Exception(
                    f"Difference of dict_edges: "\
                    f"{i}-th edge labels diff : {ds1[k1]} != {ds2[k2]}")
            return True
        # since this function is an auxilary function for S_node.__eq__ method
        # we cannot check s_nodes equalities, we just check .main_target

    @staticmethod
    def discard_sn_from_deps_of_its_users(sn):
        for user_sn in sn.users.keys():
            S_edges.discard_inplace(user_sn.deps,sn)
    @staticmethod
    def discard_sn_from_users_of_its_deps(sn):
        for req_sn in sn.deps.keys():
            S_edges.discard_inplace(req_sn.users,sn)

    @staticmethod
    def make_users_using_deps(sn):
        for (req_sn,str_set) in sn.deps.items():
            req_sn.users[sn] = set(str_set)
    @staticmethod
    def make_deps_using_users(sn):
        for (user_sn,str_set) in sn.users.items():
            user_sn.deps[sn] = set(str_set)

    @staticmethod
    def discard_edge_inplace(req_sn,user_sn):
        S_edges.discard_inplace(req_sn.users,user_sn)
        S_edges.discard_inplace(user_sn.deps,req_sn)

    @staticmethod
    def add_edge_inplace(req_sn,user_sn,str_set):
        S_edges.add_inplace(req_sn.users,user_sn,str_set)
        S_edges.add_inplace(user_sn.deps,req_sn,str_set)

    @staticmethod
    def is_subset(de1,de2):
        for (sn1,str_set1) in de1.items():
            if sn1 not in de2: return False
            if str_set1 > de2[sn1]: return False
        return True



# **********
# * S_node *
# **********

class S_node(base.Node):
    def __init__(self,
            main_target="No target",code=None,fct="",
            protected=False,
            is_rand=False,deps_rand=None,
            other_obj = None):
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
        super().__init__("S",other_obj,main_target=main_target)
        self.is_artefact = False
        self.main_code = (main_target,code)
        self.main_fct = fct
        self.inplace_code = [] # list of tar * AST
        self.body_code    = []
        self.all_targets = [main_target]
        self.tensor_targets = [] # later
        self.inplace_targets = [] # later
        self.container_targets = [] # later
        self.deps = dict()
        self.users = dict()
        self.protected = protected
        self.is_rand   = is_rand
        self.deps_rand = deps_rand if deps_rand else set()

    def get_all_standard_deps(self):
        return set(self.deps.keys())
    def get_all_standard_users(self):
        return set(self.users.keys())

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

        merged_deps = S_edges.merge(self.deps,aux_sn.deps)
        S_edges.discard_inplace(merged_deps,self)

        # -- disconnect aux_n with its children (if possible) --
        if strong: # e.g. for "view"
            S_edges.discard_sn_from_deps_of_its_users(aux_sn)
            merged_users = S_edges.merge(self.users,aux_sn.users)
            S_edges.discard_inplace(merged_users,aux_sn)
            aux_sn.users = dict()
        else: # e.g. for "size"
            for user_sn in self.users.keys():
                S_edges.discard_inplace(user_sn.deps,aux_sn)
                S_edges.discard_inplace(aux_sn.users,user_sn)
            merged_users = self.users
        # -- if aux_sn is deleted, remove it from parents' users --
        if len(aux_sn.users) == 0:
            S_edges.discard_sn_from_users_of_its_deps(aux_sn)
            aux_sn.deps = dict()
            # -> aux_sn has been fully unplugged
        else:
            aux_sn.is_artefact = True
            # -> artefact

        # -- insert aux_sn code --
        self.insert_code(aux_sn,sg)
        # -- edges --
        self.deps = merged_deps
        self.users = merged_users
        S_edges.make_users_using_deps(self)
        S_edges.make_deps_using_users(self)
        if aux_sn in sg.output_nodes:
            sg.output_nodes[sg.output_nodes.index(aux_sn)] = self
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
                        S_edges.add_edge_inplace(
                            self,other_user_sn,
                            other_user_sn.deps[art_user_sn])
                        # We add the names of the variables generated 
                        # by the code of art_user_sn (present in self)
                        # which are used in other_user_sn. Because
                        # to get those vars we will no longer use
                        # art_user_sn since we already need self for
                        # something else.
                    S_edges.discard_edge_inplace(
                        art_user_sn,other_user_sn)
                if art_user_sn.users == dict():
                    S_edges.discard_sn_from_users_of_its_deps(
                        art_user_sn)
                    art_user_sn.deps = dict()

    def clear_siblings_artefact(self):
        real_deps = set()
        for req_sn in self.deps.keys():
            if not req_sn.is_artefact:
                real_deps.add(req_sn)
        for req_sn in real_deps:
            req_sn.clear_children_artefact()


# ***********
# * S_graph *
# ***********

class S_graph(base.Graph):
    artefact_edges : list[tuple[S_node,S_node,set[str]]] = None
    def __init__(self,dg : D_graph = None):
        super().__init__("S")
        if not (dg is None):
            self.inherit_base_attributes(dg)
            self.whole_model_inputs = set(dg.inputs)
        self.init_node = None # NOT in self.nodes
        self.special_output_node = None # NOT in self.nodes
        self.dict_output_viewing_code = dict()
        self.artefact_edges = []

    def __eq__(self,sg2,force_order=False,raise_exception=False):
        sg1 = self
        return small_fcts.check_attr(sg1,sg2,[
            "inputs","outputs",
            "nodes","dict_info","dict_constants"],
            raise_exception=raise_exception)
    __hash__ = base.Node.__hash__

    def make_inputs(self):
        inputs = set()
        for used_targets in self.init_node.users.values():
            inputs.update(used_targets)
            # -> labels over the edges
        self.inputs = list(inputs)
        self.whole_model_inputs.update(set(self.inputs))

    def unhook_init_node(self):
        dict_info = self.dict_info
        init_node_users = list(self.init_node.users.items())
        for user_sn,used_targets in init_node_users:
            S_edges.discard_inplace(user_sn.deps,self.init_node)
            if all(not dict_info[used_tgt].requires_grad
                   for used_tgt in used_targets):
                S_edges.discard_inplace(self.init_node.users,user_sn)
        # We need at least one first node = one user of init_node
        # -> Otherwise weird graph 
        # -> And dangerous for Rk_get_1_separators
        if len(self.init_node.users)==0:
            all_without_deps = [
                sn for sn in self.nodes 
                if len(sn.deps)==0 ]
            first_sn = min(all_without_deps,key=lambda sn : sn.get_num())
            self.init_node.users[first_sn] = set()
                

    def unhook_special_output_node(self):
        assert(len(self.output_nodes)==1)
        output_node = self.output_nodes[0]
        output = output_node.main_target
        output_info = self.dict_info[output]
        if output_info.variable_type in [tuple,list]:
            real_output_nodes = []
            real_outputs = set()
            for req_sn,req_targets in output_node.deps.items():
                real_output_nodes.append(req_sn)
                real_outputs.update(req_targets)
                S_edges.discard_inplace(req_sn.users,output_node)
            self.output_nodes = real_output_nodes
            self.outputs = list(real_outputs)
            self.special_output_node = output_node
            # keep special_output_node.deps

        # unhook viewing operations over the outputs
        self.outputs = outputs = []
        self.dict_output_viewing_code = dict_view_code = dict() # mt -> ast code for viewing stuff
        for out in self.output_nodes:
            bc = out.make_body_code_ast()
            viewing_code = ast_add_on.make_ast_list_assign(
                bc,force_special_kwargs=True
            )
            dict_view_code[out.mt] = viewing_code
            outputs.append(out.mt)
            

    def check_artefact(self):
        for sn in self.nodes:
            if sn.is_artefact:# and not (sn is self.init_node):
                if len(sn.deps)!=1:
                    raise Exception(
                      f"{sn.main_target} is_artefact, but with "\
                      f"len(deps)={len(sn.deps)} (should be 1)")
                req_sn = list(sn.deps.keys())[0]
                if S_edges.is_subset(sn.users,req_sn.users):
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
                
    # === To handle artefacts in Ptools ===
    def discard_all_artefacts(self):
        # Do this only once the order is fixed!
        # And K_graph is generated
        # -> better do a copy first
        snodes = list(self.nodes)
        nb_nodes = len(snodes)
        artefact_edges = self.artefact_edges
        for i,sn in enumerate(snodes[::-1]):
            index = nb_nodes-i-1
            if sn.is_artefact:
                del self.nodes[index]
                real_sn = list(sn.deps.keys())[0]
                for user_sn,used_targets in sn.users.items():
                    S_edges.discard_inplace(user_sn.deps,sn)
                    S_edges.add_edge_inplace(real_sn,user_sn,used_targets)
                    artefact_edges.append((real_sn,user_sn,used_targets))
                S_edges.discard_inplace(real_sn.users,sn)
    def delete_artefact_edges(self):
        for (used_sn,user_sn,_) in self.artefact_edges:
            S_edges.discard_edge_inplace(used_sn,user_sn)
        # We do NOT set self.artefact_edges = []


    def clear(self):
        # -- (re)-sorting nodes -- 
        # due to merging, the topo order may not be correct anymore
        # by the way, remove unplugged nodes
        if len(self.output_nodes)==1:
            root_sn = self.output_nodes[0]
            real_root = True
        else:
            real_root = False
            root_sn = S_node("Tmp_root")
            root_sn.deps = dict((out_sn,set()) for out_sn in self.output_nodes)
            for out_sn in self.output_nodes:
                out_sn.users[root_sn] = set()
        self.nodes = RK_sort_based_on_deps(root_sn)
        if self.init_node in self.nodes: self.nodes.remove(self.init_node)
        if not real_root:
            self.nodes.remove(root_sn)
            for out_sn in root_sn.deps:
                del out_sn.users[root_sn]
        self.check_artefact()
        self.check_relations()
        self.make_inputs()
        
        
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


    def correct_label_over_edges(self):
        for sn in self.nodes:
            sn_code = sn.get_code()
            for req_sn,used_targets in sn.deps.items():
                _used_target = list(used_targets)
                for tar in _used_target:
                    if not tar in sn_code:
                        used_targets.discard(tar)
                req_sn.users[sn] = set(used_targets)

                    
    def make_targets_attrs(self):
        # -> tensor_targets ; inplace_targets ; container_targets
        dict_info = self.dict_info
        for sn in self.nodes:
            if not sn.is_artefact:
                tensors = []
                containers = []
                for tar in sn.all_targets:
                    info = dict_info[tar]
                    variable_type = info.variable_type
                    if variable_type == torch.Tensor:
                        if info.data_owner_name != "PARAM":
                            tensors.append(tar)
                    elif variable_type == tuple or variable_type == list:
                        containers.append(tar)
                sn.tensor_targets = tensors
                sn.container_targets = containers
                sn.inplace_targets = [c[0] for c in sn.inplace_code]

                
    def assert_ready(self):
        # check if ready to be given to S_to_K
        # ie main_targets are tensors, except if artefact -> sizes
        for sn in self.nodes:
            if not (sn.main_target in self.dict_info):
                raise Exception(
                  f"{sn.main_target} not in dict_info ??")
            info = self.dict_info[sn.main_target]
            if not (info.variable_type in [torch.Tensor,torch.Size]):
                raise Exception(
                  f"After simplifications there should "\
                  f"only be tensors or sizes, but {info.variable_type} "\
                  f"found for {sn.main_target}.")
            if info.variable_type==torch.Size and not sn.is_artefact:
                raise Exception(
                  f"After simplifications, all remaining "\
                  f"\"size\" should be \"artefacts\", but "\
                  f"{sn.main_target} isn't an artefact")


# ==========================


# ==========================
# = Init move from D to S  =
# ==========================

def D_to_S_init(dg : D_graph) -> S_graph:
    sg = S_graph(dg)
    init_node = S_node(main_target="sources",other_obj=sg)
    init_node.all_targets=[]
    s_nodes = sg.nodes
    dict_s_nodes = dict() # to translate D to S
    for dn in dg.nodes:
        sn = S_node(
                main_target=dn.target,
                code=dn.code_ast,
                fct=dn.fct,
                protected=dn.protected,
                is_rand=dn.is_rand,
                deps_rand= set(dn.deps_rand),
                other_obj=sg)
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
    # -> At the beginning (here), init_node contains only the 'real' inputs
    # -> in D_graph these nodes have a dummy code `'code = 'INPUT'`,
    # -> the "insert" method will put these dummy codes in init_node.body_code
    # -> that's why we clear init_node.body_code at the end of initialization
    sg.output_nodes = [dict_s_nodes[out] for out in dg.outputs]
    sg.clear()
    return sg

# ==========================



# ==========================
# ==== Simplification 1 ====
# === remove cheap nodes ===
# ==========================

def insert_code_ast(main_sn,sub_sn):
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
            else: kwds.append(k)
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
    # aux fct, insert n.code_ast in children's code, and then unplug it
    for user_sn in sn.users.keys():
        # -- plug user_sn directly to deps of sn --
        S_edges.merge_inplace(user_sn.deps,sn.deps)
        S_edges.discard_inplace(user_sn.deps,sn)
        for (req_sn,str_set) in sn.deps.items():
            S_edges.discard_inplace(req_sn.users,sn)
            S_edges.add_inplace(req_sn.users,user_sn,str_set)
        # -- insert the code --
        insert_code_ast(user_sn,sn)
        # -- handle randomness --
        user_sn.is_rand = user_sn.is_rand or sn.is_rand
        user_sn.deps_rand.update(sn.deps_rand)
    sn.deps  = dict()
    sn.users = dict()

def simplify_cheap(sg : S_graph):
    # from root to leaves
    for sn in sg.nodes:
        if ( not (sn in sg.output_nodes)
         and    (sn.main_fct in global_vars.list_cheap_fct
            or 
                (sn.main_fct in global_vars.list_optional_cheap_fct and not sn.protected)
         )):
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
    # give the list of child nodes of sn which are size
    ret = []
    for user_sn in sn.users.keys():
        if sg.dict_info[user_sn.main_target].variable_type == torch.Size:
            ret.append(user_sn)
    return ret


def simplify_size(sg : S_graph):
    # from leaves to root
    nodes = [sg.init_node] + list(sg.nodes) ; nodes.reverse()
    for sn in nodes:
        if not (sn in sg.output_nodes):
            list_size = size_children(sg,sn)
            if list_size != []:
                # -- merge into one node --
                size_sn = list_size[0]
                for other_sn in list_size[1:]:
                    size_sn.insert(other_sn,strong=True,sg=sg)
                # -- insert their code --
                if (sn is sg.init_node
                or sg.dict_info[sn.main_target].variable_type == torch.Size):
                    sn.insert(size_sn,strong=True,sg=sg)
                else: sn.insert(size_sn,strong=False,sg=sg)
    sg.clear()

# ==========================



# ==========================
# ==== Simplification 3 ====
# === remove view nodes ====
# ==========================

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

def simplify_view(sg : S_graph):
    # from root to leaves
    sg.init_node.is_artefact = True
    for sn in sg.nodes:
        sn_info = sg.dict_info[sn.main_target]
        if (sn_info.is_view
        or  sn.main_fct in global_vars.list_view_fct # -> in case of viewing operations over parameters
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
                # But I must avoid creating cycle dependencies, so
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
                    S_edges.merge_inplace(art_req.users,sn.users)
                    for (user_sn,str_set) in sn.users.items():
                        S_edges.add_inplace(user_sn.deps,art_req,str_set)
                    # - unplug sn -
                    S_edges.discard_inplace(art_req.users,sn)
                    S_edges.discard_sn_from_deps_of_its_users(sn)
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
    for name,code_ast in sg.dict_rand.items():
        dict_random_sn[name] = S_node(
            main_target = name,
            code       = code_ast,
            fct        = "--Random function--",
            protected  = True,
            is_rand    = True,
            other_obj  = sg)
        # -> We need to generate ".info" from def_info.py
        # -> to do so we first need to generate the variable <name>
        our_global = globals().copy()
        our_global.update(sg.dict_constants)
        if model: our_global["self"] = model
        if device: our_global["device"] = device
        dict_info[name] = def_info.VariableInfo(
            eval(ast_add_on.ast_to_str(code_ast),our_global)
        )
    for sn in sg.nodes:
        for req_rd in sn.deps_rand:
            req_sn_rd = dict_random_sn[req_rd]
            S_edges.add_inplace(sn.deps,req_sn_rd,set([req_rd]))
            S_edges.add_inplace(req_sn_rd.users,sn,set([req_rd]))
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
    sg.refresh_info_data_name()
    sg.make_targets_attrs()
    sg.correct_label_over_edges()
    sg.unhook_init_node()
    sg.unhook_special_output_node()
    sg.assert_ready()
    return sg

# ==========================



# ==========================
# ==== Cut the graph in ====
# ==== sequential parts ====
# ==========================

class S_graph_list(list):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

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
    new_sg.inherit_base_attributes(sg)
    new_sg.whole_model_inputs = sg.whole_model_inputs
    new_sg.node_unique_id_generator = copy(sg.node_unique_id_generator)
    dict_nodes = {}
    new_sg.nodes = new_nodes = []
    # dict_nodes[new_init.main_target] = new_init # TO REMOVE
    for sn in sg.nodes:
        new_sn = copy_S_node(sn)
        new_nodes.append(new_sn)
        dict_nodes[sn.main_target] = new_sn
        for (req_sn,set_str) in sn.deps.items():
            new_req_sn = dict_nodes[req_sn.main_target]
            S_edges.add_inplace(new_req_sn.users,new_sn,set_str)
            S_edges.add_inplace(new_sn.deps,new_req_sn,set_str)

    # * init_node *
    new_sg.init_node \
        = new_init \
        = copy_S_node(sg.init_node)
    new_init.users = dict(
        (dict_nodes[u.mt],set_str) \
        for u,set_str in sg.init_node.users.items())
    
    # * output_nodes *
    new_sg.dict_output_viewing_code = dict(sg.dict_output_viewing_code )
    new_sg.output_nodes = [dict_nodes[out.mt] for out in sg.output_nodes]
    if sg.special_output_node is not None:
        new_sg.special_output_node \
            = special_out \
            = copy_S_node(sg.special_output_node)
        special_out.deps = dict(
            (dict_nodes[r.mt],set_str) \
            for r,set_str in sg.special_output_node.deps.items())
        
    # * artefact edges *
    new_artefact_edges = new_sg.artefact_edges
    for (req_sn,user_sn,used_targets) in sg.artefact_edges:
        new_artefact_edges.append((
            dict_nodes[req_sn.mt],
            dict_nodes[user_sn.mt],
            set(used_targets)
        ))

    return new_sg


def cut(sg : S_graph): # -> list of S_graph
    # Note: when this function is used, sg.init_node has been unhooked
    sg = copy_S_graph(sg) # to protect from side effects
    # -> Add a temporary global source before get separators
    # -> Above all the node which don't have any deps
    sg.nodes.insert(0,sg.init_node) # it's not the original sg, no risk
    for first_sn,str_set in sg.init_node.users.items():
        S_edges.add_inplace(first_sn.deps,sg.init_node,str_set)

    seps = RK_get_1_separators(sg)
    
    # -> remove tmp_source
    for first_sn in sg.init_node.users.keys():
        S_edges.discard_inplace(first_sn.deps,sg.init_node)
    
    seps = [sg.init_node] + seps
    # multiple output_nodes
    if not (seps[-1] is sg.nodes[-1]):
        seps.append(sg.nodes[-1])

    list_sg = []
    for block_nb in range(1,len(seps)):
        new_sg = S_graph()
        new_sg.whole_model_inputs = sg.whole_model_inputs
        new_sg.node_unique_id_generator = copy(sg.node_unique_id_generator)
        new_sg.inherit_base_attributes(sg)
        list_sg.append(new_sg)
        # -- get nodes --
        first_node = seps[block_nb-1]
        last_node = seps[block_nb]
        first_i = sg.nodes.index(first_node)
        last_i = sg.nodes.index(last_node)
        nodes = sg.nodes[first_i+1:last_i+1] # last IN, first NOT
        new_sg.nodes = nodes
        print_debug(f"size of bloc {block_nb} : {last_i}-{first_i}")
        # -- input --
        if block_nb==1:
            new_sg.init_node = sg.init_node
            new_sg.inputs = sg.inputs
        else:
            ino = copy_S_node(sg.init_node)
            # -> we want the init_code but NOT the deps
            new_sg.init_node = ino
            inputs = set()
            first_node_users = list(first_node.users.items())
            for (user_sn,str_set) in first_node_users:
                inputs.update(str_set)
                S_edges.discard_inplace(user_sn.deps,first_node)
                #S_edges.add_inplace(user_sn.deps,ino,str_set)
                S_edges.add_inplace(ino.users,user_sn,str_set)
                if user_sn.is_artefact:
                    ino.insert(user_sn,strong=True,sg=sg)
                    nodes.remove(user_sn)
            for user_sn in ino.users.keys(): # Unhook ino (need due to `ino.insert`)
                S_edges.discard_inplace(user_sn.deps,ino)
            first_node.users = dict() # previous bloc's output node
            new_sg.inputs = list(inputs)
        # -- outputs --
        if block_nb == len(seps)-1:
            new_sg.output_nodes = sg.output_nodes
            new_sg.special_output_node = sg.special_output_node
            new_sg.dict_output_viewing_code = sg.dict_output_viewing_code
        else:
            new_sg.output_nodes = [last_node]
    for i in range(len(list_sg)-1):
        list_sg[i].outputs = list(list_sg[i+1].inputs)
    list_sg[-1].outputs = sg.outputs
    return S_graph_list(list_sg)

# ==========================



# ==========================
# === printing functions ===
# ==========================

def aux_print_S_graph_message(sg : S_graph):
    return f"S_graph - Simplified forward graph : {len(sg.nodes)} nodes"

def aux_print_S_graph_list_message(lsg : S_graph_list):
    s = "+".join([str(len(sg.nodes)) for sg in lsg])
    return (
        f"S_graph_list - Sequentialized simplified forward graphs, "\
        f"{len(lsg)} blocks,\n     -> with {s} = "\
        f"{sum([len(sg.nodes) for sg in lsg])} nodes"
    )

def aux_print_S_graph_name(sg : S_graph,name=None):
    if name is not None: return name
    else: return "Simplified_forward_S_graph"

def aux_print_S_graph_list_name(lsg : S_graph_list,name=None):
    if name is not None: return name
    else: return "Sequentialized_Simplified_Forward_S_graph_list"


def aux_print_graph(dot,sg : S_graph,uniq_num):
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,str_set,**kwargs):
        dot.edge(uni(i1),uni(i2),label="\n".join(str_set),**kwargs)
    for sn in sg.nodes:
        if sn.is_artefact:
            node(sn.main_target,sn.get_code(),style="dashed")
        else:
            node(sn.main_target,sn.get_code())
    for sn in sg.nodes:
        for (req_sn,str_set) in sn.deps.items():
            edge(req_sn.main_target,sn.main_target,str_set)

    # -- inputs --
    ino_mt = sg.init_node.main_target
    ino_code = sg.init_node.get_code()
    ino_users = list(sg.init_node.users.items())
    if len(ino_users)!=0:
        node("input",f"INPUT",color="green",style="dashed")
        if ino_code != "":
            # "input" -> init_node -> first_nodes
            node(ino_mt,ino_code,style="dashed")
            edge("input",ino_mt,sg.inputs)
            for user_sn,used_targets in ino_users:
                edge(ino_mt,user_sn.mt,used_targets,style="dashed")
        else:
            # "input" -> first_nodes
            for user_sn,used_targets in ino_users:
                edge("input",user_sn.mt,used_targets,style="dashed")

    # -- outputs --
    node("output",f"OUTPUT",color="green",style="dashed")
    if sg.special_output_node is None:
        assert(len(sg.output_nodes)==1)
        edge(sg.output_nodes[0].mt,"output",sg.outputs)
    else:
        for out in sg.output_nodes:
            edge(out.mt,"output",sg.special_output_node.deps[out])


def print_S_graph(sg : S_graph,name=None,open=True,render_format="svg",dot=None,uniq_num=0):
    if dot is None:
        render = True
        name = aux_print_S_graph_name(sg,name)
        dot = graphviz.Digraph(name,comment=name)
    else:
        render = False
    aux_print_graph(dot,sg,uniq_num)
    if render:
        small_fcts.graph_render(dot,open,"S",render_format)


def print_S_graph_list(lsg : S_graph_list,name=None,open=True,render_format="svg"):
    name = aux_print_S_graph_list_name(lsg,name)
    dot = graphviz.Digraph(name,comment=name)
    for i in range(len(lsg)):
        aux_print_graph(dot,lsg[i],i)
    small_fcts.graph_render(dot,open,"S",render_format)

# ==========================

