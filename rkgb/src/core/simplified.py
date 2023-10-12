# ==========================
# ====== S structure =======
# ==========================

import sys
import ast
import torch
import copy
from src.lowlevel import ast_add_on
from src.lowlevel import constants
from src.lowlevel.variable_info import VariableInfo
from src.core import base
from src.core.forward import ForwardNode,ForwardGraph

# ***********
# * SimplifiedEdgeDict *
# ***********

# S edges (ie SimplifiedNodes.deps/users) are 
# dict : (SimplifiedNode -> str set)
# /!\ By default the following operations are NOT inplace
# inplace operations names end with "_inplace". 

class SimplifiedEdgeDict(dict):
    def merge_inplace(self,sub_dict):
        for sn,set_targets2 in sub_dict.items():
            set_targets1 = self[sn] if sn in self else set()
            self[sn] = set_targets1.union(set_targets2)
    def merge(self,sub_dict):
        new_dict = SimplifiedEdgeDict(self)
        new_dict.merge_inplace(sub_dict)
        return new_dict

    def discard_inplace(self,sn_to_discard):
        if sn_to_discard in self: del self[sn_to_discard]
    def discard(self,sn_to_discard):
        return SimplifiedEdgeDict(
            (sn,set_targets) 
            for (sn,set_targets) in self.items() 
            if sn != sn_to_discard)

    def add_inplace(self,sn_to_add,set_targets_to_add):
        set_targets_before = self[sn_to_add] if sn_to_add in self else set()
        self[sn_to_add] = set_targets_before.union(set_targets_to_add)
    def add(self,sn_to_add,set_targets_to_add):
        new_dict = SimplifiedEdgeDict(self)
        new_dict.add_inplace(sn_to_add,set_targets_to_add)
        return new_dict

    @staticmethod
    def discard_sn_from_deps_of_its_users(sn):
        for user_sn in sn.users.keys():
            user_sn.deps.discard_inplace(sn)
    @staticmethod
    def discard_sn_from_users_of_its_deps(sn):
        for req_sn in sn.deps.keys():
            req_sn.users.discard_inplace(sn)

    @staticmethod
    def make_users_using_deps(sn):
        for (req_sn,set_targets) in sn.deps.items():
            req_sn.users[sn] = set(set_targets)
    @staticmethod
    def make_deps_using_users(sn):
        for (user_sn,set_targets) in sn.users.items():
            user_sn.deps[sn] = set(set_targets)

    def issubset(self,bigger_dict):
        for (sn,set_targets) in self.items():
            if sn not in bigger_dict: return False
            if set_targets > bigger_dict[sn]: return False
        return True



# **********
# * SimplifiedNode *
# **********

class SimplifiedNode(base.Node):
    def __init__(self,
            main_target="No target",
            code=None,
            fct="",
            protected=False,
            is_rand=False,
            deps_rand=None,
            simplified_graph=None):
        """
        A SimplifiedNode is composed by one "real" computation, defining the
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
        .is_artifact: bool : see Doc (useful size node)
        .deps       : (SimplifiedNode,str set) dict = dict_edges
            -> required nodes with the list of vars needed per node.
        .users      : dict_edges : reciprocal of .deps
        .is_rand    : bool
        .deps_rand  : str set : because we don't want random src nodes here
        """
        super().__init__("S",main_target,
            parent_structure_with_id_generator=simplified_graph)
        self.is_artifact = False
        self.main_code = (main_target,code)
        self.main_fct = fct
        self.inplace_code = [] # list of tar * AST
        self.body_code    = []
        self.all_targets = [main_target]
        self.tensor_targets = [] # later
        self.inplace_targets = [] # later
        self.container_targets = [] # later
        self.deps = SimplifiedEdgeDict()
        self.users = SimplifiedEdgeDict()
        self.protected = protected
        self.is_rand   = is_rand
        self.deps_rand = deps_rand if deps_rand else set()
        self.info : VariableInfo = None

    def get_all_standard_deps(self):
        return set(self.deps.keys())
    def get_all_standard_users(self):
        return set(self.users.keys())

    def insert_code(self,sn_to_insert,simplified_graph):
        sn_info = sn_to_insert.info
        # 1) main_code:
        main_code_to_insert = sn_to_insert.main_code
        if not sn_info.is_inplace:
            if main_code_to_insert is None:
                raise Exception("Try to insert empty code")
            self.body_code.append(main_code_to_insert)
        else:
            # -> we want to push 'sn' to inplace code 
            # so we have to push its deps that were in body_code to inplace_code
            # eg: a = f(x) ; v = view(a) ; ip = inplace(v) ; b = f(ip)
            # Since ip requires v, then v is pushed in inplace code
            # so the final code will look like:
            # _a = f(x) ; v = view(_a) ; ip = inplace(v)
            # a = _a.detach() ; v = view(a) ; ip = v ; b = f(ip)
            sn_data_parents = []
            parent_name = sn_info.data_direct_parent_name
            while parent_name != sn_info.data_owner_name:
                sn_data_parents.append(parent_name)
                parent_info = simplified_graph.dict_info[parent_name]
                parent_name = parent_info.data_direct_parent_name
            already_in_inplace_code = set(c[0] for c in self.inplace_code)
            for code in self.body_code:
                if (code[0] in sn_data_parents
                and code[0] not in already_in_inplace_code):
                    self.inplace_code.append(code)
            self.inplace_code.append(main_code_to_insert)
            self.body_code.append((
                sn_to_insert.main_target,
                ast.Name(sn_info.data_direct_parent_name)
            )) # the equivalent of `ip = v` in the example above

        # 2) No matter inplace or not: add the body part
        self.body_code.extend(sn_to_insert.body_code)
        self.inplace_code.extend(sn_to_insert.inplace_code)
        self.all_targets.extend(sn_to_insert.all_targets)
        self.is_rand = self.is_rand or sn_to_insert.is_rand
        self.deps_rand.update(sn_to_insert.deps_rand)


    def insert(self,sn_to_insert,strong,simplified_graph):
        # this is the fct to merge nodes
        # if strong: delete sn_to_insert else it becomes an artifact
        # in any case cut as many edges as possible

        # deps of self after the merge
        merged_deps = self.deps.merge(sn_to_insert.deps)
        merged_deps.discard_inplace(self)

        # 1) disconnect sn_to_insert from its users
        # except if artifact, in which case by definition 
        # artifact.users := sn_to_insert.users - self.users
        if strong: # e.g. for "view"; sn_to_insert.users = empty
            SimplifiedEdgeDict.discard_sn_from_deps_of_its_users(sn_to_insert)
            merged_users = self.users.merge(sn_to_insert)
            merged_users.discard_inplace(sn_to_insert)
            sn_to_insert.users = SimplifiedEdgeDict()
        else: # e.g. for "size"; sn_to_insert.users -= self.users
            for user_sn in self.users.keys():
                user_sn.deps.discard_inplace(sn_to_insert)
                sn_to_insert.users.discard_inplace(user_sn)
            merged_users = self.users

        # 2) disconnect sn_to_insert from its deps if it will 
        # be deleted, ie if: sn_to_insert.users = empty
        if sn_to_insert.users == dict():
            SimplifiedEdgeDict.discard_sn_from_users_of_its_deps(sn_to_insert)
            sn_to_insert.deps = SimplifiedEdgeDict()
            # -> aux_sn has been fully unplugged
        else:
            sn_to_insert.is_artifact = True

        # 3) insert the code; set the new edges
        # and replace sn_to_insert by self in output_nodes if needed
        self.insert_code(sn_to_insert,simplified_graph)
        self.deps = merged_deps
        self.users = merged_users
        SimplifiedEdgeDict.make_users_using_deps(self)
        SimplifiedEdgeDict.make_deps_using_users(self)
        if sn_to_insert in simplified_graph.output_nodes:
            i = simplified_graph.output_nodes.index(sn_to_insert)
            simplified_graph.output_nodes[i] = self


    def clear_children_artifact(self): # TODO
        # clean useless artifact children of self
        children = dict(self.users)
        for art_user_sn in children.keys():
            if art_user_sn.is_artifact:
                if set(art_user_sn.deps.keys()) != {self}:
                    s = ",".join(
                      [aux_sn.main_target for aux_sn in art_user_sn.deps])
                    raise Exception(
                      f"{self.main_target} should be the only "\
                      f"parent of {art_user_sn.main_target} : "\
                      f"{len(art_user_sn.deps)}\n{s}")
                for other_user_sn in self.users.keys():
                    if art_user_sn in other_user_sn.deps.keys():
                        SimplifiedEdgeDict.add_edge_inplace(
                            self,other_user_sn,
                            other_user_sn.deps[art_user_sn])
                        # We add the names of the variables generated 
                        # by the code of art_user_sn (present in self)
                        # which are used in other_user_sn. Because
                        # to get those vars we will no longer use
                        # art_user_sn since we already need self for
                        # something else.
                    SimplifiedEdgeDict.discard_edge_inplace(
                        art_user_sn,other_user_sn)
                if art_user_sn.users == dict():
                    SimplifiedEdgeDict.discard_sn_from_users_of_its_deps(
                        art_user_sn)
                    art_user_sn.deps = dict()

    def clear_siblings_artifact(self): # TODO
        real_deps = set()
        for req_sn in self.deps.keys():
            if not req_sn.is_artifact:
                real_deps.add(req_sn)
        for req_sn in real_deps:
            req_sn.clear_children_artifact()


# ***********
# * SimplifiedGraph *
# ***********

# in the description: I need to explain "init_node"
class SimplifiedGraph(base.Graph):
    def __init__(self,
            forward_graph : ForwardGraph = None,
            model=None,
            device=None
    ):
        # 2 constructors: if given a forward_graph, then move from F to S
        # otherwise return an empty simplified_graph
        super().__init__("S")
        self.init_node = None # NOT in self.nodes
        self.special_output_node = None # NOT in self.nodes
        self.dict_output_viewing_code = dict()
        self.edges_via_artifacts : list[tuple[SimplifiedNode,SimplifiedNode,set[str]]] = []

        # Move from forward.py to simplified.py
        if forward_graph is not None:
            if model is None or device is None: 
                raise Exception(
                    "You need to pass model and device to "\
                    "SimplifiedGraph.__init__ to move from F to S")
            self.inherit_base_attributes(forward_graph)
            self.whole_model_inputs = set(forward_graph.inputs)

            init_node = SimplifiedNode(
                main_target=base.Graph.default_init_target_string,
                simplified_graph=self)
            init_node.all_targets=[]

            dict_s_nodes = dict() # to translate D to S
            for dn in dg.nodes:
                sn = SimplifiedNode(
                        main_target=dn.target,
                        code=dn.code_ast,
                        fct=dn.fct,
                        protected=dn.protected,
                        is_rand=dn.is_rand,
                        deps_rand= set(dn.deps_rand),
                        other_obj=sg)
                self.nodes.append(sn)
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
            # -> in ForwardGraph these nodes have a dummy code `'code = 'INPUT'`,
            # -> the "insert" method will put these dummy codes in init_node.body_code
            # -> that's why we clear init_node.body_code at the end of initialization
            sg.output_nodes = [dict_s_nodes[out] for out in dg.outputs]
            sg.clear()
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
            SimplifiedEdgeDict.discard_inplace(user_sn.deps,self.init_node)
            if all(not dict_info[used_tgt].requires_grad
                   for used_tgt in used_targets):
                SimplifiedEdgeDict.discard_inplace(self.init_node.users,user_sn)
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
                SimplifiedEdgeDict.discard_inplace(req_sn.users,output_node)
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
            

    def check_artifact(self):
        for sn in self.nodes:
            if sn.is_artifact:# and not (sn is self.init_node):
                if len(sn.deps)!=1:
                    raise Exception(
                      f"{sn.main_target} is_artifact, but with "\
                      f"len(deps)={len(sn.deps)} (should be 1)")
                req_sn = list(sn.deps.keys())[0]
                if SimplifiedEdgeDict.issubset(sn.users,req_sn.users):
                    print(f"{sn.main_target} is a useless "\
                          f"artifact of {req_sn.main_target}")

    def check_relations(self):
        for sn in self.nodes:
            for (req_sn,set_targets) in sn.deps.items():
                if (sn not in req_sn.users) or set_targets != req_sn.users[sn]:
                    raise Exception(
                      f"{req_sn.main_target} in {sn.main_target}.deps "\
                      f"but one sided relation...")
            for (user_sn,set_targets) in sn.users.items():
                if (sn not in user_sn.deps) or set_targets != user_sn.deps[sn]:
                    raise Exception(
                      f"{user_sn.main_target} in {sn.main_target}.users "\
                      f"but one sided relation...")
                
    # === To handle artifacts in Ptools ===
    def discard_all_artifacts(self):
        # Do this only once the order is fixed!
        # And K_graph is generated
        # -> better do a copy first
        snodes = list(self.nodes)
        nb_nodes = len(snodes)
        edges_via_artifacts = self.edges_via_artifacts
        for i,sn in enumerate(snodes[::-1]):
            index = nb_nodes-i-1
            if sn.is_artifact:
                del self.nodes[index]
                real_sn = list(sn.deps.keys())[0]
                for user_sn,used_targets in sn.users.items():
                    SimplifiedEdgeDict.discard_inplace(user_sn.deps,sn)
                    SimplifiedEdgeDict.add_edge_inplace(real_sn,user_sn,used_targets)
                    edges_via_artifacts.append((real_sn,user_sn,used_targets))
                SimplifiedEdgeDict.discard_inplace(real_sn.users,sn)
    def delete_edges_via_artifacts(self):
        for (used_sn,user_sn,_) in self.edges_via_artifacts:
            SimplifiedEdgeDict.discard_edge_inplace(used_sn,user_sn)
        # We do NOT set self.edges_via_artifacts = []


    def clear(self):
        # -- (re)-sorting nodes -- 
        # due to merging, the topo order may not be correct anymore
        # by the way, remove unplugged nodes
        if len(self.output_nodes)==1:
            root_sn = self.output_nodes[0]
            real_root = True
        else:
            real_root = False
            root_sn = SimplifiedNode("Tmp_root")
            root_sn.deps = dict((out_sn,set()) for out_sn in self.output_nodes)
            for out_sn in self.output_nodes:
                out_sn.users[root_sn] = set()
        self.nodes = base.Graph.get_sorted_nodes_by_following_relation_deps(root_sn)
        if self.init_node in self.nodes: self.nodes.remove(self.init_node)
        if not real_root:
            self.nodes.remove(root_sn)
            for out_sn in root_sn.deps:
                del out_sn.users[root_sn]
        self.check_artifact()
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
                if owner_sn.is_artifact:
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
            if not sn.is_artifact:
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
        # ie main_targets are tensors, except if artifact -> sizes
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
            if info.variable_type==torch.Size and not sn.is_artifact:
                raise Exception(
                  f"After simplifications, all remaining "\
                  f"\"size\" should be \"artifacts\", but "\
                  f"{sn.main_target} isn't an artifact")


# ==========================


# ==========================
# = Init move from D to S  =
# ==========================


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
        SimplifiedEdgeDict.merge_inplace(user_sn.deps,sn.deps)
        SimplifiedEdgeDict.discard_inplace(user_sn.deps,sn)
        for (req_sn,set_targets) in sn.deps.items():
            SimplifiedEdgeDict.discard_inplace(req_sn.users,sn)
            SimplifiedEdgeDict.add_inplace(req_sn.users,user_sn,set_targets)
        # -- insert the code --
        insert_code_ast(user_sn,sn)
        # -- handle randomness --
        user_sn.is_rand = user_sn.is_rand or sn.is_rand
        user_sn.deps_rand.update(sn.deps_rand)
    sn.deps  = dict()
    sn.users = dict()

def simplify_cheap(sg : SimplifiedGraph):
    # from root to leaves
    for sn in sg.nodes:
        if ( not (sn in sg.output_nodes)
         and    (sn.main_fct in constants.list_cheap_fct
            or 
                (sn.main_fct in constants.list_optional_cheap_fct and not sn.protected)
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
#    parent, and keep them only if needed -> artifact

def size_children(sg,sn):
    # give the list of child nodes of sn which are size
    ret = []
    for user_sn in sn.users.keys():
        if sg.dict_info[user_sn.main_target].variable_type == torch.Size:
            ret.append(user_sn)
    return ret


def simplify_size(sg : SimplifiedGraph):
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
        if not req_sn.is_artifact)

def get_direct_real_deps(sn):
    deps = get_all_real_deps(sn)
    for req_sn in deps:
        if get_all_real_deps(req_sn) ==  deps-set([req_sn]):
            return set([req_sn])
    return deps

def simplify_view(sg : SimplifiedGraph):
    # from root to leaves
    sg.init_node.is_artifact = True
    for sn in sg.nodes:
        sn_info = sg.dict_info[sn.main_target]
        if (sn_info.is_view
        or  sn.main_fct in constants.list_view_fct # -> in case of viewing operations over parameters
        or  sn.main_fct == "getattr"
        or  sn_info.is_inplace):
            # ASSERTION remaining getattr are related to views !! 
            # we also consider inplace ops as views
            real_deps = get_direct_real_deps(sn)
            if len(real_deps)==1:
                req_sn = real_deps.pop()
                req_sn.insert(sn,strong=True,sg=sg)
                req_sn.clear_children_artifact()
                req_sn.clear_siblings_artifact()
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
                        inplace_real_node.clear_siblings_artifact()
            elif len(real_deps)==0 and len(sn.deps)>0:
                # experimental : I assume that views which don't 
                # require any real tensor are views over parameters
                # so mem=0 and no bwd K_node, so I can insert them
                # in their parents even if they are artifacts.
                # But artifact nodes aren't safe, they might disappear
                # if self.users sub set of self.parent.users
                # so I must share the code with artifacts' parent
                # It's not a problem to insert the code in different 
                # nodes because view operations are cheap.
                # But I must avoid creating cycle dependencies, so
                # for the moment I assert len(sn.deps)==1
                if sn_info.is_inplace: raise Exception(
                    f"Sorry we do not support inplace operations over "\
                    f"parameters (or anything that isn't a Tensor). \n"\
                    f"Here {sn.main_target} only deps on artifacts, but"\
                    f"sn_info.is_inplace=True :/")
                for art_req in sn.deps.keys():
                    if len(art_req.deps)==0:
                        assert(art_req is sg.init_node)
                        real_req = None
                    else:
                        assert(len(art_req.deps)==1) # as an artifact
                        real_req = list(art_req.deps.keys())[0]
                        real_req.insert_code(sn,sg)
                    art_req.insert_code(sn,sg)
                    # -> Insert sn's code BOTH in art_req and real_req

                    # - plug art_req to sn's users -
                    SimplifiedEdgeDict.merge_inplace(art_req.users,sn.users)
                    for (user_sn,set_targets) in sn.users.items():
                        SimplifiedEdgeDict.add_inplace(user_sn.deps,art_req,set_targets)
                    # - unplug sn -
                    SimplifiedEdgeDict.discard_inplace(art_req.users,sn)
                    SimplifiedEdgeDict.discard_sn_from_deps_of_its_users(sn)
                    sn.deps = dict()
                    sn.users = dict()
                    if real_req: real_req.clear_children_artifact()

    sg.clear()

# ==========================



# ==========================
#  Insert random operations 
#  which were in dict_rand
# ==========================
# -> Now that we simplified everything,
# -> we can insert random nodes from dict_rand

def create_random_snodes_from_dict_rand(sg,model,device):
    dict_random_sn = dict() # str -> SimplifiedNode
    dict_info = sg.dict_info
    for name,code_ast in sg.dict_rand.items():
        dict_random_sn[name] = SimplifiedNode(
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
        dict_info[name] = VariableInfo(
            eval(ast_add_on.ast_to_str(code_ast),our_global)
        )
    for sn in sg.nodes:
        for req_rd in sn.deps_rand:
            req_sn_rd = dict_random_sn[req_rd]
            SimplifiedEdgeDict.add_inplace(sn.deps,req_sn_rd,set([req_rd]))
            SimplifiedEdgeDict.add_inplace(req_sn_rd.users,sn,set([req_rd]))
    sg.nodes = list(dict_random_sn.values()) + sg.nodes

# ==========================



# ==========================
# = Move from D to S graph =
# ==========================


# ==========================



# ==========================
# ==== Cut the graph in ====
# ==== sequential parts ====
# ==========================

class SimplifiedGraph_list(list):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

def copy_SimplifiedNode(sn : SimplifiedNode): # aux for copy_SimplifiedGraph
    new_sn = SimplifiedNode()
    new_sn.is_artifact       = sn.is_artifact
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

def copy_SimplifiedGraph(sg : SimplifiedGraph):
    # -> a copy of sg with fresh nodes
    new_sg = SimplifiedGraph()
    new_sg.inherit_base_attributes(sg)
    new_sg.whole_model_inputs = sg.whole_model_inputs
    new_sg.node_unique_id_generator = copy(sg.node_unique_id_generator)
    dict_nodes = {}
    new_sg.nodes = new_nodes = []
    # dict_nodes[new_init.main_target] = new_init # TO REMOVE
    for sn in sg.nodes:
        new_sn = copy_SimplifiedNode(sn)
        new_nodes.append(new_sn)
        dict_nodes[sn.main_target] = new_sn
        for (req_sn,set_str) in sn.deps.items():
            new_req_sn = dict_nodes[req_sn.main_target]
            SimplifiedEdgeDict.add_inplace(new_req_sn.users,new_sn,set_str)
            SimplifiedEdgeDict.add_inplace(new_sn.deps,new_req_sn,set_str)

    # * init_node *
    new_sg.init_node \
        = new_init \
        = copy_SimplifiedNode(sg.init_node)
    new_init.users = dict(
        (dict_nodes[u.mt],set_str) \
        for u,set_str in sg.init_node.users.items())
    
    # * output_nodes *
    new_sg.dict_output_viewing_code = dict(sg.dict_output_viewing_code )
    new_sg.output_nodes = [dict_nodes[out.mt] for out in sg.output_nodes]
    if sg.special_output_node is not None:
        new_sg.special_output_node \
            = special_out \
            = copy_SimplifiedNode(sg.special_output_node)
        special_out.deps = dict(
            (dict_nodes[r.mt],set_str) \
            for r,set_str in sg.special_output_node.deps.items())
        
    # * artifact edges *
    new_edges_via_artifacts = new_sg.edges_via_artifacts
    for (req_sn,user_sn,used_targets) in sg.edges_via_artifacts:
        new_edges_via_artifacts.append((
            dict_nodes[req_sn.mt],
            dict_nodes[user_sn.mt],
            set(used_targets)
        ))

    return new_sg


def cut(sg : SimplifiedGraph): # -> list of SimplifiedGraph
    # Note: when this function is used, sg.init_node has been unhooked
    sg = copy_SimplifiedGraph(sg) # to protect from side effects
    # -> Add a temporary global source before get separators
    # -> Above all the node which don't have any deps
    sg.nodes.insert(0,sg.init_node) # it's not the original sg, no risk
    for first_sn,set_targets in sg.init_node.users.items():
        SimplifiedEdgeDict.add_inplace(first_sn.deps,sg.init_node,set_targets)

    seps = sg.find_cutting_points()
    
    # -> remove tmp_source
    for first_sn in sg.init_node.users.keys():
        SimplifiedEdgeDict.discard_inplace(first_sn.deps,sg.init_node)
    
    seps = [sg.init_node] + seps
    # multiple output_nodes
    if not (seps[-1] is sg.nodes[-1]):
        seps.append(sg.nodes[-1])

    list_sg = []
    for block_nb in range(1,len(seps)):
        new_sg = SimplifiedGraph()
        new_sg.whole_model_inputs = sg.whole_model_inputs
        new_sg.node_unique_id_generator = copy.copy(sg.node_unique_id_generator)
        new_sg.inherit_base_attributes(sg)
        list_sg.append(new_sg)
        # -- get nodes --
        first_node = seps[block_nb-1]
        last_node = seps[block_nb]
        first_i = sg.nodes.index(first_node)
        last_i = sg.nodes.index(last_node)
        nodes = sg.nodes[first_i+1:last_i+1] # last IN, first NOT
        new_sg.nodes = nodes
        # -- input --
        if block_nb==1:
            new_sg.init_node = sg.init_node
            new_sg.inputs = sg.inputs
        else:
            ino = copy_SimplifiedNode(sg.init_node)
            # -> we want the init_code but NOT the deps
            new_sg.init_node = ino
            inputs = set()
            first_node_users = list(first_node.users.items())
            for (user_sn,set_targets) in first_node_users:
                inputs.update(set_targets)
                SimplifiedEdgeDict.discard_inplace(user_sn.deps,first_node)
                #SimplifiedEdges.add_inplace(user_sn.deps,ino,set_targets)
                SimplifiedEdgeDict.add_inplace(ino.users,user_sn,set_targets)
                if user_sn.is_artifact:
                    ino.insert(user_sn,strong=True,sg=sg)
                    nodes.remove(user_sn)
            for user_sn in ino.users.keys(): # Unhook ino (need due to `ino.insert`)
                SimplifiedEdgeDict.discard_inplace(user_sn.deps,ino)
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
    return SimplifiedGraph_list(list_sg)

# ==========================



# ==========================
# === printing functions ===
# ==========================

def aux_print_SimplifiedGraph_message(sg : SimplifiedGraph):
    return f"SimplifiedGraph - Simplified forward graph : {len(sg.nodes)} nodes"

def aux_print_SimplifiedGraph_list_message(lsg : SimplifiedGraph_list):
    s = "+".join([str(len(sg.nodes)) for sg in lsg])
    return (
        f"SimplifiedGraph_list - Sequentialized simplified forward graphs, "\
        f"{len(lsg)} blocks,\n     -> with {s} = "\
        f"{sum([len(sg.nodes) for sg in lsg])} nodes"
    )

def aux_print_SimplifiedGraph_name(sg : SimplifiedGraph,name=None):
    if name is not None: return name
    else: return "Simplified_forward_SimplifiedGraph"

def aux_print_SimplifiedGraph_list_name(lsg : SimplifiedGraph_list,name=None):
    if name is not None: return name
    else: return "Sequentialized_Simplified_Forward_SimplifiedGraph_list"


def aux_print_graph(dot,sg : SimplifiedGraph,uniq_num):
    def uni(tar): return f"_{uniq_num}_{tar}"
    def node(i,l,**kwargs): dot.node(uni(i),l,**kwargs)
    def edge(i1,i2,set_targets,**kwargs):
        dot.edge(uni(i1),uni(i2),label="\n".join(set_targets),**kwargs)
    for sn in sg.nodes:
        if sn.is_artifact:
            node(sn.main_target,sn.get_code(),style="dashed")
        else:
            node(sn.main_target,sn.get_code())
    for sn in sg.nodes:
        for (req_sn,set_targets) in sn.deps.items():
            edge(req_sn.main_target,sn.main_target,set_targets)

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


def print_SimplifiedGraph_list(lsg : SimplifiedGraph_list,dot,name=None,open=True,render_format="svg"):
    for i in range(len(lsg)):
        aux_print_graph(dot,lsg[i],i)

# ==========================

