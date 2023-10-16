# ==========================
# ====== S structure =======
# ==========================

import sys
import copy
import warnings
import ast
import torch
from src.lowlevel import ast_add_on
from src.lowlevel import constants
from src.lowlevel.variable_info import VariableInfo
from src.core import base
from src.core.forward import ForwardNode,ForwardGraph


# **********
# * SimplifiedNode *
# **********

class SimplifiedNode(base.Node):
    def __init__(self,
            main_target="No target",
            code=None,
            fct="",
            info=None,
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
        .deps       : set[SimplifiedNode]
        .users      : set[SimplifiedNode]
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
        self.deps = set()
        self.users = set()
        self.deps_through_artifacts = set()
        self.users_through_artifacts = set()
        self.info : VariableInfo = info if info is not None else VariableInfo()
        self.protected = protected
        self.is_rand   = is_rand
        self.deps_rand = deps_rand if deps_rand else set()

    def get_all_standard_deps(self):
        return self.deps.union(self.deps_through_artifacts)
    def get_all_standard_users(self):
        return self.users.union(self.users_through_artifacts)

    # === INSERT ===
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
        merged_deps = self.deps.union(sn_to_insert.deps)
        merged_deps.discard(self)

        # 1) disconnect sn_to_insert from its users
        # except if artifact, in which case by definition 
        # artifact.users := sn_to_insert.users - self.users
        if strong: # e.g. for "view"; sn_to_insert.users = empty
            for user_sn in sn_to_insert.users:
                user_sn.deps.discard(sn_to_insert)
            merged_users = self.users.union(sn_to_insert.users)
            merged_users.discard(sn_to_insert)
            sn_to_insert.users = set()
        else: # e.g. for "size"; sn_to_insert.users -= self.users
            for user_sn in self.users:
                user_sn.deps.discard(sn_to_insert)
                sn_to_insert.users.discard(user_sn)
            merged_users = self.users

        # 2) disconnect sn_to_insert from its deps if it will 
        # be deleted, ie if: sn_to_insert.users = empty
        if sn_to_insert.users == set():
            for req_sn in sn_to_insert.deps:
                req_sn.users.discard(sn_to_insert)
            sn_to_insert.deps = set()
            # -> sn_to_insert has been fully unplugged
        else:
            sn_to_insert.is_artifact = True

        # 3) insert the code; set the new edges
        # and replace sn_to_insert by self in output_nodes if needed
        self.insert_code(sn_to_insert,simplified_graph)
        self.deps = merged_deps
        self.users = merged_users
        for req_sn in merged_deps: req_sn.users.add(self)
        for user_sn in merged_users: user_sn.deps.add(self)
        if sn_to_insert in simplified_graph.output_nodes:
            i = simplified_graph.output_nodes.index(sn_to_insert)
            simplified_graph.output_nodes[i] = self
    # === END INSERT ===


    # === SUBSTITUTE ===
    def substitute_an_id_by_a_code_in_self(
            self,id_to_replace,code_replacing,dict_info):
        """
        e.g. id_to_replace = "a"; code_replacing = AST("f(x)")
        with self's code: AST(b = g(a))
        this method change self's code, to: AST(b = g(f(a)))
        """
        main_code_expr = self.main_code[1]
        if (isinstance(main_code_expr,ast.Subscript)
        and isinstance(main_code_expr.value,ast.Name)
        and main_code_expr.value.id == id_to_replace
        and ast_add_on.is_constant(main_code_expr.slice)
        and (isinstance(code_replacing,ast.List)
          or isinstance(code_replacing,ast.Tuple))):
            # Special case for ast.Subscript,
            # e.g. "a = [x,y]" and "b = a[0]"
            # we can set "b = x", and then substitute
            # "b" by "x" in Node(b).users, so we avoid 
            # defining the useless variable "b", and 
            # instead directly use "x"
            index = main_code_expr.slice.value
            assert(isinstance(index,int)) 
            # Otherwise: slice with a string ??? Like : a = b["attr"] ???
            self.main_code = (
                self.main_target,
                code_replacing.elts[index])
            self.substitute_self_by_its_code_in_its_users(dict_info)
        else:
            self.main_code = (
                self.main_target,
                ast_add_on.substitute(
                    main_code_expr,
                    id_to_replace,
                    code_replacing))

    def substitute_self_by_its_code_in_its_users(self,dict_info):
        """
        e.g. a = f(x) ; b = g(a), calling this function 
        on 'a' => b = g(f(x)); useful notably for list 
        constructors, as we want nodes to represent tensors,
        not tuple or list of tensors.
        """
        self_info = self.info
        self_deps = self.deps
        self_target = self.main_target # to avoid getattr() inside for loops
        user_sn : SimplifiedNode
        for user_sn in self.users:
            # 1) unplug self and plug its users with its deps:
            # user_sn now directly depends on deps of self 
            # (since self's code is integrated in user_sn's code)
            user_sn.deps.update(self_deps)
            user_sn.deps.discard(self)
            for req_sn in self_deps:
                req_sn.users.discard(self)
                req_sn.users.add(user_sn)
            # 2) insert the code
            user_sn.substitute_an_id_by_a_code_in_self(
                self.target,self.main_code,dict_info)
            # 3) handle randomness
            user_sn.is_rand = user_sn.is_rand or self.is_rand
            user_sn.deps_rand.update(self.deps_rand)
            # 4) data_direct_parent_name
            if user_sn.info.data_direct_parent_name == self_target:
                if self_info.data_direct_parent_name == self_target:
                    new_direct_parent = user_sn.main_target
                else:
                    new_direct_parent = self_info.data_direct_parent_name
                user_sn.info.data_direct_parent_name = new_direct_parent

        if self_info.inplace_targets != set():
            raise Exception(
                f"A substitution (ie inserting the code inside the users) "\
                f"on a node which has some side inplace operations, "\
                f"shouldn't happen. Code substituted: {self.get_code()}"
            )

        # Correct the data_owner of views of self (as we are removing self)
        if (self_info.view_targets != set() 
        and len(self.users)==1): # Only 1 user => it's fine, otherwise impossible
            unique_user_sn : SimplifiedNode = next(iter(self.users))
            unique_user_target = unique_user_sn.main_target
            for view_target in self_info.view_targets:
                view_info = dict_info[view_target]
                view_info.data_owner_name = unique_user_target # instead of self
                # data_direct_parent_name already changed
            assert(unique_user_target in self_info.view_targets) # TO REMOVE
            view_targets = set(self_info.view_targets) # I prefer to copy; TO REMOVE
            view_targets.discard(unique_user_target)
            unique_user_sn.info.view_targets = view_targets
            unique_user_sn.info.is_view = False
        self.deps = set()
        self.users = set()
    # === END SUBSTITUTE ===

    def remove_obsolete_child_artifacts(self):
        # An artifact is obsolete if it no longer have
        # any user that isn't an user of its parent.
        child_artifacts = [
            user_sn for user_sn in self.users 
            if user_sn.is_artifact]
        for artifact_sn in child_artifacts:
            # Check everything is going fine
            if artifact_sn.deps != {self}:
                raise Exception(
                    f"Error with artifact {artifact_sn.mt}, "\
                    f"its only dependence should be {self.mt}, "\
                    f"but deps = {artifact_sn.deps}")

            # update and unplug if obsolete
            artifact_sn.users -= self.users
            if artifact_sn.users == set():
                self.users.discard(artifact_sn)
                artifact_sn.deps = set()

    def remove_obsolete_sibling_artifacts(self):
        # 1) get the deps 2) remove child artifacts of them
        # and children of deps of self are: siblings of self
        non_artifact_deps = [
            req_sn for req_sn in self.deps
            if not req_sn.is_artifact]
        for req_sn in non_artifact_deps:
            req_sn.remove_obsolete_child_artifacts()


# ***********
# * SimplifiedGraph *
# ***********

# in the description: I need to explain "init_node"
class SimplifiedGraph(base.Graph):
    node_class = SimplifiedNode
    init_node : SimplifiedNode = None # NOT in self.nodes
    wrapper_output_node : SimplifiedNode = None # NOT in self.nodes
    dict_output_viewing_code : dict[str,ast.Module] = None
    dict_of_labels_on_edges : dict[tuple[SimplifiedNode,SimplifiedNode],set[str]] = None
    edges_via_artifacts : list[tuple[SimplifiedNode,SimplifiedNode]] = None
    def __init__(self,
            forward_graph : ForwardGraph = None,
            model=None,
            device=None
    ):
        # 2 constructors: if given a forward_graph, then move from F to S
        # otherwise return an empty simplified_graph
        super().__init__("S")
        if forward_graph is not None:
            if model is None or device is None: 
                raise Exception(
                    "You need to pass model and device to "\
                    "SimplifiedGraph.__init__ to move from F to S")
            self.inherit_base_attributes(forward_graph)
            self.whole_model_inputs = set(forward_graph.inputs)

            self.init_node = init_node = SimplifiedNode(
                main_target=constants.constant_init_target_string,
                simplified_graph=self)
            init_node.all_targets=[]

            # translate each node one by one
            dict_simplified_nodes = dict()
            fn : ForwardNode
            for fn in forward_graph.nodes:
                sn = SimplifiedNode(
                    main_target=fn.target,
                    code=fn.code_ast,
                    fct=fn.fct,
                    info=fn.info,
                    protected=fn.protected,
                    is_rand=fn.is_rand,
                    deps_rand=set(fn.deps_rand),
                    simplified_graph=self)
                self.nodes.append(sn)
                dict_simplified_nodes[fn.target] = sn
                for req_fn in fn.deps:
                    req_sn = dict_simplified_nodes[req_fn.target]
                    req_sn.users.add(sn)
                    sn.deps.add(req_sn)

            # merge all the inputs in the special `init_node`
            for input_target in forward_graph.inputs:
                init_node.insert(
                    dict_simplified_nodes[input_target],
                    strong=True,simplified_graph=self)
            init_node.body_code = []
            # At the beginning (here), init_node contains only the 'real' inputs
            # in ForwardGraph these nodes have a dummy code `code = 'INPUT'`,
            # the "insert" method will put these dummy codes in init_node.body_code
            # that's why we clear init_node.body_code at the end of initialization

            self.output_nodes = [
                dict_simplified_nodes[out] 
                for out in forward_graph.outputs]
            self.clear()
            self.simplify_cheap() # TODO
            self.simplify_size() # TODO
            self.simplify_view() # TODO
            self.create_nodes_for_random_operations_from_dict_rand(model,device)
            self.check_edges_are_reciprocal()
            self.make_dict_of_labels_on_edges()
            self.make_targets_attributes_and_fix_info_data_owner_name()
            self.make_inputs()
            self.unplug_init_node()
            self.if_multiple_outputs_break_the_wrapper_in_multiple_nodes()
            self.make_dict_output_viewing_code()
            self.assert_ready()



    # ===== BLOC 1 : CLEAR and CHECK =====
    def clear(self):
        self.toposort_nodes()
        self.check_artifact()
        self.check_edges_are_reciprocal()
        
    def toposort_nodes(self):
        # As we're doing some merges, we will have to re-sort
        # 1) Find (or build) the node at the root of the deps relation
        if len(self.output_nodes)==1: # Simple case:
            root_sn = self.output_nodes[0]
            fake_tmp_root = False
        else: 
            # We need to generate a node, parent to all output_nodes
            # in the deps relation (like a very last node to the graph)
            fake_tmp_root = True
            root_sn = SimplifiedNode("Tmp_root")
            root_sn.deps = set(self.output_nodes)
            for out_sn in self.output_nodes:
                out_sn.users.add(root_sn)
        # 2) sort
        self.nodes = base.Graph.get_sorted_nodes_by_following_deps_relation(root_sn)
        # 3) remove the fake root (if created) and the init_node
        # because we don't want the init_node in self.nodes
        # but it was fetch by following deps will sorting
        if self.init_node in self.nodes: self.nodes.remove(self.init_node)
        if fake_tmp_root:
            self.nodes.remove(root_sn)
            for out_sn in root_sn.deps:
                out_sn.users.discard()

    def check_artifact(self):
        sn : SimplifiedNode
        for sn in self.nodes:
            if sn.is_artifact:
                if len(sn.deps)!=1: raise Exception(
                    f"{sn.main_target} is_artifact, but with several "\
                    f"deps ({len(sn.deps)}), should have only one.")
                parent_sn = next(iter(sn.deps))
                if sn.users.issubset(parent_sn.users):
                    warnings.warn(
                        f"{sn.main_target} is a useless "\
                        f"artifact of {parent_sn.main_target}, "\
                        f"it should have been unplugged.")

    def check_edges_are_reciprocal(self):
        sn : SimplifiedNode
        for sn in self.nodes:
            for req_sn in sn.deps:
                if sn not in req_sn.users: raise Exception(
                    f"{req_sn.mt} in {sn.mt}.deps but not reciprocal")
            for user_sn in sn.users:
                if sn not in user_sn.deps: raise Exception(
                    f"{user_sn.mt} in {sn.mt}.users but not reciprocal")

    def assert_ready(self):
        # Check if ready to build the backward graphs
        # ie main_targets are tensors, except if artifact -> sizes
        for sn in self.nodes:
            sn_info = sn.info
            if not (sn_info.variable_type in [torch.Tensor,torch.Size]):
                raise Exception(
                  f"After simplifications there should be only "\
                  f"tensors and sizes, but {sn_info.variable_type} "\
                  f"found for {sn.main_target}.")
            if sn_info.variable_type==torch.Size and not sn.is_artifact:
                raise Exception(
                  f"After simplifications, all remaining "\
                  f"\"size\" should be \"artifacts\", but "\
                  f"{sn.main_target} isn't an artifact")
    # ===== END BLOC 1 : CLEAR and CHECK =====


    # ===== BLOC 2 : ADJUST ATTRIBUTES AFTER ALL SIMPLIFICATIONS =====
    def create_nodes_for_random_operations_from_dict_rand(self,model,device):
        dict_random_nodes = dict() # str -> SimplifiedNode
        dict_info = self.dict_info
        # 1) Generate all the random nodes, via self.dict_rand
        for random_variable_name,code_ast in self.dict_rand.items():
            dict_random_nodes[random_variable_name] \
                = random_variable_node \
                = SimplifiedNode(
                    main_target=random_variable_name,
                    code=code_ast,
                    fct="--Random function--",
                    protected=True,
                    is_rand=True,
                    simplified_graph=self)
            # -> We need to generate VariableInfo
            # to do so we generate the value by running the code
            our_global = self.make_copy_of_globals(model,device)
            dict_info[random_variable_name] \
                = random_variable_node.info \
                = VariableInfo(eval(ast_add_on.ast_to_str(code_ast),our_global))
        # 2) Link them in the graph, via node.deps_rand
        # Note: by definition, these nodes don't have deps, only users
        # So they can be put at the beginning of self.nodes (the topo-order)
        sn : SimplifiedNode
        for sn in self.nodes:
            if not sn.is_artifact:
                for req_rd_target in sn.deps_rand:
                    req_rd_node = dict_random_nodes[req_rd_target]
                    req_rd_node.users.add(sn)
                    sn.deps.add(req_rd_node)
        # 3) Set them as user of self.init_node, since, by definition
        # of dict_rand, these nodes don't have dependencies. Then try
        # to insert (to be sure to keep only node<=>Tensor), otherwise
        # put in a the beginning of self.nodes.
        random_variable_node : SimplifiedNode
        init_node = self.init_node
        for random_variable_node in dict_random_nodes.values():
            if random_variable_node.info.variable_type != torch.Tensor:
                init_node.insert(random_variable_node,True,self)
            else:
                self.nodes.insert(0,random_variable_node)

    def remove_artifacts_and_replace_them_by_soft_edges(self):
        nb_nodes = len(self.nodes)
        artifact_sn : SimplifiedNode
        for index,artifact_sn in enumerate(self.nodes[::-1]):
            if artifact_sn.is_artifact:
                del self.nodes[nb_nodes-index-1]
                parent_sn = next(iter(artifact_sn.deps))
                parent_sn.users.discard(artifact_sn)
                for user_sn in artifact_sn.users:
                    user_sn.deps.discard(artifact_sn)
                    user_sn.deps_through_artifacts.add(parent_sn)
                    parent_sn.users_through_artifacts.add(user_sn)

    def make_dict_of_labels_on_edges(self):
        self.dict_of_labels_on_edges = dict_labels = dict()
        for sn in self.nodes:
            sn_code = sn.get_code()
            req_sn : SimplifiedNode
            for req_sn in sn.deps.union(sn.deps_through_artifacts):
                used_targets = set()
                for target in req_sn.all_targets:
                    if target in sn_code: used_targets.add(target)
                dict_labels[(req_sn,sn)] = used_targets
                dict_labels[(sn,req_sn)] = used_targets
                    
    def make_targets_attributes_and_fix_info_data_owner_name(self):
        # -> tensor_targets ; inplace_targets ; container_targets
        dict_info = self.dict_info
        sn : SimplifiedNode
        for sn in self.nodes:
            if not sn.is_artifact:
                sn_main_target = sn.main_target
                tensors = []
                containers = []
                for tar in sn.all_targets:
                    info = dict_info[tar]
                    target_type = info.variable_type
                    if target_type == torch.Tensor and not info.is_param:
                        tensors.append(tar)
                        info.data_owner_name = sn_main_target
                    elif target_type == tuple or target_type == list:
                        containers.append(tar)
                sn.tensor_targets = tensors
                sn.container_targets = containers
                sn.inplace_targets = [c[0] for c in sn.inplace_code]

    def make_inputs(self):
        inputs = set()
        for user_of_init_sn in self.init_node.users:
            used_targets = self.dict_of_labels_on_edges[
                (self.init_node,user_of_init_sn)]
            inputs.update(used_targets)
        self.inputs = list(inputs)

    def unplug_init_node(self):
        dict_info = self.dict_info
        init_node = self.init_node
        dict_labels = self.dict_of_labels_on_edges
        all_init_node_users = set(init_node.users)
        for user_sn in all_init_node_users:
            # 1) Remove init_node from other nodes deps ie unplug it
            user_sn.deps.discard(init_node)
            # 2) If there is no requires_grad link from init_node to user_sn
            # then we don't need to consider it in init_node.users
            # we only want to keep the nodes that produces gradients 
            # on the input (to make the backward graph)
            used_targets = dict_labels[(init_node,user_sn)]
            if not any(
                    dict_info[target].requires_grad
                    for target in used_targets):
                init_node.users.discard(user_sn)
        # 3) BUT, we need at least one node in init_node.users
        # so we have 1 first node where to start find_cutting_points.
        # Thus in case we remove all the users at stage (2),
        # we add back the first one in the topological order
        if len(self.init_node.users)==0:
            all_without_deps = [
                sn for sn in self.nodes 
                if len(sn.deps)==0 ]
            first_sn = min(all_without_deps,key=base.Node.get_num)
            self.init_node.users.add(first_sn)

    def if_multiple_outputs_break_the_wrapper_in_multiple_nodes(self):
        """
        example: a = f(x) ; b = g(y) ; return (a,b)
        before this function, due to raw.py, we have only 
        one output node, equal to the tuple: c = (a,b); return c
        in this method, we unplug Node(c) from the graph, and set:
        - self.output_nodes := [Node(a),Node(b)]
        - self.wrapper_output_node := Node(c), 
        """
        assert(len(self.output_nodes)==1)
        self.wrapper_output_node = wrapper_output_node = self.output_nodes[0]
        if wrapper_output_node.info.variable_type in [tuple,list]:
            self.output_nodes = list(wrapper_output_node.deps)
            for real_output_node in self.output_nodes:
                real_output_node.users.discard(wrapper_output_node) # unplug
            self.nodes.remove(wrapper_output_node)

    def make_dict_output_viewing_code(self):
        """
        Note: use it after "if_multiple_outputs_break_the_wrapper_in_multiple_nodes"
        Example:
        a = f(x) ; v = view(a)
        Instead of returning 'v', we decide to return 'a',
        and the viewing operation will be done outside.
        This is due to how Rockmate creates an equivalent torch.nn.Module.
        But since 'v' can still be used by other variables
        (e.g. c = g(v) and c is a 2nd output). So we don't remove 
        'v = view(a)' from Node(a), we just duplicate it in self.dict_output_viewing_code
        """
        self.outputs = []
        self.dict_output_viewing_code = dict()
        for output_node in self.output_nodes:
            body_code = output_node.make_body_code_ast()
            viewing_code = ast_add_on.make_ast_list_assign(
                body_code,force_special_kwargs=True
            )
            self.dict_output_viewing_code[output_node.mt] = viewing_code
            self.outputs.append(output_node.mt)
    # ===== END BLOC 2 : ADJUST ATTRIBUTES AFTER ALL SIMPLIFICATIONS =====


    # ===== BLOC 3 : SIMPLIFY CONSTRUCTORS AND CHEAP OPERATIONS =====
    # ===== END BLOC 3 : SIMPLIFY CONSTRUCTORS AND CHEAP OPERATIONS =====
            

        
            

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
                req_sn.remove_obsolete_child_artifacts()
                req_sn.remove_obsolete_sibling_artifacts()
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
                        inplace_real_node.remove_obsolete_sibling_artifacts()
            elif len(real_deps)==0 and len(sn.deps)>0:
                # TODO : change sn.info.is_param
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
                    if real_req: real_req.remove_obsolete_child_artifacts()

    sg.clear()

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
    if sg.wrapper_output_node is not None:
        new_sg.wrapper_output_node \
            = special_out \
            = copy_SimplifiedNode(sg.wrapper_output_node)
        special_out.deps = dict(
            (dict_nodes[r.mt],set_str) \
            for r,set_str in sg.wrapper_output_node.deps.items())
        
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
            new_sg.wrapper_output_node = sg.wrapper_output_node
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
    if sg.wrapper_output_node is None:
        assert(len(sg.output_nodes)==1)
        edge(sg.output_nodes[0].mt,"output",sg.outputs)
    else:
        for out in sg.output_nodes:
            edge(out.mt,"output",sg.wrapper_output_node.deps[out])


def print_SimplifiedGraph_list(lsg : SimplifiedGraph_list,dot,name=None,open=True,render_format="svg"):
    for i in range(len(lsg)):
        aux_print_graph(dot,lsg[i],i)

# ==========================

