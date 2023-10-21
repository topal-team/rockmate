# ==========================
# ====== S structure =======
# ==========================

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
        .is_artifact: bool : see Doc (useful size node)
        .deps       : set[SimplifiedNode]
        .users      : set[SimplifiedNode]
        .is_rand    : bool
        .deps_rand  : str set : because we don't want random src nodes here
        """
        super().__init__(main_target,
            parent_structure_with_id_generator=simplified_graph)
        self.is_artifact = False
        self.parent_sn_as_an_artifact : SimplifiedNode = None
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
            sn_to_insert.parent_sn_as_an_artifact = self

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
                self.target,self.main_code[1],dict_info)
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
            for user_sn in self.users:
                user_sn.deps.discard(artifact_sn)
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
    sequentialized_list_of_bloc_of_nodes : list[list[SimplifiedNode]] = None
    is_sequentialization_aggressive : bool = None
    def __init__(self,
            forward_graph : ForwardGraph = None,
            model=None,
            device=None
    ):
        # 2 constructors: if given a forward_graph, then move from F to S
        # otherwise return an empty simplified_graph
        super().__init__()
        if forward_graph is not None:
            if model is None or device is None: 
                raise Exception(
                    "You need to pass model and device to "\
                    "SimplifiedGraph.__init__ to move from F to S")
            self.inherit_base_attributes(forward_graph)
            self.whole_model_inputs = set(forward_graph.input_targets)

            self.init_node = init_node = SimplifiedNode(
                main_target=constants.init_target_string,
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
            for input_target in forward_graph.input_targets:
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
                for out in forward_graph.output_targets]
            self.clear()
            self.simplify_constructors()
            self.optional_simplify_cheap_operations()
            self.simplify_sizes()
            self.simplify_view()
            self.create_nodes_for_random_operations_from_dict_rand(model,device)
            self.remove_artifacts_and_replace_them_by_soft_edges()
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
        self.check_artifact() # TO REMOVE after all tests passed
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
        self.nodes = self.get_sorted_nodes_by_following_deps_relation()
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
        self.input_targets = list(inputs)

    def unplug_init_node(self):
        dict_info = self.dict_info
        init_node = self.init_node
        dict_labels = self.dict_of_labels_on_edges
        init_node.users_through_artifacts = set(init_node.users)
        # as this might change, we save original init_node's users
        for user_sn in init_node.users_through_artifacts:
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
        wrapper_output_node = self.output_nodes[0]
        if wrapper_output_node.info.variable_type in [tuple,list]:
            self.wrapper_output_node = wrapper_output_node
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
        self.output_targets = []
        self.dict_output_viewing_code = dict()
        for output_node in self.output_nodes:
            body_code = output_node.make_body_code_ast()
            viewing_code = ast_add_on.make_ast_list_assign(
                body_code,force_special_kwargs=True
            )
            self.dict_output_viewing_code[output_node.mt] = viewing_code
            self.output_targets.append(output_node.mt)
    # ===== END BLOC 2 : ADJUST ATTRIBUTES AFTER ALL SIMPLIFICATIONS =====


    # ===== BLOC 3 : SIMPLIFICATIONS =====
    def simplify_constructors(self):
        """
        Note: from root to leaves (init_node to output_nodes)
        """
        sn : SimplifiedNode
        for sn in self.nodes:
            if (sn not in self.output_nodes
            and sn.main_fct == constants.constructor_function_string):
                sn.substitute_self_by_its_code_in_its_users(self.dict_info)
        self.clear()

    def optional_simplify_cheap_operations(self):
        """
        Note: from root to leaves (init_node to output_nodes)
        """
        sn : SimplifiedNode
        for sn in self.nodes:
            if (sn not in self.output_nodes
            and sn.main_fct in constants.list_cheap_functions
            and len(sn.deps) <= 2
            and len(sn.users) == 1):
                # with this new conditions, no need to protect against over simplification
                sn.substitute_self_by_its_code_in_its_users(self.dict_info)
        self.clear()

    def simplify_sizes(self):
        """
        1) merge the size nodes which have the same parent
        2) insert the size nodes in the body code of the
           parent, and keep them only if needed -> artifact
        Note: from leaves to root (output_nodes to init_node)
        """
        nodes = [self.init_node] + self.nodes
        nodes.reverse()
        sn : SimplifiedNode
        for sn in nodes:
            if sn not in self.output_nodes:
                users_that_are_sizes = [
                    user_sn for user_sn in sn.users
                    if user_sn.info.variable_type == torch.Size
                ]
                if users_that_are_sizes != []:
                    # 1) merge all "size"-type users together
                    size_sn = users_that_are_sizes[0]
                    size_sn : SimplifiedNode
                    for other_size_sn in users_that_are_sizes[1:]:
                        size_sn.insert(
                            sn_to_insert=other_size_sn,
                            strong=True,
                            simplified_graph=self)
                    # 2) insert them in the parent node
                    if (sn is self.init_node
                    or  sn.info.variable_type == torch.Size):
                        sn.insert(size_sn,strong=True,simplified_graph=self)
                    else: # might stay as artifact
                        sn.insert(size_sn,strong=False,simplified_graph=self)
        # artifact are introduced in this method, 
        # so we finish by setting init_node has an artifact too;
        # only useful in: simplify_view, the next stage
        self.init_node.is_artifact = True
        self.clear()

    def simplify_view(self):
        """
        Note: from root to leaves (init_node to output_nodes)
        """
        sn : SimplifiedNode
        for sn in self.nodes:
            if (sn not in self.output_nodes
            and sn.main_fct != constants.constructor_function_string
            and sn.deps != set()
            and (sn.info.is_view
                or sn.info.is_inplace
                or sn.info.is_param
                or sn.main_fct == "getattr") # those not simplified via constructors
                # or sn.main_fct in constants.list_view_fct)
                # Normally, all fct in list_view_fct were already 
                # detected and info.is_view/param/inplace
            ):
                # 1) We look for one clear parent of sn
                # ie a node sn depend on, in which we could insert sn code, 
                # will being sure we don't create cycles in the graph
                sn_non_artefact_deps = [
                    req_sn for req_sn in sn.deps if not req_sn.is_artifact]
                sn_non_artefact_deps_set = set(sn_non_artefact_deps)
                parent_sn : SimplifiedNode = None
                while parent_sn is None and sn_non_artefact_deps != []:
                    req_sn = sn_non_artefact_deps.pop()
                    if req_sn.main_target == sn.info.data_direct_parent_name:
                        parent_sn = req_sn
                    else:
                        req_sn_deps = req_sn.deps.copy()
                        req_sn_deps.add(req_sn)
                        if req_sn_deps >= sn.deps:
                            parent_sn = req_sn

                if parent_sn is None and sn_non_artefact_deps_set != set():
                    # we look for the deps whose index as an element of self.nodes 
                    # is the biggest, but without using self.nodes.index() to keep it linear
                    index_node = self.nodes.index(sn) -1
                    while self.nodes[index_node] not in sn_non_artefact_deps_set:
                        index_node -= 1
                    parent_sn = self.nodes[index_node]

                if parent_sn is not None:
                    parent_sn.insert(sn,strong=True,simplified_graph=self)
                    parent_sn.remove_obsolete_child_artifacts()
                    parent_sn.remove_obsolete_sibling_artifacts()
                else:
                    # ie all deps are artifacts
                    if sn.info.is_inplace:
                        raise Exception(
                            f"sorry we don't support inplace operations over "\
                            f"parameters, or anything that look as such. \n"\
                            f"Here {sn.main_target} only depends on artifacts, "\
                            f"but sn.info.is_inplace=True.\nCode:\n {sn.get_code()}")
                    if not sn.info.is_param:
                        raise Exception(
                            "view without any dependencies except artifacts "\
                            "and it's not a parameter: I didn't thought this can exist")
                    # => View over parameters, so mem=0 and no backward node
                    # so I can insert sn's code in ALL its deps (which are all artifacts)
                    # but artifact nodes are temporary, so what is important is to 
                    # insert the code in the parent of the artifacts.
                    artifact_req_sn : SimplifiedNode
                    for artifact_req_sn in sn.deps:
                        # 1) duplicate the code in artifact's parent
                        if sn is not self.init_node:
                            artifact_req_sn \
                                .parent_sn_as_an_artifact \
                                .insert_code(sn,self)
                        # 2) put it inside the artifact too, 
                        # TO REMOVE / useless as we discard all the artifact by the end ?
                        artifact_req_sn.insert_code(sn,self)
                        # 3) plug artifact_req_sn to sn's users
                        artifact_req_sn.users.update(sn.users)
                        for user_sn in sn.users:
                            user_sn.deps.add(artifact_req_sn)
                        artifact_req_sn.users.discard(sn)
                        # 4) clear artifact if possible # TO IMPROVE
                        if sn is not self.init_node:
                            artifact_req_sn \
                                .parent_sn_as_an_artifact \
                                .remove_obsolete_child_artifacts()
                    # 5) unplug sn
                    for user_sn in sn.users:
                        user_sn.deps.discard(sn)
                    sn.deps = set()
                    sn.users = set()
        self.clear()
    # ===== END BLOC 3 : SIMPLIFICATIONS =====

    def make_sequentialized_list_of_bloc_of_nodes(self,aggressive=False):
        """
        'aggressive' is True <=> models' inputs are considered 
        as global variables usable in any bloc.
        Which isn't conventional with classic torch.nn.Sequential, 
        where inputs are simply fed in the first layer.
        So if `aggressive` we consider consider inputs as global vars,
        and so we don't have the edges, and so the graph is simpler
        and a good sequential structure can emerge. And it's always
        like that in rkgb->Rockmate, as init_node have as few `.users`
        as possible after `self.unplug_init_node`.
        But this method, `make_sequentialized...`, is purely for external
        usage, so the user might prefer to respect the conventional 
        torch.nn.Sequential.
        """
        if (self.is_sequentialization_aggressive is None # first time
        or self.is_sequentialization_aggressive != aggressive): # change aggressiveness
            init_node = self.init_node
            init_node_users_beside_this_context = init_node.users
            if aggressive:
                init_node_users_in_this_context = init_node.users_through_artifacts
            else:
                init_node_users_in_this_context = init_node.users
            init_node.users = init_node_users_in_this_context
            # 1) re-plug init_node 
            # 2) find cutting points 
            # 3) unplug init_node
            for user_sn in init_node_users_in_this_context:
                user_sn.deps.add(init_node)
            self.nodes.insert(0,init_node)
            cutting_points = self.find_cutting_points()
            self.nodes.remove(init_node)
            for user_sn in init_node.users:
                user_sn.deps.remove(init_node)
            # 4) cut self.nodes in blocs following cutting_points 
            list_blocs = []
            current_bloc = []
            for sn in self.nodes:
                current_bloc.append(sn)
                if sn is cutting_points[0]:
                    cutting_points.pop(0)
                    list_blocs.append(current_bloc)
                    current_bloc = []
            if current_bloc != []: list_blocs.append(current_bloc)
            init_node.users = init_node_users_beside_this_context
            self.sequentialized_list_of_bloc_of_nodes = list_blocs
            self.is_sequentialization_aggressive = aggressive


    def __str__(self):
        return f"Simplified Forward Graph with {len(self.nodes)} nodes."
    def render(self,
            name=None,
            view=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        # Reminder: at the end of __init__, artifacts are removed
        # and replace by soft edges "deps_through_artifacts"
        def edge(sn1,sn2,style=None):
            used_targets = self.dict_of_labels_on_edges[(sn1,sn2)]
            dot.edge(
                sn1.main_target,sn2.main_target,
                label="\n".join(used_targets),
                style=style)
        # 1) nodes and edges
        sn : SimplifiedNode
        for sn in self.nodes:
            dot.node(sn.main_target,sn.get_code())
            for req_sn in sn.deps:
                edge(req_sn,sn)
            for req_sn in sn.deps_through_artifacts:
                edge(req_sn,sn,style="dashed")
        # 2) init node
        dot.node(self.init_node.main_target,
            "INPUT\n"+self.init_node.get_code(),
            color=constants.render_color_special,
            style="dashed")
        for user_sn in self.init_node.users:
            edge(self.init_node,user_sn,style="dashed")
        # 3) output nodes
        wrapper_node = self.wrapper_output_node
        if wrapper_node is None:
            assert(len(self.output_nodes)==1)
            output_node = self.output_nodes[0]
            dot.node("output","OUTPUT",
                color=constants.render_color_special,
                style="dashed")
            dot.edge(output_node.main_target,"output",
                self.whole_module_output,style="dashed")
        else:
            dot.node(
                wrapper_node.main_target,
                "OUTPUT\n"+wrapper_node.get_code(),
                color=constants.render_color_special,
                style="dashed")
            for output_node in self.output_nodes:
                edge(wrapper_node,output_node,style="dashed")
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )

    def render_sequentialized(self,
            name=None,
            view=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        name = self._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        self.make_sequentialized_list_of_bloc_of_nodes()
        # Auxiliary functions
        def make_unique(target,bloc_id):
            return str(bloc_id)+"_"+target
        def node(bloc_id,target,label,**kwargs):
            dot.node(
                make_unique(target,bloc_id),
                label,**kwargs)
        def edge(bloc_id,sn1,sn2,style=None):
            used_targets = self.dict_of_labels_on_edges[(sn1,sn2)]
            dot.edge(
                make_unique(sn1.main_target,bloc_id),
                make_unique(sn2.main_target,bloc_id),
                label="\n".join(used_targets),
                style=style)
        # Build each bloc one by one
        blocs = self.sequentialized_list_of_bloc_of_nodes
        len_blocs = len(blocs)
        wrapper_node = self.wrapper_output_node
        for bloc_id,bloc_nodes in enumerate(blocs):
            # 1) input
            if bloc_id == 0:
                node(bloc_id,
                    self.init_node.main_target,
                    "GLOBAL INPUT\n"+self.init_node.get_code(),
                    color=constants.render_color_special,
                    style="dashed")
            else:
                node(bloc_id,
                    blocs[bloc_id-1][-1].main_target,
                    f"INPUT bloc {bloc_id+1}",
                    color=constants.render_color_special,
                    style="dashed")
            # 2) nodes and edges
            for sn in bloc_nodes:
                node(bloc_id,sn.main_target,sn.get_code())
                for req_sn in sn.deps:
                    edge(bloc_id,req_sn,sn)
                for req_sn in sn.deps_through_artifacts:
                    edge(req_sn,sn,style="dashed")
            # 3) output
            if (bloc_id == len_blocs-1
            and wrapper_node is not None):
                node(bloc_id,
                    wrapper_node.main_target,
                    "OUTPUT\n"+wrapper_node.get_code(),
                    color=constants.render_color_special,
                    style="dashed")
                for output_node in self.output_nodes:
                    edge(bloc_id,wrapper_node,output_node,style="dashed")
            else:
                node(bloc_id,
                    "output","OUTPUT",
                    color=constants.render_color_special,
                    style="dashed")
                dot.edge(
                    make_unique(bloc_nodes[-1].main_target,bloc_id),
                    make_unique("output",bloc_id),
                    style="dashed")
            # 4) special case init_node:
            if bloc_id == 0:
                for user_sn in self.init_node.users:
                    edge(bloc_id,self.init_node,user_sn,style="dashed")
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )
        
    """ WIP TODO
    def build_equivalent_torch_nn_sequential(
            self,model : torch.nn.Module,device):
        simplified_graph = self
        self.make_sequentialized_list_of_bloc_of_nodes()
        blocs_nodes = self.sequentialized_list_of_bloc_of_nodes
        # Module for one bloc:
        def make_BlocModule(input_targets,nodes,output_targets):
            # ugly way to do it, but I want to respect the input/outputs
            # and, above all, I didn't find a way to pass the kwargs
            # otherwise, I want the result to be clean.


            # The problem is that, if I store kwargs in a global dict after
            # I receive them, it's not very natural,
            # en gros est-ce que le "tgt" dans nn.Transformer 
            # devrait etre dans un global dict qu'on partage entre
            # les blocs, ou alors contraindre à qu'il ne soit soit use 
            # que dans le premier bloc; 
            # => en fait c'est meme bcp bcp plus naturel avec la notion
            # de nn.Sequential
            # mais en fait dans Rockmate on ne prend pas un nn.Sequential !
            # on considère que les inputs du 1er bloc sont des globals vars
            # intouchable et donc utilisable partout
            # mais c'est complètement faux !!!
            # D'ailleurs comme ça se passe si on a de l'inplace sur les 
            # inputs ??? Jamais prévu ça !
            # Mais du coup, est-ce qu'on se restreint au nn.Sequential normale
            # ie celui où les inputs ne sont pas des global_vars
            # mais du coup nn.Transformer est vraiment qu'un gros bloc.

            # l'autre problème c'est les _modules;
            # est-ce que je mets tous les _modules à tout le monde ?


        class BlocModule(torch.nn.Module):
            def __init__(self,input_targets,nodes,output_targets):
                super().__init__()
                self.input_targets = input_targets
                self.code_strings = [rn.get_code() for rn in nodes]
                self.output_targets = output_targets
            def forward(self,*args):
                dict_local = locals()
                dict_local.update(simplified_graph.dict_constants)
                for input_target,input_value in zip(self.input_targets,args):
                    dict_local[input_target] = input_value
                for line_of_code in self.code_strings:
                    exec(line_of_code)
                if len(self.output_targets)==1:
                    return dict_local[self.output_targets[0]]
                else:
                    return tuple(
                        dict_local[output_target] 
                        for output_target in self.output_targets)
        # Build the sequence
        if blocs_nodes
    """

