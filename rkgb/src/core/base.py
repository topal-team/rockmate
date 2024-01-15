"""
======================================
= Define base class for rk-GB graphs =
=      with some useful methods      =
======================================
"""

import ast
import copy
import sys
import torch
try:
    import graphviz
    has_graphviz = True
except ModuleNotFoundError:
    has_graphviz = False
from src.lowlevel import ast_add_on
from src.lowlevel import variable_info
# import lowlevel.ast_add_on
# from lowlevel import ast_add_on

# ==============================================
# = Auxiliary class : Node_unique_id_generator =
# -> This class is responsible to give each node a unique __hash__
# -> We cannot use __hash__ = id(self) when building a graph
# -> because it would create nondeterminism when iterating over 
# -> a set of node, such as deps/users, resulting nondeterminism 
# -> in S_node order auxiliary operations in the body code for instance.
# -> But we cannot use any attribute (.main_target, .num etc) because
# -> it may create some collisions, when anonymizing graphs for instance.
class Node_unique_id_generator():
    def __init__(self):
        self.gen = 0
    def __copy__(self):
        c = Node_unique_id_generator()
        c.gen = self.gen
        return c
    def use(self):
        u = self.gen
        self.gen = u+1
        return u
    @staticmethod
    def get_unique_id_one_way_or_the_other(
            object,
            structure_with_id_generator = None, # to get unique_id from it
            unique_id_generator = None):
        if structure_with_id_generator is not None:
            if hasattr(structure_with_id_generator,"node_unique_id_generator"):
                return structure_with_id_generator\
                    .node_unique_id_generator.use()
            elif isinstance(structure_with_id_generator,Node_unique_id_generator):
                return structure_with_id_generator.use()
        elif unique_id_generator is not None:
            return unique_id_generator.use()
        else:
            return id(object)


# ==============================================



# =============================
# =====                   =====
# =====     Base node     =====
# =====                   =====
# =============================

class Node():
    no_target_string = "__No_Target__"
    def __init__(
            self,
            main_target : str = None,
            target = None, mt = None, # aliases
            parent_structure_with_id_generator = None, # to get unique_id from it
            unique_id_generator : Node_unique_id_generator = None):
        # == init main_target ==
        if not (main_target is None):
            self.main_target = main_target
        elif not (target is None):
            self.main_target = target
        elif not (mt is None):
            self.main_target = mt
        else:
            self.main_target = "/!\\ No target /!\\"
        # == init unique_id ==
        self.unique_id = Node_unique_id_generator\
            .get_unique_id_one_way_or_the_other(
                self,
                parent_structure_with_id_generator,
                unique_id_generator
            )

    def get_all_standard_deps(self):
        if hasattr(self,"deps"):
            return self.deps
        else: raise Exception(
            f"{type(self).__name__} should overwrite "\
            f"the method `get_all_standard_deps`.")
    def get_all_standard_users(self):
        if hasattr(self,"users"):
            return self.users
        else: raise Exception(
            f"{type(self).__name__} should overwrite "\
            f"the method `get_all_standard_users`.")

    # =================================
    # === main_target / mt / target ===
    # -> For any type of node
    @property
    def mt(self):
        return self.main_target
    @mt.setter
    def mt(self,mt):
        self.main_target = mt
    @property
    def target(self):
        return self.main_target
    @target.setter
    def target(self,target):
        self.main_target = target
    # =================================


    # ========================================
    # === target/node/name number AND sort ===
    # -> For any type of node
    @staticmethod
    def get_num_tar(tar):
        try:    return int(tar.split('_')[2])
        except: return (-1)
    get_num_cst = get_num_tar
    def get_num(self):
        if hasattr(self,"_topological_number"):
            return self._topological_number 
            # -> for SimplifiedNode / (Hierarchical)ComputationNodes
        else:
            return Node.get_num_tar(self.main_target)

    sort_nodes   = lambda s : sorted(s,key=Node.get_num)
    sort_targets = lambda s : sorted(s,key=Node.get_num_tar)
    # ========================================

    # =============================
    # === generate ast/str code ===
    # -> For B, D, S, WC
    def make_body_code_ast(self):
        dict_ic = dict(self.inplace_code)
        bc = [
            (tar,dict_ic[tar] if tar in dict_ic else acode)
            for (tar,acode) in self.body_code]
        return bc
    def get_code_ast(self,force_special_kwargs=False):
        if hasattr(self,"code_ast"):
            return ast_add_on.make_ast_assign(
                (self.main_target,self.code_ast),
                force_special_kwargs=force_special_kwargs
            )
        else:
            if not (hasattr(self,"main_code") 
                    and hasattr(self,"inplace_code")
                    and hasattr(self,"body_code")):
                raise Exception(
                    "Not standard `code` attribute(s), "\
                    "should overwrite `Node.get_code()`")
            mc = self.main_code
            mc = [] if mc is None or mc[1] is None else [mc]
            bc = self.make_body_code_ast()
            code = mc + bc
            return ast_add_on.make_ast_list_assign(code,
                force_special_kwargs=force_special_kwargs)
    def get_code(self,force_special_kwargs=False):
        code_ast = self.get_code_ast(force_special_kwargs)
        try:
            return ast_add_on.ast_to_str(code_ast)
        except:
            raise Exception(
                "Problem to ast.unparse code:\n"
                + ast.dump(code_ast,indent=4))
    
    # -> For S, WC
    # This function is a way to see what the final
    # code will look like (including detach). But it's
    # never used in Rockmate, the translator/compiler isn't that simple.
    def full_code(self,force_special_kwargs=False):
        main_code = ast_add_on.make_str_assign(self.main_code,prefix="_",
            force_special_kwargs=force_special_kwargs)
        inplace_code = ast_add_on.make_str_list_assign(self.inplace_code,
            force_special_kwargs=force_special_kwargs)
        body_code = ast_add_on.make_str_list_assign(self.body_code,
            force_special_kwargs=force_special_kwargs)
        if main_code == "":
            return body_code
        else:
            mt = self.main_target
            code = f"{main_code}\n{mt} = _{mt}\n"
            code += inplace_code+"\n" if inplace_code != "" else ""
            code += f"{mt} = _{mt}.detach().requires_grad_()\n"
            code += body_code
            return code
    # =============================


    # =====================
    # === requires_grad ===
    # -> For any type of node
    def does_requires_grad(self):
        return (
            self.info is not None
            and hasattr(self.info,"requires_grad")
            and self.info.requires_grad
        )
        # Overwritten in partitioned.py and hierarchical.py
        # where a node might represent a sub_graph.
    # =====================

    # ================
    # === __hash__ ===
    # -> For any type of node
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self) # When init via pickle
    # ================

    def __repr__(self):
        if hasattr(self,"name"):
            s = self.name
        else: 
            s = self.main_target
        return f"{type(self).__name__}({s})"
# ============================


# ============================
class ParameterNode():
    """
    self.param_str : 'self.wpe[0].weight' as it appears in the code
    self.param_name : 'wpe.0.weight' as it appears in model.named_parameters
    """
    def __init__(self,
            param_str = None,
            node_to_clone = None,
            parent_structure_with_id_generator = None, # to get unique_id from it
            unique_id_generator : Node_unique_id_generator = None):
        if node_to_clone is not None:
            for attr in [
                    "param_str","param_name",
                    "view_targets","view_code",
                    "requires_grad","is_buffer",
                    "unique_id"]:
                setattr(self,attr,copy.deepcopy(getattr(node_to_clone,attr)))
        else:
            assert param_str[:4] == "self"
            self.param_str = param_str
            self.param_name = param_str[5:].replace('[','.').replace(']','')
            self.view_targets = []
            self.view_code = []
            self.requires_grad = None
            self.is_buffer = None
            self.unique_id = Node_unique_id_generator\
                .get_unique_id_one_way_or_the_other(
                    self,
                    parent_structure_with_id_generator,
                    unique_id_generator
                )
        self.users = set()
    def get_code(self):
        return ast_add_on.make_str_list_assign(self.view_code)
    def get_value(self,model):
        if self.is_buffer:
            return model.get_buffer(self.param_name)
        else:
            return model.get_parameter(self.param_name)
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self) # When init via pickle
    def __repr__(self):
        return f"ParameterNode({self.param_name})"

# ============================



# ==============================
# =====                    =====
# =====     Base Graph     =====
# =====                    =====
# ==============================

class Graph():
    node_class = Node
    def __init__(
            self,
            other_object_with_id_generator = None,
            node_unique_id_generator : Node_unique_id_generator = None):
        # == base attribute ==
        self.tracer_used = ""
        self.input_targets = []
        self.output_targets = []
        self.original_mod_input_targets = []
        self.original_mod_output_targets = []
        self.sources_req_grad = None
        self.dict_constants = dict()
        self.dict_info : dict[str,variable_info.VariableInfo] = dict()
        self.dict_rand = dict() # empty after S
        self.nodes = []
        self.output_nodes = []
        # == init node_unique_id_generator ==
        if other_object_with_id_generator is not None:
            if hasattr(
            other_object_with_id_generator,"node_unique_id_generator"):
                self.node_unique_id_generator = \
                    other_object_with_id_generator.node_unique_id_generator
            elif isinstance(
            other_object_with_id_generator,Node_unique_id_generator):
                self.node_unique_id_generator = other_object_with_id_generator
        elif node_unique_id_generator is None:
            self.node_unique_id_generator = Node_unique_id_generator()
        else:
            self.node_unique_id_generator = node_unique_id_generator


    # =================================
    # === nodes / list_nodes / iter ===
    # -> For any type of graph
    @property
    def list_nodes(self):
        return self.nodes
    @list_nodes.setter
    def list_nodes(self,list_nodes):
        self.nodes = list_nodes

    def __iter__(self):
        return iter(self.nodes)
        # for graphs with Computation/Allocation nodes,
        # iter over self.computation_nodes
    # =================================

    # ===============================
    # === inherit base attributes ===
    # -> For any type of graph
    def inherit_base_attributes(self,other_graph):
        for attr in [
            "tracer_used",
            "input_targets",
            "output_targets",
            "original_mod_input_targets",
            "original_mod_output_targets",
            "sources_req_grad",
            "dict_constants",
            "dict_info",
            "dict_rand"
        ]:
            setattr(self,attr,copy.copy(getattr(other_graph,attr)))
    # ===============================

    def make_copy_of_globals(self,original_mod,device):
        our_global = globals().copy()
        our_global.update(self.dict_constants)
        our_global["self"] = original_mod
        our_global["device"] = device
        return our_global


    def make_temporary_global_root_node_to_deps_relation(self):
        """return bool * Node
        True <=> it's a fresh node (=> it must be removed after)
        Note : In backward.py and hierarchical.py cases, 
        we are interested by ComputationNodes
        """
        if len(self.output_nodes)==1:
            return False,self.output_nodes[0]
        else:
            fresh_root = self.node_class()
            if not hasattr(fresh_root,"deps") or not hasattr(fresh_root,"users"):
                raise Exception(
                    f"{type(self).__name__} should overwrite the method: "\
                    f"`make_temporary_global_root_node_to_deps_relation`.")
            fresh_root.deps = set(self.output_nodes)
            for out_node in self.output_nodes:
                out_node.users.add(fresh_root)
            return True,fresh_root
    def remove_temporary_global_root_node(self,fresh_root):
        for out_node in self.output_nodes:
            out_node.users.discard(fresh_root)

    def get_sorted_nodes_by_following_deps_relation(self):
        """
        Note : In backward.py and hierarchical.py cases, 
        we are interested by ComputationNodes
        """
        is_it_a_tmp_fresh_root , root_node \
            = self.make_temporary_global_root_node_to_deps_relation()
        # /!\ root_node is the source of .deps relation 
        # /!\ => e.g. the output node of the graph

        # Compute incoming degree (= len(users) (not len(deps)))
        degree = dict()
        degree[root_node] = 0
        to_visit = [root_node]
        while to_visit != []:
            n = to_visit.pop()
            for req_n in n.get_all_standard_deps():
                if req_n not in degree:
                    d = 0
                    degree[req_n] = 0
                    to_visit.append(req_n)
                else:
                    d = degree[req_n]
                degree[req_n] = d+1

        debug = False
        if hasattr(req_n,"deps_real"):
            debug = True

        # Explore nodes by increasing lexicographic-order of their n.main_target
        # BUT a node is explored iff all its users are explored => toposort
        sorted_list = []
        to_explore = set([root_node]) # TO CHANGE: to a max heap structure
        while to_explore: # not empty
            n = max(to_explore,key=lambda n : n.get_num())
            to_explore.discard(n)
            sorted_list.append(n)
            for req_n in n.get_all_standard_deps():
                if req_n in sorted_list: raise Exception(
                    f"Cycle in the graph ! (found while trying to toposort):\n"\
                    f"{req_n.mt} and {n.mt} are members of this cycle.")
                d = degree[req_n]
                if d == 1:
                    to_explore.add(req_n)
                else:
                    degree[req_n] = d-1

        if is_it_a_tmp_fresh_root:
            self.remove_temporary_global_root_node(root_node)
            sorted_list.remove(root_node)

        # return from first to last
        return sorted_list[::-1]


    def find_cutting_points(self):
        """
        self MUST HAVE a global SINK to deps relation
        ie a very first node / like SimplifiedGraph.init_node.

        Note : We don't want a block where nothing requires_grad.
        Because it implies that we don't have a output_bdn_grad 
        and that Fe/Be make no sense.
        Thus a cutting point must requires_grad.
        """
        is_it_a_tmp_fresh_root , root_node \
            = self.make_temporary_global_root_node_to_deps_relation()
        # root_node is the source of .deps relation => e.g. output_node
        to_be_visited = [root_node]
        seen = set(to_be_visited)
        dict_nb_usages = dict(
            [(m,len(m.get_all_standard_users())) 
             for m in self.nodes])

        separators = []
        while to_be_visited!=[]:
            n = max(to_be_visited,key=lambda n : n.get_num())
            to_be_visited.remove(n)
            seen.remove(n)
            if seen==set():
                if n.does_requires_grad():
                    separators.append(n)
            for req_n in n.get_all_standard_deps():
                seen.add(req_n)
                dict_nb_usages[req_n]-=1
                if dict_nb_usages[req_n]==0:
                    to_be_visited.append(req_n)
        separators.reverse()

        if is_it_a_tmp_fresh_root:
            self.remove_temporary_global_root_node(root_node)
        return separators


    # RENDER :
    default_render_directory = "graphviz_dir"
    default_render_format = "svg"
    def _get_render_name(self,name):
        if name is not None: 
            return name
        else:
            return type(self).__name__
    @staticmethod
    def _get_graphviz_dot(name,dot=None):
        if dot is not None:
            return dot
        elif not has_graphviz:
            raise Exception(
                "To render rkGB graphs you need graphviz.\n"\
                "Please install the [draw] variant of rkgb:"\
                "\n pip install rkgb[draw]")
        else:
            return graphviz.Digraph(name,comment=name)
    @staticmethod
    def _call_graphviz_to_render(dot,
            view,
            directory,
            render_format):
        try:
            dot.render(
                view=view,
                directory=directory,
                format=render_format,
                quiet=True)
        except: print(
            f"Warning : issue with graphviz to render.\n"\
            f"Python graphviz package is installed, but maybe "\
            f"the Graphviz software isn't installed. \nThe python "\
            f"package is just an interface which generates the .gv file,"\
            f"but you need to install the software (maybe via apt) "\
            f"to generate the .{render_format}",
            file = sys.stderr)
    def render(self):
        raise NotImplementedError(
            "rkGB graph classes should overwrite 'render' method"
        )
    """ SKELETON FOR RENDER :
    def __str__(self):
        return f"Raw Graph with {len(self.nodes)} nodes."
    def render(self,
            name=None,
            view=True,
            directory=base.Graph.default_render_directory,
            render_format=base.Graph.default_render_format,
            render=True,
            dot=None):
        name = base.Graph._get_render_name(name)
        dot = base.Graph._get_graphviz_dot(name,dot)
        if render:
            base.Graph._call_graphviz_to_render(
                dot,view,directory,render_format
            )
    """

