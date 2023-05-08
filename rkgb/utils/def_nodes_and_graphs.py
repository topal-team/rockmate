# ======================================
# = Define base class for rk-GB graphs =
# =      with some useful methods      =
# ======================================

from rkgb.utils.imports import *
from rkgb.utils.global_vars import print_debug
from rkgb.utils import ast_add_on

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
# ==============================================



# ===========================
# =====                 =====
# =====     RK_NODE     =====
# =====                 =====
# ===========================

class RK_node():
    def __init__(
            self,
            node_type : str,
            other_obj = None,
            main_target : str = None,
            target = None, mt = None, # aliases
            unique_id_generator : Node_unique_id_generator = None):
        self.node_type = node_type # string: B, D, S, P, KC, KD, HC, HD
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
        if other_obj is not None:
            if hasattr(other_obj,"node_unique_id_generator"):
                self.unique_id = other_obj.node_unique_id_generator.use()
            elif isinstance(other_obj,Node_unique_id_generator):
                self.unique_id = other_obj.use()
        elif unique_id_generator is not None:
            self.unique_id = unique_id_generator.use()
        else:
            self.unique_id = id(self)

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
        if hasattr(self,"number"):
            return self.number # -> for H
        return RK_node.get_num_tar(self.main_target)

    # -> For K
    def get_num_name(self):
        name = self.name
        if (name.startswith("fwd_")
        or  name.startswith("bwd_")):
            return RK_node.get_num_tar(name[4:])
        elif (name.endswith("data")
        or    name.endswith("grad")):
            return RK_node.get_num_tar(name[:-4])
        elif name.endswith("phantoms"):
            return RK_node.get_num_tar(name[:-8])
        
    sort_nodes   = lambda s : sorted(s,key=RK_node.get_num)
    sort_targets = lambda s : sorted(s,key=RK_node.get_num_tar)
    sort_names   = lambda s : sorted(s,key=RK_node.get_num_name)
    # ========================================


    # =========================================
    # === General way to get deps and users ===
    # -> For any type of node
    def get_deps(self):
        # match self.node_type ; case # Not available py_version < 3.10
        nt = self.node_type
        if   nt == "B":  return self.deps
        elif nt == "D":  return self.deps
        elif nt == "S":  return set(self.deps.keys())
        elif nt == "P":  return self.deps
        elif nt == "KC": return set().union(
                                    *[kdn.deps for kdn in self.deps_real],
                                    self.deps_through_size_artefacts)
        elif nt == "KD": return set().union(
                                    *[kcn.deps_real for kcn in self.deps])
        elif nt == "HC": return set().union(
                                    *[hdn.deps for hdn in self.deps])
        elif nt == "HD": return set().union(
                                    *[hcn.deps for hcn in self.deps])
        else: raise Exception(f"Unrecognized node type : {nt}")

    def get_users(self):
        nt = self.node_type
        if   nt == "B":  return self.users
        elif nt == "D":  return self.users
        elif nt == "S":  return set(self.users.keys())
        elif nt == "P":  return self.users
        elif nt == "KC": return set().union(
                                    *[kdn.users_real for kdn in self.users],
                                    self.users_through_size_artefacts)
        elif nt == "KD": return set().union(
                                    *[kcn.users for kcn in self.users_real])
        elif nt == "HC": return set().union(
                                    *[hdn.users for hdn in self.users])
        elif nt == "HD": return set().union(
                                    *[hcn.users for hcn in self.users])
        else: raise Exception(f"Unrecognized node type : {nt}")
    # =====================================


    # =============================
    # === generate ast/str code ===
    # -> For B, D, S, KC
    def get_code_ast(self,force_special_kwargs=False):
        nt = self.node_type
        if nt == "B" or nt == "D":
            return ast_add_on.make_ast_assign(
                (self.main_target,self.ast_code),
                force_special_kwargs=force_special_kwargs
            )
        else:
            mc = self.main_code
            mc = [] if mc is None or mc[1] is None else [mc]
            dict_ic = dict(self.inplace_code)
            bc = [
                (tar,dict_ic[tar] if tar in dict_ic else acode)
                for (tar,acode) in self.body_code]
            code = mc + bc
            return ast_add_on.make_ast_list_assign(code,
                force_special_kwargs=force_special_kwargs)
    def get_code(self,force_special_kwargs=False):
        return ast_add_on.ast_to_str(self.get_code_ast(force_special_kwargs))
    
    # -> For S, KC
    # This function is a way to see what the final
    # code will look like (including detach). But it's
    # never used in RK, the translator isn't that simple.
    def full_code(self,force_special_kwargs=False):
        mt = self.main_target
        mc = ast_add_on.make_str_assign(self.main_code,prefix="_",
            force_special_kwargs=force_special_kwargs)
        ic = ast_add_on.make_str_list_assign(self.inplace_code,
            force_special_kwargs=force_special_kwargs)
        bc = ast_add_on.make_str_list_assign(self.body_code,
            force_special_kwargs=force_special_kwargs)
        if mc == "":
            return bc
        else:
            s = f"{mc}\n{mt} = _{mt}\n"
            s += ic+"\n" if ic != "" else ""
            s += f"{mt} = _{mt}.detach().requires_grad_()\n"
            s += bc
            return s
    # =============================


    # =====================
    # === requires_grad ===
    # -> For any type of node
    def does_requires_grad(self,dict_info):
        tar = self.main_target
        nt = self.node_type
        if tar is None:
            if nt == "P" or nt == 'HC':
                if self.is_leaf:
                    raise Exception(
                        "'main_target == None' should imply 'not self.is_leaf'"
                    )
                return True # a subgraph always requires_grad
            else:
                raise Exception(
                    f"Apart from P_nodes and H_C_nodes,\n"\
                    f"self.main_target shouldn't be None, "\
                    f"error with a {nt}_node"
                )
        else:
            return (tar in dict_info # -> otherwise : special nodes
                and hasattr(dict_info[tar],"requires_grad")
                and dict_info[tar].requires_grad)
    # =====================

    # ================
    # === __hash__ ===
    # -> For any type of node
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self) # When init via pickle
    # ================
# ============================



# ============================
# =====                  =====
# =====     RK_GRAPH     =====
# =====                  =====
# ============================

class RK_graph():
    def __init__(
            self,
            graph_type : str,
            other_obj = None,
            node_unique_id_generator : Node_unique_id_generator = None):
        self.graph_type = graph_type # string: B, D, S, P, K, H
        # - base attribute -
        self.inputs = [] ; self.outputs = [] # str list -> interfaces
        self.sources_req_grad = None
        self.dict_constants = dict()
        self.dict_info = dict()
        self.dict_rand = dict() # empty after S
        self.nodes = []
        self.output_nodes = []

        if other_obj is not None:
            if hasattr(other_obj,"node_unique_id_generator"):
                self.node_unique_id_generator = other_obj.node_unique_id_generator
            elif isinstance(other_obj,Node_unique_id_generator):
                self.node_unique_id_generator = other_obj
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
        if self.graph_type == "K": return iter(self.list_kcn)
        else: return iter(self.nodes)
    # =================================

    # ===============================
    # === inherit base attributes ===
    # -> For any type of graph
    def inherit_base_attributes(self,other_graph):
        for attr in [
            "inputs",
            "outputs",
            "sources_req_grad",
            "dict_constants",
            "dict_info",
            "dict_rand"
        ]:
            setattr(self,attr,copy(getattr(other_graph,attr)))
    # ===============================

    # =====================
    # === small methods ===
    def does_node_requires_grad(self,n : RK_node):
        return n.does_requires_grad(self.dict_info)

    def __hash__(self):
        return id(self)
# ============================



# ==========================
# ==== PERFECT TOPOSORT ====
# ==========================

def RK_sort_based_on_deps(origin_node : RK_node):
    # -> For B, S, K, H
    # /!\ origin_node is the root of .deps relation 
    # /!\ => e.g. the output node of the graph

    # Compute incoming degree
    degree = {}
    def count_edges(n : RK_node):
        for sub_n in n.get_deps():
            if sub_n not in degree:
                d = 0
                degree[sub_n] = 0
                count_edges(sub_n)
            else:
                d = degree[sub_n]
            degree[sub_n] = d+1
    count_edges(origin_node)

    # Explore nodes by increasing lexicographic-order of their n.main_target
    # BUT a node is explored iff all its users are explored => toposort
    sorted_list = []
    to_explore = set([origin_node])
    while to_explore: # not empty
        n = max(to_explore,key=lambda n : n.get_num())
        to_explore.discard(n)
        sorted_list.append(n)
        for req_n in n.get_deps():
            if req_n in sorted_list: raise Exception(
                f"Cycle in the graph ! (found while trying to toposort):\n"\
                f"{req_n.mt} and {n.mt} are members of this cycle.")
            d = degree[req_n]
            if d == 1:
                to_explore.add(req_n)
            else:
                degree[req_n] = d-1

    # return from first to last
    return sorted_list[::-1]

# ==========================



# ==========================
# ======= CUT GRAPHS =======
# ==========================

# Note : We don't want a block where nothing requires_grad.
# Because it implies that we don't have a output_kdn_grad 
# and that Fe/Be make no sense. So the first cut should 
# happen after the begging of "requires_grad". To do this,
# the rule is: a separator must requires_grad.

def RK_get_1_separators(g: RK_graph):
    # DONT FORGET TO ADD A SOURCE IF NECESSARY

    # used on D (to protect), S (to cut), P (to partition)
    # returns the list of all 1-separators of the 'deps' relation
    to_be_visited = list(g.output_nodes)
    seen = set(to_be_visited)
    dict_nb_usages = dict([(m,len(m.get_users())) for m in g.nodes])

    separators = []
    while to_be_visited!=[]:
        n = to_be_visited.pop()
        seen.remove(n)
        if seen==set():
            if g.does_node_requires_grad(n):
                separators.append(n)
        for req_n in n.get_deps():
            seen.add(req_n)
            dict_nb_usages[req_n]-=1
            if dict_nb_usages[req_n]==0:
                to_be_visited.append(req_n)
    separators.reverse()
    return separators

# ==========================

