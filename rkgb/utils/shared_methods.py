# ========================================
# = Useful functions for rk-GB's graphs
# = for instance methods with similar code
# ========================================
from rkgb.utils.imports import *
from rkgb.utils.global_vars import print_debug
from rkgb.utils.ast_add_on import (
    make_str_assign, 
    make_str_list_assign,
    make_ast_list_assign,
    ast_to_str)

# ======================================
# ==== GENERATE STR CODE FOR S AND K==== 
# ======================================

def get_code_ast(n,force_special_kwargs=False): # For S_node or KCN
    mc = n.main_code
    mc = [] if mc is None or mc[1] is None else [mc]
    dict_ic = dict(n.inplace_code)
    bc = [
        (tar,dict_ic[tar] if tar in dict_ic else acode)
        for (tar,acode) in n.body_code]
    code = mc + bc
    return make_ast_list_assign(code,
        force_special_kwargs=force_special_kwargs)

def get_code(n,force_special_kwargs=False): # For S_node or KCN
    return ast_to_str(get_code_ast(n,force_special_kwargs))

def full_code(n,force_special_kwargs=False): # For S_node or KCN
    # This function is a way to produce what the final
    # code will look like (including detach). But it's
    # never used in RK, the translator isn't that simple.
    mt = n.main_target
    mc = make_str_assign(n.main_code,prefix="_",
        force_special_kwargs=force_special_kwargs)
    ic = make_str_list_assign(n.inplace_code,
        force_special_kwargs=force_special_kwargs)
    bc = make_str_list_assign(n.body_code,
        force_special_kwargs=force_special_kwargs)
    if mc == "":
        return bc
    else:
        s = f"{mc}\n{mt} = _{mt}\n"
        s += ic+"\n" if ic != "" else ""
        s += f"{mt} = _{mt}.detach().requires_grad_()\n"
        s += bc
        return s

# ==========================



# ===============================
# = GENERAL FUNCTIONS TO GET    =
# = NODE'S TARGET, NUM AND DEPS =
# ===============================

def get_target(n):
    if hasattr(n,"target"): return n.target # BN and DN
    elif hasattr(n,"main_target"): return n.main_target # SN, KN and HN
    else: raise Exception(
        "A node without .target nor .main_target ?! Sorry bug")

def get_num_tar(tar):
    try:    return int(tar.split('_')[2])
    except: return (-1)
get_num_cst = get_num_tar
def get_num_name(name): # for KCN or KDN's name
    if (name.startswith("fwd_")
    or  name.startswith("bwd_")):
        return get_num_tar(name[4:])
    elif (name.endswith("data")
    or    name.endswith("grad")):
        return get_num_tar(name[:-4])
    elif name.endswith("phantoms"):
        return get_num_tar(name[:-8])
def get_num(n): # can be used on B, D, S, K and H
    if hasattr(n,"number"):
        return n.number # -> for H
    return get_num_tar(get_target(n))

sort_nodes = lambda s : sorted(s,key=get_num)
sort_targets = lambda s : sorted(s,key=get_num_tar)
sort_names = lambda s : sorted(s,key=get_num_name)

def get_deps(n):
    # To be compatible with different type/name of attribute "deps"
    t = str(type(n))
    if   "B_node" in t:   return n.deps
    elif "D_node" in t:   return n.deps
    elif "S_node" in t:   return set(n.deps.keys())
    elif "K_C_node" in t: return set().union(
        *[kdn.deps for kdn in n.deps_real],
        n.deps_through_size_artefacts)
    elif "K_D_node" in t: return set().union(
        *[kcn.deps_real for kcn in n.deps])
    elif "H_C_node" in t: return set().union(
        *[hdn.deps for hdn in n.deps]
    )
    else: raise Exception(f"Unrecognize node type : {t}")

# ==========================



# ==========================
# ==== PERFECT TOPOSORT ====
# ==========================

def sort_based_on_deps(origin_node): # used on B, S, K and H
    # /!\ origin_node is the root of .deps relation 
    # /!\ => e.g. the output node of the graph

    # Compute incomming degree
    degree = {}
    def count_edges(n):
        for sub_n in get_deps(n):
            if sub_n not in degree:
                d = 0
                degree[sub_n] = 0
                count_edges(sub_n)
            else:
                d = degree[sub_n]
            degree[sub_n] = d+1
    count_edges(origin_node)

    # Explore nodes by increasing lexi-order of their n.target
    # BUT a node is explored iff all its users are explored => toposort
    sorted_list = []
    to_explore = set([origin_node])
    while to_explore: # not empty
        n = max(to_explore,key=lambda n : get_num(n))
        to_explore.discard(n)
        sorted_list.append(n)
        for req_n in get_deps(n):
            if req_n in sorted_list: raise Exception(
                f"Cycle in the graph ! (found while trying to "\
                f"toposort), {get_target(req_n)} and "\
                f"{get_target(n)} are members of this cycle.")
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
# happend after the begging of "requires_grad". To do this,
# the rule is: a separator must requires_grad.

def cut_based_on_deps(g): # used on D and S
    # returns the list of all 1-separator of the graph.
    dict_info = g.dict_info
    to_be_visited = [g.output_node]
    seen = set([g.output_node])
    dict_nb_usages = dict([(m , len(m.users)) for m in g.nodes])
    separators = []
    while to_be_visited!=[]:
        n = to_be_visited.pop()
        seen.remove(n)
        if seen==set():
            tar = get_target(n)
            if (tar in dict_info
            and hasattr(dict_info[tar],"requires_grad")
            and dict_info[tar].requires_grad):
                separators.append(n)
        for req_n in get_deps(n):
            seen.add(req_n)
            dict_nb_usages[req_n]-=1
            if dict_nb_usages[req_n]==0:
                to_be_visited.append(req_n)
    separators.reverse()
    return separators

# ==========================

