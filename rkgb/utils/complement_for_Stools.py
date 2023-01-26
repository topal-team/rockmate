# ==========================
# ==== OP OVER S EDGES =====
# ==========================
from rkgb.utils.imports import *
from rkgb.utils import shared_methods

# S edges (ie S_nodes.deps/users) are (S_node -> str set) dict

# /!\ By default the following operations are NOT inplace
# inplace op end up with "_inplace". 

def dict_edges_merge_inplace(de_main,de_sub):
    for k in de_sub.keys():
        s = de_main[k] if k in de_main else set()
        de_main[k] = s.union(de_sub[k])
def dict_edges_merge(de1,de2):
    d = dict(de1)
    dict_edges_merge_inplace(d,de2)
    return d

def dict_edges_discard(de,sn):
    return dict((n,s) for (n,s) in de.items() if n != sn)
def dict_edges_discard_inplace(de,sn):
    if sn in de: del de[sn]


def dict_edges_add_inplace(de,sn,str_set):
    s = de[sn] if sn in de else set()
    de[sn] = s.union(str_set)
def dict_edges_add(de,sn,str_set):
    d = dict(de) # not inplace
    dict_edges_add_inplace(d,sn,str_set)
    return d

def dict_edges_eq(de1,de2,raise_exception=False):
    ds1 = dict((n.main_target,s) for (n,s) in de1.items())
    ds2 = dict((n.main_target,s) for (n,s) in de2.items())
    if not raise_exception:
        return ds1 == ds2
    else:
        keys1 = shared_methods.sort_targets(ds1.keys())
        keys2 = shared_methods.sort_targets(ds2.keys())
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

def dict_edges_discard_sn_from_deps_of_its_users(sn):
    for user_sn in sn.users.keys():
        dict_edges_discard_inplace(user_sn.deps,sn)
def dict_edges_discard_sn_from_users_of_its_deps(sn):
    for req_sn in sn.deps.keys():
        dict_edges_discard_inplace(req_sn.users,sn)

def dict_edges_make_users_using_deps(sn):
    for (req_sn,str_set) in sn.deps.items():
        req_sn.users[sn] = set(str_set)
def dict_edges_make_deps_using_users(sn):
    for (user_sn,str_set) in sn.users.items():
        user_sn.deps[sn] = set(str_set)

def dict_edges_discard_edge_inplace(req_sn,user_sn):
    dict_edges_discard_inplace(req_sn.users,user_sn)
    dict_edges_discard_inplace(user_sn.deps,req_sn)

def dict_edges_add_edge_inplace(req_sn,user_sn,str_set):
    dict_edges_add_inplace(req_sn.users,user_sn,str_set)
    dict_edges_add_inplace(user_sn.deps,req_sn,str_set)

def dict_edges_is_subset(de1,de2):
    for (sn1,str_set1) in de1.items():
        if sn1 not in de2: return False
        if str_set1 > de2[sn1]: return False
    return True

# ==========================

