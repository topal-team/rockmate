from read_trace_code import *

class D_node(B_node):
    def __init__(self,target="",code=""):
        super().__init__(target,code)
        self.used_by_nodes = []

class D_graph():
    def __init__(self):
        self.inputs  = [] # str list
        self.nodes   = [] # D_node list -> topo sorted
        self.outputs = [] # str list
        self.dict_outputs = {} # dict
        self.dict_rand = {}

def sort_nodes(g : B_graph): # -> B_node list 
    # use outputs' nodes, not B_graph.nodes
    # by the way, I take the opportunity to change g.nodes
    assert(len(g.outputs)==1) # tmp
    o_var = g.outputs[0]
    if not o_var.has_node:
        g.nodes = []
        return []
    else:
        dict_done = {}
        nodes = []
        def visit(n):
            if n not in dict_done:
                dict_done[n]=False
                for sub_n in n.req:
                    visit(sub_n)
                dict_done[n]=True
                nodes.append(n)
            elif not dict_done[n]:
                raise Exception("Cycle in the B_graph. How could this happened ??")
        visit(o_var.node)
        g.nodes = nodes
        return nodes

def print_all_nodes(g):
    for n in g.nodes:
        print(f"({n.target}) : [{n.fct}] : {n.code}")

def B_to_D(bg : B_graph) -> D_graph:
    inputs = []
    d_nodes = []
    b_nodes = sort_nodes(bg)
    dict_nodes = {}
    for n in b_nodes:
        dn = D_node(n.target,n.code_without_target)
        if n.is_input:
            inputs.append(n.target)
            dn.is_input = True
        req = list(set(n.req))
        for sub_n in req:
            sub_dn = dict_nodes[sub_n]
            dn.req.append(sub_dn)
            sub_dn.used_by_nodes.append(dn)
        dict_nodes[n] = dn
        d_nodes.append(dn)
    dg = D_graph()
    dg.nodes = d_nodes
    dg.inputs = inputs
    for v in bg.outputs:
        if v.has_node:
            dg.dict_outputs[v.val] = dict_nodes[v.node]
        dg.outputs.append(v.val)
    dg.dict_rand = bg.dict_rand
    return dg

def print_code(g : D_graph):
    print(g.dict_rand)
    str_input = ','.join(g.inputs)
    str_output = ','.join(g.outputs)
    print(f"def main({str_input}):")
    for n in g.nodes:
        if not n.is_input: print(f"\t{n.code}")
    print(f"\treturn {str_output}")

import torch
from torch import tensor

def test_code(g : D_graph,nn_mod,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = nn_mod
    for inp in g.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for v in g.dict_rand:
        exec(g.dict_rand[v], globals(), loc_dict)
    for n in g.nodes:
        if n.is_rand:
            for sub_t in n.req_rand:
                exec(g.dict_rand[sub_t])
        if not n.is_input: exec(n.code, globals(), loc_dict)
    ret = []
    for out in g.outputs:
        ret.append(loc_dict[out])
    if len(ret)==1: return ret[0]
    else: return tuple(ret)

import graphviz

def print_graph(g : D_graph,name=None):
    if name is None:
        name = "forward D-graph"
    dot = graphviz.Digraph(name,comment="D_graph = forward graph")
    for n in g.nodes:
        if n.is_input:
            dot.node(n.target,n.code,color="blue")
        elif n.target in g.outputs:
            dot.node(n.target,n.code,color="red")
        else: dot.node(n.target,n.code)
    for n in g.nodes:
        for sub_n in n.req:
            dot.edge(sub_n.target,n.target)
    dot.render(directory="graphviz_dir",view=True)
