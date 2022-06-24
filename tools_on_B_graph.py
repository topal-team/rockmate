from read_trace_code import *

class D_node(B_node):
    def __init__(self,target="",code=""):
        super().__init__(target,code) # is_input is now useless
        self.used_by_nodes = []

class D_graph():
    def __init__(self):
        self.inputs  = [] # str list
        self.nodes   = [] # D_node list -> topo sorted
        self.outputs = [] # str list

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
                for sub_n in n.required_nodes:
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
        print(f"({n.target}) : {n.code}")

def B_to_D(bg : B_graph) -> D_graph:
    inputs = []
    d_nodes = []
    b_nodes = sort_nodes(bg)
    dict_nodes = {}
    for n in b_nodes:
        if n.is_input:
            inputs.append(n.target)
        else:
            dn = D_node(n.target,n.code)
            for sub_n in n.required_nodes:
                if sub_n.target not in inputs:
                    sub_dn = dict_nodes[sub_n]
                    dn.required_nodes.append(sub_dn)
                    sub_dn.used_by_nodes.append(dn)
            dict_nodes[n] = dn
            d_nodes.append(dn)
    dg = D_graph()
    dg.nodes = d_nodes
    dg.inputs = inputs
    dg.outputs = [v.val for v in bg.outputs]
    return dg

def print_code(g : D_graph):
    str_input = ','.join(g.inputs)
    str_output = ','.join(g.outputs)
    print(f"def main({str_input}):")
    for n in g.nodes:
        print(f"\t{n.code}")
    print(f"\treturn {str_output}")

import torch
from torch import tensor

def test_code(g : D_graph,nn_mod,dict_inputs : dict):
    self=nn_mod
    for inp in g.inputs:
        assert(inp in dict_inputs)
        exec(f"{inp} = {dict_inputs[inp]}")
    for n in g.nodes:
        exec(n.code)
    ret = []
    for out in g.outputs:
        exec(f"global btools_extract_result ; btools_extract_result = {out}")
        ret.append(globals()["btools_extract_result"])
    return ret
    exec(f"global result ; result = {g.outputs[0]}")
    return globals()["result"]
    # return [globals()[out] for out in g.outputs]

import graphviz

def print_graph(g : D_graph):
    dot = graphviz.Digraph('calc-graph',comment="The forward D_graph")
    for n in g.nodes:
        dot.node(n.target,n.code)
    for n in g.nodes:
        for sub_n in n.required_nodes:
            dot.edge(sub_n.target,n.target)
    dot.render(directory="dottest",view=True)
