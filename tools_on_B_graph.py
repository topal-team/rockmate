from read_trace_code import *

class D_node(B_node):
    def __init__(self):
        super().__init__()
        self.used_by_nodes = []

class D_graph():
    def __init__(self):
        self.inputs  = [] # str list
        self.nodes   = [] # D_node list -> topo sorted
        self.outputs = [] # str list

def sort_nodes(g : B_graph): # in place -> change g.nodes
    assert(len(g.outputs)==1)
    o_var = g.outputs[0]
    if not o_var.has_node:
        g.nodes = []
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
                raise Exception("Cycle in the B_graph")
        visit(o_var.node)
        g.nodes = nodes

def print_all_nodes(g):
    for n in g.nodes:
        print(f"({n.target}) : {n.code}")

def print_code(g : B_graph):
    sort_nodes(g) # done in place
    inputs = []
    body = []
    for n in g.nodes:
        if n.is_input:
            inputs.append(n.target)
        else:
            body.append(n)
    str_input = ','.join(inputs)
    print(f"def main({str_input}):")
    for n in body:
        print(f"\t{n.code}")
    print(f"\treturn {g.outputs[0].val}")

def test_code(g : B_graph):
    sort_nodes(g) # done in place
    pass
