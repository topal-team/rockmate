import D_gr_to_K_gr
import graphviz

def print_K_nodes(dict_Kfwd,dict_Kbwd,name=None):
    if name is None:
        name = "K-nodes graph"
    dot = graphviz.Digraph(name,comment="K_nodes")
    def print_node(n,default_color):
        if n.code=="INPUT":
            dot.node(n.name,f"{n.name} = INPUT",color="green")
        else:
            dot.node(n.name,n.code,color=default_color)
    fwd_nodes = dict_Kfwd.values()
    bwd_nodes = dict_Kbwd.values()
    all_nodes = fwd_nodes + bwd_nodes
    for n in fwd_nodes:
        print_node(n,"blue")
    for n in bwd_nodes:
        print_node(n,"red")
    for n in all_nodes:
        for sub_n in n.req:
            dot.edge(sub_n.name,n.name)
    dot.render(directory="graphviz_dir",view=True)

