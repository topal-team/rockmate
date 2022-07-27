# Forward + Backward graph builder

### Quick use :
```python
import pytorch_checkmate as pk

# consider an instance of your module
mymod = Mymod()

# make a dictionary with an example for each required input
# you can use pk.print_inputs(mymod) to list the required inputs
inputs = {"src" : torch.Tensor([[464,5440,4534]])}

# generate the graphs you want
# this function create them all
graphs = pk.make_all_graphs(mymod,inputs)

# then you can access to them using
# graphs.B_graph / C_graph / D_graph / K_graph
# -> if you want the whole module
# graphs.S_graph_list / K_graph_list
# -> if you want to sequentialize it
# if you want to generate a visualization of them you can
# use printing function (e.g. pk.Dtools.print_D_graph)
# or print them all using pk.print_all_graphs
pk.print_all_graphs(graphs)
```

This tool use a module and an example of inputs to it, to create
representations of the forward/backward graph of the module.

Graphs are built in 4 steps :
1. **B\_graph**:
the module and the example of inputs are given to `torch.jit.trace`,
so we optain a string python code, on which we use `ast.parse`.
Base on this we create the first forward graph, a *B\_graph*.
This is just a non ordered list of nodes, which just contain a piece
of code, each node are linked using their *.req* attribute. 
Each piece of code is an `ast.Assign`, ie one variable computed per node, 
using one function, there isn't any nested operations.
2. **D\_graph**:
