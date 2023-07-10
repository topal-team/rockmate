
# rk-GraphBuilder

```python
# Example of how to use rkgb
import torch
import rkgb
from models.GPT import GPT2

device = torch.device("cuda")
model = GPT2(nlayers=12,dropout=0.1)
model.to(device)
sample = torch.randint(5400,(100,20),device=device)

rkgb_result = rkgb.make_all_graphs(model,sample)

rkgb.print_all_graphs(rkgb_result,name="GPT2_12",render_format="pdf")
# To render the graphs in pdf you need Graphviz

# You can also try:
rkgb_result = rkgb.test_rkgb(model,sample)
```

## Tests provided:
You can run the Python Notebook : `test_rkgb.ipynb`, 
which include some tests over GPT2, Resnet101, Regnetx32, MLP_Mixer and nn.transformer.
rk-GB works on these modules, but Rockmate fails on nn.transformer Rockmate.

## rk-GB graphs:
- `B_graph` stands for Basic Graph, object built by processing `torch.jit.trace_module` output. It just a raw graph, consisting simply in a list of operations. Therefore, it cannot be rendered. Everything concerning this structure, and the way it's computed is in `Btools.py`.
- `D_graph` is the first useful DAG graph, data-flow of the forward computation. Each node consists of one assignment, defining one variable using one primitive operation. To generate it you need a `B_graph` via `B_to_D`. See `Dtools.py`.In particular, each operation is run to collect basic information (dtype, shape, views etc). 
- `S_graph` is the simplified forward graph, where each node consist of one real operation, and a body code (shapes, viewing or in-place operations). You need a `D_graph` to generate it, see `Stools.py`. Note that you can manually apply each simplification step one by one, and print intermediate results using `rkgb.stools.print_S_graph`, check the code of `D_to_S`.
- The `S_graph` can be cut using `Stools.cut` to obtain the sequence of blocks, as needed by `rk-Rotor`.
- `Atools.py` handle anonymization stuff, to recognize equivalent blocks.
- Finally, you can generate `K_graphs`, which are graphs containing bacKward nodes, and everything you need for rk-Checkmate, see `Ktools.py`.

Thus the main function of `rkgb` (`rkgb.make_all_graphs`) runs :
```python
bg = Btools.make_B(model,samples,device)
dg = Dtools.B_to_D(bg,model,samples,device)
sg = Stools.D_to_S(dg,model,device)
kg = Ktools.S_to_K(sg,model,device)

# For sequential graphs:
list_sg = Stools.cut(sg)
equivalent_classes,list_kg,list_ano_sg = Atools.S_list_to_K_list_eco(list_sg,model,device)

# You can print each graph using their respective function. Example:
Stools.print_S_graph(sg)
# Or the general function
rkgb.print_graph(sg)
```
