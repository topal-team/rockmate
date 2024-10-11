
# How to run

Singularity image at Plafrim

`/beegfs/ygusak/singularity_images/hiremate_twremat.sif` containing Gurobi and Twremat Haskell to run with HiRemate based on `hiremate_twremat.def`

Run Singularity shell
`sungularity shell --nv beegfs/ygusak/singularity_images/hiremate_twremat.sif`

Inside the shell
```
cd ~/rockmate-private/rockmate && pip install -e . 
cd ~/rockmate-private/rkgb && pip install -e .

```

# TwRemat example
```
import rkgb, rockmate, torch
from torchvision.models import resnet18
from rockmate.solvers.twremat_utils import *

model, sample = resnet18(), torch.randn(2, 3, 224, 224)
budget = 1024**9


h_cluster = get_rockmate_graphs(model, sample)

node_info, targets, loss_node = get_twremat_graph(h_cluster) 

steps = runtwremat(node_info, budget, targets, loss_node)

```

