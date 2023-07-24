# Rockmate

Given a module, a sample input and a memory budget, `Rockmate` builds a new `torch.nn.Module` with equal forward and backward results while keeping the memory usage of activations under the given budget.

For more details of our algorithm, see our paper at: https://openreview.net/pdf?id=wLAMOoL0KD

- The model and sample should be on the same GPU device.
- **Warning**: Currently, Rockmate relies on Gurobi to solve the Integer Linear Programming model. 

# Installation

You can simply use pip:
```
pip install rockmate
```

Or clone the repository and install locally (we recommand using editable mode)
```
git clone https://github.com/topal-team/rockmate.git
cd rockmate
pip install -e ./rockmate -e ./rkgb
```

# Examples

## Rockmate

```python
import torch
from rockmate import Rockmate
from torchvision.models import resnet101

device = torch.device("cuda")

resnet = resnet101().cuda()
optimizer = torch.optim.Adam(resnet.parameters())
sample = torch.randn([100, 3, 128, 128]).cuda()
m_budget = 2 * 1024**3 # 2GB

rk_resnet = Rockmate(resnet, sample, m_budget)

for data, target in dataset:
    y = rk_resnet(data) # use rk_resnet as resnet
    loss = loss_function(y, target)
    loss.backward()
    rk_resnet.backward()
    optimizer.step() # parameters in resnet are updated
```

Implementation will be soon updated so that `rk_resnet.backward()` is not needed.

## rk-GraphBuilder

**rk-GB** generates the graphs needed by Rockmate. It can be used on its own, in particular as a way to visualize PyTorch modules without requiring any annotations.

```python
# Example of how to use rkgb
import torch
import rkgb
from torchvision.models import resnet101

device = torch.device("cuda")
model = resnet101().cuda()
sample = torch.randn([100, 3, 128, 128]).cuda()

rkgb_result = rkgb.make_all_graphs(model,sample)

rkgb.print_all_graphs(rkgb_result,name="resnet101",render_format="pdf")
# To render the graphs in pdf you need Graphviz

# You can also try:
rkgb_result = rkgb.test_rkgb(model,sample)
```

# Next release soon

Cleaner implementation and detail documentation to be released soon.
