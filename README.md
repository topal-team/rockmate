# Rockmate

This repository contains the code for the [ICML 2023 paper (oral) "Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch"](https://openreview.net/pdf?id=wLAMOoL0KD). It demonstrates how a PyTorch neural network can be trained under the given GPU budget constraint using the proposed automatic re-materialization (activation checkpointing) technique.

Given a PyTorch model, a sample input, and a GPU memory budget, 
`Rockmate` builds a new `torch.nn.Module`, which performs forward and backward pass keeping activations under the given budget. 

- The new model produces the same outputs and gradients as the original one.
- Model training with a budget constraint, which is lower than the one required by PyTorch Autodiff, is achieved by re-computing some of the activations instead of storing them for gradient calculation.
- Depending on the budget, `Rockmate` defines automatically which activations should be recomputed. 

<!-- Given a module, sample input, and a memory budget, `Rockmate` builds a new `torch.nn.Module` with equal forward and backward results while keeping the memory usage of activations under the given budget. -->

<!-- For more details of our algorithm, see our paper at: https://openreview.net/pdf?id=wLAMOoL0KD -->

Note:
- The model and sample should be on the same GPU device.
- **Warning**: Currently, Rockmate relies on [Gurobi](https://www.gurobi.com/documentation/quickstart.html) optimization library to solve the Integer Linear Programming model that defines a recomputation schedule for a given neural network architecture. This requires a license to Gurobi, which is free for academic use. 

# Installation

You can simply use pip:
```
pip install rockmate
```

Or clone the repository and install locally (we recommend using editable mode)
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

# Citing
If you used our research, we kindly ask you to cite the corresponding [paper](https://openreview.net/pdf?id=wLAMOoL0KD).

```
@inproceedings{zhao2023rockmate,
  title={Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch},
  author={Zhao, Xunyi and Le Hellard, Th{\'e}otime and Eyraud-Dubois, Lionel and Gusak, Julia and Beaumont, Olivier},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

# Further research and release

Rockmate is in heavy development, with documentation and more features. Stay tuned for future updates coming soon.


