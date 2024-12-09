# Rockmate

The `Rockmate` framework is designed for training a PyTorch neural network within a given GPU budget
constraint using automatic re-materialization (activation checkpointing) technique.

Given a PyTorch model, a sample input, and a GPU memory budget, `Rockmate` builds a new
`torch.nn.Module`, which performs forward and backward pass keeping activations under the given
budget.

- The new model produces the same outputs and gradients as the original one.
- Model training with a budget constraint, which is lower than the one required by PyTorch Autodiff,
  is achieved by re-computing some of the activations instead of storing them for gradient
  calculation.
- Depending on the budget, `Rockmate` defines automatically which activations should be recomputed.

<!-- Given a module, sample input, and a memory budget, `Rockmate` builds a new `torch.nn.Module`
with equal forward and backward results while keeping the memory usage of activations under the
given budget. -->

<!-- For more details of our algorithm, see our paper at: https://openreview.net/pdf?id=wLAMOoL0KD
-->

Notes:

- The model and sample should be on the same GPU device.
- The `Rockmate` framework contains a variety of optimization algorithms, highly configurable, with
  three main default behaviors:
  - The original **Rockmate** algorithm designed for sequential-like neural networks, described in the
    the [ICML 2023 paper (oral) "Rockmate: an Efficient, Fast, Automatic and Generic Tool for
    Re-materialization in PyTorch"](https://openreview.net/pdf?id=wLAMOoL0KD). The code for the
    paper is available standalone as the `v1` tag of this repository, but the algorithm is also part
    of the complete repository, accessible vie the `PureRockmate` class.
  - The hierarchical approach **HiRemate** can be applied to any kind of neural network, without the
    sequential-like restriction of `Rockmate`. It is described in [ HiRemate: Hierarchical Approach
    for Efficient Re-materialization of Large Neural Networks
    ](https://inria.hal.science/hal-04403844), and accessible via the `Hiremate` class.
  - The **OffMate** specialization also includes activation and weight offloading to further reduce
    memory consumption. It is described in [OffMate: full fine-tuning of LLMs on a single GPU by
    re-materialization and offloading](https://inria.hal.science/hal-04660745), and accessible via
    the `Offmate` class.

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
from rockmate import PureRockmate, Hiremate
from torchvision.models import resnet101

device = torch.device("cuda")

resnet = resnet101().cuda()
optimizer = torch.optim.Adam(resnet.parameters())
sample = torch.randn([100, 3, 128, 128]).cuda()
m_budget = 2 * 1024**3 # 2GB

if use_hiremate:
	rk_resnet = PureHiremate(resnet, sample, m_budget)
else:
	rk_resnet = PureRockmate(resnet, sample, m_budget)

for data, target in dataset:
    y = rk_resnet(data) # use rk_resnet as resnet
    loss = loss_function(y, target)
    loss.backward()
    optimizer.step() # parameters in resnet are updated
```

## Offmate

The usage of `Offmate` is slightly different, because in this configuration the framework also
handles the optimizer step. Furthermore, the framework does not assume that the model is on GPU at
the start, which allows using models whose parameters do not fit in the GPU memory.

```python
import torch
from rockmate import PureRockmate, Hiremate
from torchvision.models import resnet101

device = torch.device("cuda")

resnet = resnet101()
optimizer = torch.optim.Adam(resnet.parameters())
sample = torch.randn([100, 3, 128, 128]).cuda()
m_budget = 2 * 1024**3 # 2GB

rk_resnet = Offmate(resnet, sample, m_budget)

for data, target in dataset:
    y = rk_resnet(data) # use rk_resnet as resnet
    loss = loss_function(y, target)
    loss.backward()
```

## Configurations

The `Rockmate` framework also provides a configuration mechanism, accessible with the following functions:

* `generate_config(config_type)` generates a complete configuration, where `config_type` can be any
  of `"rotor"`, `"rockmate"`, `"checkmate"`, `"hilp"`, `"hiremate"`, `"offmate"`, `"noremat"`.
* This configuration can be modified, and also saved and loaded with `save_config()` and
  `load_config()` functions.
* Then, the `from_config(model, sample, budget, config)` function builds the appropriate `Rockmate()` module.

# Citing
If you used our research, we kindly ask you to cite the corresponding
[paper](https://openreview.net/pdf?id=wLAMOoL0KD).

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
