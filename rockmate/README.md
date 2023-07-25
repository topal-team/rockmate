# Rockmate

Warning: Currently, Rockmate relies on Gurobi to solve the Integer Linear Programming model. 

Given a module and a sample (*i.e.* example input for it) and a memory budget, 
`Rockmate` builds a new `torch.nn.Module` with equal forward and backward results while 
keeping the memory peak under the given budget.

Backward pass updates original model parameters.

The model and sample should be on the GPU device.

### Complete example

```python
import torch
from rockmate import Rockmate
from torchvision.models import resnet101

device = torch.device("cuda")
model = resnet101().to(device)
x = torch.randn([100, 3, 128, 128]).to(device)
m_budget = 2 * 1024**3

rkMod = Rockmate(model, x, m_budget)

loss = rkMod(x).mean()
loss.backward()
rkMod.backward()
```
