import torch
import numpy as np
from test.utils import compare_model
from test.test_models import *
from torchvision.models import resnet18

model = resnet18().to(torch.float64)
x = torch.randn(100, 3, 32, 32).to(torch.float64)
for budget in np.arange(10, 20, 2) * 10 * 1024 ** 2:
    compare_model(model, x, budget)

model = get_GPT("GPT2-small")
x = torch.randint(0, 600, [2, 64])
for budget in np.arange(2, 6) * 100 * 1024 ** 2:
    compare_model(model, x, budget)
