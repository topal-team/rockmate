import time
from exp_utils import *
import torch
import pgb
import rockmate as rk
from copy import deepcopy
from rotor import timing
import numpy as np
import pickle
from mem_tools import *
from torchvision.models import resnet101
from models.GPT import get_GPT

tmp = torch.ones(3, 1024, 1024, 256, device=device)
del tmp

model = get_GPT("GPT2-medium")
x = torch.randint(0, 1000, [16, 128])
results = []
budget = np.arange(0.8, 4, 0.4) * 1024 ** 3
for (nbar, nall) in [(3, 3,), (5, 5), (10, 10), (20, 20), (30, 30)]:
    results += copy_run_rk(model, x, budget, repeat=10, nbar=nbar, nall=nall)
    with open("options_exp_v100_GPT2.pkl", "wb") as f:
        pickle.dump(results, f)

model = resnet101()
x = torch.randn([200, 3, 128, 128])
budget = np.arange(1, 8, 1) * 1024 ** 3
results = []
for (nbar, nall) in [(3, 3,), (5, 5), (10, 10), (20, 20), (30, 30)]:
    results += copy_run_rk(model, x, budget, repeat=10, nbar=nbar, nall=nall)
    with open("options_exp_v100_resnet.pkl", "wb") as f:
        pickle.dump(results, f)

