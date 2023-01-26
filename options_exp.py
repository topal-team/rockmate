import time
from exp_utils import *
import torch
import rkgb
import rockmate as rk
from copy import deepcopy
from rotor import timing
import numpy as np
import pickle
from mem_tools import *

from models.GPT import get_GPT

tmp = torch.ones(3, 1024, 1024, 256, device=device)
del tmp

model = get_GPT("GPT2-small")
x = torch.randint(0, 1000, [16, 128])
# res = copy_run(model,x)
results = []
budget = np.arange(0.6, 2.5, 0.2) * 1024 ** 3
# for (nbar, nall) in [(3, 3,), (5, 5), (10, 10), (20, 20), (30, 30)]:
for nbar in np.arange(5, 21, 5):
    for nall in np.arange(5, 21, 5):
        results += copy_run_rk(
            model, x, budget, repeat=10, nbar=nbar, nall=nall
        )
        with open("options_exp_small_step.pkl", "wb") as f:
            pickle.dump(results, f)
