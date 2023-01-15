import torch
import rockmate as rk
from models.GPT import get_GPT

from exp_utils import sanity_check, throughput_exp

device = torch.device("cuda")

from models.GPT import get_GPT
import numpy as np

model = get_GPT("GPT2-large").to(device)
x = torch.randint(0, 600, [10, 512]).to(device)

throughput_res = throughput_exp(model, x, np.arange(2, 20, 2), mem_limit=7e9)

import pickle

with open("throughput.pkl", "wb") as f:
    pickle.dump(throughput_res, f)
