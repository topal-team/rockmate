import torch
import rockmate as rk
from models.GPT import get_GPT

from exp_utils import sanity_check, throughput_exp

device = torch.device("cuda")

from models.GPT import get_GPT
import numpy as np

model = get_GPT("GPT2-large").to(device)
x = torch.randint(0, 600, [6, 256]).to(device)

throughput_res = throughput_exp(model, x, np.arange(8, 32, 1), mem_limit=7.5e9)


# model = get_GPT("GPT2-xl").to(device)
# x = torch.randint(0, 600, [2, 256]).to(device)

# throughput_res = throughput_exp(model, x, np.arange(1, 7, 1), mem_limit=4.5e9)


import pickle

with open("throughput.pkl", "wb") as f:
    pickle.dump(throughput_res, f)
