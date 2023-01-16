import torch
import rockmate as rk
from models.GPT import get_GPT

from exp_utils import sanity_check, throughput_exp
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda")

from models.GPT import get_GPT
import numpy as np

throughput_res = {}

name = "GPT2-medium"
input_size = [6, 512]
batch_sizes = np.arange(6, 16, 1)
budget = 11.6e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

name = "GPT2-large"
input_size = [6, 256]
batch_sizes = np.arange(6, 16, 1)
budget = 7.8e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

name = "GPT2-xl"
input_size = [1, 256]
batch_sizes = np.arange(1, 8, 1)
budget = 2.4e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)


import pickle

# with open("throughput.pkl", "wb") as f:
#     pickle.dump(throughput_res, f)
