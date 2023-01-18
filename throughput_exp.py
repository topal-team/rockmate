import torch
import rockmate as rk
from models.GPT import get_GPT
import pickle
from torchvision.models import regnet_y_1_6gf, resnet18

from exp_utils import sanity_check, throughput_exp
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda")

from models.GPT import get_GPT
import numpy as np

tmp = torch.ones(14, 1024, 1024, 256, device=device)
del tmp

throughput_res = {}

name = "regnet_y_1_6gf"
input_size = [250, 3, 128, 128]
batch_sizes = np.arange(5, 16, 1) * 50
budget = 12.0e9
model = regnet_y_1_6gf().to(device).to(torch.float64)
x = torch.randn(input_size).to(device).to(torch.float64)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)
with open("throughput_compiled_v100.pkl", "wb") as f:
    pickle.dump(throughput_res, f)


name = "GPT2-medium"
input_size = [6, 512]
batch_sizes = np.arange(6, 16, 1)
budget = 11.6e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

with open("throughput_compiled_v100.pkl", "wb") as f:
    pickle.dump(throughput_res, f)

name = "GPT2-large"
input_size = [6, 256]
batch_sizes = np.arange(6, 16, 1)
budget = 7.8e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

with open("throughput_compiled_v100.pkl", "wb") as f:
    pickle.dump(throughput_res, f)

name = "GPT2-xl"
input_size = [1, 256]
batch_sizes = np.arange(1, 8, 1)
budget = 2.4e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)


with open("throughput_compiled_v100.pkl", "wb") as f:
    pickle.dump(throughput_res, f)

