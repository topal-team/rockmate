import torch
import rockmate as rk
from models.GPT import get_GPT
import pickle
from torchvision.models import regnet_y_1_6gf, resnet18
from mlp_mixer_pytorch import MLPMixer

from exp_utils import sanity_check, throughput_exp
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda")

from models.GPT import get_GPT
import numpy as np

tmp = torch.ones(14, 1024, 1024, 256, device=device)
del tmp

# throughput_res = {}
file_name = "throughput_v100_b_4.pkl"
with open(file_name, "rb") as f:
    throughput_res = pickle.load(f)


# name = "MLPMixer"
# model = MLPMixer(
#     image_size=256,
#     channels=3,
#     patch_size=16,
#     dim=512,
#     depth=12,
#     num_classes=1000,
# ).to(device)
# input_size = [120, 3, 256, 256]
# batch_sizes = np.arange(6, 16, 1) * 20
# budget = 10.5e9
# x = torch.randn(input_size).to(device)
# throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

# with open(file_name, "wb") as f:
#     pickle.dump(throughput_res, f)

# name = "regnet_y_1_6gf"
# input_size = [250, 3, 128, 128]
# batch_sizes = np.arange(5, 16, 1) * 50
# budget = 12.0e9
# model = regnet_y_1_6gf().to(device).to(torch.float64)
# x = torch.randn(input_size).to(device).to(torch.float64)

# throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)
# with open(file_name, "wb") as f:
#     pickle.dump(throughput_res, f)

# name = "GPT2-large"
# input_size = [6, 256]
# batch_sizes = np.arange(6, 16, 1)
# budget = 7.8e9
# model = get_GPT(name).to(device)
# x = torch.randint(0, 600, input_size).to(device)


# name = "GPT2-large"
# input_size = [4, 332]
# batch_sizes = np.arange(4, 17, 1)
# budget = 7.5e9
# model = get_GPT(name).to(device)
# x = torch.randint(0, 600, input_size).to(device)


# throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

# with open(file_name, "wb") as f:
#     pickle.dump(throughput_res, f)

# # name = "GPT2-medium"
# # input_size = [6, 512]
# # batch_sizes = np.arange(6, 16, 1)
# # budget = 11.6e9
# # model = get_GPT(name).to(device)
# # x = torch.randint(0, 600, input_size).to(device)

name = "GPT2-medium"
input_size = [4, 640]
batch_sizes = np.arange(4, 17, 1)
budget = 10.8e9
model = get_GPT(name).to(device)
x = torch.randint(0, 600, input_size).to(device)

throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)

with open(file_name, "wb") as f:
    pickle.dump(throughput_res, f)

# name = "GPT2-xl"
# input_size = [1, 256]
# batch_sizes = np.arange(1, 8, 1)
# budget = 2.4e9
# model = get_GPT(name).to(device)
# x = torch.randint(0, 600, input_size).to(device)


# name = "GPT2-xl"
# input_size = [4, 72]
# batch_sizes = np.arange(4, 17, 1)
# budget = 2.02e9
# model = get_GPT(name).to(device)
# x = torch.randint(0, 600, input_size).to(device)

# throughput_res[name] = throughput_exp(model, x, batch_sizes, mem_limit=budget)


# with open(file_name, "wb") as f:
#     pickle.dump(throughput_res, f)

