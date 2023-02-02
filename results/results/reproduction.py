from models.GPT import *
from rotor_exp.rotor.models import resnet101, resnet50
from torchvision.models import regnet_y_1_6gf, regnet_x_3_2gf
from exp_utils import *

file_name = "resnet_exp_v100.pkl"
results = {}

model = resnet50()
x = torch.randn([200, 3, 128, 128])
budgets = np.concatenate(
    (
        np.arange(1.2, 2, 0.1) * 1024 ** 3,
        np.arange(2, 5, 0.2) * 1024 ** 3,
        np.arange(5, 7, 0.5) * 1024 ** 3,
    )
)
results["resnet50"] = exp(model, x, budget=budgets, run_rotor=True)

model = resnet101()
x = torch.randn([200, 3, 128, 128])
budgets = np.concatenate(
    (
        np.arange(1.2, 2, 0.1) * 1024 ** 3,
        np.arange(2, 5, 0.2) * 1024 ** 3,
        np.arange(5, 9, 0.5) * 1024 ** 3,
    )
)
results["resnet101"] = exp(model, x, budget=budgets, run_rotor=True)
with open(file_name, "wb") as f:
    pickle.dump(results, f)


file_name = "GPT2_exp_v100.pkl"
results = {}

input_sizes = {}
input_sizes["GPT2-medium"] = [2, 1024]
input_sizes["GPT2-large"] = [2, 512]

budgets = {}

budgets["GPT2-medium"] = np.concatenate(
    (np.arange(0.6, 2, 0.1) * 1024 ** 3, np.arange(2, 12, 0.4) * 1024 ** 3,)
)
budgets["GPT2-large"] = np.concatenate(
    (
        np.arange(0.45, 1, 0.1) * 1024 ** 3,
        np.arange(1, 2, 0.2) * 1024 ** 3,
        np.arange(2, 8, 0.4) * 1024 ** 3,
    )
)

for name in ["GPT2-medium", "GPT2-large"]:
    results[name] = {}
    model = get_GPT(name)
    x = torch.randint(0, 600, input_sizes[name])
    mbudget = budgets[name]
    results[name]["original"] = copy_run(model, x, repeat=10)
    results[name]["rotor"] = copy_run_rt_GPT(model, x, mbudget, repeat=10)
    results[name]["rockmate"] = copy_run_rk(model, x, mbudget, repeat=10)
    with open(file_name, "wb") as f:
        pickle.dump(results, f)


file_name = "GPT2_exp_v100_seq.pkl"
results = {}

input_sizes = {}
input_sizes["GPT2-medium"] = [4, 512]
input_sizes["GPT2-large"] = [4, 256]

budgets = {}

budgets["GPT2-medium"] = np.concatenate(
    (np.arange(0.6, 2, 0.1) * 1024 ** 3, np.arange(2, 8, 0.4) * 1024 ** 3,)
)
budgets["GPT2-large"] = np.concatenate(
    (
        np.arange(0.45, 1, 0.1) * 1024 ** 3,
        np.arange(1, 2, 0.2) * 1024 ** 3,
        np.arange(2, 6, 0.4) * 1024 ** 3,
    )
)

for name in ["GPT2-medium", "GPT2-large"]:
    results[name] = {}
    model = get_GPT(name)
    x = torch.randint(0, 600, input_sizes[name])
    mbudget = budgets[name]

    results[name]["original"] = copy_run(model, x, repeat=10)

    results[name]["rotor"] = copy_run_rt_GPT(model, x, mbudget, repeat=10)

    results[name]["rockmate"] = copy_run_rk(model, x, mbudget, repeat=10)

    with open(file_name, "wb") as f:
        pickle.dump(results, f)


file_name = "otherNN_exp_v100.pkl"
results = {}

model = regnet_y_1_6gf()
x = torch.randn([300, 3, 128, 128])
budgets = np.concatenate(
    (
        np.arange(1.5, 2.5, 0.1) * 1024 ** 3,
        np.arange(2.5, 4, 0.2) * 1024 ** 3,
        np.arange(4, 9, 0.5) * 1024 ** 3,
    )
)

results["regnet_y_1_6gf"] = exp(model, x, budget=budgets, run_rotor=False)
with open(file_name, "wb") as f:
    pickle.dump(results, f)

model = regnet_x_3_2gf()
x = torch.randn([300, 3, 128, 128])
budgets = np.concatenate(
    (
        np.arange(1.5, 2.5, 0.1) * 1024 ** 3,
        np.arange(2.5, 4, 0.2) * 1024 ** 3,
        np.arange(4, 10, 0.5) * 1024 ** 3,
    )
)

results["regnet_x_3_2gf"] = exp(model, x, budget=budgets, run_rotor=False)
with open(file_name, "wb") as f:
    pickle.dump(results, f)
