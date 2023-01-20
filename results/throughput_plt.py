import pandas as pd
import pickle
import seaborn

seaborn.set_theme()
import numpy as np
import matplotlib.pyplot as plt

with open("throughput_p100_b_4.pkl", "rb") as f:
    res = pickle.load(f)

fig, ax1 = plt.subplots(figsize=(12, 6), dpi=100)
for name in ["GPT2-medium", "GPT2-large", "GPT2-xl"]:
    r = res[name]
    o_mean = np.mean([4 / t for t in r[0][1:]])
    batch = [b for b, v in r.items() if b > 0 and v != "infeasible"]
    mean = []
    std = []
    for k, v in r.items():
        if v == "infeasible":
            continue
        if k > 0:
            mean.append(np.mean([k / t / o_mean for t in v[1:]]))
            std.append(np.std([k / t / o_mean for t in v[1:]]))
    plt.errorbar(batch, mean, yerr=std, label=name)
plt.legend()
plt.xticks(batch, ["{:.0%}".format(b / 4) for b in batch])
ys = np.arange(0.9, 1.1, 0.02)
plt.yticks(ys, ["{:.0%}".format(y) for y in ys])
plt.xlabel("relative batch size")
plt.ylabel("relative throughput")
plt.savefig("throughput.pdf", format="pdf")
