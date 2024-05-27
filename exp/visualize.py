import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn
seaborn.set_theme()
# seaborn.set(font="Nimbus Roman")
# seaborn.set_style("whitegrid", {'grid.linestyle': '-.'})
import matplotlib as mpl
# mpl.rcParams['font.family'] = 'tImeS nEw ROmaN'
def read_res(file_name):
    with open(file_name, "rb") as f:
        results = pickle.load(f)
    return results

def get_label(key):
    d = {
    "offmate":"OffMate",
    "offmate_no_act_offload":"OffMate w/o Act Offload",
    "offmate_no_cpu_optim":"OffMate w/o CPU Optimize",
    "llama7b": "Llama2 (7B)",
    "llama7b_lora": "Llama2 (7B) LoRA",
    "phi2-3b": "Phi-2 (2.7B)",
    "float32": "fp32",
    "bfloat16": "bf16",
    "zero-3": "ZeRO-Infinity",
    "zero-2": "ZeRO-2",
    "paged_optim": "Paged Optimizer",
    "llama13b": "Llama2 (13B)",
    "rockmate": "Rockmate",
    "Torch_exp": "Torch (Extrapolated)"
    }
    if key in d:
        return d[key]
    return key

def get_color(key):
    c = {
        "Torch":"deeppink",
        "Torch_exp":"deeppink",
        "offmate": "green",
        "zero-3":"brown",
        "zero-2": "violet",
        "paged_optim":"blue",
        "rockmate": "darkorange"
        }
    return c[key]

class Viz:
    def __init__(self, model="llama7b", dtype="float32", id="0", device="RTX 3060 12GB"):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.id = id
        self.time_norm = 1000
        pass

    def plot_method(self, ax, method):
        results = read_res(f"exp_results/{method}-{self.model}-{self.dtype}-{self.id}.pkl")
        self.nlayers = list(sorted([k for k in results.keys() if "time" in results[k]]))
        x = self.nlayers
        y = np.array([results[n]["time"] for n in self.nlayers])
        # print(method, x, y)
        ax.plot(x,
                y/self.time_norm, 
                label=f"{get_label(method)}",
                color=get_color(method))
        # ax.set_xticks([])
        # ax.set_yticks([])
        
    def plot_torch(self, ax, ax2, i = 1, j = 2, match_coloer = "deeppink"):
        method = "torch"
        results = read_res(f"exp_results/{method}-{self.model}-{self.dtype}-{self.id}.pkl")
        nlayers1 = list(sorted([k for k in results.keys() if "time" in results[k]]))
        t = [results[n]["time"]/self.time_norm for n in nlayers1]
        m = [results[n]["peak_mem"] for n in nlayers1]

        # nlayers2 = [, 4, 8, 12, 16, 20, 24, 28, 32]
        j = max(nlayers1)
        nlayers2 = [max(nlayers1)] + [n for n in self.nlayers if n >max(nlayers1)]
        k_t = (results[j]["time"]/self.time_norm - results[i]["time"]/self.time_norm)/(j-i)
        k_m = (results[j]["peak_mem"] - results[i]["peak_mem"])/(j-i)


        t_exp = [(results[i]["time"]/self.time_norm + k_t * (n-i)) for n in nlayers2]
        ax.plot(nlayers2, t_exp, linestyle="dashed", color=get_color("Torch_exp"), label=get_label("Torch_exp"))
        ax.plot(nlayers1, t, color=get_color("Torch"), label=get_label("Torch"))

        m_exp = [(results[i]["peak_mem"]/self.time_norm + k_m * (n-i)) for n in nlayers2]
        
        def generate_ticks(max_mem):
            unit = 1024**3*10
            while max_mem//unit > 7:
                unit *= 2
            max_l = max_mem//unit + 1
            gb_unit = unit/1024**3
            return np.arange(0, (max_l+1)*gb_unit, gb_unit, dtype=int), gb_unit
        
        # mlabels = range(10,110, 10)
        mlabels, unit = generate_ticks(max(m_exp))
        mticks = np.array([(mem*1024**3 -results[i]["peak_mem"])/k_m for mem in mlabels])+i
        # print(mticks, mlabels, k_m)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(mticks)
        ax2.set_xticklabels(mlabels)
        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel("Memory requirement Torch (GiB)")
        ax2.set_yticks([])
        # ax2.set_xticks([])

    def plot(self, methods=[], save_fig=False):
        f = plt.figure()
        ax_mem = f.add_subplot(111, label="2")
        ax_layer = f.add_subplot(111, label="1",
                                 frame_on=False
                       )
        ax_layer.grid(axis="x")
        for method in methods:
            self.plot_method(ax_layer, 
                            method=method)
        ax_layer.set_ylabel("Iteration time (s)")
        ax_layer.set_xlabel("Number of hidden layers")
        self.plot_torch(ax_layer, ax_mem)
        title = f"{get_label(self.model)} {get_label(self.dtype)} on {self.device}"
        ax_layer.legend()
        # ax_layer.set_xticks([])
        # ax_layer.set_yticks([])
        ax_mem.set_axisbelow(True)
        f.suptitle(title)
        f.tight_layout()
        if save_fig:
            f.savefig(f"{title.replace(' ', '_')}.pdf", format="pdf", dpi=100)

def plot_batch_sel(results, save_fig=False):
    f = plt.figure()
    ax = f.add_subplot(111, label="2")
    batch = np.array([k for k in sorted(results.keys()) if k >0])
    time = np.array([results[k]["time"] for k in batch])/1000
    throughput = batch/time

    sched_time = np.array([results[k]["sched_stats"]["Schedule_total_time"] for k in batch])/1000
    sched_throughput = batch/sched_time

    ilp_time = np.array([float(results[k]["ilp_objective"]) for k in batch])/1000
    ilp_throughput = batch/ilp_time

    ax.plot(batch, throughput, label="Measured Throughput")
    ax.plot(batch, sched_throughput, label="Schedule Throughput")
    ax.plot(batch, ilp_throughput, label="ILP solution Throughput")

    optim_batch = results[0]["input_size"][0][0]/results[0]["req_w"]
    optim_time = float(results[0]["ilp_objective"])/1000/results[0]["req_w"]
    ax.scatter([optim_batch], [optim_batch/optim_time], color="r", label="Optim Batch size")

    title = "Phi-2 FP32 on RTX 3060 12GB"
    ax.set_title(title)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (sample/s)")
    ax.set_ylim(0, max(throughput)+0.05)
    ax.legend()
    if save_fig:
        f.savefig(f"{title.replace(' ', '_')}_batch.pdf", format="pdf", dpi=100)
