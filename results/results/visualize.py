import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd

seaborn.set_theme()


def plot_vs_rotor(file_name="results/vs_rotor.pdf"):
    with open("results/resnet_exp_v100.pkl", "rb") as f:
        results = pickle.load(f)
    with open("results/GPT2_exp_v100.pkl", "rb") as f:
        results = {**pickle.load(f), **results}
    names = ["resnet50", "resnet101"]
    names += ["GPT2-medium", "GPT2-large"]

    NN_names = ["ResNet-50", "ResNet-101", "GPT2-medium", "GPT2-large"]
    fig = plt.figure(figsize=(6, 6), dpi=100)
    n = len(names)
    for i in range(n):
        name = names[i]
        ax1 = fig.add_subplot(2, n // 2, i + 1)

        rt_time_mean = []
        rt_time_std = []
        rk_time_mean = []
        rk_time_std = []
        rt_peak_mem = []
        rk_peak_mem = []
        capsize = 2

        original_time = np.mean(results[name]["original"]["times"][1:])
        original_peak = results[name]["original"]["peak_mem"]

        for r in results[name]["rotor"]:
            if r["feasible"]:
                rt_time_mean.append(np.mean(r["times"][1:]))
                rt_time_std.append(np.std(r["times"][1:]))
                rt_peak_mem.append(r["peak_mem"])
        for r in results[name]["rockmate"]:
            if r["feasible"]:
                rk_time_mean.append(np.mean(r["times"][1:]))
                rk_time_std.append(np.std(r["times"][1:]))
                rk_peak_mem.append(r["peak_mem"])

        ax1.errorbar(
            np.array(rt_peak_mem),
            np.array(rt_time_mean) / original_time,
            yerr=np.array(rt_time_std) / original_time,
            label="Rotor",
            capsize=capsize,
            color="b",
        )

        ax1.errorbar(
            np.array(rk_peak_mem),
            np.array(rk_time_mean) / original_time,
            yerr=np.array(rk_time_std) / original_time,
            label="Rockmate",
            capsize=capsize,
            color="r",
        )

        x = rt_peak_mem + rk_peak_mem
        ax1.scatter([original_peak], [1], color="g", label="PyTorch")
        ax1.plot(np.linspace(min(x), max(x), 100), [1] * 100, "--", color="g")
        ax1.legend(loc="upper right")
        ax1.set_title(NN_names[i])
        ax1.set_xlabel("Peak mem (GiB)")

        xticks = np.arange(min(x) // 1024 ** 3, (max(x) // 1024 ** 3) + 1)
        xtickslabel = [str(t) for t in xticks]
        ax1.set_xticks(xticks * 1024 ** 3)
        ax1.set_xticklabels(xtickslabel)

        if i % 2 == 0:
            ax1.set_ylabel("Overhead")
    fig.subplots_adjust(hspace=0.45, wspace=0.25, left=0.1, right=0.9)
    fig.show()
    fig.savefig("results/vs_rotor.pdf", format="pdf")


def plot_vs_checkmate(log=False):
    with open("results/checkmate_exp_noapprox.pkl", "rb") as f:
        results_chk = pickle.load(f)
    with open("results/rockmate_exp_solve_time.pkl", "rb") as f:
        results_rkm = pickle.load(f)
    with open("results/rotor_exp_solve_time.pkl", "rb") as f:
        results_rot = pickle.load(f)

    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    overhead = []
    solve_time = []
    nlayers = np.arange(2, 12, 2)
    for n in nlayers:
        res = [r for r in results_chk if r[0] == n]
        if len(res) < 1:
            continue
        overhead.append(
            max([((r[4] - r[5]) / r[5]) // 0.0001 / 100 for r in res])
        )
        if log:
            solve_time.append(np.log(max([r[2] for r in res]) / 60))
        else:
            solve_time.append(max([r[2] for r in res]))
    ax1.scatter(nlayers, 1 / np.array(overhead), label="Checkmate", marker="o")
    ax2.scatter(nlayers, solve_time, label="Checkmate", marker="o")
    chk_solve_time = solve_time[0:]

    overhead = []
    solve_time = []
    nlayers = np.arange(2, 12, 2)
    for n in nlayers:
        res = [r for r in results_rkm if r[0] == n]
        if len(res) < 1:
            continue
        overhead.append(
            max([((r[4] - r[5]) / r[5]) // 0.0001 / 100 for r in res])
        )
        if log:
            solve_time.append(np.log(max([r[2] for r in res]) / 60))
        else:
            solve_time.append(max([r[2] for r in res]))
    ax1.scatter(nlayers, 1 / np.array(overhead), label="Rockmate", marker="v")
    ax2.scatter(nlayers, solve_time, label="Rockmate", marker="v")

    overhead = []
    solve_time = []
    nlayers = np.arange(2, 12, 2,)
    for n in nlayers:
        res = [r for r in results_rot if r[0] == n]
        if len(res) < 1:
            continue
        overhead.append(
            max([((r[4] - r[5]) / r[5]) // 0.0001 / 100 for r in res])
        )
        if log:
            solve_time.append(np.log(max([r[2] for r in res]) / 60))
        else:
            solve_time.append(max([r[2] for r in res]))
    ax1.scatter(nlayers, 1 / np.array(overhead), label="Rotor", marker="^")
    ax2.scatter(nlayers, solve_time, label="Rotor", marker="^")

    ax1.legend()
    ax2.legend()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax1.set_xticks(nlayers)
    ax1.set_xticklabels([str(num) for num in nlayers])
    ax2.set_xticks(nlayers)
    ax2.set_xticklabels([str(num) for num in nlayers])
    ax1.set_ylabel("Throughput")
    if log:
        ax2.set_ylabel("Solve time (min)")
    else:
        ax2.set_ylabel("Solve time (hour)")
    ax1.set_xlabel("nlayers")
    ax2.set_xlabel("nlayers")
    if not log:
        ax2.set_yticks(list(np.arange(6, 31, 5) * 3600))
        ax2.set_yticklabels(np.arange(6, 31, 5))
    else:
        yt_label = np.array([0.05, 0.5, 5, 50, 500])
        yt_label = np.array([0.01, 0.1, 1, 10, 100, 1000])
        yt = np.log(yt_label)
        ax2.set_yticks(yt)
        ax2.set_yticklabels(
            [np.format_float_positional(l, 2, trim="-") for l in yt_label]
        )
    fig.subplots_adjust(wspace=0.1, bottom=0.18, left=0.1, right=0.85)
    if log:
        fig.savefig("results/vs_checkmate_log.pdf", format="pdf")
    else:
        fig.savefig("results/vs_checkmate.pdf", format="pdf")


def plot_options():
    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    with open("results/options_exp_v100_GPT2.pkl", "rb") as f:
        results_opt = pickle.load(f)
    budget = []

    for nbar in [3, 5, 10, 20, 30]:
        nall = nbar
        res = [
            r
            for r in results_opt
            if (
                r["feasible"]
                and r["nb_budget_abar"] == nbar
                and r["nb_budget_all"] == nall
            )
        ]
        budget = [r["budget"] / 1024 ** 3 for r in res]
        mean = [np.mean(r["times"][2:]) for r in res]
        std = [np.std(r["times"][2:]) for r in res]
        ax1.errorbar(budget, mean, yerr=std, label=f"({nbar}, {nall})")
    ax1.legend()
    ax1.set_title("GPT2-medium")
    ax1.set_xlabel("Budget (GiB)")
    ax1.set_ylabel("Makespan (ms)")

    with open("results/options_exp_v100_resnet.pkl", "rb") as f:
        results_opt = pickle.load(f)
    budget = []

    for nbar in [3, 5, 10, 20, 30]:
        nall = nbar
        res = [
            r
            for r in results_opt
            if (
                r["feasible"]
                and r["nb_budget_abar"] == nbar
                and r["nb_budget_all"] == nall
            )
        ]
        budget = [r["budget"] / 1024 ** 3 for r in res]
        mean = [np.mean(r["times"][2:]) for r in res]
        std = [np.std(r["times"][2:]) for r in res]
        ax2.errorbar(budget, mean, yerr=std, label=f"({nbar}, {nall})")
    ax2.legend()
    ax2.set_title("ResNet-101")
    ax2.set_xlabel("Budget (GiB)")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    fig.subplots_adjust(wspace=0.1, bottom=0.18, left=0.1, right=0.85)
    fig.savefig("results/options.pdf", format="pdf")


def get_table():
    with open("results/GPT2_exp_v100.pkl", "rb") as f:
        results = pickle.load(f)
    with open("results/GPT2_exp_v100_seq.pkl", "rb") as f:
        results2 = pickle.load(f)
    features = [
        "Model",
        "Input_size",
        "Algorihtm",
        "Budget (GiB)",
        "Peak_mem (GiB)",
        "Makespan_mean (ms)",
        "Makespan_std (ms)",
    ]
    l_data = []
    m_unit = 1024 ** 3
    for model_name in ["GPT2-medium", "GPT2-large"]:
        for result in [results, results2]:
            exp = result[model_name]["original"]
            l_data.append(
                [
                    model_name,
                    exp["input_size"],
                    "PyTorch",
                    0,
                    exp["peak_mem"] / m_unit,
                    np.mean(exp["times"][1:]),
                    np.std(exp["times"][1:]),
                ]
            )
            for alg in ["rotor", "rockmate"]:
                res = result[model_name][alg]
                res = [res[4], res[len(res) // 2], res[-1]]
                for exp in res:
                    if exp["feasible"]:
                        l_data.append(
                            [
                                model_name,
                                exp["input_size"],
                                alg,
                                exp["budget"] / m_unit,
                                exp["peak_mem"] / m_unit,
                                np.mean(exp["times"][1:]),
                                np.std(exp["times"][1:]),
                            ]
                        )
    df = pd.DataFrame(l_data, columns=features)
    print(
        df.sort_values(["Model", "Input_size", "Budget (GiB)"]).to_latex(
            index=False, float_format="%.3f"
        )
    )


def plot_regnet():
    with open("results/otherNN_exp_v100.pkl", "rb") as f:
        results = pickle.load(f)
    names = ["regnet_y_1_6gf", "regnet_x_3_2gf"]
    NN_names = ["Regnet_y_1_6gf", "Regnet_x_3_2gf"]
    fig = plt.figure(figsize=(6, 3), dpi=100)
    n = len(names)
    for i in range(n):
        name = names[i]
        ax1 = fig.add_subplot(1, n, i + 1)

        rt_time_mean = []
        rt_time_std = []
        rk_time_mean = []
        rk_time_std = []
        rt_peak_mem = []
        rk_peak_mem = []
        capsize = 2

        for r in results[name]["rockmate"]:
            if r["feasible"]:
                rk_time_mean.append(np.mean(r["times"][1:]))
                rk_time_std.append(np.std(r["times"][1:]))
                rk_peak_mem.append(r["peak_mem"])
        ax1.errorbar(
            rk_peak_mem,
            rk_time_mean,
            yerr=rk_time_std,
            label="rockmate",
            capsize=capsize,
            color="r",
        )

        x = rt_peak_mem + rk_peak_mem
        original_time = np.mean(results[name]["original"]["times"][1:])
        original_peak = results[name]["original"]["peak_mem"]
        ax1.scatter(
            [original_peak], [original_time], color="g", label="PyTorch"
        )
        ax1.plot(
            np.linspace(min(x), max(x), 100),
            [original_time] * 100,
            "--",
            color="g",
        )
        ax1.legend(loc="upper right")
        ax1.set_title(NN_names[i])
        ax1.set_xlabel("Peak mem (GiB)")

        xticks = np.arange(min(x) // 1024 ** 3, (max(x) // 1024 ** 3) + 1)
        xtickslabel = [str(t) for t in xticks]
        ax1.set_xticks(xticks * 1024 ** 3)
        ax1.set_xticklabels(xtickslabel)

        if i == 0:
            ax1.set_ylabel("Makespan (ms)")
    fig.subplots_adjust(wspace=0.2, bottom=0.18, left=0.1, right=0.9)
    fig.savefig("results/regnet.pdf", format="pdf")


plot_vs_rotor()
plot_vs_checkmate(log=True)
plot_vs_checkmate()
plot_options()
plot_regnet()
