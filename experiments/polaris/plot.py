import itertools
import os
import pathlib

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deephyper.analysis._matplotlib import figure_size, update_matplotlib_rc

update_matplotlib_rc()
figsize = figure_size(252 * 1.8, 1.0)

from tqdm.notebook import tqdm

from deephyper.skopt.moo import pareto_front, non_dominated_set, hypervolume
from deephyper.skopt.utils import cook_objective_scaler
from deephyper.analysis.hpo import plot_worker_utilization

objective_columns = [f"objective_{i}" for i in range(3)]


@ticker.FuncFormatter
def minute_major_formatter(x, pos):
    x = float(f"{x/60:.2f}")
    if x % 1 == 0:
        x = str(int(x))
    else:
        x = f"{x:.2f}"
    return x


t_max = 3600 * 2.5
source_dir = "jobs/output"
figures_dir = "figures"
pathlib.Path(figures_dir).mkdir(parents=False, exist_ok=True)

data_path = {
    # 10 nodes -> 40 workers
    "Random-10": f"{source_dir}/random-10/results.csv",
    "NSGAII-10": f"{source_dir}/nsgaii-10/results.csv",
    "NSGAII (P)-10": f"{source_dir}/nsgaii-P-10/results.csv",
    "MoTPE-10": f"{source_dir}/motpe-10/results.csv",
    "MoTPE (P)-10": f"{source_dir}/motpe-P-10/results.csv",
    "D-MoBO-10": f"{source_dir}/dmobo-10/results.csv",
    "D-MoBO (P)-10": f"{source_dir}/dmobo-P-10/results.csv",
    # 40 nodes -> 160 workers
    "Random-40": f"{source_dir}/random-40/results.csv",
    "NSGAII-40": f"{source_dir}/nsgaii-40/results.csv",
    "NSGAII (P)-40": f"{source_dir}/nsgaii-P-40/results.csv",
    "MoTPE-40": f"{source_dir}/motpe-40/results.csv",
    "MoTPE (P)-40": f"{source_dir}/motpe-P-40/results.csv",
    "D-MoBO-40": f"{source_dir}/dmobo-40/results.csv",
    "D-MoBO (P)-40": f"{source_dir}/dmobo-P-40/results.csv",
    # 160 nodes -> 640 workers
    "Random-160": f"{source_dir}/random-160/results.csv",
    "NSGAII-160": f"{source_dir}/nsgaii-160/results.csv",
    "NSGAII (P)-160": f"{source_dir}/nsgaii-P-160/results.csv",
    "MoTPE-160": f"{source_dir}/motpe-160/results.csv",
    "MoTPE (P)-160": f"{source_dir}/motpe-P-160/results.csv",
    "D-MoBO-160": f"{source_dir}/dmobo-160/results.csv",
    "D-MoBO (P)-160": f"{source_dir}/dmobo-P-160/results.csv",
}


def load_csv(path):

    print(f"Loading: {path}")
    df = pd.read_csv(path)

    t0 = df["m:timestamp_start"].min()

    df["m:timestamp_end"] = df["m:timestamp_end"] - t0
    df["m:timestamp_start"] = df["m:timestamp_start"] - t0
    df = df[df["m:timestamp_end"] <= t_max]

    return df


# Loading the data
data_df = {k: load_csv(v) for k, v in data_path.items()}

color_mapping = {
    # 10
    "Random-10": "silver",
    "NSGAII-10": "coral",
    "NSGAII (P)-10": "coral",
    "MoTPE-10": "violet",
    "MoTPE (P)-10": "violet",
    "D-MoBO-10": "yellowgreen",
    "D-MoBO (P)-10": "yellowgreen",
    # 40
    "Random-40": "gray",
    "NSGAII-40": "orangered",
    "NSGAII (P)-40": "orangered",
    "MoTPE-40": "orchid",
    "MoTPE (P)-40": "orchid",
    "D-MoBO-40": "limegreen",
    "D-MoBO (P)-40": "limegreen",
    # 160
    "Random-160": "black",
    "NSGAII-160": "crimson",
    "NSGAII (P)-160": "crimson",
    "MoTPE-160": "purple",
    "MoTPE (P)-160": "purple",
    "D-MoBO-160": "seagreen",
    # "D-MoBO-160": "fuchsia",
    "D-MoBO (P)-160": "seagreen",
}

linestyle_mapping = {
    "10": ":",
    "40": "--",
    "160": "-",
}

marker_mapping = {
    "Random": "x",
    "NSGAII": "^",
    "NSGAII (P)": ">",
    "MoTPE": "s",
    "MoTPE (P)": "d",
    "D-MoBO": "h",
    "D-MoBO (P)": "o",
}

# Print failures and evaluations
for label, df in data_df.items():
    print(f"Results for {label}:")
    num_failures = len([1 for v in df["objective_0"] if v == "F"])

    print(f"\t - Number of evaluations: {len(df)}")
    print(f"\t - Number of failures: {num_failures}")

    print()

# Normalize the data
data_filtered = []

for label, df in data_df.items():
    print(label)
    df = df[df["objective_0"] != "F"].copy()
    df[objective_columns] = df[objective_columns].astype(float)
    df = df[np.isfinite(df[objective_columns].values).all(axis=1)]

    df["m:search"] = "-".join(label.split("-")[:-1])
    df["m:num_nodes"] = int(label.split("-")[-1])
    mask = non_dominated_set(-df[objective_columns].values)
    df["m:pf"] = mask

    data_filtered.append(df)

data_filtered = pd.concat(data_filtered, ignore_index=True)

tmp = data_filtered.copy()
tmp["objective_1"] = -tmp["objective_1"]
tmp["objective_2"] = -tmp["objective_2"]
tmp = tmp.rename(
    columns={
        "objective_0": "Valid. R2",
        "objective_1": "Num. Parameters",
        "objective_2": "Latency",
    }
)

### PLOT Normalization ###
scaler = cook_objective_scaler("quantile-uniform", None)
data_filtered[objective_columns] = data_filtered[objective_columns].astype(float)
data_filtered.loc[:, objective_columns] = -data_filtered[objective_columns].values

# Fit -> Transform
scaler.fit(data_filtered[objective_columns].values)
data_filtered.loc[:, objective_columns] = scaler.transform(
    data_filtered[objective_columns].values
)

# The penalty/constraint
bound = 0.85
ref_point = [1, 1, 1]
ref_point[0] = scaler.transform([[-bound, 0, 0]])[0][0]
max_volume = np.prod(ref_point)
print(f"{ref_point=}")
print(f"{max_volume=}")


### Hypervolume curves


def plot_hypervolume_vs_time(fname, labels):
    plt.figure()

    for label in labels:
        print(label)
        df = data_filtered[
            (data_filtered["m:search"] == "-".join(label.split("-")[:-1]))
            & (data_filtered["m:num_nodes"] == int(label.split("-")[-1]))
        ]
        df = df[df["m:timestamp_end"] <= t_max]
        num_total_eval = len(df)

        df = df[df["objective_0"] < ref_point[0]]
        num_valid_eval = len(df)
        print(f"  valid={num_valid_eval/num_total_eval*100:.2f}")

        # Sort by time of completion
        df = df.sort_values(by=["m:timestamp_end"], ascending=True)

        t = df["m:timestamp_end"].values.tolist()
        objectives = df[objective_columns].values

        hv = []
        for i in range(len(objectives)):
            pf = pareto_front(objectives[: i + 1])
            hv.append(hypervolume(pf, ref=ref_point))

        t.append(t_max)
        if len(hv) > 0:
            hv.append(hv[-1])
        else:
            hv.append(0)

        t = np.asarray(t)
        hv = np.asarray(hv) / max_volume

        t = np.concatenate([[0], t])
        hv = np.concatenate([[0], hv])

        print(f"  hvi={hv[-1]:.2f}")

        plabel = "-".join(label.split("-")[:-1])
        pnodes = label.split("-")[-1]

        if "scaling" in fname:
            label_ = int(pnodes) * 4  # Scaling
        else:
            label_ = plabel  # Comparing algorithms

        plt.step(
            t,
            hv,
            where="post",
            label=label_,
            color=color_mapping[label],
            linestyle=linestyle_mapping[pnodes],
            marker=marker_mapping[plabel],
            markevery=max(int(len(hv) / 5), 1),
        )

    if "scaling" in fname:
        plt.legend(title=plabel, ncols=1, fontsize=7, loc="upper left")  # Scaling
    else:
        plt.legend(ncols=1, fontsize=7)  # Comparing algorithms
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle=":")
    plt.xlabel("Time (min.)")
    plt.ylabel("HVI")

    plt.xlim(0, t_max)
    plt.ylim(0, 0.8)

    ax = plt.gca()
    ticker_freq = t_max / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(ticker_freq / 2))

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/hypervolume-vs-time-polaris-combo-{fname}.png")
    plt.close()


## Comparing all without constraints
for nodes in [10, 40, 160]:
    fname = f"all-{nodes}"
    labels = [f"Random-{nodes}", f"NSGAII-{nodes}", f"MoTPE-{nodes}", f"D-MoBO-{nodes}"]
    plot_hypervolume_vs_time(fname, labels)

## Comparing with/without constraints
for nodes in [10, 40, 160]:
    fname = f"d-mobo-constraint-{nodes}"
    labels = [f"D-MoBO-{nodes}", f"D-MoBO (P)-{nodes}"]
    plot_hypervolume_vs_time(fname, labels)

## Comparing all with constraints
for nodes in [10, 40, 160]:
    fname = f"all-constraint-{nodes}"
    labels = [
        f"Random-{nodes}",
        f"NSGAII (P)-{nodes}",
        f"MoTPE (P)-{nodes}",
        f"D-MoBO (P)-{nodes}",
    ]
    plot_hypervolume_vs_time(fname, labels)

## SCALING
fname = "random-scaling"
labels = ["Random-10", "Random-40", "Random-160"]
plot_hypervolume_vs_time(fname, labels)

fname = "d-mobo-scaling"
labels = ["D-MoBO-10", "D-MoBO-40", "D-MoBO-160"]
plot_hypervolume_vs_time(fname, labels)

fname = "d-mobo-const-scaling"
labels = ["D-MoBO (P)-10", "D-MoBO (P)-40", "D-MoBO (P)-160"]
plot_hypervolume_vs_time(fname, labels)

fname = "nsgaii-scaling"
labels = ["NSGAII-10", "NSGAII-40", "NSGAII-160"]
plot_hypervolume_vs_time(fname, labels)

fname = "nsgaii-const-scaling"
labels = ["NSGAII (P)-10", "NSGAII (P)-40", "NSGAII (P)-160"]
plot_hypervolume_vs_time(fname, labels)

fname = "motpe-scaling"
labels = ["MoTPE-10", "MoTPE-40", "MoTPE-160"]
plot_hypervolume_vs_time(fname, labels)

fname = "motpe-const-scaling"
labels = ["MoTPE (P)-10", "MoTPE (P)-40", "MoTPE (P)-160"]
plot_hypervolume_vs_time(fname, labels)


### Cumulated Regret Curves


def plot_cumulated_regret_vs_time(fname, labels):
    plt.figure()
    for label in labels:
        print(label)
        df = data_filtered[
            (data_filtered["m:search"] == "-".join(label.split("-")[:-1]))
            & (data_filtered["m:num_nodes"] == int(label.split("-")[-1]))
        ]
        df = df[df["m:timestamp_end"] <= t_max]
        df = df[df["objective_0"] < ref_point[0]]

        # Sort by time of completion
        df = df.sort_values(by=["m:timestamp_end"], ascending=True)

        t = df["m:timestamp_end"].values.tolist()
        objectives = df[objective_columns].values

        hv = []
        for i in range(len(objectives)):
            pf = pareto_front(objectives[: i + 1])
            hv.append(hypervolume(pf, ref=ref_point))

        t.append(t_max)
        if len(hv) > 0:
            hv.append(hv[-1])
        else:
            hv.append(0)

        t = np.asarray(t)
        hv = np.asarray(hv) / max_volume

        t = np.concatenate([[0], t])
        hv = np.concatenate([[0], hv])

        cum_hv = [0]
        cum_hv_aux = 0
        for i in range(0, len(hv) - 1):
            cum_hv_aux += (1 - hv[i]) * (t[i + 1] - t[i]) / t_max
            cum_hv.append(cum_hv_aux)

        print(f"final cum. regret: {cum_hv[-1]:.2f}")
        print()

        plabel = "-".join(label.split("-")[:-1])
        pnodes = label.split("-")[-1]

        if "scaling" in fname:
            label_ = int(pnodes) * 4  # Scaling
        else:
            label_ = plabel  # Comparing algorithms

        plt.plot(
            t,
            cum_hv,
            label=label_,
            color=color_mapping[label],
            linestyle=linestyle_mapping[pnodes],
            marker=marker_mapping[plabel],
            markevery=max(int(len(hv) / 5), 1),
        )

    if "scaling" in fname:
        plt.legend(title=plabel, ncols=1, fontsize=7, loc="upper left")  # Scaling
    else:
        plt.legend(ncols=1, fontsize=7)  # Comparing algorithms
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle=":")
    plt.xlabel("Time (min.)")
    plt.ylabel("Cumulated HVI Regret")

    plt.xlim(0, t_max)
    plt.ylim(0, 1.0)

    ax = plt.gca()
    ticker_freq = t_max / 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticker_freq))
    ax.xaxis.set_major_formatter(minute_major_formatter)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(ticker_freq / 2))

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/cumulated-regret-vs-time-polaris-combo-{fname}.png")
    plt.close()


## Same Scale
for nodes in [10, 40, 160]:
    fname = f"all-{nodes}"
    labels = [f"Random-{nodes}", f"NSGAII-{nodes}", f"MoTPE-{nodes}", f"D-MoBO-{nodes}"]
    plot_cumulated_regret_vs_time(fname, labels)

## Comparing with/without constraint
for nodes in [10, 40, 160]:
    fname = f"d-mobo-constraint-{nodes}"
    labels = [f"D-MoBO-{nodes}", f"D-MoBO (P)-{nodes}"]
    plot_cumulated_regret_vs_time(fname, labels)

## Comparing all with constraint
for nodes in [10, 40, 160]:
    fname = f"all-constraint-{nodes}"
    labels = [
        f"Random-{nodes}",
        f"NSGAII (P)-{nodes}",
        f"MoTPE (P)-{nodes}",
        f"D-MoBO (P)-{nodes}",
    ]
    plot_cumulated_regret_vs_time(fname, labels)

## SCALING
fname = "random-scaling"
labels = ["Random-10", "Random-40", "Random-160"]
plot_cumulated_regret_vs_time(fname, labels)

fname = "d-mobo-scaling"
labels = ["D-MoBO-10", "D-MoBO-40", "D-MoBO-160"]
plot_cumulated_regret_vs_time(fname, labels)

fname = "d-mobo-const-scaling"
labels = ["D-MoBO (P)-10", "D-MoBO (P)-40", "D-MoBO (P)-160"]
plot_cumulated_regret_vs_time(fname, labels)

fname = "nsgaii-scaling"
labels = ["NSGAII-10", "NSGAII-40", "NSGAII-160"]
plot_cumulated_regret_vs_time(fname, labels)

fname = "nsgaii-const-scaling"
labels = ["NSGAII (P)-10", "NSGAII (P)-40", "NSGAII (P)-160"]
plot_cumulated_regret_vs_time(fname, labels)

fname = "motpe-scaling"
labels = ["MoTPE-10", "MoTPE-40", "MoTPE-160"]
plot_cumulated_regret_vs_time(fname, labels)

fname = "motpe-const-scaling"
labels = ["MoTPE (P)-10", "MoTPE (P)-40", "MoTPE (P)-160"]
plot_cumulated_regret_vs_time(fname, labels)
