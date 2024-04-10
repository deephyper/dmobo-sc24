import os
import pathlib
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deephyper.analysis._matplotlib import figure_size, update_matplotlib_rc

update_matplotlib_rc()
figsize = figure_size(252 * 1.8, 1.0)

from tqdm.notebook import tqdm

from deephyper.skopt.moo import pareto_front, hypervolume
from deephyper.skopt.utils import cook_objective_scaler
from deephyper.analysis import rank

source_dir = "jobs/output"
figures_dir = "figures"
pathlib.Path(figures_dir).mkdir(parents=False, exist_ok=True)

tasks = [
    "navalpropulsion",
    # "parkinsonstelemonitoring",
    # "proteinstructure",
    # "slicelocalization",
]
objective_columns = ["objective_0", "objective_1"]
n_objectives = 2
scalers = [
    # "identity",
    # "minmaxlog",
    "quantile-uniform",
]

strategies = [
    "Linear",
    # "Chebyshev",
    # "PBI",
]
repetitions = list(range(10))

other_results = [
    "random",
    "nsgaii",
    # "motpe",
    # "smac",
    # "botorch"
]


def load_data_from_task(task_name, other_results=None):
    """
    New colums are:
    - strategy: the name of the strategy.
    - scaler: the name of the scaler.
    - task: the name of the task.
    """

    df_results = []

    for scaler, strategy in itertools.product(scalers, strategies):

        for i in repetitions:
            path = f"{source_dir}/{task_name}/{i}/dmobo-{strategy}-{scaler}/results.csv"

            if not (os.path.exists(path)):
                print(f"Skipping: {path} because not found.")
                continue

            df = pd.read_csv(path)

            if len(df) < 200:
                print(f"Skipping: {path} because imcomplete.")
                continue

            df["repetition"] = i
            df["strategy"] = f"{strategy}"
            df["scaler"] = scaler
            df_results.append(df)

    if other_results is None:
        other_results = []
    for label in other_results:

        for i in repetitions:
            path = f"{source_dir}/{task_name}/{i}/{label}/results.csv"

            if not (os.path.exists(path)):
                print(f"Skipping: {path}")
                continue

            df = pd.read_csv(path)

            if len(df) < 200:
                print(f"Skipping: {path} because imcomplete.")
                continue

            df["repetition"] = i
            df["strategy"] = label
            df["scaler"] = "identity"
            df_results.append(df)

    df = pd.concat(df_results, ignore_index=True)
    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    except:
        pass
    df["task"] = task_name

    return df


scalers_mapping = {"identity": "Id", "minmaxlog": "MML", "quantile-uniform": "QU"}
# scalers = {k: v for k, v in scalers_mapping.items() if k in scalers}

for scaler_key, scaler_label in scalers_mapping.items():

    print(scaler_label)

    # Load data
    df = pd.concat(
        [load_data_from_task(task_name) for task_name in tasks], ignore_index=True
    )

    # Scale data
    scaler = cook_objective_scaler(scaler_key, None)
    for (task_name,), group_df in df.groupby(["task"]):
        df.loc[group_df.index, objective_columns] = scaler.fit_transform(
            -group_df[objective_columns].values
        )

    for (task_name,), group_df in df.groupby(["task"]):
        objectives = group_df[objective_columns].values
        pf = pareto_front(objectives, sort=True)
        ref = np.max(objectives, axis=0)
        hv_x = pf[:, 0].tolist()
        hv_y = pf[:, 1].tolist()
        hv_x = [ref[0]] + [hv_x[0]] + hv_x + [ref[0]] + [ref[0]]
        hv_y = [ref[1]] + [ref[1]] + hv_y + [hv_y[-1]] + [ref[1]]
        plt.figure()
        plt.scatter(
            objectives[:, 0],
            objectives[:, 1],
            color="royalblue",
            s=5,
            alpha=0.1,
            label="Samples",
        )
        plt.plot(pf[:, 0], pf[:, 1], color="crimson", linewidth=2, label="Pareto-Front")
        plt.fill(
            hv_x,
            hv_y,
            facecolor="gray",
            edgecolor="none",
            alpha=0.25,
            linewidth=2,
            label="Hypervolume",
        )
        plt.xlabel(r"$y_1$ (Validation Error)")
        plt.ylabel(r"$y_2$ (Training Time)")

        # handles, labels =  plt.gca().get_legend_handles_labels()
        # handle_scatter = mpatches.Patch(color=handles[0].get_facecolor(), label=handles[0].get_label())
        # handles[0] = handle_scatter

        legend = plt.legend(loc="upper right")
        legend.legend_handles[0].set_sizes([20])
        legend.legend_handles[0].set_alpha(1)

        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/scaler_{scaler_key}_{task_name}.png", dpi=300)
        plt.close()

# Scale objectives to be uniformly distributed in [0, 1]

# Load data
df = pd.concat(
    [load_data_from_task(task_name, other_results=other_results) for task_name in tasks], ignore_index=True
)

scaler = cook_objective_scaler("quantile-uniform", None)
for (task_name,), group_df in df.groupby(["task"]):
    df.loc[group_df.index, objective_columns] = scaler.fit_transform(
        -group_df[objective_columns].values
    )


def hypervolume_curve(y, ref_point):
    assert np.shape(y)[1] == np.shape(ref_point)[0]

    hv = []
    for i in range(len(y)):
        pf = pareto_front(y[: i + 1])
        hv.append(hypervolume(pf, ref=ref_point))
    return hv


# Compute the hypervolume curve
df["hypervolume"] = None
for group_values, group_df in tqdm(
    df.groupby(["task", "strategy", "scaler", "repetition"])
):
    hv = hypervolume_curve(
        group_df[objective_columns].values, ref_point=[1.0 for _ in range(n_objectives)]
    )
    df.loc[group_df.index, "hypervolume"] = hv


scaler_linestyle = {
    "identity": "-",
    "minmaxlog": ":",
    "quantile-uniform": "--",
    "log": (5, (10, 3)),
}

strategy_color = {
    "Linear": "seagreen",
    "Chebyshev": "orange",
    "PBI": "royalblue",
    "Linear_mf": "lime",
    "Chebyshev_mf": "yellow",
    "PBI_mf": "cyan",
    "nsgaii": "crimson",
    "random": "black",
    "motpe": "pink",
    "smac": "cyan",
    "botorch": "lime",
}

label_mapping = {
    "Linear": "L",
    "Chebyshev": "CH",
    "PBI": "PBI",
    "nsgaii": "NSGAII (Optuna)",
    "random": "Random",
    "motpe": "MoTPE (Optuna)",
    "smac": "ParEGO (SMAC)",
    "botorch": "qEHVI (BoTorch)",
    "identity": "Id",
    "minmaxlog": "MML",
    "quantile-uniform": "QU",
}


### Plot for DMOBO
bbox_to_anchor = (1.21, 1.025)
postfix = "_dmobo"

filter = df["scaler"].isin(scalers) & df["strategy"].isin(strategies)

task_rankings = []
task_scores = []
for labels, group_df in df[filter].groupby(["task", "repetition"]):
    group_labels = []
    group_hv = []
    for gv, gdf in group_df.groupby(["strategy", "scaler"]):
        group_labels.append("-".join(gv))
        group_hv.append(gdf["hypervolume"].values)

    group_hv = np.array(group_hv)

    # Simple ranking which does not take into account ties given a tolerance
    # ranks = group_hv.shape[0] - np.argsort(group_hv, axis=0)

    # Ranking which takes into account ties given a tolerance
    ranks = np.zeros_like(group_hv).astype(int)
    for i in range(group_hv.shape[1]):
        r = group_hv.shape[0] - rank(group_hv[:, i], decimals=5) + 1
        ranks[:, i] = r

    task_scores.append(group_hv)
    task_rankings.append(ranks)

task_rankings = np.array(task_rankings).astype(float)
task_scores = np.array(task_scores).astype(float)

conf = 1.96
n = task_rankings.shape[0]

average_rankings = np.mean(task_rankings, axis=0)
stde_rankings = conf * np.std(task_rankings, axis=0) / np.sqrt(n)

average_scores = np.mean(task_scores, axis=0)
stde_scores = conf * np.std(task_scores, axis=0) / np.sqrt(n)

fig = plt.figure()
for i, label in enumerate(group_labels):
    flabel = label
    if (
        "nsgaii" in label
        or "random" in label
        or "motpe" in label
        or "smac" in label
        or "botorch" in label
    ):
        flabel = flabel.replace(
            flabel.split("-")[0], label_mapping[flabel.split("-")[0]]
        )
        flabel = flabel.split("-")[0]
    else:
        flabel = flabel.replace(
            flabel.split("-")[0],
            label_mapping.get(flabel.split("-")[0], flabel.split("-")[0]),
        )
        flabel = flabel.replace(
            flabel[flabel.index("-") + 1 :],
            label_mapping[flabel[flabel.index("-") + 1 :]],
        )
        if flabel == "L-QU":
            flabel = "L-QU (D-MoBO)"

    x = np.arange(len(average_rankings[i])) + 1
    plt.plot(
        x,
        average_rankings[i],
        linestyle=scaler_linestyle.get("-".join(label.split("-")[1:]), "-"),
        color=strategy_color.get(label.split("-")[0]),
        label=flabel,
    )
    plt.fill_between(
        x,
        average_rankings[i] - stde_rankings[i],
        average_rankings[i] + stde_rankings[i],
        alpha=0.1,
        color=strategy_color.get(label.split("-")[0]),
    )
plt.xlabel("Evaluations")
plt.ylabel("Ranking")
fig.legend(ncols=1, bbox_to_anchor=bbox_to_anchor, fontsize=7)
plt.grid()
plt.xlim(1, average_rankings.shape[1])
plt.tight_layout()
plt.savefig(
    f"{figures_dir}/average_ranking{postfix}.png",
    bbox_inches="tight",
    dpi=360,
)
plt.close()

fig = plt.figure()

for i, label in enumerate(group_labels):
    flabel = label
    if (
        "nsgaii" in label
        or "random" in label
        or "motpe" in label
        or "smac" in label
        or "botorch" in label
    ):
        flabel = flabel.replace(
            flabel.split("-")[0], label_mapping[flabel.split("-")[0]]
        )
        flabel = flabel.split("-")[0]
    else:
        flabel = flabel.replace(
            flabel.split("-")[0],
            label_mapping.get(flabel.split("-")[0], flabel.split("-")[0]),
        )
        flabel = flabel.replace(
            flabel[flabel.index("-") + 1 :],
            label_mapping[flabel[flabel.index("-") + 1 :]],
        )
        if flabel == "L-QU":
            flabel = "L-QU (D-MoBO)"

    x = np.arange(len(average_scores[i])) + 1
    plt.plot(
        x,
        average_scores[i],
        linestyle=scaler_linestyle.get("-".join(label.split("-")[1:]), "-"),
        color=strategy_color.get(label.split("-")[0]),
        label=flabel,
    )
    plt.fill_between(
        x,
        average_scores[i] - stde_scores[i],
        average_scores[i] + stde_scores[i],
        alpha=0.1,
        color=strategy_color.get(label.split("-")[0]),
    )
plt.ylim(0, 1)
plt.xlabel("Evaluations")
plt.ylabel("HVI")
# plt.legend(ncols=3, loc="lower right", fontsize=7)
fig.legend(ncols=1, bbox_to_anchor=bbox_to_anchor, fontsize=7)
plt.grid(True, which="major")
plt.grid(True, which="minor", linestyle=":")
plt.xlim(1, average_rankings.shape[1])
plt.tight_layout()
plt.savefig(
    f"{figures_dir}/average_hypervolume{postfix}.png",
    bbox_inches="tight",
    dpi=360,
)
plt.close()

### Plot to compare with other algorithms
bbox_to_anchor = (1.28, 0.8)
postfix = "_algorithms"

filter = (
    df["scaler"].isin(["quantile-uniform"]) & df["strategy"].isin(["Linear"])
) | df["strategy"].isin(other_results)

task_rankings = []
task_scores = []
for labels, group_df in df[filter].groupby(["task", "repetition"]):
    group_labels = []
    group_hv = []
    for gv, gdf in group_df.groupby(["strategy", "scaler"]):
        group_labels.append("-".join(gv))
        group_hv.append(gdf["hypervolume"].values)

    group_hv = np.array(group_hv)

    # Simple ranking which does not take into account ties given a tolerance
    # ranks = group_hv.shape[0] - np.argsort(group_hv, axis=0)

    # Ranking which takes into account ties given a tolerance
    ranks = np.zeros_like(group_hv).astype(int)
    for i in range(group_hv.shape[1]):
        r = group_hv.shape[0] - rank(group_hv[:, i], decimals=5) + 1
        ranks[:, i] = r

    task_scores.append(group_hv)
    task_rankings.append(ranks)

task_rankings = np.array(task_rankings).astype(float)
task_scores = np.array(task_scores).astype(float)

conf = 1.96
n = task_rankings.shape[0]

average_rankings = np.mean(task_rankings, axis=0)
stde_rankings = conf * np.std(task_rankings, axis=0) / np.sqrt(n)

average_scores = np.mean(task_scores, axis=0)
stde_scores = conf * np.std(task_scores, axis=0) / np.sqrt(n)

fig = plt.figure()
for i, label in enumerate(group_labels):
    flabel = label
    if (
        "nsgaii" in label
        or "random" in label
        or "motpe" in label
        or "smac" in label
        or "botorch" in label
    ):
        flabel = flabel.replace(
            flabel.split("-")[0], label_mapping[flabel.split("-")[0]]
        )
        flabel = flabel.split("-")[0]
    else:
        flabel = flabel.replace(
            flabel.split("-")[0],
            label_mapping.get(flabel.split("-")[0], flabel.split("-")[0]),
        )
        flabel = flabel.replace(
            flabel[flabel.index("-") + 1 :],
            label_mapping[flabel[flabel.index("-") + 1 :]],
        )
        if flabel == "L-QU":
            flabel = "L-QU (D-MoBO)"

    x = np.arange(len(average_rankings[i])) + 1
    plt.plot(
        x,
        average_rankings[i],
        linestyle=scaler_linestyle.get("-".join(label.split("-")[1:]), "-"),
        color=strategy_color.get(label.split("-")[0]),
        label=flabel,
    )
    plt.fill_between(
        x,
        average_rankings[i] - stde_rankings[i],
        average_rankings[i] + stde_rankings[i],
        alpha=0.1,
        color=strategy_color.get(label.split("-")[0]),
    )
plt.xlabel("Evaluations")
plt.ylabel("Ranking")
# fig.legend(ncols=3, bbox_to_anchor=(0.92, 1.3), fontsize=7)
# fig.legend(ncols=2, bbox_to_anchor=(1.45, 0.8), fontsize=7)
fig.legend(ncols=1, bbox_to_anchor=bbox_to_anchor, fontsize=7)
plt.grid()
plt.xlim(1, average_rankings.shape[1])
plt.tight_layout()
plt.savefig(
    f"{figures_dir}/average_ranking{postfix}.png",
    bbox_inches="tight",
    dpi=360,
)
plt.close()

fig = plt.figure()

for i, label in enumerate(group_labels):
    flabel = label
    if (
        "nsgaii" in label
        or "random" in label
        or "motpe" in label
        or "smac" in label
        or "botorch" in label
    ):
        flabel = flabel.replace(
            flabel.split("-")[0], label_mapping[flabel.split("-")[0]]
        )
        flabel = flabel.split("-")[0]
    else:
        flabel = flabel.replace(
            flabel.split("-")[0],
            label_mapping.get(flabel.split("-")[0], flabel.split("-")[0]),
        )
        flabel = flabel.replace(
            flabel[flabel.index("-") + 1 :],
            label_mapping[flabel[flabel.index("-") + 1 :]],
        )
        if flabel == "L-QU":
            flabel = "L-QU (D-MoBO)"

    x = np.arange(len(average_scores[i])) + 1
    plt.plot(
        x,
        average_scores[i],
        linestyle=scaler_linestyle.get("-".join(label.split("-")[1:]), "-"),
        color=strategy_color.get(label.split("-")[0]),
        label=flabel,
    )
    plt.fill_between(
        x,
        average_scores[i] - stde_scores[i],
        average_scores[i] + stde_scores[i],
        alpha=0.1,
        color=strategy_color.get(label.split("-")[0]),
    )
plt.ylim(0, 1)
plt.xlabel("Evaluations")
plt.ylabel("HVI")
# plt.legend(ncols=3, loc="lower right", fontsize=7)
fig.legend(ncols=1, bbox_to_anchor=bbox_to_anchor, fontsize=7)
plt.grid(True, which="major")
plt.grid(True, which="minor", linestyle=":")
plt.xlim(1, average_rankings.shape[1])
plt.tight_layout()
plt.savefig(
    f"{figures_dir}/average_hypervolume{postfix}.png",
    bbox_inches="tight",
    dpi=360,
)
plt.close()
