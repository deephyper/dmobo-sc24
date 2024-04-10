# Parallel Multi-Objective Bayesian Hyperparameter Optimization with Normalized and Bounded Objectives

Experimental repository for the paper on "Parallel Multi-Objective Bayesian Hyperparameter Optimization with Normalized and Bounded Objectives".


## Installation

### Local

From the root of this repository:

```console
mkdir build && cd build/
./install/local.sh
```

### Polaris (ALCF)

From the root of this repository:

```console
mkdir build && cd build/
../install.polaris.sh
```

## Easy local reproducible examples

Experiments from Figures 3 and 4 are easy to reproduce locally by running the following commands after following the [Local Installation](#local).

```console
cd experiments/local/jobs/
./run-all.sh
```

The produced outputs are placed in the `experiments/local/output/` folder.
Then the plotting of the figures can be done with:

```console
cd ..
python plot.py
```

The produced figures are placed in the `experiments/local/figures/` folder.
