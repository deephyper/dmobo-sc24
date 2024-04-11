# Parallel Multi-Objective Bayesian Hyperparameter Optimization with Normalized and Bounded Objectives

Experimental repository for the paper on "Parallel Multi-Objective Bayesian Hyperparameter Optimization with Normalized and Bounded Objectives".


## Installation

The installation requires Miniconda or Anaconda. See [Miniconda Documentation](https://docs.anaconda.com/free/miniconda/miniconda-install/).

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
../install/local.sh
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

## Large-scale experiments on Polaris

This section explains how to reproduce experiments run on Polaris on the Combo benchmark. An example installation script is provided for Polaris at the Argonne Leadership Computing Facility. This script can be used as an example for other HPC systems.

From a login node of Polaris and the root of this repository, the following commands can be run to install:

```console
mkdir build && cd build/
../install/polaris.sh
```

Once the installation is complete go to Polaris experimental directory:

```console
cd experiments/polaris/jobs/
```

From this directory, each job can be submitted with the command `qsub some-job-script.sh`. For example, if we want to submit D-MoBO with 10 nodes we run `qsub dmobo-10.sh`. Similarly, if we want to submit NSGAII with enforced constraints/penalty with 40 nodes we run `qsub nsgaii-P-40.sh`.

Once the experiments are completed the graphs can be created from `experiments/polaris` by running:

```console
source ../../activate-dhenv.sh
python plot.py
```

The figures will then appear in the `experiments/polaris/figures` folder.