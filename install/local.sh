#!/bin/bash

conda create -n dhenv-sc24 python=3.11 -y
conda activate oneEpoch

# Clone DeepHyper 0.7.0
git clone --depth 1 --branch 0.7.0 https://github.com/deephyper/deephyper.git

# Install DeepHyper with MPI and Redis backends
pip install -e "deephyper/[hps]"

# Install Benchmarks
git clone --depth 1 --branch 0.0.1 https://github.com/deephyper/benchmark.git deephyper-benchmark
pip install -e "deephyper-benchmark/"

# Install the dhexp package
pip install -e ".."

# To install the `HPOBench/tabular` (the 4 regression problems) with `deephyper_benchmark`
python -c "import deephyper_benchmark as dhb; dhb.install('HPOBench/tabular');"
