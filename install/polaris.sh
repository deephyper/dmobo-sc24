#!/bin/bash

set -xe

module load llvm
module load conda/2022-09-08

conda create -p dhenv --clone base -y
conda activate dhenv/
pip install --upgrade pip

# Install Spack
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh

git clone --depth 1 --branch 0.1.0 https://github.com/deephyper/deephyper-spack-packages.git

# Install RedisJson With Spack
spack env create redisjson
spack env activate redisjson
spack repo add deephyper-spack-packages
spack add redisjson
spack install

# Clone DeepHyper 0.7.0
git clone --depth 1 --branch 0.7.0 https://github.com/deephyper/deephyper.git

# Install DeepHyper with MPI and Redis backends
pip install -e "deephyper/[hps,mpi,redis]"

# Install Benchmarks
git clone --depth 1 --branch 0.0.1 https://github.com/deephyper/benchmark.git deephyper-benchmark
pip install -e "deephyper-benchmark/"

# Install the dhexp package
pip install -e ".."

# Copy activation of environment file
cp ../install/env/polaris.sh activate-dhenv.sh
echo "" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh

# Activate Spack env
echo "" >> activate-dhenv.sh
echo ". $PWD/spack/share/spack/setup-env.sh" >> activate-dhenv.sh
echo "spack env activate redisjson" >> activate-dhenv.sh

# Redis Configuration
cp ../install/env/redis.conf redis.confg
cat $(spack find --path redisjson | grep -o "/.*/redisjson.*")/redis.conf >> redis.conf

# Install the Combo Benchmark
python -c "from deephyper_benchmark import *; install('ECP-Candle/Pilot1/Combo');"