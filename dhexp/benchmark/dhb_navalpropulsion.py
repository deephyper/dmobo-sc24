import os

os.environ["DEEPHYPER_BENCHMARK_TASK"] = "navalpropulsion"
os.environ["DEEPHYPER_BENCHMARK_MOO"] = "1"
os.environ["OPTUNA_N_OBJECTIVES"] = "2"

import deephyper_benchmark as dhb

dhb.load("HPOBench/tabular")

from deephyper_benchmark.lib.hpobench.tabular import hpo

hp_problem = hpo.problem
run = hpo.run
