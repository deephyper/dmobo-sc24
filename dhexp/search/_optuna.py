import functools
import getpass
import logging
import pathlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

import numpy as np
import optuna

from deephyper_benchmark.search import MPIDistributedOptuna

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def execute_optuna(
    problem,
    timeout,
    max_evals,
    random_state,
    log_dir,
    method,
    lower_bounds=None,
    **kwargs,
):
    """Execute algorithms from Optuna.
    """
    hp_problem = problem.hp_problem
    run = problem.run

    pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)

    path_log_file = os.path.join(log_dir, f"deephyper.{rank}.log")
    if rank == 0:
        logging.basicConfig(
            filename=path_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )


    if os.environ.get("OPTUNA_DB_HOST") is None:
        assert size == 1, "OPTUNA_DB_HOST is not set, so only one process can be used."
        storage = None
    else:
        username = getpass.getuser()
        host = os.environ["OPTUNA_DB_HOST"]
        storage = f"postgresql://{username}@{host}:5432/hpo"

    n_objectives = int(os.environ.get("OPTUNA_N_OBJECTIVES", 1))

    logging.info(f"storage={storage}")

    if "TPE" in method:
        sampler = "TPE"
    elif "NSGAII" in method:
        sampler = "NSGAII"
    else:
        sampler = "DUMMY"

    study_name = os.path.basename(log_dir)

    if lower_bounds is not None:
        lower_bounds = [float(lb) if lb != "None" else None for lb in lower_bounds.split(",")]

    search = MPIDistributedOptuna(
        hp_problem,
        run,
        random_state=random_state,
        log_dir=log_dir,
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        comm=comm,
        n_objectives=n_objectives,
        moo_lower_bounds=lower_bounds,
        verbose=0,
    )
    results = search.search(max_evals=max_evals, timeout=timeout)

    logging.info("Search is done")
    
    MPI.Finalize()
