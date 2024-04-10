import logging
import pathlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False

import numpy as np

from deephyper.search.hps import MPIDistributedBO
from deephyper.evaluator.callback import TqdmCallback

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def execute(
    problem,
    acq_func,
    timeout,
    max_evals,
    random_state,
    verbose,
    log_dir,
    n_jobs,
    model,
    scheduler=False,
    scheduler_periode=25,
    scheduler_rate=0.1,
    filter_duplicated=False,
    objective_scaler="identity",
    scalarization="Chebyshev",
    lower_bounds=None,
    acq_func_optimizer="sampling",
    **kwargs,
):

    pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)

    path_log_file = os.path.join(log_dir, f"deephyper.{rank}.log")
    if rank == 0:
        logging.basicConfig(
            filename=path_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    rs = np.random.RandomState(random_state)

    hp_problem = problem.hp_problem
    run = problem.run

    if os.environ.get("DEEPHYPER_DB_HOST") is None:
        assert size == 1, "DEEPHYPER_DB_HOST is not set, so only one process can be used."
        callbacks = []
        if verbose:
            callbacks.append(TqdmCallback())
        evaluator = MPIDistributedBO.bootstrap_evaluator(
            run,
            evaluator_type="serial",  # one worker to evaluate the run-function per rank
            evaluator_kwargs={"callbacks": callbacks},
            storage_type="memory",
            comm=comm,
            root=0,
        )
    else:
        evaluator = MPIDistributedBO.bootstrap_evaluator(
            run,
            evaluator_type="serial",  # one worker to evaluate the run-function per rank
            storage_type="redis",
            storage_kwargs={
                "host": os.environ["DEEPHYPER_DB_HOST"],
                "port": 6379,
            },
            comm=comm,
            root=0,
        )

    logging.info("Creation of the search instance...")

    if scheduler:
        scheduler = {
            "type": "periodic-exp-decay",
            "period": scheduler_periode,
            "rate": scheduler_rate,
        }
    else:
        scheduler = None

    if lower_bounds is not None:
        lower_bounds = [
            float(lb) if lb != "None" else None for lb in lower_bounds.split(",")
        ]

    search = MPIDistributedBO(
        hp_problem,
        evaluator,
        n_jobs=n_jobs,
        log_dir=log_dir,
        random_state=rs,
        acq_func=acq_func,
        acq_optimizer=acq_func_optimizer,
        acq_optimizer_freq=1,
        surrogate_model=model,
        filter_duplicated=filter_duplicated,
        filter_failures="min",
        scheduler=scheduler,
        objective_scaler=objective_scaler,
        moo_scalarization_strategy=scalarization,
        moo_lower_bounds=lower_bounds,
        verbose=0,
    )
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(max_evals=max_evals, timeout=timeout)
    else:
        search.search(max_evals=max_evals, timeout=timeout)
    logging.info("Search is done")
