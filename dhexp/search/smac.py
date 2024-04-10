from deephyper_benchmark.search import SMAC


def execute(
    problem,
    timeout,
    max_evals,
    random_state,
    log_dir,
    lower_bounds=None,
    **kwargs,
):
    """Execute the SMAC algorithm."""

    SMAC(
        problem.hp_problem,
        problem.run,
        random_state=random_state,
        log_dir=log_dir,
        acq_func="EI",
        n_objectives=2,
        verbose=0,
    ).search(max_evals=max_evals)
