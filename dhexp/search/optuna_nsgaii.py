from ._optuna import execute_optuna


def execute(
    problem,
    timeout,
    max_evals,
    random_state,
    log_dir,
    lower_bounds=None,
    **kwargs,
):
    """Execute the NSGAII algorithm."""
    execute_optuna(
        problem=problem,
        timeout=timeout,
        max_evals=max_evals,
        random_state=random_state,
        log_dir=log_dir,
        method="NSGAII",
        lower_bounds=lower_bounds,
        **kwargs,
    )
