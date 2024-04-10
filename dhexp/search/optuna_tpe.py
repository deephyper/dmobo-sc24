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
    """Execute the TPE algorithm with HB."""
    execute_optuna(
        problem=problem,
        timeout=timeout,
        max_evals=max_evals,
        random_state=random_state,
        log_dir=log_dir,
        method="TPE",
        lower_bounds=lower_bounds,
        **kwargs,
    )
