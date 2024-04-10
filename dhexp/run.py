import warnings

warnings.simplefilter("ignore")

import argparse
import pathlib
import importlib
import sys

PROBLEMS = {
    "dhb_combo": "dhexp.benchmark.dhb_combo",
    "dhb_slicelocalization": "dhexp.benchmark.dhb_slicelocalization",
    "dhb_navalpropulsion": "dhexp.benchmark.dhb_navalpropulsion",
    "dhb_proteinstructure": "dhexp.benchmark.dhb_proteinstructure",
    "dhb_parkinsonstelemonitoring": "dhexp.benchmark.dhb_parkinsonstelemonitoring",
}

SEARCHES = {
    "DMOBO": "dhexp.search.dbo",
    "MOTPE": "dhexp.search.optuna_tpe",
    "NSGAII": "dhexp.search.optuna_nsgaii",
}


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--problem",
        type=str,
        choices=list(PROBLEMS.keys()),
        required=True,
        help="Problem on which to experiment.",
    )
    parser.add_argument(
        "--search",
        type=str,
        choices=list(SEARCHES.keys()),
        required=True,
        help="Search the experiment must be done with.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["RF", "GP", "DUMMY", "ET"],
        default=None,
        help="Surrogate model used by the Bayesian optimizer.",
    )
    parser.add_argument(
        "--acq-func",
        type=str,
        default="UCB",
        help="Acquisition funciton to use.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="cl_max",
        help="The strategy for multi-point acquisition.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Search maximum duration (in min.) for each optimization.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=-1,
        help="Number of iterations to run for each optimization.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Control the random-state of the algorithm.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="output",
        help="Logging directory to store produced outputs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        help="Wether to activate or not the verbose mode.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="The number of parallel processes to use to fit the surrogate model.",
    )
    parser.add_argument(
        "--scheduler",
        type=bool,
        default=False,
    )
    parser.add_argument("--scheduler-periode", type=int, default=25)
    parser.add_argument("--scheduler-rate", type=float, default=0.1)
    parser.add_argument(
        "--filter-duplicated",
        type=bool,
        default=False,
    )
    parser.add_argument("--objective-scaler", type=str, default="identity")
    parser.add_argument("--scalarization", type=str, default="Chebyshev")
    parser.add_argument("--lower-bounds", type=str, default=None)
    parser.add_argument("--acq-func-optimizer", type=str, default="sampling")
    return parser


def main(args):
    args = vars(args)

    # load the problem
    args["problem"] = importlib.import_module(PROBLEMS.get(args["problem"]))
    search = importlib.import_module(SEARCHES.get(args.pop("search")))

    pathlib.Path(args["log_dir"]).mkdir(parents=True, exist_ok=True)

    search.execute(**args)


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    # delete arguments to avoid conflicts
    sys.argv = [sys.argv[0]]

    main(args)
