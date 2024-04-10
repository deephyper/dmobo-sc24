#!/bin/bash

set -e

### A few variables that can be uncommented for testing
# export tasks=("navalpropulsion" "parkinsonstelemonitoring" "proteinstructure" "slicelocalization")
# export tasks=("navalpropulsion")
# export repetitions=(0 1 2 3 4 5 6 7 8 9)
# export max_evals=200


exec_search_ () {
    export log_dir="output/$task/$repetition_i/smac"

    rm -rf $log_dir && mkdir -p $log_dir

    echo "Experiment $log_dir"

    python -m dhexp.run \
        --problem "dhb_${task}" \
        --search SMAC \
        --log-dir $log_dir \
        --random-state $repetition_i \
        --max-evals $max_evals
}

for task in ${tasks[@]}; do
    for repetition_i in ${repetitions[@]}; do
        exec_search_;
    done
done