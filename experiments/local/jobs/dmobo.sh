#!/bin/bash

set -e

# export tasks=("navalpropulsion" "parkinsonstelemonitoring" "proteinstructure" "slicelocalization")
export tasks=("navalpropulsion")
export repetitions=(0 1 2 3 4 5 6 7 8 9)
export max_evals=200
export scalarization="Linear"
export objective_scaler="quantile-uniform"




exec_search_ () {
    export log_dir="output/$task/$repetition_i/dmobo-$scalarization-$objective_scaler"

    rm -rf $log_dir && mkdir -p $log_dir

    echo "Experiment $log_dir"

    python -m dhexp.run \
        --problem "dhb_${task}" \
        --search DMOBO \
        --model ET \
        --acq-func UCBd \
        --acq-func-optimizer sampling \
        --objective-scaler $objective_scaler \
        --scalarization $scalarization \
        --log-dir $log_dir \
        --max-evals $max_evals \
        --random-state $repetition_i \
        --verbose 1
}

for task in ${tasks[@]}; do
    for repetition_i in ${repetitions[@]}; do
        exec_search_;
    done
done