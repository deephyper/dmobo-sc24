#!/bin/bash

export tasks=("navalpropulsion" "parkinsonstelemonitoring" "proteinstructure" "slicelocalization")
export repetitions=(0 1 2 3 4 5 6 7 8 9)
export max_evals=200

./random.sh

./nsgaii.sh

# TODO: setup variables for different variants of DMOBO
./dmobo.sh