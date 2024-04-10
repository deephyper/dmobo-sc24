#!/bin/bash

export tasks=("navalpropulsion" "parkinsonstelemonitoring" "proteinstructure" "slicelocalization")
export repetitions=(0 1 2 3 4 5 6 7 8 9)
export max_evals=200

./random.sh

./nsgaii.sh

./motpe.sh

# TODO: setup variables for different variants of DMOBO
# D-MOBO with Linear scalarization
export scalarization="Linear"
export objective_scaler="identity"
./dmobo.sh

export objective_scaler="minmaxlog"
./dmobo.sh

export objective_scaler="quantile-uniform"
./dmobo.sh

# D-MOBO with Chebyshev scalarization
export scalarization="Chebyshev"
export objective_scaler="identity"
./dmobo.sh

export objective_scaler="minmaxlog"
./dmobo.sh

export objective_scaler="quantile-uniform"
./dmobo.sh

# D-MOBO with Chebyshev scalarization
export scalarization="PBI"
export objective_scaler="identity"
./dmobo.sh

export objective_scaler="minmaxlog"
./dmobo.sh

export objective_scaler="quantile-uniform"
./dmobo.sh

./smac.sh

./botorch.sh