#!/bin/bash
#PBS -l select=160:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:10:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}


source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export problem="dhb_combo"
export search="DMOBO"
export model="DUMMY"
export acq_func="UCB"
export scheduler_periode=48
export scheduler_rate=0.1
export objective_scaler="minmaxlog"
export timeout=10800
export random_state=42
#!!! CONFIGURATION - END

export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH


export log_dir="output/random-160"
mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server $REDIS_CONF &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 5

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    ../set_affinity_gpu_polaris.sh python -m dhexp.run --problem $problem \
    --search $search \
    --model $model \
    --acq-func $acq_func \
    --objective-scaler $objective_scaler \
    --scheduler 1 \
    --scheduler-periode $scheduler_periode \
    --scheduler-rate $scheduler_rate \
    --random-state $random_state \
    --log-dir $log_dir \
    --timeout $timeout \
    --filter-duplicated 1
