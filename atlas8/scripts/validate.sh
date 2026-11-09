#!/usr/bin/env bash
set -euo pipefail
# Run inside a PBS allocation with at least two MPI slots, 2 threads per rank.
mkdir -p results
args=(--points 10003 --clusters 7 --dims 3 --iterations 12 --seed 17 --print-centroids)
./bin/serial "${args[@]}" > results/validate.serial.log
mpiexec -np 2 ./bin/mpi "${args[@]}" > results/validate.mpi.log
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE mpiexec -np 2 ./bin/mpi_openmp "${args[@]}" > results/validate.hybrid.log
python3 scripts/check_results.py \
  results/validate.serial.log results/validate.mpi.log results/validate.hybrid.log
