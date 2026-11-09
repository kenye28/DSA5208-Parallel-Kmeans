# DSA5208 Project 3 — Parallel K-means

C++ implementations of serial K-means, MPI K-means, hybrid MPI+OpenMP K-means, and CUDA K-means, developed and benchmarked for NUS Atlas8 and Vanda HPC systems.

## Repository contents

```text
.
├── README.md
├── .gitignore
├── atlas8/
│   ├── Makefile
│   ├── src/                  # serial, MPI, MPI+OpenMP, shared helpers
│   ├── jobs/                 # Atlas8 PBS job scripts
│   └── scripts/              # Atlas8 setup and CSV/validation helpers
└── vanda/
    ├── Makefile
    ├── src/                  # serial, MPI, MPI+OpenMP, CUDA, shared helpers
    └── jobs/                 # Vanda PBS job scripts
```

This repository contains **source code and execution scripts**. The full benchmark CSVs, figures, and final PDF report are maintained separately for coursework submission. Generated binaries, raw scheduler logs, and run outputs are not committed.

## Requirements

- C++17 compiler (`g++`)
- Open MPI (`mpicxx`, `mpiexec`)
- OpenMP support (`-fopenmp`) for the hybrid implementation
- NVIDIA CUDA toolkit and a compatible NVIDIA GPU for the CUDA implementation
- PBS Pro for the included HPC batch scripts

The recorded experiments used **Atlas8** and **Vanda**, each with its own site-specific modules and scheduler configuration. Run large experiments on allocated compute nodes, **not login nodes**.

## Build and run CPU versions

On Vanda, from the cloned repository root:

```bash
cd vanda
module load gcc/13.1.0 openmpi/gcc/64/4.1.8
make
```

On Atlas8, run `make` from `atlas8/` after preparing the environment described in `atlas8/scripts/atlas8_env.sh`.

These small examples are intended for an **allocated compute environment** where MPI is permitted:

```bash
./bin/serial --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
mpiexec -np 4 ./bin/mpi --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
OMP_NUM_THREADS=2 mpiexec -np 2 ./bin/mpi_openmp --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
```

For multi-node runs, prefer the PBS scripts, which specify allocation, MPI launch options and placement.

## Build the CUDA version (Vanda)

From `vanda/`, on an appropriate NVIDIA GPU allocation:

```bash
module load gcc/13.1.0 cuda12.4/toolkit/12.4.1
mkdir -p bin
nvcc -O3 -std=c++17 -arch=sm_86 -ccbin "$(command -v g++)" \
  -I src src/kmeans_cuda.cu -o bin/kmeans_cuda
./bin/kmeans_cuda --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
```

The `sm_86` architecture flag is for the NVIDIA A40 tested on Vanda; change it for other GPU architectures.

## Submit batch experiments

Submit from the **matching cluster subdirectory** (so that `$PBS_O_WORKDIR` resolves correctly). First compile the executables as above. Check the queue/project directives against your own HPC account before submitting.

Vanda examples:

```bash
cd vanda
qsub jobs/vanda_weak_controlled.pbs
qsub jobs/vanda_kmeans_256.pbs
qsub jobs/vanda_gpu_bench16m.pbs
```

Atlas8 example:

```bash
cd atlas8
qsub jobs/atlas8_serial_sweep_2026.pbs
```

Submit only the experiment you intend to run; the examples above are not meant to be launched all at once. Job scripts write measurement CSVs under their respective `results/` trees (which are intentionally excluded from this repository). Scripts may have site-specific resource requests and PBS project settings.

## Algorithm and reproducibility

The implementations use Lloyd's K-means algorithm: assign each point to the nearest centroid, accumulate cluster sums and counts, and update the centroids over a fixed number of iterations. MPI distributes points between ranks and uses global collective reductions for centroid updates; MPI+OpenMP also parallelizes CPU work within ranks; CUDA performs data-parallel work on a GPU.

Synthetic points are generated deterministically from the global point index and random seed (`src/common.hpp`), so an MPI rank decomposition does not change the underlying dataset. The principal benchmark setup used **8 clusters, 8 dimensions, 20 iterations, and seed 1234**. Correctness output includes clustering accuracy relative to known synthetic labels, inertia, and a deterministic assignment hash.

The timing results in the final coursework report distinguish CPU iteration time, MPI communication time, CUDA kernel time, and GPU transfer-inclusive time. These metrics should not be conflated with end-to-end application time.

## Notes

- Run results can depend on CPU node type, rank binding, memory bandwidth, network and allocation conditions.
- A verified 256-rank K-means execution is not, on its own, a matched 256-core strong- or weak-scaling efficiency measurement.
