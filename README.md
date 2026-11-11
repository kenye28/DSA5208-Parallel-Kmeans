# DSA5208 Project 3 — Reproducing the K-means Experiments

This repository provides the C++ source and PBS scripts for serial, MPI, hybrid MPI+OpenMP, and CUDA K-means. Experiments were run on the NUS **Atlas8** and **Vanda** HPC systems.

## 1. Requirements and setup

- Access to Atlas8 or Vanda and the PBS job scheduler (`qsub`, `qstat`).
- C++17 compiler, Open MPI, and OpenMP; Vanda CUDA experiments additionally require the CUDA 12.4 toolkit and an NVIDIA GPU (A40 for the reported experiments).
- The PBS resource and project directives must be valid for your HPC account.

Clone the project on the **HPC system** where you intend to run the jobs:

```bash
git clone https://github.com/kenye28/DSA5208-Parallel-Kmeans.git
cd DSA5208-Parallel-Kmeans
```

Compile and submit from the respective `atlas8/` or `vanda/` directory. **Do not run large benchmarks on login nodes.**

## 2. Build the executables

### Atlas8 — CPU implementations

```bash
cd atlas8
source scripts/atlas8_env.sh
make
```

### Vanda — CPU implementations

```bash
cd vanda
module load gcc/13.1.0
module load openmpi/gcc/64/4.1.8
make
```

### Vanda — CUDA implementation

```bash
module load gcc/13.1.0
module load cuda12.4/toolkit/12.4.1
mkdir -p bin
nvcc -O3 -std=c++17 -arch=sm_86 \
  -ccbin "$(command -v g++)" -I src \
  src/kmeans_cuda.cu -o bin/kmeans_cuda
```

## 3. Reproduce Atlas8 experiments

From the **`atlas8/`** directory, after running `make`:

```bash
# Correctness validation
qsub jobs/atlas8_validation.pbs

# Serial runtime versus dataset size
qsub jobs/atlas8_serial_sweep_2026.pbs

# MPI strong scaling: fixed 16,000,000 points
qsub -l select=1:ncpus=12:mpiprocs=12 \
  -v MODE=strong,RANKS=12 jobs/atlas8_mpi_scaling.pbs

# MPI weak scaling: 800,000 points per rank
qsub -l select=1:ncpus=12:mpiprocs=12 \
  -v MODE=weak,RANKS=12 jobs/atlas8_mpi_scaling.pbs

# MPI versus MPI+OpenMP, with 12 total workers
qsub -l select=1:ncpus=12:mpiprocs=12 \
  jobs/atlas8_hybrid_compare16m_2026.pbs
```

For other MPI rank counts, set `RANKS` and request enough PBS slots.
## 4. Reproduce Vanda experiments

From the **`vanda/`** directory, after compiling the required executable(s):

```bash
# MPI strong scaling: fixed 64,000,000 points
qsub jobs/vanda_scaling_64.pbs        # 64 ranks; 1 node, Baseline for both strong and weak scaling
qsub jobs/vanda_scaling_128.pbs       # 128 ranks; 2 nodes
qsub jobs/vanda_scaling_256.pbs       # 256 ranks; 4 nodes

# MPI weak scaling: 1,000,000 points per rank
qsub jobs/vanda_weak_scaling_128.pbs  # 128 ranks; 128,000,000 points
qsub jobs/vanda_kmeans_256.pbs        # 256 ranks; 256,000,000 points

# CUDA correctness and CPU-vs-GPU runtime comparison
qsub jobs/vanda_gpu_validate.pbs
qsub jobs/vanda_gpu_bench16m.pbs
```

Submit jobs individually, taking into account queue limits and cluster availability. The GPU jobs require `bin/kmeans_cuda`; the MPI jobs require `bin/mpi`.

## 5. Experiment output
Find the resulting CSV files from the corresponding `atlas8/` or `vanda/` directory:

```bash
find results -type f -name 'results.csv' | sort
```

For example, Vanda scaling jobs write to:

```text
results/vanda/strong_scaling/64_<JOB_ID>/results.csv
results/vanda/strong_scaling/128_<JOB_ID>/results.csv
results/vanda/strong_scaling/256_<JOB_ID>/results.csv
results/vanda/weak_scaling/128_<JOB_ID>/results.csv
results/vanda/256core/<JOB_ID>/results.csv
```
