# DSA5208 Project 3 — Parallel K-means (Atlas8 + Vanda)

## What is included

- **Source code:** serial C++, MPI, hybrid MPI/OpenMP on both HPC systems, plus Vanda CUDA.
- **Execution scripts:** all 16 original PBS scripts, and the five Atlas8 helper scripts
  required by those scripts for environment setup, CSV parsing, and validation.

## Quick-start: build and run

Requirements: C++17 compiler, Open MPI (with `mpicxx` and `mpiexec`), GNU OpenMP support for the hybrid binary, and NVIDIA CUDA toolkit + GPU only for the CUDA variant. Compute-intensive runs belong in PBS jobs on the HPC clusters, **not login nodes**.

For a small correctness-oriented example, from the **repository root**:

```bash
cd vanda                         # or atlas8 for its CPU variants
module load gcc/13.1.0 openmpi/gcc/64/4.1.8   # Vanda environment
make
./bin/serial --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
mpiexec -np 4 ./bin/mpi --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
OMP_NUM_THREADS=2 mpiexec -np 2 ./bin/mpi_openmp --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
```

The MPI commands above are **small examples**; production multi-node runs use the supplied PBS scripts and MPI prefix/binding options. To compile the Vanda CUDA implementation on an NVIDIA A40-equipped node:

```bash
module load gcc/13.1.0 cuda12.4/toolkit/12.4.1
mkdir -p bin
nvcc -O3 -std=c++17 -arch=sm_86 -ccbin "$(command -v g++)" -I src src/kmeans_cuda.cu -o bin/kmeans_cuda
./bin/kmeans_cuda --points 200000 --clusters 8 --dims 8 --iterations 20 --seed 1234
```

To rerun existing PBS configurations, **from the platform subdirectory** on the matching cluster, use for example:

```bash
# On Vanda, from repository_root/vanda/ (only when PBS is available):
qsub jobs/vanda_weak_controlled.pbs
qsub jobs/vanda_gpu_bench16m.pbs
# On Atlas8, from repository_root/atlas8/:
qsub jobs/atlas8_serial_sweep_2026.pbs
```

These may require site-specific PBS project permission and node allocation; do not run Atlas8 scripts on Vanda or vice versa. Original results remain under `atlas8/results/` and `vanda/results/`. Do not overwrite them when rerunning benchmarks.

## Rebuild CPU and CUDA binaries

On either HPC system, enter the platform directory and run `make`, ensuring
`g++`, `mpicxx`, and OpenMP support are available through the appropriate modules.
The PBS scripts preserve the validated site-specific module and MPI commands.

For Vanda CUDA on an NVIDIA A40 GPU allocation:

```bash
cd vanda
module load gcc/13.1.0 cuda12.4/toolkit/12.4.1
mkdir -p bin
nvcc -O3 -std=c++17 -arch=sm_86 -ccbin "$(command -v g++)" -I src src/kmeans_cuda.cu -o bin/kmeans_cuda
```
