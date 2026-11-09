# DSA5208 Project 3 — Parallel K-means (Atlas8 + Vanda)

**GitHub repository:** https://github.com/kenye28/DSA5208-Parallel-Kmeans

This is the **merged, submission-focused project**. It combines the earlier cleaned
Atlas8/Vanda backup with the newly downloaded Vanda essential archive.
Original uploaded archives are unchanged.

## What is included

- **One editable report:** `Kmeans_Final_Report.tex` (five figures embedded by reference from `figures/*.pdf`).
- **Source code:** serial C++, MPI, hybrid MPI/OpenMP on both HPC systems, plus Vanda CUDA.
- **Execution scripts:** all 16 original PBS scripts, and the five Atlas8 helper scripts
  required by those scripts for environment setup, CSV parsing, and validation.
- **Measurements:** all preserved CSVs, including historical run series, the Vanda
  256-rank/4-node original `results.csv`, and a cross-experiment `analysis/benchmark_summary.csv`.
- **Figures:** five vector-PDF charts; regenerate from measurements with `scripts/generate_plots.py`.

No compiled executables, intermediate LaTeX files, scheduler output, terminal logs,
PBS-generated `*.log` files, or duplicate report versions are included.

## Key paths to cross-check

- Atlas8 serial sweep: `atlas8/results/atlas8/serial_sweep/1032959/results.csv`
- Atlas8 strong scaling: `atlas8/results/atlas8/scaling/strong/{4_1032008,8_1032018,12_1031261,24_1031275}/results.csv`
- Atlas8 hybrid comparison: `atlas8/results/atlas8/hybrid_compare16m/1032961/results.csv`
- Vanda **controlled** weak scaling: `vanda/results/vanda/weak_controlled/1518721/{results.csv,summary.csv}`
- Vanda **separate two-node** 24-rank weak scaling: `vanda/results/vanda/weak_scaling/24_1518615/results.csv`
- Vanda CPU/CUDA comparison: `vanda/results/vanda/gpu_benchmark16m/1518595/{results.csv,timing.csv}`
- Vanda **256-core** run: `vanda/results/vanda/256core/1518514/results.csv`
- Derived 256-core CPU-affinity summary: `vanda/results/vanda/256core/1518514/affinity_summary.csv`

**Provenance:** The entire fresh Vanda archive matched the overlapping Vanda files in
our prior cleaned ZIP byte-for-byte. The 256-core measurement CSV in this package
is from that fresh Vanda download, not merely a retyped transcript. The optional
`affinity_summary.csv` was reconstructed from the user-provided job output;
its provenance differs from the original results CSV. The CSV `source` columns
refer to historical `.log` paths on the HPC systems; these logs are deliberately
not bundled, following Lecture 12's submission instructions.

**Scaling interpretation:** The 256-core benchmark verifies a 256-rank execution
on 256 million points (median iteration time 1.152116 s), but it does not provide
matched 256-core strong/weak scaling efficiency. The controlled Vanda 4/8/12-rank
weak-scaling data, the separately collected two-node 24-rank data, and the
256-rank run must not be treated as a controlled homogeneous scaling experiment.

## Compile the report

Run from the package root (not inside `atlas8` or `vanda`):

```bash
pdflatex -interaction=nonstopmode -halt-on-error Kmeans_Final_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error Kmeans_Final_Report.tex
```

Requires a LaTeX installation with `graphicx`, `amsmath`, `booktabs`, `hyperref`,
`geometry`, `microtype`, `caption`, `float`, `lmodern`. Overleaf supports these.
The report references the bundled vector-PDF figures; keep `figures/` alongside
`Kmeans_Final_Report.tex`.

To regenerate all five figures from the original CSVs:

```bash
python3 -m pip install matplotlib  # if needed
python3 scripts/generate_plots.py
```

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

## Submission reminders

Fill in student name/ID in the report; read and confirm the AI-assistance
declaration; compile the report to PDF if required by your submission portal;
and recheck the 256-core job's provenance and conclusion before submission.
