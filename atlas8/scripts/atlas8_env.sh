#!/usr/bin/env bash
# Load the compiler/MPI combination observed in the student's Atlas8 module list.
# Source this file in a shell or PBS job; do not execute it as a separate process.
if [[ -r /etc/profile.d/rec_modules.sh ]]; then
  # NUS PBS help recommends sourcing this in batch jobs.
  source /etc/profile.d/rec_modules.sh
fi
if ! type module >/dev/null 2>&1; then
  echo 'ERROR: Environment Modules is unavailable. Check the Atlas login/compute environment.' >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -d /app1/modules/centos6.3 ]]; then
    module use /app1/modules/centos6.3
fi

module use /app1/modules/centos6.3
module use /app1/modules/centos7

module load gmp/4.3.2
module load mpfr/2.4.2
module load gcc/7.3.0
module load openmpi/4.1.2
module load python3.7

echo "Running on: $(hostname)"
module list
