#!/usr/bin/env python3
"""Compare a serial run with MPI and MPI+OpenMP runs of identical inputs."""
import math
import sys
from pathlib import Path


def read_result(path):
    lines = Path(path).read_text().splitlines()
    result_lines = [line for line in lines if line.startswith("RESULT ")]
    if len(result_lines) != 1:
        raise ValueError(f"{path}: expected exactly one RESULT line, got {len(result_lines)}")
    result = dict(part.split("=", 1) for part in result_lines[0].split()[1:])
    centers = {}
    for line in lines:
        if line.startswith("CENTROID "):
            _, k, *dims = line.split()
            centers[int(k)] = [float(x) for x in dims]
    if not centers:
        raise ValueError(f"{path}: no CENTROID lines; run with --print-centroids")
    return result, centers


def compare(ref, trial, path):
    base, base_centers = ref
    got, centers = trial
    for field in ("points", "clusters", "dims", "iterations"):
        if got[field] != base[field]:
            raise AssertionError(f"{path}: differing {field}")
    if got["assignment_hash"] != base["assignment_hash"]:
        raise AssertionError(f"{path}: assignment hashes differ")
    if set(centers) != set(base_centers):
        raise AssertionError(f"{path}: centroid labels differ")
    max_centroid_diff = 0.0
    for k in centers:
        if len(centers[k]) != len(base_centers[k]):
            raise AssertionError(f"{path}: different dimensionality")
        for observed, expected in zip(centers[k], base_centers[k]):
            if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-8):
                raise AssertionError(
                    f"{path}: centroid {k} differs: {observed} vs {expected}")
            max_centroid_diff = max(max_centroid_diff, abs(observed - expected))
    if not math.isclose(float(got["accuracy"]), float(base["accuracy"]),
                        rel_tol=0, abs_tol=1e-12):
        raise AssertionError(f"{path}: synthetic ground-truth accuracy differs")
    if float(got["accuracy"]) < 0.999999:
        raise AssertionError(f"{path}: unexpected synthetic ground-truth accuracy")
    inertia = float(got["inertia"])
    ref_inertia = float(base["inertia"])
    if not math.isclose(inertia, ref_inertia, rel_tol=1e-9, abs_tol=1e-8):
        raise AssertionError(f"{path}: inertia differs: {inertia} vs {ref_inertia}")
    print(f"PASS {got['variant']:12s}  max_centroid_error={max_centroid_diff:.3e}  "
          f"inertia={inertia:.9g}  assignment_hash={got['assignment_hash']}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/check_results.py serial.log mpi.log [hybrid.log ...]")
        return 2
    ref = read_result(sys.argv[1])
    print(f"Reference: {sys.argv[1]}")
    for path in sys.argv[2:]:
        compare(ref, read_result(path), path)
    print("All supplied implementations agree within tolerance.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, AssertionError, KeyError, OSError) as err:
        print(f"FAIL: {err}", file=sys.stderr)
        sys.exit(1)
