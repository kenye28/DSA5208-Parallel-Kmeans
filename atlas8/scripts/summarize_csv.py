#!/usr/bin/env python3
"""Print grouped medians, ranges, MPI collective fraction and optional strong/weak efficiencies."""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


def read(path):
    groups = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["variant"], int(row["ranks"]), int(row["threads"]),
                   int(row["points"]), int(row["clusters"]), int(row["dims"]))
            groups[key].append(row)
    return groups


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 scripts/summarize_csv.py FILE.csv [MORE.csv ...]")
    for filename in sys.argv[1:]:
        path = Path(filename)
        print(f"\n{path}:")
        print(f"{'Variant':<13} {'ranks×thr':<10} {'N':>10} {'K':>4} {'D':>3} {'runs':>5} {'median_s':>12} {'min_s':>12} {'max_s':>12} {'MPI_%':>8}")
        for (variant, ranks, threads, n, k, d), rows in sorted(read(path).items()):
            times = [float(r["time_s"]) for r in rows]
            comm = [float(r["communication_s"]) for r in rows]
            print(f"{variant:<13} {f'{ranks}x{threads}':<10} {n:>10} {k:>4} {d:>3} {len(rows):>5} "
                  f"{median(times):>12.6f} {min(times):>12.6f} {max(times):>12.6f} {100*median(comm)/median(times):>7.2f}%")

if __name__ == "__main__":
    main()
