#!/usr/bin/env python3
"""Print a CSV of RESULT lines found in supplied PBS output/log files."""
import csv
import sys
from pathlib import Path

FIELDS = ["variant", "points", "clusters", "dims", "iterations", "ranks",
          "threads", "time_s", "communication_s", "inertia", "accuracy", "assignment_hash"]
writer = csv.DictWriter(sys.stdout, fieldnames=["source"] + FIELDS)
writer.writeheader()
for filename in sys.argv[1:]:
    for line in Path(filename).read_text(errors="replace").splitlines():
        if not line.startswith("RESULT "):
            continue
        values = dict(part.split("=", 1) for part in line.split()[1:])
        writer.writerow({"source": filename, **values})
