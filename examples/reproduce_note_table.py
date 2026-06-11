#!/usr/bin/env python3
"""Reproduce the headline table of the basin warm-start note (v2.4) from the
canonical released CSV — no training required.

Expected output (from results/basin_warmstart_v2.4_postfix.csv, 120 rows):
    exp d=3: blind 7/20 (35%), warm 20/20, curriculum 20/20
    ln  d=5: blind 5/20 (25%), warm 20/20, curriculum 20/20
"""
import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "results" / "basin_warmstart_v2.4_postfix.csv"

cells = defaultdict(lambda: [0, 0, set()])
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        key = (r["function"], int(r["depth"]), r["init_mode"])
        cells[key][1] += 1
        if r["valid_snap"] == "1":
            cells[key][0] += 1
            cells[key][2].add(r["symbolic_form"])

print(f"{'func':<5} {'d':<2} {'mode':<11} {'valid':>9}  forms")
for (func, d, mode), (v, n, forms) in sorted(cells.items()):
    print(f"{func:<5} {d:<2} {mode:<11} {v:>4}/{n:<4}  {sorted(forms)}")
