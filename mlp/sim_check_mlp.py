"""
RTL vs Python bit-equality for the quantized-MLP cores (symbolic-vs-MLP study).

Same harness shape as hardware/sim_check.py: compile each generated <name>.v +
<name>_tb.v with Icarus, run the 256-point sweep, and compare every (x_raw,
y_raw) against mlp/fixed_point_mlp.py::QuantMLP.forward_raw on the same raw input
code. Any mismatch is a divergence between the golden model and the emitter.

Covers both the bare pipelined core and its byte-serial streaming wrapper, for
every trained cell. Requires iverilog on PATH (OSS CAD Suite).

    python -m mlp.sim_check_mlp                      # all cells, core + stream
    python -m mlp.sim_check_mlp --func exp --h 8 --depth 1
"""

import argparse
import os
import subprocess
import sys

from mlp.fixed_point_mlp import QuantMLP, FUNC_FMT, load_cell
from mlp.run_mlp import cell_name, emit_cell
from mlp.train_mlp import HIDDENS, DEPTHS, FUNCS

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(os.path.dirname(HERE), "hardware", "rtl")


def run_one(name: str, qm: QuantMLP, extra_src=()) -> int:
    """Compile name.v + name_tb.v (+extra), diff every output vs the golden model."""
    sim = os.path.join(RTL_DIR, f"{name}_sim")
    srcs = [f"{name}.v", f"{name}_tb.v", *extra_src]
    subprocess.run(["iverilog", "-g2012", "-o", sim, *srcs], cwd=RTL_DIR, check=True)
    out = subprocess.run(["vvp", sim], cwd=RTL_DIR, check=True,
                         capture_output=True, text=True).stdout
    mismatches = n = 0
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        x_raw, y_rtl = int(parts[0]), int(parts[1])
        y_py = qm.forward_raw(x_raw)
        n += 1
        if y_rtl != y_py:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH {name}: x_raw={x_raw} rtl={y_rtl} py={y_py}")
    if n != 256:
        print(f"{name}: INCOMPLETE -- {n}/256 results")
        return 256 - n + mismatches
    status = "BIT-EXACT" if mismatches == 0 else f"{mismatches} MISMATCHES"
    print(f"{name}: {n} points, {status}")
    return mismatches


def check_cell(func, hidden, depth) -> int:
    layers, _ = load_cell(func, hidden, depth)
    qm = QuantMLP(layers, FUNC_FMT[func])
    emit_cell(func, hidden, depth, verbose=False)         # ensure RTL is current
    name = cell_name(func, hidden, depth)
    total = run_one(name, qm)                              # bare pipelined core
    total += run_one(f"{name}_stream", qm, extra_src=[f"{name}.v"])  # wrapper
    return total


def main():
    ap = argparse.ArgumentParser(description="Bit-equality check for MLP cores")
    ap.add_argument("--func", choices=list(FUNCS))
    ap.add_argument("--h", type=int)
    ap.add_argument("--depth", type=int)
    args = ap.parse_args()
    funcs = [args.func] if args.func else list(FUNCS)
    hiddens = [args.h] if args.h else HIDDENS
    depths = [args.depth] if args.depth else DEPTHS

    total = 0
    for func in funcs:
        for d in depths:
            for h in hiddens:
                try:
                    total += check_cell(func, h, d)
                except FileNotFoundError:
                    print(f"{cell_name(func, h, d)}: (not trained)")
    if total:
        sys.exit(f"FAILED: {total} total mismatches")
    print("All MLP RTL outputs match the Python golden model bit-for-bit.")


if __name__ == "__main__":
    main()
