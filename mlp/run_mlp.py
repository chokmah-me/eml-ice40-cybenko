"""
Emit MLP RTL (core + streaming wrapper + tb) for every trained cell.

Run from repo root (after `python -m mlp.train_mlp`):
    python -m mlp.run_mlp                 # all trained cells
    python -m mlp.run_mlp --func exp --h 32 --depth 2

Writes hardware/rtl/mlp_<func>_h<H>_d<D>{,_stream}.v (+ _tb.v), reusing the
snapped-EML byte-serial wrapper so the cores share one sim/synth/flash flow.
Then verify with: python -m mlp.sim_check_mlp
"""

import argparse
import os

from hardware.verilog_gen_stream import emit_stream_wrapper
from mlp.fixed_point_mlp import QuantMLP, FUNC_FMT, load_cell
from mlp.train_mlp import HIDDENS, DEPTHS, FUNCS
from mlp.verilog_gen_mlp import emit_verilog_mlp

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(os.path.dirname(HERE), "hardware", "rtl")


def cell_name(func, hidden, depth):
    return f"mlp_{func}_h{hidden}_d{depth}"


def emit_cell(func, hidden, depth, verbose=True):
    layers, meta = load_cell(func, hidden, depth)
    fmt = FUNC_FMT[func]
    qm = QuantMLP(layers, fmt)
    name = cell_name(func, hidden, depth)
    info = emit_verilog_mlp(qm, name, RTL_DIR)
    winfo = emit_stream_wrapper(name, fmt, info["latency"], RTL_DIR)
    if verbose:
        print(f"{name}: {info['n_layers']} layers, latency {info['latency']} "
              f"-> {os.path.relpath(info['rtl'])}  (+ {winfo['name']}, "
              f"{winfo['bytes_per_sample']} B/sample)")
    return info, winfo


def main():
    ap = argparse.ArgumentParser(description="Emit MLP RTL for the symbolic-vs-MLP study")
    ap.add_argument("--func", choices=list(FUNCS))
    ap.add_argument("--h", type=int)
    ap.add_argument("--depth", type=int)
    args = ap.parse_args()
    funcs = [args.func] if args.func else list(FUNCS)
    hiddens = [args.h] if args.h else HIDDENS
    depths = [args.depth] if args.depth else DEPTHS
    for func in funcs:
        for d in depths:
            for h in hiddens:
                try:
                    emit_cell(func, h, d)
                except FileNotFoundError:
                    print(f"{cell_name(func, h, d)}: (not trained -- run mlp.train_mlp)")


if __name__ == "__main__":
    main()
