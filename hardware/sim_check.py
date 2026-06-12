"""
RTL vs Python bit-equality check (Track C stage 3).

Compiles each generated <name>.v + <name>_tb.v with Icarus Verilog, runs the
256-point sweep testbench, and compares every (x_raw, y_raw) pair against the
bit-exact Python model evaluated on the same raw input codes. Any mismatch is
a divergence between hardware/fixed_point.py and hardware/verilog_gen.py.

The tb sweep is x_raw = (i-128) << (F-6), i.e. ~[-2, 2): it exercises the exp
LUT interior, the ln clamp path (x <= 0), and the range-reduction for m both
above and below 1. Full-domain accuracy is covered by tests/test_hardware.py;
this script's job is RTL == model, bit for bit.

Requires iverilog on PATH (OSS CAD Suite). Run from repo root:
    python -m hardware.sim_check
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.fixed_point import FixedPointFormat, FixedEML
from hardware.form_parser import parse_form

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")

CASES = [
    ("exp_d2", "eml(x,1)", FixedPointFormat(8, 8)),
    ("ln_d4", "eml(1,eml(eml(1,x),1))", FixedPointFormat(10, 12)),
]


def eval_raw(net, fe, x_raw: int) -> int:
    """eval_netlist_fixed, but on a raw input code (exactly what the RTL sees)."""
    cache = {}
    for g in net.gates:
        if not g.on_path:
            continue
        if g.const_val is not None:
            cache[g.idx] = fe.fmt.quantize(g.const_val)
            continue
        def raw(kind_child):
            kind, child = kind_child
            if kind == 'const1':
                return fe.one
            if kind == 'x':
                return x_raw
            return cache[child]
        cache[g.idx] = fe.eml_fix(raw(g.left), raw(g.right))
    return cache[net.out_idx]


def run_case(name: str, form: str, fmt: FixedPointFormat) -> int:
    net = parse_form(form)
    fe = FixedEML(fmt)
    sim = os.path.join(RTL_DIR, f"{name}_sim")
    subprocess.run(["iverilog", "-g2012", "-o", sim, f"{name}.v", f"{name}_tb.v"],
                   cwd=RTL_DIR, check=True)
    out = subprocess.run(["vvp", sim], cwd=RTL_DIR, check=True,
                         capture_output=True, text=True).stdout
    mismatches = 0
    n = 0
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue  # vvp status lines ($finish message etc.)
        x_raw, y_rtl = int(parts[0]), int(parts[1])
        y_py = eval_raw(net, fe, x_raw)
        n += 1
        if y_rtl != y_py:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH {name}: x_raw={x_raw}  rtl={y_rtl}  py={y_py}")
    status = "BIT-EXACT" if mismatches == 0 else f"{mismatches} MISMATCHES"
    print(f"{name} (Q{fmt.int_bits}.{fmt.frac_bits}): {n} points, {status}")
    return mismatches


def main():
    total = sum(run_case(*c) for c in CASES)
    if total:
        sys.exit(f"FAILED: {total} total mismatches")
    print("All RTL outputs match the Python model bit-for-bit.")


if __name__ == "__main__":
    main()
