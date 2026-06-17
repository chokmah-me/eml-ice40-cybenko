"""
Emit the pipelined (EBR-friendly, clocked) RTL variants for the canonical forms.

Run from repo root:  python -m hardware.run_pipelined
Then: python -m hardware.sim_check  (bit-equality incl. pipelined designs)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.fixed_point import FixedPointFormat, FixedEML
from hardware.form_parser import parse_form
from hardware.verilog_gen_pipelined import emit_verilog_pipelined
from hardware.verilog_gen_stream import emit_stream_wrapper

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(HERE, "rtl")

CASES = [
    ("exp_d2_pipe", "eml(x,1)", FixedPointFormat(8, 8)),
    ("ln_d4_pipe", "eml(1,eml(eml(1,x),1))", FixedPointFormat(10, 12)),
]


def main():
    for name, form, fmt in CASES:
        fe = FixedEML(fmt)
        info = emit_verilog_pipelined(parse_form(form), fe, name, RTL_DIR)
        print(f"{name}: {info['active_gates']} gates, latency {info['latency']} cycles "
              f"-> {info['rtl']}")
        winfo = emit_stream_wrapper(name, fe.fmt, info['latency'], RTL_DIR)
        print(f"{winfo['name']}: {winfo['bytes_per_sample']} bytes/sample, 19 pins "
              f"-> {winfo['rtl']}")


if __name__ == "__main__":
    main()
