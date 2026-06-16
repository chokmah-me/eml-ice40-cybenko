"""
iCEstick board demo (Track C stage 6): hardware bit-exactness over UART.

Streams the same 256-point sweep that hardware/sim_check.py uses to the exp_d2
core running on a Lattice iCEstick (iCE40-HX1K), reads the result codes back over
the FT2232H USB-serial port, and asserts byte-for-byte equality with the Python
fixed-point model (hardware/fixed_point.py). This is the on-silicon confirmation
of the sim_check result.

Protocol (icestick_exp_top.v): 2 bytes little-endian per x sample in, 2 bytes
little-endian per result out (Q8.8 codes, output sign-extended).

Requires the `board` extra (pyserial). Flash first, then run from the repo root:
    python -m hardware.host_demo --port COM3
On Linux/macOS the port is e.g. /dev/ttyUSB1 (iCEstick exposes two FTDI channels;
the UART is the second one).
"""

import argparse
import struct
import sys

from hardware.fixed_point import FixedPointFormat, FixedEML
from hardware.form_parser import parse_form
from hardware.sim_check import eval_raw, EXP_FORM, Q8_8

BAUD = 115200
F = Q8_8.frac_bits


def sweep_inputs():
    """Same sweep as the testbench / sim_check: x_raw = (i-128) << (F-6)."""
    return [(i - 128) << (F - 6) for i in range(256)]


def to_signed16(raw: int) -> int:
    return raw - 0x10000 if raw & 0x8000 else raw


def main() -> int:
    ap = argparse.ArgumentParser(description="iCEstick exp_d2 hardware bit-exact check")
    ap.add_argument("--port", required=True, help="serial port, e.g. COM3 or /dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--timeout", type=float, default=2.0, help="per-read timeout (s)")
    args = ap.parse_args()

    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("pyserial not installed -- run: pip install -e .[board]")

    net = parse_form(EXP_FORM)
    fe = FixedEML(Q8_8)

    mismatches = 0
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        ser.reset_input_buffer()
        for x_raw in sweep_inputs():
            ser.write(struct.pack("<h", x_raw))  # 2 bytes little-endian, signed
            ser.flush()
            resp = ser.read(2)
            if len(resp) != 2:
                sys.exit(f"timeout: x_raw={x_raw} got {len(resp)} bytes (check port/baud/flash)")
            y_hw = to_signed16(resp[0] | (resp[1] << 8))
            y_py = eval_raw(net, fe, x_raw)
            if y_hw != y_py:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  MISMATCH x_raw={x_raw}  hw={y_hw}  py={y_py}")

    if mismatches:
        sys.exit(f"FAILED: {mismatches}/256 mismatches")
    print("BIT-EXACT 256/256 -- iCEstick exp_d2 matches the Python model.")
    return 0


if __name__ == "__main__":
    main()
