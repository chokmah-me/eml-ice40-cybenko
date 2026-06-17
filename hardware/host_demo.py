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
import os
import sys

from hardware.fixed_point import FixedPointFormat, FixedEML
from hardware.form_parser import parse_form
from hardware.sim_check import eval_raw, EXP_FORM, Q8_8

BAUD = 115200


def sweep_inputs(frac_bits: int):
    """Same sweep as the testbench / sim_check: x_raw = (i-128) << (F-6)."""
    return [(i - 128) << (frac_bits - 6) for i in range(256)]


def to_signed(raw: int, bits: int) -> int:
    return raw - (1 << bits) if raw & (1 << (bits - 1)) else raw


def symbolic_model():
    """exp_d2 golden: (fmt, predict(x_raw) -> y_raw, label)."""
    net = parse_form(EXP_FORM)
    fe = FixedEML(Q8_8)
    return Q8_8, (lambda x_raw: eval_raw(net, fe, x_raw)), "exp_d2"


def mlp_model():
    """MLP board core golden, read from hardware/rtl/mlp_board_cell.txt."""
    from mlp.fixed_point_mlp import QuantMLP, FUNC_FMT, load_cell
    cell_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "rtl", "mlp_board_cell.txt")
    if not os.path.exists(cell_file):
        sys.exit("no mlp_board_cell.txt -- run: python -m mlp.board --func exp --h 8 --depth 1")
    func, hidden, depth = open(cell_file).read().split()
    layers, _ = load_cell(func, int(hidden), int(depth))
    fmt = FUNC_FMT[func]
    qm = QuantMLP(layers, fmt)
    return fmt, qm.forward_raw, f"mlp_{func}_h{hidden}_d{depth}"


def main() -> int:
    ap = argparse.ArgumentParser(description="iCEstick hardware bit-exact check (symbolic or MLP)")
    ap.add_argument("--port", required=True, help="serial port, e.g. COM3 or /dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--timeout", type=float, default=2.0, help="per-read timeout (s)")
    ap.add_argument("--model", choices=["symbolic", "mlp"], default="symbolic",
                    help="which board core is flashed (default: symbolic exp_d2)")
    args = ap.parse_args()

    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("pyserial not installed -- run: pip install -e .[board]")

    fmt, predict, label = mlp_model() if args.model == "mlp" else symbolic_model()
    W = fmt.total_bits
    n = (W + 7) // 8                      # bytes per sample, both directions

    mismatches = 0
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        ser.reset_input_buffer()
        for x_raw in sweep_inputs(fmt.frac_bits):
            ser.write((x_raw & ((1 << W) - 1)).to_bytes(n, "little"))  # little-endian
            ser.flush()
            resp = ser.read(n)
            if len(resp) != n:
                sys.exit(f"timeout: x_raw={x_raw} got {len(resp)}/{n} bytes (check port/baud/flash)")
            y_hw = to_signed(int.from_bytes(resp, "little"), 8 * n)
            y_py = to_signed(predict(x_raw) & ((1 << W) - 1), W)
            if y_hw != y_py:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  MISMATCH x_raw={x_raw}  hw={y_hw}  py={y_py}")

    if mismatches:
        sys.exit(f"FAILED: {mismatches}/256 mismatches")
    print(f"BIT-EXACT 256/256 -- iCEstick {label} matches the Python model.")
    return 0


if __name__ == "__main__":
    main()
