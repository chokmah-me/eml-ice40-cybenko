"""
Build the iCEstick bitstream for the exp_d2 UART demo (Track C stage 6).

Runs the open-source iCE40 flow (yosys -> nextpnr-ice40 -> icepack) on
icestick_exp_top for the iCE40-HX1K/tq144, then optionally flashes with iceprog.
Requires the OSS CAD Suite on PATH (prepend its bin/ and lib/).

    python -m hardware.build_icestick           # synth + pnr + pack
    python -m hardware.build_icestick --flash    # also iceprog the board
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
PCF = os.path.join(HERE, "icestick.pcf")
TOP = "icestick_exp_top"
SRCS = ["icestick_exp_top.v", "exp_d2_pipe_stream.v", "exp_d2_pipe.v",
        "uart_rx.v", "uart_tx.v"]


def run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=RTL, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/flash the iCEstick exp_d2 demo")
    ap.add_argument("--flash", action="store_true", help="iceprog the board after packing")
    args = ap.parse_args()

    run(["yosys", "-q", "-p", f"synth_ice40 -top {TOP} -json {TOP}.json", *SRCS])
    run(["nextpnr-ice40", "--hx1k", "--package", "tq144", "--pcf", PCF,
         "--json", f"{TOP}.json", "--asc", f"{TOP}.asc"])
    run(["icepack", f"{TOP}.asc", f"{TOP}.bin"])
    print(f"\nBitstream: {os.path.join(RTL, TOP + '.bin')}")
    if args.flash:
        try:
            run(["iceprog", f"{TOP}.bin"])
        except subprocess.CalledProcessError:
            sys.exit(
                "\niceprog could not access the board.\n"
                "If you saw \"Can't find iCE FTDI USB device\": on Windows the FTDI\n"
                "serial driver is bound to the iCEstick's config interface, so libusb\n"
                "cannot claim it. Fix with Zadig (https://zadig.akeo.ie):\n"
                "  Options -> List All Devices, select the *Interface 0* entry\n"
                "  (Dual RS232-HS / iCEstick Interface 0), set driver to libusbK,\n"
                "  Replace Driver. Leave Interface 1 as USB Serial (COM) for the UART.\n"
                "Then re-run with --flash. (The bitstream above is already built.)"
            )
        print("Flashed. Now run: python -m hardware.host_demo --port <COMx>")
    else:
        print("Flash with: python -m hardware.build_icestick --flash")
    return 0


if __name__ == "__main__":
    main()
