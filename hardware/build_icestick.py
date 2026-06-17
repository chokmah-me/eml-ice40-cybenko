"""
Build the iCEstick bitstream for the exp_d2 UART demo (Track C stage 6).

Runs the open-source iCE40 flow (yosys -> nextpnr-ice40 -> icepack) on
icestick_exp_top for the iCE40-HX1K/tq144, then optionally flashes with iceprog.
Requires the OSS CAD Suite on PATH (prepend its bin/ and lib/).

    python -m hardware.build_icestick           # synth + pnr + pack
    python -m hardware.build_icestick --flash    # also iceprog the board

Bring-up diagnostics (one flash cycle each) isolate a dead UART link:
    python -m hardware.build_icestick --top icestick_tx_heartbeat --flash
    python -m hardware.build_icestick --top icestick_echo --flash
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")

# Board/device profiles. The UART demo tops (clk/rx/tx/led, 12 MHz -> CPB=104)
# are device-agnostic, so a board is just nextpnr device+package + a .pcf.
DEVICES = {
    "hx1k": {"package": "tq144", "pcf": "icestick.pcf"},       # iCEstick (default)
    "hx8k": {"package": "ct256", "pcf": "hx8k_breakout.pcf"},  # HX8K-B-EVN breakout
}

# top module -> Verilog sources it needs
BUILDS = {
    "icestick_exp_top": ["icestick_exp_top.v", "exp_d2_pipe_stream.v",
                         "exp_d2_pipe.v", "uart_rx.v", "uart_tx.v"],
    # MLP baseline (symbolic-vs-MLP study): emit the core first with
    #   python -m mlp.board --func exp --h <H> --depth <D>
    "icestick_mlp_top": ["icestick_mlp_top.v", "mlp_board_core_stream.v",
                         "mlp_board_core.v", "uart_rx.v", "uart_tx.v"],
    "icestick_tx_heartbeat": ["icestick_tx_heartbeat.v", "uart_tx.v"],
    "icestick_echo": ["icestick_echo.v", "uart_rx.v", "uart_tx.v"],
}


def run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=RTL, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/flash an iCE40 bitstream (iCEstick or HX8K breakout)")
    ap.add_argument("--top", default="icestick_exp_top", choices=sorted(BUILDS),
                    help="top module to build (default: the exp_d2 demo)")
    ap.add_argument("--device", default="hx1k", choices=sorted(DEVICES),
                    help="target board/device (default: hx1k = iCEstick)")
    ap.add_argument("--pcf", help="override the device's default .pcf")
    ap.add_argument("--flash", action="store_true", help="iceprog the board after packing")
    args = ap.parse_args()

    top = args.top
    srcs = BUILDS[top]
    dev = DEVICES[args.device]
    pcf = os.path.join(HERE, args.pcf) if args.pcf else os.path.join(HERE, dev["pcf"])
    run(["yosys", "-q", "-p", f"synth_ice40 -top {top} -json {top}.json", *srcs])
    run(["nextpnr-ice40", f"--{args.device}", "--package", dev["package"], "--pcf", pcf,
         "--json", f"{top}.json", "--asc", f"{top}.asc"])
    run(["icepack", f"{top}.asc", f"{top}.bin"])
    print(f"\nBitstream: {os.path.join(RTL, top + '.bin')}")
    if args.flash:
        try:
            run(["iceprog", f"{top}.bin"])
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
