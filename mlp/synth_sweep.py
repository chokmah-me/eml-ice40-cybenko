"""
Synthesis + place-and-route sweep -> Pareto data (symbolic-vs-MLP study).

For every trained MLP cell, run yosys synth_ice40 -> nextpnr-ice40 on the
byte-serial *stream* wrapper (19 pins, so it fits the UP5K sg48 package -- the
same variant the snapped-EML stream rows in hardware/HARDWARE.md were measured
on), and parse logic-cell / BRAM / DSP utilization and routed Fmax. Pairs each
area point with the fixed-point max-error from mlp/fixed_point_mlp.py.

Emits results/mlp_pareto.csv with the two snapped-EML reference points appended
(from HARDWARE.md stage-5 stream table). Requires yosys + nextpnr-ice40 on PATH
(OSS CAD Suite). Targets iCE40 UP5K sg48 to match the symbolic stream numbers.

    python -m mlp.synth_sweep                  # full sweep (slow for big cells)
    python -m mlp.synth_sweep --func exp --h 8 --depth 1
"""

import argparse
import csv
import os
import re
import subprocess

from mlp.fixed_point_mlp import error_report
from mlp.run_mlp import cell_name, emit_cell
from mlp.train_mlp import HIDDENS, DEPTHS, FUNCS

HERE = os.path.dirname(os.path.abspath(__file__))
RTL_DIR = os.path.join(os.path.dirname(HERE), "hardware", "rtl")
OUT_CSV = os.path.join(os.path.dirname(HERE), "results", "mlp_pareto.csv")
META_TXT = os.path.join(os.path.dirname(HERE), "results", "mlp_pareto_meta.txt")

DEVICE = ("--up5k", "--package", "sg48")

# Snapped-EML reference points (HARDWARE.md stage-5 stream table, UP5K sg48) +
# their fixed-point max-err (hardware_poc_report.md). These dominate the frontier.
# area_source="hardware.md" flags that these are from the prior stage-5 run rather
# than re-measured in this sweep (same tool versions; see mlp_pareto_meta.txt).
SYMBOLIC_REF = [
    {"unit": "symbolic_exp", "func": "exp", "hidden": 0, "depth": 0, "fmt": "Q8.8",
     "lcs": 316, "lc_est": 316, "area_source": "hardware.md", "ebr": 3, "dsp": 0,
     "fmax_mhz": 28.3, "fp_maxerr": 0.014979, "bit_exact": 1},
    {"unit": "symbolic_ln", "func": "ln", "hidden": 0, "depth": 0, "fmt": "Q10.12",
     "lcs": 1964, "lc_est": 1964, "area_source": "hardware.md", "ebr": 13, "dsp": 0,
     "fmax_mhz": 16.7, "fp_maxerr": 0.001071, "bit_exact": 1},
]

FIELDS = ["unit", "func", "hidden", "depth", "fmt", "lcs", "lc_est", "area_source",
          "ebr", "dsp", "fmax_mhz", "fp_maxerr", "bit_exact"]


def _grep_int(pat, text, default=0):
    m = re.search(pat, text)
    return int(m.group(1)) if m else default


# Per-tool wall-clock caps. yosys-abc mapping on the huge fabric-multiplier cells
# (e.g. h32 depth-2 = ~1000 signed multiplies) is slow, but those cells are exactly
# the ones that could challenge the dominance claim, so we let yosys finish (large
# cap) and record a real pre-P&R LC proxy (SB_LUT4 count from `stat`) even when the
# design overflows the device and nextpnr can't place it.
YOSYS_TIMEOUT = 3600
PNR_TIMEOUT = 600


def synth_cell(func, hidden, depth) -> dict:
    emit_cell(func, hidden, depth, verbose=False)         # ensure RTL current
    name = cell_name(func, hidden, depth)
    top = f"{name}_stream"
    srcs = [f"{name}.v", f"{top}.v"]
    json_out = f"{top}.json"
    er = error_report(func, hidden, depth)
    row = {"unit": name, "func": func, "hidden": hidden, "depth": depth,
           "fmt": er["fmt"].split()[0], "lcs": "", "lc_est": "", "area_source": "",
           "ebr": "", "dsp": "", "fmax_mhz": "", "fp_maxerr": round(er["fp_maxerr"], 6),
           "bit_exact": 1}

    # synth + stat: SB_LUT4 count is the pre-P&R LC proxy (each iCE40 LC holds one
    # LUT4; carries/FFs pack into the same cells, so #LCs >= #SB_LUT4 -- a lower bound
    # that is still a real, reproducible area number for cells too big to place).
    try:
        ys = subprocess.run(
            ["yosys", "-q", "-p",
             f"synth_ice40 -top {top} -json {json_out}; stat", *srcs],
            cwd=RTL_DIR, check=True, capture_output=True, text=True,
            timeout=YOSYS_TIMEOUT)
    except subprocess.TimeoutExpired:
        row["lcs"] = "yosys-timeout"      # only if it blows past the large cap
        row["area_source"] = "none"
        return row
    lc_est = _grep_int(r"SB_LUT4\s+(\d+)", ys.stdout + ys.stderr, default=None)
    if lc_est is not None:
        row["lc_est"] = lc_est

    try:
        pnr = subprocess.run(
            ["nextpnr-ice40", *DEVICE, "--json", json_out, "--freq", "12"],
            cwd=RTL_DIR, capture_output=True, text=True, timeout=PNR_TIMEOUT)
        log = pnr.stdout + pnr.stderr
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""

    # The utilization line prints even when placement overflows (>5280 LC), so an
    # over-100% lcs here is real data ("does not fit"), not an error.
    lcs = _grep_int(r"ICESTORM_LC:\s*(\d+)/", log, default=None)
    if lcs is not None:
        row["lcs"] = lcs
        row["area_source"] = "pnr"
        row["ebr"] = _grep_int(r"ICESTORM_RAM:\s*(\d+)/", log)
        row["dsp"] = (_grep_int(r"ICESTORM_DSP:\s*(\d+)/", log)
                      or _grep_int(r"DSP:\s*(\d+)/", log))
    else:
        # nextpnr could not place/route (overflow or timeout): fall back to the
        # yosys SB_LUT4 proxy so the row still carries a real area number.
        row["lcs"] = row["lc_est"]
        row["area_source"] = "yosys-stat"
    fm = re.search(r"Max frequency for clock '[^']*':\s*([\d.]+)\s*MHz", log)
    if fm:
        row["fmax_mhz"] = round(float(fm.group(1)), 1)
    return row


def _numeric(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fill_overflow_lower_bounds(rows):
    """Give every MLP row a numeric area, even cells whose synth did not finish.

    The cells that time out are the largest of their function (most parameters ->
    most fabric multipliers), so their true area is >= the largest *measured*
    same-function cell. We record that measured maximum as a documented lower bound
    (area_source='overflow-lb') rather than leaving a bare 'yosys-timeout'. These
    cells lose on accuracy anyway, so the exact count is not load-bearing -- the
    lower bound (already several x over the 7680-LC HX8K) suffices for 'does not fit'.
    """
    by_func_max = {}
    for r in rows:
        if not r["unit"].startswith("mlp_"):
            continue
        lc = _numeric(r.get("lcs"))
        if lc is not None:
            by_func_max[r["func"]] = max(by_func_max.get(r["func"], 0), lc)
    for r in rows:
        if not r["unit"].startswith("mlp_") or _numeric(r.get("lcs")) is not None:
            continue
        lb = by_func_max.get(r["func"])
        if lb is not None:
            r["lcs"] = lb
            r["lc_est"] = lb
            r["area_source"] = "overflow-lb"
    return rows


def synth_rom(func) -> dict:
    """Direct-ROM baseline row: emit + synth the feasible ROM, or report the
    block-RAM requirement for the infeasible one (referee hole #3, sec 3.5)."""
    from hardware.verilog_gen_stream import emit_stream_wrapper
    from mlp.fixed_point_mlp import FUNC_FMT
    from mlp import rom_baseline as rb

    fmt = FUNC_FMT[func]
    spec = rb.rom_spec(func, fmt)
    er = rb.error_report(func, fmt)
    name = f"rom_{func}"
    row = {"unit": name, "func": func, "hidden": 0, "depth": 0,
           "fmt": repr(fmt).split()[0], "lcs": "", "lc_est": "", "area_source": "",
           "ebr": spec["n_ebr"], "dsp": 0, "fmax_mhz": "",
           "fp_maxerr": round(er["fp_maxerr"], 6), "bit_exact": 1}

    if spec["n_ebr"] > 32:        # larger than any iCE40's block RAM -- do not synth
        row["lcs"] = ""
        row["area_source"] = "rom-infeasible"
        return row

    info = rb.emit_rom_verilog(func, fmt, name, RTL_DIR)
    top = f"{name}_stream"
    emit_stream_wrapper(name, fmt, info["latency"], RTL_DIR)
    try:
        subprocess.run(["yosys", "-q", "-p", f"synth_ice40 -top {top} -json {top}.json",
                        f"{name}.v", f"{top}.v"], cwd=RTL_DIR, check=True,
                       capture_output=True, text=True, timeout=YOSYS_TIMEOUT)
        pnr = subprocess.run(["nextpnr-ice40", *DEVICE, "--json", f"{top}.json", "--freq", "12"],
                             cwd=RTL_DIR, capture_output=True, text=True, timeout=PNR_TIMEOUT)
        log = pnr.stdout + pnr.stderr
        row["lcs"] = _grep_int(r"ICESTORM_LC:\s*(\d+)/", log, default="")
        row["ebr"] = _grep_int(r"ICESTORM_RAM:\s*(\d+)/", log, default=spec["n_ebr"])
        row["area_source"] = "pnr"
        fm = re.search(r"Max frequency for clock '[^']*':\s*([\d.]+)\s*MHz", log)
        if fm:
            row["fmax_mhz"] = round(float(fm.group(1)), 1)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        row["area_source"] = "rom-emit-only"
    return row


def _tool_versions() -> str:
    import platform
    lines = [f"host: {platform.node()} ({platform.platform()})"]
    for tool in (["yosys", "--version"], ["nextpnr-ice40", "--version"]):
        try:
            out = subprocess.run(tool, capture_output=True, text=True, timeout=30)
            lines.append(f"{tool[0]}: {(out.stdout or out.stderr).strip().splitlines()[0]}")
        except Exception as e:                       # noqa: BLE001 - best-effort provenance
            lines.append(f"{tool[0]}: (unavailable: {e})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Synthesis sweep for the symbolic-vs-MLP Pareto")
    ap.add_argument("--func", choices=list(FUNCS))
    ap.add_argument("--h", type=int)
    ap.add_argument("--depth", type=int)
    args = ap.parse_args()
    funcs = [args.func] if args.func else list(FUNCS)
    hiddens = [args.h] if args.h else HIDDENS
    depths = [args.depth] if args.depth else DEPTHS

    rows = []
    for func in funcs:
        for d in depths:
            for h in hiddens:
                try:
                    row = synth_cell(func, h, d)
                except FileNotFoundError:
                    print(f"{cell_name(func, h, d)}: (not trained)")
                    continue
                except subprocess.CalledProcessError as e:
                    print(f"{cell_name(func, h, d)}: SYNTH FAILED ({e})")
                    continue
                rows.append(row)
                print(f"{row['unit']}: {row['lcs']} LC, {row['ebr']} EBR, "
                      f"{row['dsp']} DSP, {row['fmax_mhz']} MHz, "
                      f"fp_maxerr {row['fp_maxerr']}")

    fill_overflow_lower_bounds(rows)
    for func in funcs:                       # direct-ROM baseline (sec 3.5)
        try:
            rows.append(synth_rom(func))
        except Exception as e:               # noqa: BLE001 - ROM row is best-effort
            print(f"rom_{func}: skipped ({e})")
    rows.extend(SYMBOLIC_REF)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {os.path.relpath(OUT_CSV)}")

    import datetime
    with open(META_TXT, "w") as f:
        f.write("Provenance for results/mlp_pareto.csv (symbolic-vs-MLP synthesis sweep)\n")
        f.write(f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"device: nextpnr-ice40 {' '.join(DEVICE)}\n")
        f.write(f"yosys_timeout_s: {YOSYS_TIMEOUT}  pnr_timeout_s: {PNR_TIMEOUT}\n")
        f.write("lc_est = yosys SB_LUT4 count (pre-P&R proxy); lcs = nextpnr "
                "ICESTORM_LC when placed, else lc_est (area_source column).\n\n")
        f.write(_tool_versions() + "\n")
    print(f"Wrote provenance -> {os.path.relpath(META_TXT)}")


if __name__ == "__main__":
    main()
