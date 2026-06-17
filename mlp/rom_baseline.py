"""
Direct-ROM baseline for the symbolic-vs-MLP study (referee hole #3).

A scalar function of one fixed-point input is, in the limit, just a lookup table:
ROM[x_in] = quantize(f(x_in)). This is the sharpest alternative to *both* the
snapped-EML unit and the MLP -- it is bit-exact to the true function's fixed-point
image (output-quantization error only) and *exhaustively* verifiable (the table is
the spec). The question this module answers with real numbers is whether that
collapse actually disposes of the symbolic unit on iCE40.

The catch is the address width: a direct ROM is indexed by the input *code*, so its
size is set by the input wordlength and the function's domain, not by the function's
complexity. Over the domains/formats used in the study:

  exp on [-2, 2], Q8.8 : 1024 input codes -> 1024 x 16b = 16 kbit = 4 EBR (FITS)
  ln  on [0.1, 10], Q10.12 : ~40960 input codes -> ~40960 x 22b ~= 0.9 Mbit
                             ~= 320 EBR (INFEASIBLE: HX8K has 32 EBR)

So the direct ROM is a genuine competitor for exp (narrow input, fits a few EBR, and
*more* accurate than the symbolic unit) but blows past every iCE40 for ln, where the
22-bit input makes the table ~10x the largest device. The snapped-EML unit's range
reduction (ln) and interpolated 257-entry LUT (exp) are precisely the *compression*
of this ROM -- which is why the symbolic unit survives the ROM-collapse argument on
ln and on any multi-input function (table size grows as the product of per-input
domains; the symbolic datapath does not). See notes/symbolic_vs_mlp.md sec 3.5.

This module mirrors the interface contract of mlp/verilog_gen_mlp.py: the emitted
core is (.clk, .x_in[W-1:0], .y_out[W-1:0]) with latency 1 (registered ROM read),
so hardware/verilog_gen_stream.py wraps it unchanged and sim_check diffs it the same.
"""

import math
import os

from hardware.fixed_point import FixedPointFormat
from mlp.fixed_point_mlp import FUNC_FMT

# Stated function domains (same intervals the released CSV solutions were recovered
# on; see notes/symbolic_vs_mlp.md sec 1).
FUNC_DOMAIN = {"exp": (-2.0, 2.0), "ln": (0.1, 10.0)}
TRUE_FN = {"exp": math.exp, "ln": math.log}

EBR_BITS = 4096          # iCE40 block RAM: 4 kbit each (256 x 16 config)


def rom_spec(func: str, fmt: FixedPointFormat) -> dict:
    """Address/size spec for a direct ROM of `func` over its domain at `fmt`."""
    lo, hi = FUNC_DOMAIN[func]
    W, F = fmt.total_bits, fmt.frac_bits
    lo_raw = int(math.floor(lo * (1 << F)))
    hi_raw = int(math.ceil(hi * (1 << F)))
    span = hi_raw - lo_raw + 1                 # input codes spanned by the domain
    addr_bits = max(1, math.ceil(math.log2(span)))
    n_entries = 1 << addr_bits                  # power-of-2 table for clean addressing
    rom_bits = n_entries * W
    # EBR count: 256 entries per 4kbit BRAM, ceil(W/16) BRAMs wide for W>16.
    n_ebr = math.ceil(n_entries / 256) * math.ceil(W / 16)
    return {"func": func, "fmt": fmt, "lo_raw": lo_raw, "addr_bits": addr_bits,
            "n_entries": n_entries, "rom_bits": rom_bits, "n_ebr": n_ebr,
            "domain": (lo, hi), "W": W, "F": F}


def rom_table(spec: dict) -> list:
    """Output raw codes: rom[idx] = quantize(f((lo_raw+idx) / 2^F)), saturated."""
    func, fmt, lo_raw, F = spec["func"], spec["fmt"], spec["lo_raw"], spec["F"]
    fn = TRUE_FN[func]
    table = []
    for idx in range(spec["n_entries"]):
        x = (lo_raw + idx) / (1 << F)
        try:
            y = fn(x)
        except ValueError:                      # ln(x<=0): clamp to the low rail
            y = fn(FUNC_DOMAIN[func][0])
        table.append(fmt.quantize(y))           # quantize() rounds + saturates
    return table


def rom_eval(spec: dict, table: list, x_raw: int) -> int:
    """Golden: saturate x to the domain, index the table. Bit-exact to the RTL."""
    lo_raw = spec["lo_raw"]
    idx = x_raw - lo_raw
    idx = 0 if idx < 0 else (spec["n_entries"] - 1 if idx >= spec["n_entries"] else idx)
    return table[idx]


def error_report(func: str, fmt: FixedPointFormat = None, npts: int = 2001) -> dict:
    """Fixed-point max-err/MAE of the direct ROM vs the true function over the domain."""
    fmt = fmt or FUNC_FMT[func]
    spec = rom_spec(func, fmt)
    table = rom_table(spec)
    lo, hi = spec["domain"]
    fn = TRUE_FN[func]
    F = spec["F"]
    errs = []
    for i in range(npts):
        x = lo + i * (hi - lo) / (npts - 1)
        y_raw = rom_eval(spec, table, int(round(x * (1 << F))))
        errs.append(abs(y_raw / (1 << F) - fn(x)))
    return {"func": func, "fp_mae": sum(errs) / len(errs), "fp_maxerr": max(errs),
            "n_ebr": spec["n_ebr"], "n_entries": spec["n_entries"],
            "feasible_hx8k": spec["n_ebr"] <= 32}


def emit_rom_verilog(func: str, fmt: FixedPointFormat, name: str, out_dir: str) -> dict:
    """Emit a synthesizable direct-ROM core (+ .hex + delay-matched tb).

    Registered read (latency 1) so yosys infers EBR; input is saturated to the
    domain then offset to the table index. Same port/tb shape as the MLP cores.
    """
    spec = rom_spec(func, fmt)
    table = rom_table(spec)
    W, F = spec["W"], spec["F"]
    A = spec["addr_bits"]
    N = spec["n_entries"]
    lo_raw, hi_raw = spec["lo_raw"], spec["lo_raw"] + N - 1
    os.makedirs(out_dir, exist_ok=True)

    hex_name = f"{name}.hex"
    with open(os.path.join(out_dir, hex_name), "w") as f:
        mask = (1 << W) - 1
        for y in table:
            f.write(f"{y & mask:0{(W + 3) // 4}x}\n")

    # Saturate x_in to [lo_raw, hi_raw], then idx = x - lo_raw (0..N-1).
    rtl = f"""// Generated by mlp/rom_baseline.py -- do not edit by hand.
// Direct ROM for {func} over [{spec['domain'][0]}, {spec['domain'][1]}], Q{fmt.int_bits}.{F}
// ({W} bits). {N} entries x {W} b = {spec['rom_bits']} bit ~= {spec['n_ebr']} EBR.
// Bit-exact to quantize({func}(x)); latency 2 (registered addr+data). Exhaustively
// verifiable: the {N}-entry table IS the spec.

module {name} (
    input clk,
    input  signed [{W-1}:0] x_in,
    output signed [{W-1}:0] y_out    // valid 2 cycles after x_in
);
    localparam LATENCY = 2;          // registered address + registered data (BRAM)
    (* rom_style = "block" *) reg [{W-1}:0] rom [0:{N-1}];
    initial $readmemh("{hex_name}", rom);

    // saturate to the domain, then offset to a 0-based table index
    wire signed [{W-1}:0] xs =
        (x_in < {lo_raw}) ? {lo_raw} : (x_in > {hi_raw}) ? {hi_raw} : x_in;
    wire [{A-1}:0] idx = (xs - {lo_raw});

    reg [{A-1}:0] idx_r;
    reg [{W-1}:0] y_r;
    always @(posedge clk) begin
        idx_r <= idx;                // stage 1: register the address
        y_r   <= rom[idx_r];         // stage 2: registered synchronous read
    end
    assign y_out = y_r;
endmodule
"""

    tb = f"""// Streaming testbench for {name}: prints x_raw,y_raw CSV (delay-matched).
`timescale 1ns/1ps
module {name}_tb;
    localparam LAT = 2;
    reg clk = 0;
    always #5 clk = ~clk;
    reg signed [{W-1}:0] x = 0;
    wire signed [{W-1}:0] y;
    {name} dut (.clk(clk), .x_in(x), .y_out(y));

    reg signed [{W-1}:0] xd [0:LAT-1];
    integer j;
    always @(posedge clk) begin
        xd[0] <= x;
        for (j = LAT-1; j > 0; j = j - 1) xd[j] <= xd[j-1];
    end

    integer i;
    initial begin
        for (i = 0; i < 256 + LAT; i = i + 1) begin
            x = $signed((i < 256 ? i : 255) - 128) <<< ({F} - 6);
            @(posedge clk); #1;
            if (i >= LAT - 1 && i <= 255 + LAT - 1)
                $display("%0d,%0d", xd[LAT-1], y);
        end
        $finish;
    end
endmodule
"""

    rtl_path = os.path.join(out_dir, f"{name}.v")
    tb_path = os.path.join(out_dir, f"{name}_tb.v")
    with open(rtl_path, "w") as f:
        f.write(rtl)
    with open(tb_path, "w") as f:
        f.write(tb)
    return {"rtl": rtl_path, "tb": tb_path, "hex": hex_name, "name": name,
            "latency": 2, "spec": spec}


# --------------------------------------------------------------------------- #
# 2-input direct ROM (arity experiment).
#
# A direct table for f(x, y) = exp(x) - ln(y) is addressed by BOTH input codes:
# addr = {x_code, y_code}, so its entry count is the *product* 2^(Wx+Wy). This is
# the multiplicative-vs-additive contrast with the symbolic unit (mlp/symbolic_2in.py),
# whose two LUT legs total 514 entries regardless of input wordlength. We sweep a
# common per-input wordlength Win and report, at matched input precision, the ROM's
# size/feasibility and its accuracy -- the table fits at coarse Win and explodes
# (infeasible on any iCE40) well before it reaches the symbolic unit's accuracy.

TWO_IN_DOMAIN = {"x": (-2.0, 2.0), "y": (0.1, 10.0)}     # exp leg, ln leg
HX8K_EBR = 32


def _true_2in(x: float, y: float) -> float:
    return math.exp(x) - math.log(y)


def rom_spec_2d(win: int, fmt: FixedPointFormat) -> dict:
    """Spec for a 2-input ROM: each input quantized to `win` codes over its domain."""
    W = fmt.total_bits
    n_per = 1 << win                       # codes per input axis
    n_entries = n_per * n_per              # product -> multiplicative growth
    rom_bits = n_entries * W
    n_ebr = math.ceil(n_entries / 256) * math.ceil(W / 16)
    return {"win": win, "fmt": fmt, "n_per": n_per, "n_entries": n_entries,
            "rom_bits": rom_bits, "n_ebr": n_ebr, "W": W}


def error_report_2d(win: int, fmt: FixedPointFormat = None, npts: int = 129) -> dict:
    """Fixed-point max-err/MAE of the 2-input ROM at per-input resolution `win`.

    The ROM stores quantize(f(x_k, y_l)) at the grid of code centers; a query
    snaps each input to its nearest code (the address), so error has two parts:
    input-quantization (coarse for small win) and output-quantization.
    """
    fmt = fmt or FixedPointFormat(8, 8)
    spec = rom_spec_2d(win, fmt)
    (xlo, xhi), (ylo, yhi) = TWO_IN_DOMAIN["x"], TWO_IN_DOMAIN["y"]
    n_per = spec["n_per"]

    def code(v, lo, hi):
        idx = int(round((v - lo) / (hi - lo) * (n_per - 1)))
        return 0 if idx < 0 else (n_per - 1 if idx >= n_per else idx)

    def center(idx, lo, hi):
        return lo + idx * (hi - lo) / (n_per - 1)

    errs = []
    for i in range(npts):
        x = xlo + i * (xhi - xlo) / (npts - 1)
        xc = center(code(x, xlo, xhi), xlo, xhi)
        for j in range(npts):
            y = ylo + j * (yhi - ylo) / (npts - 1)
            yc = center(code(y, ylo, yhi), ylo, yhi)
            stored = fmt.dequantize(fmt.quantize(_true_2in(xc, yc)))
            errs.append(abs(stored - _true_2in(x, y)))
    return {"win": win, "n_entries": spec["n_entries"], "n_ebr": spec["n_ebr"],
            "fp_mae": sum(errs) / len(errs), "fp_maxerr": max(errs),
            "feasible_hx8k": spec["n_ebr"] <= HX8K_EBR}


def two_in_sweep(wins=(4, 6, 8, 10), fmt: FixedPointFormat = None) -> list:
    """The arity table: ROM size/feasibility/accuracy vs per-input wordlength."""
    return [error_report_2d(w, fmt) for w in wins]


if __name__ == "__main__":
    print(f"{'func':5} {'entries':>8} {'EBR':>5} {'fp_maxerr':>10} {'feasible(HX8K)':>15}")
    for func in ("exp", "ln"):
        r = error_report(func)
        print(f"{func:5} {r['n_entries']:8d} {r['n_ebr']:5d} {r['fp_maxerr']:10.5f} "
              f"{str(r['feasible_hx8k']):>15}")

    print("\n2-input direct ROM for f(x,y) = exp(x) - ln(y), Q8.8 output:")
    print(f"{'Win/in':>6} {'entries':>10} {'EBR':>6} {'fp_maxerr':>10} {'feasible(HX8K)':>15}")
    for r in two_in_sweep():
        print(f"{r['win']:6d} {r['n_entries']:10d} {r['n_ebr']:6d} "
              f"{r['fp_maxerr']:10.5f} {str(r['feasible_hx8k']):>15}")
