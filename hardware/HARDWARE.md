# Track C: Hardware Realization of Snapped EML Trees (iCE40)

This directory ports the original `prior_versions/dev/fixed_point_converter.py`
pipeline to the v2 `EMLTree` and ties it to the *actual* valid snapped forms
from the released data, per ROADMAP Track C. Estimation-first: no Lattice
toolchain is required to validate the numerics.

## Why snapped trees are cheap hardware

After `snap_all()`, every `InputSelector` has committed to one of `{1, x, f}`.
The soft simplex mixing disappears: there are **no weights to store and no
multipliers for selection** — the "weights" become pure routing. Each remaining
gate is `eml(a,b) = exp(clamp(a,-8,8)) - ln(b)` with inputs wired from the
constant 1.0, the primary input `x`, or a child gate. Off-path and
x-independent gates (the deliberate dangling capacity in over-depth trees)
are constant-folded away at generation time.

## Pipeline

```
EMLTree (snapped)                          eml_layer_v2.py
   │  extract_netlist()                    hardware/converter.py
   ▼
Netlist (gates + routing, constant-folded)
   │  eval_netlist_fixed()  ──────────────  bit-exact error report
   │  emit_verilog()                       hardware/verilog_gen.py
   ▼
RTL + LUT hex + testbench                  hardware/rtl/
```

Run end-to-end from the repo root:

```
python -m hardware.run_poc
```

Outputs `hardware/rtl/{exp_d2,ln_d4}.v` (+ `_tb.v`, `_*_lut.hex`) and
`results/hardware_poc_report.md`.

## Fixed-point design (changes vs the 2026-04 prototype)

`hardware/fixed_point.py` is an **integer-only model that mirrors the emitted
Verilog operation-for-operation**, so the Python error report *is* the hardware
error report — no HDL simulation needed for numeric validation.

- **exp(a)**: 257-entry LUT direct-indexed over the gate's clamp domain
  [-8, 8]. The domain width 16 = 2^4 makes index extraction a bit-slice
  (no divide): for Q*m*.*f* input, index = top 8 bits of `raw + 8·2^f`,
  with linear interpolation on the remaining bits (up to 8).
- **ln(b)**: the old prototype indexed its ln LUT in log-space, which is not
  synthesizable (you'd need a log to index a log table). Replaced with standard
  **range reduction**: `ln(m·2^k) = ln_lut[m] + k·ln2`, `m ∈ [1,2)`, using a
  leading-one detector. Accurate both near zero and far above 1 — the ln d=4
  chain needs the latter (its intermediate `exp` output reaches `e^e/x ≈ 151`
  at `x = 0.1`, which is also why Q8.8 cannot carry that chain: see report).
- Saturating arithmetic throughout; `ln(≤0)` clamps to the smallest positive code.

## Results (see `results/hardware_poc_report.md` for the full table)

On the canonical recovered forms — `eml(x,1)` for exp d=2 and
`eml(1,eml(eml(1,x),1))` for ln d=4:

| Case | Format emitted | MAE vs target | max err | Active gates |
|---|---|---|---|---|
| exp d=2, x∈[-2,2] | Q8.8 (16b) | 0.0027 | 0.0150 | 1 of 1 |
| ln d=4, x∈[0.1,10] | Q10.12 (22b) | 0.0003 | 0.0011 | 3 of 7 |

At Q10.12 both meet the paper's 0.01 validity threshold **pointwise**, not just
in MAE. The Q8.8 exp max error (0.015 at x=2) is the input-quantization floor
(exp′(2)/2⁹), not a LUT artifact.

## Resource estimate (per `eml_gate`, iCE40)

- 2 ROMs × 257 × W bits → ~2 EBR per gate (UP5K has 30, HX8K has 32).
- One (W+1)×8 interpolation multiply per LUT + one small `k·ln2` multiply →
  fabric or shared SB_MAC16; leading-one detect ≈ W LUT4s.
- exp_d2 folds to **1 gate** (~2 EBR); ln_d4 to **3 gates** (~6 EBR).

The generated RTL is combinational (PoC). For timing closure on silicon:
register the EBR reads (1-cycle ROM) and pipeline one stage per gate level —
numerics are unchanged.

## Synthesis flow (when the toolchain is available)

Open-source iCE40 flow: `yosys -p "synth_ice40 -json out.json" exp_d2.v`,
then `nextpnr-ice40` (place/route, timing), `icepack`, `iceprog`. The
`$readmemh` ROM init is supported by yosys EBR inference. Not run here.

## CSV-driven flow (stage 2)

```
python -m hardware.run_csv
```

`hardware/form_parser.py` parses any `symbolic_form` string from the released
CSVs directly into a netlist (no tree reconstruction or retraining), so the
flow is data-driven for future targets. `run_csv.py` scans the canonical CSVs
(`snapping_v2_final.csv`, `results/basin_warmstart_v2.4_postfix.csv`),
evaluates every unique valid form, picks the cheapest evaluated format meeting
max error < 0.01, and emits RTL per form (`hardware/rtl/csv_*.v`); report in
`results/hardware_csv_forms_report.md`.

Finding: **all 152 valid rows across both canonical CSVs collapse to exactly
two unique forms** (`eml(x,1)`, `eml(1,eml(eml(1,x),1))`) — the stage-1 RTL
already covers 100% of released valid solutions. The parsed netlists reproduce
the tree-extracted stage-1 error numbers bit-for-bit (cross-validation of the
extraction path).

## Stage 3: HDL simulation + synthesis (measured, OSS CAD Suite 2026-06-11)

**Bit-equality (Icarus Verilog)**: `python -m hardware.sim_check` compiles each
generated design + testbench, runs the 256-point sweep, and compares every
output code against the Python model evaluated on the same raw inputs.
Result: **both designs bit-exact, 256/256 points each** (exp_d2 at Q8.8,
ln_d4 at Q10.12) — including the ln clamp path and range reduction on both
sides of m=1.

**Synthesis + place-and-route (yosys `synth_ice40` → nextpnr-ice40, UP5K sg48)**:

| Design | SB_LUT4 | SB_CARRY | Placed LCs | UP5K utilization | EBR |
|---|---|---|---|---|---|
| exp_d2 (Q8.8) | 572 | 50 | 586 / 5280 | 11% | 0 |
| ln_d4 (Q10.12) | 3121 | 245 | 3193 / 5280 | 60% | 0 |

Both fit a UP5K outright. **Zero EBRs were inferred**: iCE40 block RAM only
supports registered (synchronous) reads, so the combinational `$readmemh` ROMs
were mapped into fabric LUTs. That is the dominant LUT cost — the pipelined
variant (registered ROM reads, one stage per gate level) would move 2 ROMs per
gate into EBRs and shrink the LUT count dramatically. Timing was not
constrained (pure combinational design, no clock); meaningful Fmax comes with
the pipelined variant.

Toolchain note: OSS CAD Suite (Windows build 2026-06-11) installed at
`%USERPROFILE%\tools\oss-cad-suite`; prepend its `bin` and `lib` to PATH.
This build resolved the yosys datadir to `<suite>\share` instead of
`<suite>\share\yosys`; fixed once by junctioning the contents of
`share\yosys` up into `share`.

## Stage 4: pipelined variant (EBR + Fmax, measured)

`hardware/verilog_gen_pipelined.py` (`python -m hardware.run_pipelined`) emits
clocked, fully streaming RTL: ROM reads are registered (the pattern yosys
needs to infer EBR), each LUT is split into two banks (A = entries 0..255,
B = 1..256) because iCE40 EBR has one read port and interpolation needs two
adjacent entries per cycle, each gate is a 3-stage pipeline, and chained gates
get delay-matched inputs (balancing registers + an x delay line). One sample
per clock; latency = 3 cycles x chain depth (exp_d2: 3, ln_d4: 9).

`python -m hardware.sim_check` covers all four designs: **bit-exact 256/256
points each**, pipelined included — same math, different timing.

| Design | LCs | EBR | Fmax (routed) | Device |
|---|---|---|---|---|
| exp_d2_pipe (Q8.8) | 255 / 5280 (4%) | 3 / 30 (10%) | 29.9 MHz | UP5K sg48 |
| ln_d4_pipe (Q10.12) | 1932 (25%) | 13 (40%) | 39.1 MHz | HX8K ct256 |

Pipelining moved the ROMs into block RAM and cut fabric logic sharply
(exp_d2: 586 → 255 LCs; ln_d4: 3193 → 1932). At ~30 MHz streaming this is
~30 Msamples/s. ln_d4_pipe synthesizes to 36% LC / 43% EBR on the UP5K too,
but its bare 22-bit ports need 45 I/O pins vs the sg48's 39 — a packaging
artifact of the port-per-bit PoC top, resolved by any serial/streaming
wrapper; HX8K ct256 numbers shown for routed timing.

## Stage 5: streaming wrapper + UP5K bitstreams (measured)

`hardware/verilog_gen_stream.py` (emitted by `python -m hardware.run_pipelined`)
wraps each pipelined core in a byte-serial interface: `in_data[7:0]+in_valid` /
`out_data[7:0]+out_valid`, ceil(W/8) bytes per sample little-endian (output
sign-extended), one new sample every N_IN cycles. 19 pins regardless of W —
this resolves the stage-4 pin blocker (45 > 39 for bare Q10.12 ports).
Deserializer → core (free-running, with a LATENCY-deep sample_valid shift
register marking real outputs) → serializer; N_OUT == N_IN guarantees the
serializer can never be overrun at the max input rate.

`sim_check` now covers **6 designs, all bit-exact 256/256** (combinational,
pipelined, wrapped).

| Design (UP5K sg48) | LCs | EBR | I/O | Fmax (routed) | Throughput |
|---|---|---|---|---|---|
| exp_d2_pipe_stream (Q8.8) | 316 (5%) | 3 (10%) | 19 (48%) | 28.3 MHz | 1 sample / 2 clk ≈ 14 MS/s |
| ln_d4_pipe_stream (Q10.12) | 1964 (37%) | 13 (43%) | 19 (48%) | 16.7 MHz | 1 sample / 3 clk ≈ 5.6 MS/s |

`nextpnr --asc` + `icepack` produce valid bitstreams for both (104,090 bytes
each — UP5K bitstreams are fixed-size). The `.asc`/`.bin` files are
gitignored build artifacts: pin assignments are auto-placed, and a real board
(e.g. iCEBreaker) needs its `.pcf` before flashing with `iceprog`.

## Next steps (staged)

1. Board demo: iCEBreaker pcf (UART bridge via the FTDI), `iceprog` flash,
   host-side script feeding samples and checking against the Python model.
