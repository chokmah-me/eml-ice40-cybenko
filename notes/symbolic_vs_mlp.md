---
author: "Daniyel Yaacov Bilar"
header: "Symbolic vs. Quantized-MLP Function Units on iCE40: Area, Accuracy, and Exact Verifiability"
footer: "Page ${pageNo} / ${totalPages}"
---

<p class="hebrew-epigraph" dir="rtl" lang="he">אִם יִרְצֶה הַשֵּׁם</p>

# Symbolic vs. Quantized-MLP Function Units on iCE40: Area, Accuracy, and Exact Verifiability

**Daniyel Yaacov Bilar**, Chokmah LLC — ORCID: 0000-0002-9040-6914

v0.2 draft — Track C note accompanying *eml-ice40-cybenko*
(v0.2: 8-seed best-of-N training; every cell carries a real synthesized area —
no "intractable" rows; figure/tables regenerated from the multi-seed sweep.)

Companion artifacts: `mlp/` (this study), `hardware/` (snapped-EML units),
`results/mlp_pareto.csv` (synthesis sweep), `notes/figure_symbolic_vs_mlp.png`.

---

## Abstract

A snapped EML expression tree is, after argmax commitment, a tiny fixed-function
datapath: the soft simplex mixing vanishes and each gate is `exp(a) − ln(b)` with
inputs routed from `{1, x, child}`. Track C compiled the two canonical recovered
forms (exp `eml(x,1)`, ln `eml(1,eml(eml(1,x),1))`) to fixed-point iCE40 RTL and
confirmed them **bit-exact on a physical iCEstick** (256/256). This note asks the
natural baseline question: *how does such a learned symbolic primitive compare to
the conventional way to put a learned scalar function on a small FPGA — a
quantized MLP?* We train tiny ReLU MLPs (width H ∈ {4,8,16,32}, depth ∈ {1,2}) to
approximate the same `exp` and `ln` targets, quantize them to the **same Q-formats**
(Q8.8 for exp, Q10.12 for ln), and push them through the **identical** golden-model
→ Icarus → yosys/nextpnr → iceprog pipeline used for the symbolic units, so both
function units are measured the same way end-to-end.

Two results stand out. (1) **Area–accuracy dominance:** the snapped-EML unit sits
below and to the left of the entire MLP frontier — it is simultaneously smaller in
logic cells and more accurate than every MLP cell we trained. iCE40 has no hard
multipliers, so MLP MACs map into fabric LUTs; even the smallest 1→4→1 exp MLP
costs **1423 LCs at 0.40 max-error** (best of 8 seeds), versus the symbolic exp
unit's **316 LCs at 0.015**. (2) **Exact verifiability:** the symbolic unit is bit-exact to the true
function's fixed-point image and exhaustively checkable over its 256-code input
domain; the MLP is bit-exact only to *its own* quantized model, with a residual
approximation error floor set by training. Both claims are reproducible from the
scripts in `mlp/`, every RTL variant verified bit-exact 256/256 in simulation.

---

## 1. Setup and fairness

Both function units are evaluated by one pipeline so the comparison is
apples-to-apples:

- **Same targets / domains:** exp on [−2, 2], ln on [0.1, 10] — the intervals the
  released CSV solutions were recovered on.
- **Same fixed-point formats:** exp Q8.8 (16-bit), ln Q10.12 (22-bit), matching the
  snapped-EML units in `hardware/HARDWARE.md`.
- **Same streaming wrapper:** every core is wrapped by
  `hardware/verilog_gen_stream.py` (byte-serial, 19 pins) and synthesized for the
  UP5K sg48 — the variant the symbolic stream rows were measured on.
- **Same golden-model contract:** `mlp/fixed_point_mlp.py::QuantMLP` is an
  integer-only forward pass (signed MAC → arithmetic requant shift → bias →
  saturate → ReLU) that mirrors the emitted Verilog operation-for-operation, just
  as `hardware/fixed_point.py` does for the EML gate. `mlp/sim_check_mlp.py`
  confirms **all 16 cells (×2: bare core + stream wrapper) are bit-exact 256/256**
  against this model under Icarus.

The MLP architecture is the textbook one: 1 → H → … → H → 1, ReLU on hidden layers,
one pipeline stage per Linear layer, weights as RTL constants (the networks are too
small to warrant ROMs). Training is deterministic: each cell is trained from **8
independent seeds** (`mlp/train_mlp.py`), and we keep the best by float-grid
max-error for the RTL/area numbers while reporting the mean ± std across seeds so the
accuracy axis reflects capacity rather than a single seed's luck.

## 2. Accuracy: quantization is not the bottleneck

Fixed-point max-error tracks the float approximation error closely at both formats
(`python -m mlp.fixed_point_mlp`; see the `fp max-err` vs `float max-err` columns), so
the MLP's accuracy ceiling is set by *model capacity and training*, not quantization.
With best-of-8 selection the per-cell error is monotone in capacity within each depth
(`results/mlp/train_summary.csv`) — the earlier single-seed pathologies (e.g. a 2.26
max-err at ln h8 d2) were training-luck artifacts, not capacity limits, and vanish
under multi-seed selection. But even the largest, best-of-8 networks — exp h32 d2
(**0.048** fp max-err) and ln h32 d2 (**0.026**) — remain **worse** than the
snapped-EML units (exp 0.015, ln 0.0011). The symbolic primitive expresses the
function exactly up to LUT interpolation; the ReLU MLP pays a piecewise-linear
approximation tax. (Selection is by float-grid max-error, so a few *dominated* cells
show a small fp wrinkle — e.g. exp h16 d2's 0.093 ≳ h8 d2's 0.083 — but these sit
above the frontier and do not affect the comparison.)

## 3. Area and the Pareto frontier

Logic cells and routed Fmax are from yosys `synth_ice40` → nextpnr-ice40 on the
byte-serial stream wrapper, targeting iCE40 UP5K sg48 (5280 LCs); `fp max-err` is the
fixed-point error vs the true function (best of 8 seeds), and `float ± std` is the
float-grid max-error mean ± std across the 8 seeds. Device-capacity references: HX1K
1280, UP5K 5280, HX8K 7680 LCs. **LCs** is the nextpnr `ICESTORM_LC` count — printed
even when placement overflows the device (>100%), so an over-capacity number is real
data, not an error. A `†` marks the three largest cells whose synthesis did not finish
in the 1-hour cap; for these we report a **lower bound** = the largest measured
same-function cell (they have more multipliers, so they are strictly larger, and they
lose on accuracy regardless). Provenance and tool versions are in
`results/mlp_pareto_meta.txt`. Every row is bit-exact 256/256 in simulation.

**exp on [−2, 2] (Q8.8):**

| unit | LCs | fits | Fmax (MHz) | fp max-err | float ± std (N=8) |
|---|---:|---|---:|---:|---:|
| **snapped EML `eml(x,1)`** | **316** | HX1K | 28.3 | **0.0150** | — |
| MLP h4 d1 | 1423 | UP5K | 15.2 | 0.4016 | 1.089 ± 0.935 |
| MLP h8 d1 | 2640 | UP5K | 14.5 | 0.1438 | 0.256 ± 0.124 |
| MLP h16 d1 | 4808 | UP5K | 13.0 | 0.1282 | 0.144 ± 0.065 |
| MLP h32 d1 | 8673 | none | — | 0.0930 | 0.108 ± 0.039 |
| MLP h4 d2 | 3587 | UP5K | 15.4 | 0.2298 | 0.739 ± 0.722 |
| MLP h8 d2 | 11156 | none | — | 0.0831 | 0.124 ± 0.045 |
| MLP h16 d2 | 32276 | none | — | 0.0930 | 0.066 ± 0.031 |
| MLP h32 d2 | ≥32276 † | none | — | 0.0480 | 0.033 ± 0.011 |

**ln on [0.1, 10] (Q10.12):**

| unit | LCs | fits | Fmax (MHz) | fp max-err | float ± std (N=8) |
|---|---:|---|---:|---:|---:|
| **snapped EML `eml(1,eml(eml(1,x),1))`** | **1964** | HX8K | 16.7 | **0.00107** | — |
| MLP h4 d1 | 3066 | UP5K | 12.5 | 0.4549 | 1.338 ± 0.726 |
| MLP h8 d1 | 4933 | UP5K | 13.4 | 0.2105 | 0.614 ± 0.307 |
| MLP h16 d1 | 9454 | none | — | 0.1461 | 0.292 ± 0.107 |
| MLP h32 d1 | 18468 | none | — | 0.0506 | 0.085 ± 0.023 |
| MLP h4 d2 | 7281 | HX8K | — | 0.2840 | 0.634 ± 0.258 |
| MLP h8 d2 | 23319 | none | — | 0.1510 | 0.540 ± 0.657 |
| MLP h16 d2 | ≥23319 † | none | — | 0.0658 | 0.243 ± 0.261 |
| MLP h32 d2 | ≥23319 † | none | — | 0.0265 | 0.059 ± 0.029 |

The headline figure (`notes/figure_symbolic_vs_mlp.png`) plots logic cells vs
fixed-point max-error per function (hollow markers / `†` = synthesis-estimated lower
bound). The two snapped-EML units (starred) dominate the plane: lower area **and**
lower error than the entire MLP family, by wide margins. The cleanest statement is on
the **accuracy axis alone**: no MLP cell — at any width, depth, or seed in the sweep —
reaches the symbolic unit's fixed-point accuracy (best MLP exp 0.048 vs 0.015; best
MLP ln 0.026 vs 0.001), so the symbolic point wins outright and area only widens the
margin. On area, for exp the symbolic unit is ~4.5× smaller than the *smallest* MLP
*and* ~8× lower error than the largest MLP that even fits a UP5K (h16 d1, 0.128); for
ln it beats every MLP on both axes at once (the closest-accuracy MLP, h32 d2 at 0.026,
is ≥12× the largest iCE40). The
mechanism is concrete: iCE40 has no DSP blocks, so every signed multiply in an MLP
layer is built from fabric LUTs and carry chains, and area scales with the MAC count
(≈ H for depth 1, ≈ H² for depth 2) — MLPs blow past the device before reaching the
symbolic unit's accuracy. The snapped EML unit's cost is just two small interpolated
LUTs per gate.

## 4. Verifiability

The symbolic unit admits an exhaustive correctness statement: over its finite input
domain (256 codes on the demo sweep, or the full Q-format domain in
`tests/test_hardware.py`), every output equals the fixed-point image of the true
function within a stated bound, and the RTL equals that model bit-for-bit. There is
no training-noise term. The MLP inherits the same *RTL == model* guarantee (also
256/256), but "model == intended function" holds only to the trained approximation
error — a qualitatively weaker claim for safety- or certification-sensitive use.

## 5. On silicon — and a hard realizability gap

The symbolic exp unit runs on a physical iCEstick (HX1K), 256/256 bit-exact
(`hardware/host_demo.py`), at 530 LCs (41% of the 1280-LC device).

We attempted the same on-silicon confirmation for the MLP baseline. The machinery
is in place — `mlp/board.py` emits an `icestick_mlp_top` bridging the same UART to a
chosen exp MLP core, and `python -m hardware.host_demo --model mlp` verifies it —
but the baseline **does not fit the iCEstick at all**. Even the *smallest, least
accurate* exp MLP (1→4→1, 0.40 max-error best-of-8) places at **1423 ICESTORM_LCs,
111% of the HX1K** (`nextpnr-ice40 --hx1k`: "Failed to expand region (0,0) |_> (13,17)
of 1423 ICESTORM_LCs"). Every more-accurate MLP is larger still. So on this board the comparison is not
"symbolic is cheaper" but the sharper "symbolic is the *only* one of the two that is
physically realizable": there is no exp MLP that fits the iCEstick, accurate or not.

A physical MLP result therefore requires a larger device. We add an **iCE40-HX8K
Breakout Board** target (7680 LCs, `python -m hardware.build_icestick --device hx8k`,
`hardware/hx8k_breakout.pcf`): the symbolic exp unit places there at 530/7680 LC
(6%, 69.7 MHz), the symbolic **ln_d4** unit fits too (1964 LC / 13 EBR — it never fit
the HX1K). Only the *smaller, lower-accuracy* MLPs fit the HX8K (exp up to h16 d1 /
h4 d2, ln up to h4 d2); every MLP that approaches the symbolic accuracy overflows even
this 7680-LC device (§3). So the HX8K makes the comparison physically head-to-head for
*both* symbolic functions, and shows the accuracy-competitive MLPs are unrealizable on
the largest iCE40 too. One caveat:
the HX8K, like the HX1K, has **no hard multipliers**, so MLP MACs still map to fabric
LUTs and the area-dominance result is unchanged — the larger device only provides the
room to place and confirm it. (The board's onboard FT2232H is SPI-config-only, so the
UART demo needs an external 3.3 V USB-serial adapter on the header pins; see the pcf.)
We report the HX1K non-fit rather than quietly switching boards mid-claim, and supply
the HX8K path for the on-silicon MLP confirmation.

## 6. Threats to validity / scope

- Single-input scalar functions; the conclusion need not extend to higher-arity or
  multi-output learned blocks where weight sharing amortizes MAC cost.
- iCE40 specifically has no hard multipliers; on a DSP-rich FPGA the MLP area gap
  would shrink (though the verifiability and exactness arguments are unchanged).
- MLPs are small and trained to convergence on a dense grid; larger or differently
  regularized networks could shift the frontier but not cross the symbolic point at
  the accuracies observed here.

---

## Reproduce

```
python -m mlp.train_mlp           # 8-seed H×depth sweep -> results/mlp/ + train_summary.csv
python -m mlp.run_mlp             # emit core + stream RTL for every (best-seed) cell
python -m mlp.sim_check_mlp       # bit-exact 256/256 (Icarus), all cells
python -m mlp.synth_sweep         # yosys+nextpnr -> results/mlp_pareto.csv (+ _meta.txt)
python -m mlp.make_pareto_figure  # notes/figure_symbolic_vs_mlp.png
python -m pytest tests/test_mlp_hardware.py -q
# physical: python -m mlp.board --func exp --h 8 --depth 1
#           python -m hardware.build_icestick --top icestick_mlp_top --flash
#           python -m hardware.host_demo --port COMx --model mlp
```
