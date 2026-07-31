# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Session-recall and hashline conventions live in the global `~/.claude/CLAUDE.md`.

## Project Overview

**eml-ice40-cybenko** is a two-track project combining symbolic regression research with hardware synthesis:

- **Track A/B**: Empirical study of EML (Elementwise ML) expression trees with a focus on the basin-selection problem. Compares blind vs warm-start vs curriculum approaches to achieve valid symbol recovery (post-snap MAE < 0.01). Published v2.3 paper + v1.0 technical note on Zenodo.
- **Track C**: Hardware realization—converts valid snapped trees to fixed-point iCE40 RTL. All 152 valid forms from released CSVs canonicalize to 2 designs (exp d=2, ln d=4). exp d=2 runs on a physical Lattice iCEstick with UART bridge; v2.8 confirms 256/256 bit-exact match to Python model on silicon.

## Quick Start

```bash
pip install -e .                                    # Install the package
pip install -e .[dev]                              # + pytest for tests
pip install -e .[board]                            # + pyserial for iCEstick
pip install -e .[figures]                          # + matplotlib

python examples/quickstart.py                      # ~1 min: blind vs warm on exp(x)
python examples/reproduce_note_table.py            # Headline table from released data
python -m pytest tests/ -q                         # 50-test suite
```

## Common Commands

### Software (Track A/B: EML Training & Analysis)

```bash
# Training experiments (use basin-warmstart CLI or direct script)
basin-warmstart exp --seeds 20 --epochs 2000     # Warm-start / curriculum comparison (CLI)
python basin_warmstart.py --seeds 8 --epochs 1500  # Quick test (~2-3 min)
python experiment_v2.py                          # Reproduce v2.2 paper (240 runs)

# Analysis & figures
python analyze_basin_rates.py                    # Parse results CSV for statistics
python make_basin_figures.py                     # Generate Figure 3 for notes/
python make_figure2.py                           # Figure 2: blind vs warm rates
```

**Key modules:**
- `eml_layer_v2.py`: EMLTree, InputSelector, train_eml, initialize_to_target, grow_from_shallow — core tree + 3-phase training + warm-start helpers
- `basin_warmstart.py`: Main driver for warm-start/curriculum experiments. CLI entry point `basin-warmstart` (see `--help` for init modes: blind, warm, grow-from-shallow, etc.)
- `experiment_v2.py`: Original v2.2 paper reproduction (deterministic, 2000 epochs, all 240 runs)
- `analyze_basin_rates.py`: Parse CSV (stdout table per function/depth + valid%)
- `make_basin_figures.py`: Matplotlib Figure 3 (success rates heatmap)

### Hardware (Track C: Fixed-Point RTL & Synthesis)

```bash
# Error reports + RTL generation (no toolchain needed)
python -m hardware.run_poc                         # PoC: error report + combinational RTL
python -m hardware.run_csv                         # Same, driven from all valid CSV forms
python -m hardware.run_pipelined                   # Pipelined + streaming RTL (EBR-aware)

# HDL verification (requires iverilog from OSS CAD Suite)
python -m hardware.sim_check                       # Bit-equality: Icarus vs Python fixed-point model

# iCEstick physical demo (requires OSS CAD Suite on PATH)
python -m hardware.build_icestick                  # Synthesize + P&R (yosys → nextpnr → icepack)
python -m hardware.build_icestick --flash          # ^ then flash via iceprog
python -m hardware.host_demo --port COM3           # Stream 256-point sweep, verify bit-exact on-silicon
```

**Key modules:**
- `hardware/fixed_point.py`: Bit-exact integer Python model (ground truth for hardware)
- `hardware/converter.py`: Extract netlist from snapped EMLTree
- `hardware/form_parser.py`: Parse symbolic_form string → netlist (CSV-driven)
- `hardware/verilog_gen*.py`: RTL emission (combinational, pipelined, streaming)
- `hardware/sim_check.py`: Icarus simulation harness
- `hardware/build_icestick.py`: yosys/nextpnr/icepack orchestration + flashing
- `hardware/host_demo.py`: UART demo (requires pyserial)

### Testing

```bash
# Run full suite (50 tests, ~2 min)
python -m pytest tests/ -q
python -m pytest tests/ -v                         # Verbose

# Run specific test class or function
python -m pytest tests/test_hardware.py::TestFixedPointFormat -v
python -m pytest tests/test_hardware.py -k "bit_equal" -v  # Tests matching "bit_equal"

# Run with output capture disabled (see print statements)
python -m pytest tests/ -s
```

**Test strategy:** `test_hardware.py` locks regression bounds from measured PoC numbers. The fixed-point model is the numeric ground truth standing in for HDL simulation (yosys/iverilog not always available on dev machines). When you modify hardware code, re-run tests to confirm no numeric regressions.

## Code Architecture

### Track A/B: Software Pipeline

```
snapping_v2_final.csv / basin_warmstart_v2.*.csv
    ↓
experiment_v2.py or basin_warmstart.py
    ↓ (torch training, Adam + entropy + temperature anneal)
EMLTree (fully trained, then snapped)
    ↓ post_snap_loss, valid_snap, symbolic_form
results/basin_warmstart*.csv
    ↓
analyze_basin_rates.py / make_basin_figures.py
    ↓
Figure 3 + summary tables
```

**Key distinction:** Pre-snap `final_loss` systematically understates basin-selection failure. Use `post_snap_loss < 0.01` + `valid_snap == 1` as the correctness signal (paper Sec. 2.5).

### Track C: Hardware Pipeline

```
snapped EMLTree or symbolic_form (CSV)
    ↓ [extract_netlist or parse_form]
Netlist (gates + routing, constant-folded)
    ↓ [eval_netlist_fixed, emit_verilog]
RTL + hex LUTs + testbench
    ↓ [sim_check (Icarus)]
Bit-equal vs Python fixed_point.py model
    ↓ [yosys synth_ice40 → nextpnr → icepack]
Valid bitstreams (.asc / .bin)
    ↓ [iceprog or build_icestick.py --flash]
On-silicon (iCEstick, UP5K, or UPduino)
```

**Fixed-point design principles:**
- exp(a): 257-entry LUT indexed over clamp domain [-8, 8], linear interpolation
- ln(b): range reduction via leading-one detect; `ln(m·2^k) = ln_lut[m] + k·ln2`
- Saturating arithmetic; off-path gates constant-fold away at generation time
- See `hardware/HARDWARE.md` for full numeric derivation and synthesis metrics

## Critical Files & CSV Columns

### Released Data (Zenodo + OSF mirrors)

- `snapping_v2_final.csv`: **240-run paper results** (v2.3 publication data). Columns: function, depth, seed, symbolic_form, post_snap_loss, valid_snap. Use post_snap_loss (not final_loss) to evaluate basin-selection success.
- `results/basin_warmstart_v2.4_postfix.csv`: v2.4 canonical warm-start baseline (120 rows, 20-seed per function/depth cell). Post-fix data after hyperparameter tuning.
- `results/basin_warmstart_v2.5_unbalanced.csv`: v2.5 top-aligned curriculum results (20-seed per cell, showcases ln d=5 reaching 20/20 for first time). Parallel growing strategy (grow_from_shallow mode).

**CSV columns to use:**
- `post_snap_loss` (not `final_loss`): MAE of the argmax-snapped form evaluated on [-1, 1] grid. This is the correctness signal; pre-snap loss systematically underestimates basin-selection failure (see paper Sec. 2.5).
- `valid_snap`: 1 if snapped (all selectors max-weight >= 0.9) AND post_snap_loss < 0.01
- `symbolic_form`: argmax-snapped expression string. e.g., `eml(x,1)` for exp d=2, `eml(1,eml(eml(1,x),1))` for ln d=4. All 152 valid forms canonicalize to 2 unique types.

### Hardware-Related

- `results/hardware_poc_report.md`: PoC error report (measured quantization, MAE, max error)
- `results/hardware_csv_forms_report.md`: CSV-driven report showing all 152 rows collapse to 2 unique forms
- `hardware/HARDWARE.md`: Full design walkthrough, synthesis numbers, Windows flashing gotchas, OSS CAD Suite flow

### Research Context & Notes

- `notes/basin_selection_warmstart_note.md`: **Technical note (v1.0, published)**. Demonstrates warm-start and top-aligned curriculum achieve 20/20 (100%) valid recovery for exp d=3 and ln d=5 (blind: 25-35%). Includes Section 4 limitations on sqrt and future directions.
- `dyb-2026m-elm-basin.md`: **Paper (v2.3, published)**. Main contribution: distinguishes valid snaps (vertex commitment + correct form) from false snaps via post_snap_loss < 0.01 criterion.
- `dyb-2026m-elm-basin_tldr.md`: Five-perspective TL;DR summaries + AI Utilization Statement.
- `notes/symbolic_vs_mlp.md`: Comparison of EML selectors vs. classical MLP routing.
- `notes/symbolic_function_units_note.md`: Design rationale for canonical forms as reusable function units.
- `ROADMAP.md`: Prioritized tracks (basin-selection improvements, forests, verification, governance) and next-iteration decision framework.

## End-to-End Workflow (Software → Hardware)

```
Train EML trees (basin_warmstart.py or experiment_v2.py)
  ↓
Output CSV: results/basin_warmstart_v*.csv
  ↓
Extract post_snap_loss + valid_snap (analyze_basin_rates.py)
  ↓
Canonical symbolic_form strings (only 2 unique in released data)
  ↓
hardware/run_csv.py: For each valid form, generate netlist + fixed-point model + RTL
  ↓
hardware/sim_check.py: Verify RTL matches fixed-point (Icarus)
  ↓
hardware/build_icestick.py: Synthesize + P&R + flash physical board
```

**Key insight**: Once a tree snaps validly, the selectors are "pure routing" (no weights to implement). The netlist is fully determined by the `symbolic_form` string. All 152 valid rows reduce to 2 designs because exp and ln have fixed canonical architectures.

## Key Architectural Decisions

1. **Temperature annealing vs. entropy penalty**: Phase 3 temperature anneal (T: 1.0 → 0.05) *hardens* softmax toward vertex commitment. Necessary but not sufficient for basin selection—warm-start init is the main lever.

2. **Post-snap loss as validity criterion**: The paper's 0.01 threshold already excludes most competing basins (e.g., exp: eml(x,x) ≈ 0.688 MAE). This is the filter used for hardware canonicalization.

3. **Fixed-point ground truth for hardware**: `fixed_point.py` mirrors Verilog bit-for-bit, so its output *is* the golden reference for RTL verification until HDL simulation toolchain is available.

4. **Canonical forms via CSV scan**: All 152 valid rows from both CSVs collapse to exactly 2 unique `symbolic_form` strings (exp d=2, ln d=4). The stage-5 RTL (byte-streaming on UP5K) covers 100% of released valid solutions.

5. **Warm-start and curriculum as the solution**: Initialize top-gate weights to target function (exp: e^1, ln: e) and grow shallow subtrees first. This removes the 65-75% failure rate for difficult cells (exp d=3, ln d=5) and achieves 100% valid recovery at over-representational depth.

## Development Workflow

### Adding a New Training Experiment

1. Create a new driver script or extend `basin_warmstart.py` with new init mode / curriculum variant
2. Output results to `results/basin_warmstart_v*.csv` (compatible schema)
3. Run `analyze_basin_rates.py` to compute valid% per (func, depth)
4. If figures change, regenerate with `make_basin_figures.py`

### Hardware Verification After Changes

1. Modify `verilog_gen*.py` or `fixed_point.py`
2. Run `python -m hardware.run_poc` to regenerate RTL and error report
3. Run `python -m pytest tests/test_hardware.py -v` to confirm no regression
4. If changes affect synthesis: run `python -m hardware.sim_check` (requires iverilog)

### iCEstick Flashing & Debugging

1. Install OSS CAD Suite (yosys, nextpnr, icepack) and add to PATH
2. Zadig: rebind iCEstick Interface 0 to libusbK; leave Interface 1 as FTDI VCP
3. `python -m hardware.build_icestick --flash` (synthesizes, P&Rs, flashes)
4. If UART fails: run `python -m hardware.build_icestick --top icestick_heartbeat --flash`, then check baud/pins
5. `python -m hardware.host_demo --port COM3` to stream 256-point sweep and verify 256/256 bit-exact

## Notes on Toolchain Dependencies

- **No toolchain required** for data analysis, training, or error report generation
- **Icarus Verilog + yosys + nextpnr** (OSS CAD Suite) needed for `sim_check` and `build_icestick`
- Windows: OSS CAD Suite v2026-06-11 installed at `%USERPROFILE%\tools\oss-cad-suite`; prepend `bin` and `lib` to PATH
- Linux: `apt install iverilog yosys nextpnr-ice40 icestorm`

## Status

Version history and memory anchors are in `STATUS.md`.
