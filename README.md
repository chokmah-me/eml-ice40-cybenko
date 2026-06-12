[![Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19736174.svg)](https://doi.org/10.5281/zenodo.19736174)
[![Code DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FVRYNM-blue)](https://doi.org/10.17605/OSF.IO/VRYNM)

# eml-ice40-cybenko

Code and data for:

**Valid and False Snapping in EML Expression Trees: The Basin Selection Problem**
Daniyel Yaacov Bilar, Chokmah LLC, April 2026.

An empirical study of symbolic recovery in balanced-binary EML expression
trees trained with three-phase annealing (Adam, entropy penalty,
temperature anneal). The paper distinguishes *valid snaps* (vertex
commitment + correct symbolic form, post-snap MAE < 0.01) from *false
snaps* (vertex commitment, wrong form) across 240 runs over three target
functions (exp, ln, sqrt) and four tree depths.

## Contents

```
dyb-2026m-elm-basin.pdf       rendered paper (add after export)
figure1_heatmap_v2.png    Figure 1
snapping_v2_final.csv     full 240-run results
eml_layer_v2.py           EML layer + tree implementation
experiment_v2.py          training driver (2000 epochs, 3 phases)
CITATION.cff              citation metadata
LICENSE                   code license (MIT recommended)
.zenodo.json              Zenodo record metadata
```

## CSV columns

`snapping_v2_final.csv` has one row per run (240 total, 238 non-NaN).

| column | meaning |
|---|---|
| function | target function: exp, ln, sqrt |
| depth | tree depth (2-5) |
| seed | deterministic RNG seed |
| snapped | 1 if every selector max-weight >= 0.9 |
| final_loss | pre-snap loss (best task MAE during training) |
| snappability | mean max-weight across selectors (1.000 everywhere) |
| nan_epoch | epoch at which training diverged to NaN (-1 if none) |
| converged | 1 if training completed without NaN |
| symbolic_form | argmax-snapped expression string |
| expected_depth | minimum representational depth for this target |
| post_snap_loss | MAE of the snapped expression evaluated on the grid |
| valid_snap | 1 if snapped AND post_snap_loss < 0.01 |

**Use post_snap_loss and valid_snap, not final_loss, as the correctness
signal.** The paper (Sec. 2.5) explains why pre-snap loss systematically
understates the basin-selection failure mode.

## Install & quickstart

```bash
pip install -e .          # exposes eml_layer_v2 (EMLTree, train_eml) + basin-warmstart CLI
python examples/quickstart.py            # blind vs warm-start on exp(x), ~1 min
python examples/reproduce_note_table.py  # headline v2.4 table from released data, no training
```

See `examples/basin_walkthrough.ipynb` for a narrated version (released-data
rates + one live warm-start and curriculum run), and `requirements.txt` for
the exact versions used to produce the released data.

## Hardware (Track C: iCE40)

The `hardware/` package realizes the repo name: snapped EML trees compiled to
iCE40 FPGA designs. Any valid `symbolic_form` from the released CSVs becomes a
netlist (the snapped selectors are pure routing — no weight storage), then
fixed-point RTL with LUT-based exp/ln (range-reduced ln, interpolated), in
combinational, pipelined (EBR-inferring), and 19-pin byte-streaming variants.
A bit-exact integer Python model mirrors the Verilog operation-for-operation;
Icarus simulation confirms all six generated designs match it bit-for-bit.
Both canonical forms route on an iCE40 UP5K with valid `icepack` bitstreams:
ln d=4 at 1964 LCs / 13 EBR / ~5.6 Msamples/s, exp d=2 at 316 LCs / 3 EBR /
~14 Msamples/s.

```bash
python -m hardware.run_poc        # fixed-point error report + combinational RTL
python -m hardware.run_csv        # same, driven from every valid CSV form
python -m hardware.run_pipelined  # pipelined + streaming-wrapper RTL
python -m hardware.sim_check      # RTL vs Python bit-equality (needs iverilog)
python -m pytest tests/ -q        # 50-test suite incl. quantization regression bounds
```

See [hardware/HARDWARE.md](hardware/HARDWARE.md) for design details, measured
synthesis numbers, and the OSS CAD Suite flow; error reports in
`results/hardware_poc_report.md` and `results/hardware_csv_forms_report.md`.

## Reproducing

```
python experiment_v2.py
```

Deterministic per (function, depth, seed). Two exp/d=5 runs (seeds 0, 2)
diverge to NaN; this is reported in the CSV via `nan_epoch` and
`converged`.

## Citing

See `CITATION.cff`. The archival DOI is minted by Zenodo on release and
will appear as a badge at the top of this README once the first release
is published.

## License

Code released under MIT. See `LICENSE`.
Paper and figures released under CC BY 4.0 (see Zenodo record metadata).

## Changelog

- **v2.6** (2026-06-11): Track C hardware realization. `hardware/` package:
  CSV-form → netlist → fixed-point model → Verilog (combinational, pipelined,
  byte-streaming); bit-exact RTL verification (6/6 designs, Icarus); measured
  UP5K synthesis (ln d=4: 1964 LCs / 13 EBR / 16.7 MHz; exp d=2: 316 LCs /
  3 EBR / 28.3 MHz) with valid bitstreams; 50-test suite. See CHANGELOG.md
  and `hardware/HARDWARE.md`.

- **v2.5** (2026-06-11): Top-aligned `grow_from_shallow` removes the ln
  over-depth curriculum structural limit (ln d=5 curriculum 20/20); Track F
  packaging (pip install -e ., examples, notebook). See CHANGELOG.md.

- **v2.4** (2026-06-11): Track B released. Warm-start note
  (`notes/basin_selection_warmstart_note.md`) + canonical 120-row post-fix
  dataset (`results/basin_warmstart_v2.4_postfix.csv`): warm/curriculum
  20/20 (100%) for exp d=3 and ln d=5, blind 25–35%. See CHANGELOG.md.

- **v2.2** (2026-04-24): Response to review feedback. Tones down novelty framing relative to Odrzywolek (no longer claims prior work conflated commitment and validity). Removes unsupported gradient-escape-routes mechanism claim. Softens "solves completely" to "solves across tested conditions". Explains 0.000 variance in exp d=2 false snaps.

- **Track B: Basin Selection Warm-Start & Curriculum (2026-04/06)**: Major empirical and tooling progress on the basin-selection problem identified in v2.2. 
  - Added `initialize_to_target(func, noise=...)` (refined top-gate embedding for exp; spine for ln), `grow_from_shallow(...)` curriculum helper, and `reanneal_extra_capacity(...)` (short targeted re-anneal on extra selectors only, spine frozen) in `eml_layer_v2.py`.
  - New `basin_warmstart.py` driver with blind/warm/curriculum modes, --reanneal-*/--seeds etc, and higher-N support.
  - Strong results (234 rows): exp d=2 warm 100%; exp d=3 warm ~88% cumul (100% in 12-seed batches); curriculum + reanneal 100% (12/12 batch) / 50% cumul on exp d=3 (reanneal pass solves extra-level collapse). ln d=5 0% (tuning in flight).
  - `plan.md` created saving the 5-item next-iteration list verbatim. `analyze_basin_rates.py` (pure stdlib) and `make_basin_figures.py` for stats/figures.
  - Professional note skeleton in `notes/basin_selection_warmstart_note_skeleton.md` (Table 1, abstract, repro with exact 20-seed CLIs, figure embed, limitations); figure3 dropped to notes/.
  - Updated `ROADMAP.md`, `results/track_b_initial.md`, snapshot `results/v2.3_basin_snapshot/` (dated CSVs + README), and this changelog.
  - Data accumulated in `results/basin_warmstart.csv` (234 rows; 20-seed control + ln:5 400-epoch tuning launched in bg on feature branch; CSV grows live).
  - See `plan.md`, `notes/basin_selection_warmstart_note_skeleton.md`, `results/v2.3_basin_snapshot/README.md` and the Track B section of `results/track_b_initial.md` for full details, commands, and next steps (integrate 20-seed rates, final polish, snapshot refresh).

## Roadmap & Future Directions

See [ROADMAP.md](ROADMAP.md) for expansions, prioritized tracks (analysis of the released CSV, basin-selection improvements, hardware/iCE40 revival, forests/Cybenko scaling, verification/governance, packaging), decision frameworks, and concrete next actions. It synthesizes the focused v2.2 contribution with the broader original EML + Cybenko + iCE40 vision preserved in `prior_versions/dev/`.
