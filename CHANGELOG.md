# Changelog

- **Unreleased** (2026-08-17): Tufte figure pass. No data change.
  - `figure1_heatmap_v2.png` is now a table of the 12 valid-snap cells
    (Table 1 already printed the counts). New `make_figure1.py`.
  - `figure2_rate_comparison.png` is a Cleveland dot plot; `make_figure2.py`
    writes into the repo (was a `/home/claude/` path).
  - Fig. 2 caption updated; `dyb-2026m-elm-basin.pdf` rebuilt in Typora.

- **v2.8** (2026-06-16): Physical iCEstick (iCE40-HX1K) board demo of the exp d=2
  core, hardware-confirmed bit-exact (256/256).
  - New UART bridge RTL: `hardware/rtl/icestick_exp_top.v` wraps the unmodified
    `exp_d2_pipe_stream` core with `uart_rx.v`/`uart_tx.v` (8N1, 115200 from the
    12 MHz oscillator); the core's back-to-back output bytes are FIFO-buffered for
    the slower UART. Pins in `hardware/icestick.pcf` (clk 21, rx 9, tx 8, led 95).
  - `hardware/host_demo.py` (pyserial, new `board` extra) streams the 256-point
    sweep over USB-serial and asserts byte-for-byte equality with the Python model;
    `hardware/build_icestick.py` runs synth/pnr/pack and optionally flashes.
  - UART bring-up diagnostics `icestick_tx_heartbeat.v` (0x55 baud check) and
    `icestick_echo.v` (loopback), selectable via `build_icestick.py --top`.
  - nextpnr HX1K: 530/1280 LCs, 3/16 EBR, 69.6 MHz. ln d=4 does not fit the HX1K
    (needs a UP5K board). sim_check still 6/6, pytest 50 pass.
  - HARDWARE.md documents the two Windows gotchas: Zadig libusbK on Interface 0 for
    `iceprog`, and "Load VCP" on Interface 1 for the UART COM port.

- **v2.7** (2026-06-12): Zenodo publication release.
  - Paper revised to v2.3 and published (DOI 10.5281/zenodo.20671038, concept
    DOI 10.5281/zenodo.19736173): residual causal phrasing for the exp(x)
    over-depth improvement replaced with correlational language per the paper's
    own Sect. 3.4 disclaimer; duplicated abstract sentence removed; Hebrew date
    updated to 27 Sivan 5786; YAML frontmatter + ORCID added. No data changed.
  - Technical note (`notes/basin_selection_warmstart_note.md`) rewritten for
    publication and published as v1.0 (DOI 10.5281/zenodo.20671272, concept
    DOI 10.5281/zenodo.20671271): v2.5 top-aligned curriculum merged in as a
    first-class result, canonical 20-seed datasets only (v2.4_postfix + v2.5),
    pre-fix bug story condensed to Sect. 3.1, version-neutral references to
    the companion paper, AI Utilization Statement added, exported PDF tracked.
  - `make_basin_figures.py` overhauled: reads the canonical
    `basin_warmstart_v2.5_unbalanced.csv`, plots both cells (exp d=3, ln d=5)
    x three init modes with v/n labels, legend below axes, copies to notes/.
    `figure3_valid_rates_exp.png` regenerated.
  - TL;DR companion updated to v2.3, renamed `dyb-2026m-elm-basin_tldr.md`
    (version-neutral), AI Utilization Statement appended (incl. Claude Fable 5
    release-prep disclosure).
  - Metadata: README paper badge corrected to the concept DOI; note badge
    added; `.zenodo.json` bad related identifier (arXiv:2604.01369, unrelated
    paper) replaced and note DOI added; CITATION.cff dual-license note +
    DOI identifiers.
  - Both records mirrored to OSF with PDF, wiki, license, and tags: paper at
    https://osf.io/648um/ (DOI 10.17605/OSF.IO/648UM), note at
    https://osf.io/dws85/ (DOI 10.17605/OSF.IO/DWS85).

- **v2.6** (2026-06-11): Track C hardware realization (the "ice40" in the repo
  name). New `hardware/` package: any valid `symbolic_form` from the released
  CSVs parses to a netlist (snapped selectors are pure routing; off-path gates
  constant-fold away) and compiles to fixed-point iCE40 RTL. Bit-exact integer
  Python model (`fixed_point.py`: 257-entry interpolated exp LUT over the gate
  clamp domain; range-reduced ln via leading-one detect + mantissa LUT + k·ln2)
  mirrors the Verilog operation-for-operation. Three RTL generators:
  combinational (`verilog_gen.py`), pipelined with EBR-inferring registered ROM
  reads + split-bank LUTs + delay-matched chaining (`verilog_gen_pipelined.py`),
  and a 19-pin byte-serial streaming wrapper (`verilog_gen_stream.py`).
  Verification: Icarus bit-equality vs the Python model, 6/6 designs, 256/256
  points each (`sim_check.py`); 50-test pytest suite locking measured
  quantization numbers as regression bounds (exp d=2 Q8.8 MAE 0.0027; ln d=4
  Q10.12 MAE 0.0003 / max 0.0011 — under the paper's 0.01 validity threshold
  pointwise). Measured on iCE40 UP5K (yosys + nextpnr, OSS CAD Suite):
  ln d=4 streaming 1964 LCs (37%) / 13 EBR (43%) / 16.7 MHz ≈ 5.6 Msamples/s;
  exp d=2 streaming 316 LCs / 3 EBR / 28.3 MHz ≈ 14 Msamples/s; valid icepack
  bitstreams for both. Finding: all 152 valid rows across the canonical CSVs
  collapse to the two canonical forms, so the emitted RTL covers 100% of
  released valid solutions. Remaining: physical board demo (pcf + iceprog).
  Reports: `results/hardware_poc_report.md`, `results/hardware_csv_forms_report.md`;
  docs: `hardware/HARDWARE.md`.
- **v2.5** (2026-06-11): Top-aligned `grow_from_shallow` removes the ln
  over-depth curriculum structural limit. ln d=5 curriculum now 20/20 (100%)
  via real grow-from-trained-shallow (fallback audit: 20/20 on the grow path);
  exp d=3 and blind/warm controls reproduced exactly. Data:
  `results/basin_warmstart_v2.5_unbalanced.csv` + snapshot; addendum:
  `notes/v2.5_unbalanced_curriculum_addendum.md`. Also Track F packaging:
  `pyproject.toml` (pip install -e ., `basin-warmstart` CLI), requirements.txt,
  `examples/` (quickstart, table reproduction, walkthrough notebook).
- **v2.4** (2026-06-11): Track B basin-selection follow-on release. New short
  technical note `notes/basin_selection_warmstart_note.md` with canonical
  120-row post-fix dataset (`results/basin_warmstart_v2.4_postfix.csv`,
  snapshot in `results/v2.4_postfix_snapshot/`): targeted warm-start
  (`EMLTree.initialize_to_target`) achieves 100% valid recovery for exp d=3
  and ln d=5 (20/20 each; blind 25–35%); curriculum via `grow_from_shallow`
  + `reanneal_extra_capacity`, with ln over-depth delegating to direct init
  (balanced-tree structural limit, documented). Adds `basin_warmstart.py`
  driver, `analyze_basin_rates.py`, `make_basin_figures.py`, Figure 3,
  pretrain_form auditing column, deterministic seeding. v2.3 data retained
  as pre-fix diagnostic only.
- **v2.2** (2026-04-26): Review-response patches, metadata bump, repo cleanup.
- **v2.1** (2026-04-24): Section 4 connection to Odrzywolek SI warm-start
  evidence (SI Table S7 cited as independent confirmation of basin-selection
  framing); Figure 2 blind-recovery rate comparison (this work vs Odrzywolek
  SI Table S5); Odrzywolek SI Table S5 transcribed to
  `odrzywolek_si_table_s5.csv` with `make_figure2.py` as reproducibility
  script; reference block split into main-text and SI citations;
  TL;DR summaries updated to reflect the SI cross-check.
- **v2.0.1** (2026-04-24): Hebrew font and typo fixes; URL to arXiv:2603.21852v2.
- **v2.0** (2026-04-24): Initial public release. 240-run valid-snap sweep
  across (function, depth, seed). Methodological contribution: valid vs
  false snap distinction.

---

## Files new or changed in v2.1

| File | Status |
|---|---|
| `dyb-2026m-elm-basin_v2.1.md` | new (supersedes `_v2.0.1.md`) |
| `dyb-2026m-elm-basin_v2.1.pdf` | new (supersedes `_v2.0.1.pdf`) |
| `dyb-2026m-elm-basin_v2.1_tldr.md` | new |
| `figure2_rate_comparison.png` | new |
| `odrzywolek_si_table_s5.csv` | new |
| `make_figure2.py` | new |
| `CITATION.cff` | version bump to 2.1.0 |
| `prior_versions/` | contains superseded v2.0.1 files |
| `eml_layer_v2.py`, `experiment_v2.py` | unchanged |
| `snapping_v2_final.csv`, `figure1_heatmap_v2.png` | unchanged |
