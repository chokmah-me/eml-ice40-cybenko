# Changelog snippet for README.md

Paste this at the top of your existing Changelog section (or create one if
it doesn't exist yet). The file is a fragment, not a full README.

---

## Changelog

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
