# v2.3 Basin Selection Data Snapshot

This directory contains the data snapshot for the Track B short note iteration (post curriculum re-anneal tuning pass).

**Snapshot date**: 2026-06-10 (refreshed at launch of final 20-seed controls; ~234 rows in source CSV at copy time (live grew to 244 during initial bg progress). 20-seed control + ln:5 longer-reanneal (400 epochs) tuning runs launched as background tasks in this continue session; live `../basin_warmstart.csv` will grow.)

**Source (live at time of snapshot)**: ../basin_warmstart.csv (234 rows accumulated across all Track B batches, including the 12-seed reanneal batch with 100% curriculum forms on exp d=3 and priors).

**Included in this snapshot**:
- The full accumulated CSV at the moment of the "next iteration" 20-seed launch (curriculum tuning + final control run launch).
- Key cells: exp d=2, exp d=3, ln d=5.
- Modes: blind, warm (refined top-gate), curriculum (grow_from_shallow + reanneal_extra_capacity).
- Helper: `analyze_basin_rates.py` (stdlib CSV stats) + `make_basin_figures.py` used for rates/figure.

**Key results visible in the data (as of snapshot creation / ~234 rows at launch; live CSV has since grown)**:
- exp d=2: blind 17% (1/6), warm 100% (6/6)
- exp d=3: blind ~6–15% cumul, warm 88–91% cumul (100% in multiple 12-seed batches), curriculum ~50–57% cumul (100% 12/12 in the reanneal-tuned batch with clean `eml(x,1)`)
- ln d=5 (first-class boundary case): 0% across blind/warm/curriculum (current ~117 rows; 46/46/25). *Every* curriculum run collapses to the identical form `eml(1,eml(1,1))`. The note revision treats this as an equal partner to the exp wins: the spine is successfully embedded by the curriculum machinery, yet the optimizer still selects the strong constant attractor. Longer reanneal (400 epochs) + full 20-seed control (including ln curriculum) are in progress.

**Reproducibility (documented in the note skeleton and manifest)**:
See `notes/basin_selection_warmstart_note_skeleton.md`, `results/track_b_initial.md`, and the parent `results/basin_warmstart_v2.3_manifest.md` (or the live CSV header comments) for the exact commands used (including the 20-seed and 400-epoch ln variants launched here).

**Next iteration note**: This snapshot captures state at 20-seed launch time. The live CSV will grow with the real bg runs (task ids in `results/track_b_initial.md`; ignore the 1.7s stub "completions"). When complete, cp a final dated copy here + update this README + the main note (which now gives the ln d=5 case equal weight as a boundary result) with integrated N=20 rates/forms. Re-run `python make_basin_figures.py` for updated PNG.

The note is the primary consumer of this data. Per user request the ln d=5 results (consistent collapse despite spine embedding) are now presented as a first-class, informative part of the story rather than an afterthought.

For the full experimental log and diagnostics, see `results/track_b_initial.md`.
