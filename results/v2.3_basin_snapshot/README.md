# v2.3 Basin Selection Data Snapshot

This directory contains the data snapshot for the Track B short note iteration (post curriculum re-anneal tuning pass).

**Snapshot date**: 2026-06-10 (refreshed at launch of final 20-seed controls; ~234 rows in source CSV at copy time (live grew to 244 during initial bg progress). 20-seed control + ln:5 longer-reanneal (400 epochs) tuning runs launched as background tasks in this continue session; live `../basin_warmstart.csv` will grow.)

**Source (live at time of snapshot)**: ../basin_warmstart.csv (234 rows accumulated across all Track B batches, including the 12-seed reanneal batch with 100% curriculum forms on exp d=3 and priors).

**Included in this snapshot**:
- The full accumulated CSV at the moment of the "next iteration" 20-seed launch (curriculum tuning + final control run launch).
- Key cells: exp d=2, exp d=3, ln d=5.
- Modes: blind, warm (refined top-gate), curriculum (grow_from_shallow + reanneal_extra_capacity).
- Helper: `analyze_basin_rates.py` (stdlib CSV stats) + `make_basin_figures.py` used for rates/figure.

**Key results visible in the data (as of snapshot creation / 234 rows)**:
- exp d=2: blind 17% (1/6), warm 100% (6/6)
- exp d=3: blind ~6% (3/51 cumul), warm 88% (45/51 cumul; 100% in multiple 12-seed batches), curriculum 50% cumul (12/24; 100% 12/12 in the reanneal-tuned batch with clean `eml(x,1)`)
- ln d=5: 0% across blind/warm/curriculum (36/36/24 seeds; all collapse to constants e.g. `eml(1,eml(1,1))`). Longer reanneal tuning in flight.

**Reproducibility (documented in the note skeleton and manifest)**:
See `notes/basin_selection_warmstart_note_skeleton.md`, `results/track_b_initial.md`, and the parent `results/basin_warmstart_v2.3_manifest.md` (or the live CSV header comments) for the exact commands used (including the 20-seed and 400-epoch ln variants launched here).

**Next iteration note**: This snapshot captures state at 20-seed launch time. The live CSV will grow with the bg runs (task ids in track log); when complete, cp a final dated copy here + update this README + the main skeleton with integrated N=20 rates/forms. Re-run `python make_basin_figures.py` for updated PNG.

The note skeleton is the primary consumer of this data.

For the full experimental log and diagnostics, see `results/track_b_initial.md`.
