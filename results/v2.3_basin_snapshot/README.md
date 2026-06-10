# v2.3 Basin Selection Data Snapshot

This directory contains the data snapshot for the Track B short note iteration (post curriculum re-anneal tuning pass).

**Snapshot date**: 2026-06-10 / final (post 20-seed control completion). The 20-seed control (task 019eb042-9cf8-...) and ln:5 400-epoch tuning (task 019eb042-9f97-...) have both completed. Final snapshot CSV copied after the main control run (384 total rows in live CSV).

**Source (live at time of snapshot)**: ../basin_warmstart.csv (234 rows accumulated across all Track B batches, including the 12-seed reanneal batch with 100% curriculum forms on exp d=3 and priors).

**Included in this snapshot**:
- The full accumulated CSV at the moment of the "next iteration" 20-seed launch (curriculum tuning + final control run launch).
- Key cells: exp d=2, exp d=3, ln d=5.
- Modes: blind, warm (refined top-gate), curriculum (grow_from_shallow + reanneal_extra_capacity).
- Helper: `analyze_basin_rates.py` (stdlib CSV stats) + `make_basin_figures.py` used for rates/figure.

**Key results visible in the data (final, post 20-seed control)**:
- exp d=2: blind 17% (1/6), warm 100% (6/6)
- exp d=3 (20-seed control batch for the new 20 seeds): blind 8/20 (40%), warm 20/20 (100%, all `eml(x,1)`), curriculum 20/20 (100%, all `eml(x,1)`). Cumulative: blind ~15.5% (11/71), warm ~91.5% (65/71), curriculum ~73% (32/44).
- ln d=5 (first-class boundary case, now with full 20-seed + 400-epoch data): 0% for warm (66/66) and curriculum (54/54); blind ~7.6% (5/66, a few sporadic). *Every* curriculum and warm run collapses to the identical form `eml(1,eml(1,1))`. The 20-seed control (reanneal 200) gave ln curriculum 0/20 and warm 0/20, all same form. The 400-epoch tuning also 0/10 curriculum. The note treats this as an equal partner: spine embedding succeeds (see code), yet the constant attractor dominates for ln at d=5.

**Reproducibility (documented in the note skeleton and manifest)**:
See `notes/basin_selection_warmstart_note_skeleton.md`, `results/track_b_initial.md`, and the parent `results/basin_warmstart_v2.3_manifest.md` (or the live CSV header comments) for the exact commands used (including the 20-seed and 400-epoch ln variants launched here).

**Final snapshot note**: This is the post-20-seed control snapshot (copied after the main control run completed). Both the 20-seed control and the ln:5 400-epoch tuning are now done. The main note has been updated with the final batch + cumulative numbers (exp curriculum 100% in the 20-seed batch for the new seeds; ln d=5 0% with 100% identical collapse across 54 curriculum rows). Re-run `python make_basin_figures.py` and `python analyze_basin_rates.py` for figures/rates from the final CSV.

The note is the primary consumer of this data. The ln d=5 boundary (consistent collapse despite successful spine embedding via grow_from_shallow + reanneal) is presented as a first-class informative result alongside the exp wins.

For the full experimental log and diagnostics, see `results/track_b_initial.md`.
