# v2.3 Basin Selection Data Snapshot

This directory contains the data snapshot for the Track B short note iteration (post curriculum re-anneal tuning pass).

**Snapshot date**: 2026-06-10 / final (post 20-seed control completion). The 20-seed control (task 019eb042-9cf8-...) and ln:5 400-epoch tuning (task 019eb042-9f97-...) have both completed. Final snapshot CSV copied after the main control run (384 total rows in live CSV).

**Source (final snapshot)**: ../basin_warmstart.csv (final 384 completed rows after the 20-seed control and 400-epoch ln tuning).

**Included in this snapshot**:
- The full accumulated CSV at completion of the 20-seed control (the definitive v2.3 data).
- Key cells: exp d=2, exp d=3, ln d=5.
- Modes: blind, warm (refined top-gate), curriculum (grow_from_shallow + reanneal_extra_capacity).
- Helper: `analyze_basin_rates.py` (stdlib CSV stats) + `make_basin_figures.py` used for rates/figure.

**Key results visible in the data (v2.3 diagnostic snapshot, post 20-seed control)**:
- exp d=2/3 results stand (warm/curric strong rescue; forms clean on the curriculum side).
- ln d=5: 0% warm (66/66 all `eml(1,eml(1,1))`), 0% curriculum (54/54 same); blind 5/66 with varied real forms (incl. one valid `eml(1,eml(eml(1,x),1))`). 
- **Audit finding (post-capture):** the ln warm/curriculum 100% collapse was caused by the embedding code (initialize_to_target + first-gate grow produced trivial pretrain form; f did not pull the core chain). Blind > "informed". The tree/driver now fixed (connected core for any d, full-block grow, pretrain_form column, verify guard, hash-free seed). This snapshot is the pre-fix diagnostic corpus. Re-run ln:5 (and control) with corrected source + fresh CSV for the Zenodo numbers. See note for exact before/after and code pointers.

**Reproducibility (documented in the note and manifest)**:
See `notes/basin_selection_warmstart_note.md`, `results/track_b_initial.md`, and the parent `results/basin_warmstart_v2.3_manifest.md` (or the live CSV header comments) for the exact commands used (including the 20-seed control and 400-epoch ln tuning).

**Final snapshot note**: This is the post-20-seed control snapshot (copied after the main control run completed). Both the 20-seed control and the ln:5 400-epoch tuning are now done. The main note has been updated with the final batch + cumulative numbers (exp curriculum 100% in the 20-seed batch for the new seeds; ln d=5 0% with 100% identical collapse across 54 curriculum rows). Re-run `python make_basin_figures.py` and `python analyze_basin_rates.py` for figures/rates from the final CSV.

The note is the primary consumer of this data. The ln d=5 boundary (consistent collapse despite successful spine embedding via grow_from_shallow + reanneal) is presented as a first-class informative result alongside the exp wins.

For the full experimental log and diagnostics, see `results/track_b_initial.md`.
