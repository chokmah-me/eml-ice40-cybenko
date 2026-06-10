# Notes / Follow-up Work for eml-ice40-cybenko (Track B: Basin Selection)

This directory contains working documents, skeletons, and artifacts from the post-v2.2 Track B iteration on basin selection improvements via warm-start and curriculum initialization.

## Contents

- `basin_selection_warmstart_note_skeleton.md` — Professional skeleton for a short technical note / workshop paper. Includes real results from the first two batches of experiments (exp d=2 at 100% warm, exp d=3 lift after refined embedding, curriculum runs in progress for ln d=5). Ready to be filled with larger-N numbers, figures, and tightened writing.

- (Future) `basin_warmstart_v2.3_manifest.md` or similar — when we cut a cleaned data release.

## Relation to v2.2

All work here directly builds on and cites the released v2.2 paper, code (`eml_layer_v2.py`, `experiment_v2.py`), and `snapping_v2_final.csv`. The new contributions are:
- `initialize_to_target` (refined embedding)
- `grow_from_shallow` (curriculum)
- `basin_warmstart.py` comparison driver
- Empirical demonstration that cheap symbolic warm-start + curriculum can dramatically improve valid recovery on cells the v2.2 paper showed were hard.

## Reproducibility

See the skeleton for example commands. The main driver is `basin_warmstart.py` at the repo root. New code lives in `eml_layer_v2.py`.

## Current Status (as of latest update)

- Strong positive result on exp at d=2 and d=3.
- Curriculum experiments running / recently completed for the remaining hard cell (ln d=5).
- Note skeleton created and partially fleshed (figures, methods, venue, data section).
- Next: incorporate final curriculum numbers, produce 1–2 figures, decide on exact release format (arXiv note + Zenodo data deposit vs. workshop submission).

## How to contribute / iterate

- Run `python basin_warmstart.py --seeds 20 --epochs 2000 ...` for fuller numbers.
- Edit the skeleton directly.
- Add new experiment variants under `basin_warmstart.py` (unbalanced trees, different noise schedules, etc.).
- Update this README and the main `results/track_b_initial.md` when new data arrives.

Contact / issues: follow the main repo process.

---

*This is working material toward a follow-up note that strengthens the central methodological claim of the v2.2 basin-selection paper.*