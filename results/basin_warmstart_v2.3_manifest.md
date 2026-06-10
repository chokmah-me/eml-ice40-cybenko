# v2.3 Basin Selection Data Snapshot (Track B)

**Purpose**: Supporting data for the short technical note / workshop paper "Basin Selection in EML Expression Trees: Targeted Warm-Start Initialization Achieves Full Valid Recovery at Low Depths".

**Source CSV**: `results/basin_warmstart.csv` (accumulated across all Track B experiment batches, 162 rows as of latest iteration).

**Included runs (key cells highlighted in v2.2 as difficult)**:
- exp d=2 (representational depth, classic hard case dominated by `eml(x,x)` basin in blind runs)
- exp d=3 (over-depth)
- ln d=5 (over representational depth; the drop-off cell)

**Init modes compared**:
- blind: original `randomize(0.1)` (matches v2.2 baseline)
- warm: `initialize_to_target` with refined top-gate embedding for exp (direct x/1 at the root gate, everything else constant-1) + spine for ln
- curriculum: `grow_from_shallow` after pre-training a shallow tree at the function's representational depth with warm init, then embedding the structure + training the deeper tree

**Key results (cumulative + latest higher-N batches)**:
- exp d=2: blind ~17%, warm **100%** (multiple batches, 6–15 seeds)
- exp d=3: blind ~8% cumulative, warm **~85%** cumulative / **100%** in 12- and 15-seed batches (1500–1800 epochs). Curriculum batches show successful injection of the shallow `eml(x,1)` core (forms contain inner `eml(x,1)` wrapped by extra layers).
- ln d=5: 0% across blind/warm/curriculum in current data (all modes collapse to constants such as `eml(1,eml(1,1))`). Curriculum successfully plants structure on exp but has not yet overcome collapse on this deeper over-representational case.

**Protocol notes** (identical to v2.2 except initialization):
- Balanced binary EML trees, `eml_layer_v2.py` + `train_eml`.
- Three-phase (Adam 60% task loss, entropy penalty 20%, temperature anneal 20%).
- Post-snap validity: snapped + post-snap MAE < 0.01 on the training grid + correct symbolic form.
- Data generators, learning rates, epochs, and validity definition match the released `snapping_v2_final.csv` and `experiment_v2.py`.

**For the note**:
- Use the latest batches for the primary "warm delivers 100% on exp d=2 and d=3" claim.
- Curriculum results on exp d=3 are presented as mechanistic evidence that `grow_from_shallow` works (core planted) + clear signal for the next tuning target (extra-level collapse / more epochs on new capacity).
- ln d=5 remains the outstanding challenge (future work / curriculum refinements).

**Reproducibility**:
```bash
# Reproduce the exact batches used in the note
python basin_warmstart.py --seeds 12 --epochs 1500 --noise 0.4 --cells exp:3,ln:5
python basin_warmstart.py --seeds 15 --epochs 1800 --noise 0.4 --cells exp:3,ln:5

# Regenerate figures
python make_basin_figures.py
```

**v2.3 release intent**: This CSV + the two scripts (`basin_warmstart.py`, `make_basin_figures.py`) + the note constitute the "v2.3 basin selection" data + method release accompanying the short note. A cleaned subset or Zenodo deposit can be cut from this snapshot once the note is finalized.

**Limitations acknowledged in the note**:
- Seed counts per cell are still modest (6–15) compared with the original v2.2 20-seed design (full 20-seed control runs recommended before final submission).
- Only exp and ln tested so far.
- Curriculum for ln d=5 and extra-level collapse on exp over-depth still need work.

See `notes/basin_selection_warmstart_note_skeleton.md` for the current draft that consumes this data, and `results/track_b_initial.md` for the full experimental log and diagnostic details.
