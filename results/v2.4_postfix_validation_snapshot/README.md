# v2.4 Post-Fix Validation Snapshot (first data with corrected embedding)

**Generated with fixed code after SME review of 6ac4ee3 + followup a43c0e1**

**Command used (validation, 3 seeds):**
python basin_warmstart.py --seeds 3 --epochs 600 --noise 0.4 --cells ln:5 --reanneal-epochs 100 --csv results/basin_warmstart_v2.4_postfix_validation.csv

**Key post-fix results (ln d=5):**
- warm: 3/3 (100%) valid, all final symbolic_form `eml(1,eml(eml(1,x),1))`, all pretrain_form exactly the same correct core.
- curriculum: 3/3 (100%) same correct form (driver reduces over-depth ln curriculum to the fixed direct init).
- blind: 0/3 valid in this batch, but all snapped to varied non-trivial forms (real exploration, no forced `eml(1,eml(1,1))` collapse).

This CSV has the `pretrain_form` column (absent in v2.3). It provides direct evidence that the wiring bug (MUST-FIX #1) is fixed and warm now starts from the correct basin.

Full 20-seed control with the same fixed code + deterministic seeding is recommended for final publication numbers (see note reproducibility section).

See the main note for full interpretation and the followup patch diff.
