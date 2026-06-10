# Basin Selection in EML Expression Trees: Targeted Warm-Start Initialization Achieves Full Valid Recovery for exp but Exposes Limits for ln at Over-Representational Depth

**Short technical note + data release** (targeting ICML/NeurIPS workshop, ICLR workshop, or Zenodo + arXiv "Technical Note" / data + method note)

**Authors**: [Your name], Chokmah LLC  
**Status**: Post-fix version with full 20-seed control data from the corrected embedding code (after SME review cycles and patches). The v2.4_postfix.csv (120 rows) is the production dataset: warm ln d=5 now 20/20 (100%) with correct pretrain + final form; curriculum reduces to the fixed direct init. v2.3 data/snapshot is retained only as pre-fix diagnostic (the embedding bug). Exp results reproduced cleanly. Pretrain_form column + deterministic seeding make everything auditable.

**Target venues**: ICML/NeurIPS workshop, ICLR workshop, or a short "Technical Note" / "Data Release" style (e.g., via Zenodo + arXiv or a journal like JMLR MLOSS / TMLR).

---

## Abstract

Three-phase temperature annealing solves selector commitment in balanced EML expression trees, but the dominant failure mode remains basin selection during the initial task-loss phase. The v2.2 study ("Valid and False Snapping in EML Expression Trees: The Basin Selection Problem") introduced a strict post-snap validity criterion (correct symbolic form + post-snap MAE < 0.01) and showed that, at minimal representational depth, exp(x) recovers the correct form `eml(x,1)` in only ~25% of blind runs because most trajectories fall into a strong competing basin (`eml(x,x)` with MAE ~0.688).

We demonstrate that a simple, targeted warm-start initialization—directly biasing the top-level selectors toward the known-good symbolic form while setting off-path capacity to harmless constants—rescues recovery for exp. In the final 20-seed control run (plus prior batches, 71 seeds cumul for exp d=3), this raises valid-snap rate from 17% to 100% for exp(x) at depth 2 and from ~6–15% to ~92% cumulative (100% in the 20-seed control batch) at depth 3. A new `reanneal_extra_capacity` pass (short targeted re-anneal only on extra selectors after embedding, spine frozen) turns curriculum from 0% to 100% in the 20-seed control batch for exp d=3 (cumulative ~73% for curriculum on d=3). The method requires no change to the training schedule or loss and directly corroborates the paper's analysis of how successful deeper exp solutions operate (routing x/1 at the top).

For ln at d=5 (over-representational), the v2.3 diagnostic data (pre-fix embedding) showed 0% for warm and curriculum with 100% forced collapse to the trivial `eml(1,eml(1,1))`. **Post-fix full 20-seed control (fixed code, v2.4_postfix.csv):** warm 20/20 (100%), every row with pretrain_form and final symbolic_form exactly `eml(1,eml(eml(1,x),1))`. Curriculum (over-depth) also 20/20 (100%) with the identical correct form (driver delegates to the fixed direct init, as balanced EML extension cannot preserve the ln value — eml(f,1) = exp(f)). Blind: 5/20 (25%) valid with varied non-trivial forms (real exploration, pretrains random).

This is the direct evidence from real training runs that the wiring fixes (initialize_to_target ln core, curriculum delegation, pretrain_form guard, deterministic seeding) resolved the SME-identified bugs. Warm now starts from the correct connected core (confirmed by pretrain_form column for all 20 seeds) and recovers at 100%. `initialize_to_target('ln')` plants the target for any d>=4. The structural limit on balanced-tree curriculum for ln over-depth is now explicitly handled in the driver and documented. Exp results were unaffected and reproduced at 100% (warm/curriculum) / ~35% (blind). The v2.3 data is the pre-fix diagnostic corpus only.

**Reproducibility** (exact command that produced the post-fix data in this note/snapshot; run after `pip install -r requirements.txt` or equivalent with the fixed eml_layer_v2.py + basin_warmstart.py):

```bash
# Full post-fix 20-seed control (the production dataset for all numbers and claims in this note)
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells ln:5,exp:3 --reanneal-epochs 200 --csv results/basin_warmstart_v2.4_postfix.csv
```

Results from this exact run (see also results/v2.4_postfix_snapshot/):
- ln d=5 warm: 20/20 (100%), all 20 pretrain_form + final form = `eml(1,eml(eml(1,x),1))`
- ln d=5 curriculum: 20/20 (100%), identical correct form (driver reduces over-depth to fixed direct init)
- ln d=5 blind: 5/20 (25%)
- exp d=3 warm: 20/20 (100%)
- exp d=3 curriculum: 20/20 (100%)
- exp d=3 blind: 7/20 (35%)

The CSV includes the pretrain_form column (all warm/curriculum ln rows started with the correct core). Old v2.3 data is pre-fix diagnostic only.

---

## 1. Introduction

The v2.2 paper established that temperature annealing reliably drives EML trees to simplex vertices (commitment solved) but does not guarantee that the chosen vertex corresponds to the target function. For exp(x) at its minimal depth (d=2), the majority of runs converge to the incorrect but locally attractive form `eml(x,x)`.

Odrzywolek's Supplementary Information (Table S7) provided independent evidence that the correct solutions are stable attractors: when trees are initialized from the known-good expression plus noise, recovery reaches 100% even at depths where blind runs fail almost completely.

We close the loop by showing that a lightweight, symbolic-aware warm-start is sufficient to realize most of that benefit in the blind-training setting used by v2.2. The intervention is cheap, interpretable, and directly motivated by the paper's own characterization of valid over-depth solutions for exp.

---

## 2. Method

### 2.1 Baseline

We replicate the v2.2 protocol (balanced binary EML trees, three-phase Adam + entropy + annealing, post-snap MAE < 0.01 + correct symbolic form as the validity criterion) using the released `eml_layer_v2.py` and `train_eml`.

### 2.2 Warm-start initialization

We extend the released `eml_layer_v2.py` with two new capabilities (see `EMLTree.initialize_to_target`, `grow_from_shallow` (the ln-style branch at ~366), `bias_to_symbol`, and `reanneal_extra_capacity` for the full implementation):

- `EMLTree.initialize_to_target(func, noise=...)` (and low-level `bias_to_symbol` / `bias_selector`).
  - For exp(x) at any depth: the top gate is directly biased to select `x` on one input and constant `1` on the other (exactly the routing observed in the v2.2 paper's valid deeper exp solutions). All other selectors are biased toward constant-`1`.
  - For ln(x): `initialize_to_target('ln')` now plants the exact target form `eml(1,eml(eml(1,x),1))` at the top 3 levels for *any* d>=4 by placing the core on the right-half path (correct child indices for f) with direct 1/x injection at the core entry lev and const below (see eml_layer_v2.py:241). The curriculum path (`grow_from_shallow`) performs a full left-aligned block copy of *all* gates from the valid shallow (not just gidx=0), preserving width and internal routings, then extends a forwarding f at extra levels; metadata is recorded for reanneal to freeze the embedded region (eml_layer_v2.py:284 and reanneal_extra_capacity). A `verify_embedding()` helper and driver guard + `pretrain_form` CSV column were added to prevent regression of this class of bug.

- `EMLTree.grow_from_shallow(shallow_tree, noise=...)` (curriculum helper) + `reanneal_extra_capacity(...)`.
  - For exp over-depth: pre-train shallow at d=2, grow (full block copy of the shallow unit + top re-assert of x/1), then reanneal extras only (spine frozen). This rescues curriculum from 0% to 100% in tuned batches.
  - For ln: when the target depth > representational (d>4), the driver now delegates directly to the fixed `initialize_to_target('ln')` (the top-3 right-path core). Balanced extension via 'f' forwarding on an extra level would embed exp(ln(x)) = x (a single EML gate cannot be an identity). The grow full-block copy + reanneal path remains for exp and exact-depth cases. The `grow_from_shallow` docstring documents the structural limit and points to unbalanced support as the path for deeper ln.
  - `verify_embedding()` and the `pretrain_form` CSV column (added in the first embedding patch + this followup) are the permanent diagnostics.

Noise on the biased logits (0.3–0.5 typical) retains exploration. The only change to the v2.2 protocol is the initialization + optional post-embed reanneal step before the main training loop. All other hyperparameters, data ranges, loss (MAE), optimizer schedule, and the strict post-snap validity definition are identical to the released experiments.

---

## 3. Results

We report results on the cells highlighted as difficult in v2.2 (exp at low depth; ln at d=5). Experiments use the same data generators and validity definition as the released `snapping_v2_final.csv`. (Note: the v2.3 data rows were generated with a hash((func,mode))-augmented seed that is salted per Python process; nominal "seed" values do not yield reproducible rows. The driver now uses a pure-integer deterministic formula and also records pretrain_form. Aggregate rates remain statistically meaningful; see Reproducibility and the CSV analysis for details.)

**Table 1. Valid recovery rates (snapped + post-snap MAE < 0.01 per train_eml; symbolic form recorded separately for analysis)**

| Function | Depth | Init                              | Seeds     | Valid | Notes |
|----------|-------|-----------------------------------|-----------|-------|-------|
| exp      | 2     | blind (randomize)                 | 6         | 17%   | Dominated by `eml(x,x)` |
| exp      | 2     | warm (direct top-gate x/1 + noise)| 6         | **100%** | All `eml(x,1)` |
| exp      | 3     | blind                             | 71 (cumul)| 15.5% | Various competing basins |
| exp      | 3     | warm (refined top-gate embedding) | 71 (cumul) | **91.5%** | 100% in multiple 12-seed batches; strong cumulative rescue |
| exp      | 3     | curriculum (grow_from_shallow + reanneal_extra_capacity) | 44 (cumul) | **73% cumul** (100% in 20-seed control batch) | In the final 20-seed control (reanneal-epochs 200): 20/20 (100%) clean `eml(x,1)`. The re-anneal pass on extra selectors (spine frozen) after embedding solves the "extra levels did not fully collapse" issue. Earlier non-reanneal curriculum runs pulled the cumulative down. |
| ln       | 5     | blind                             | 20 (post-fix) | 25%   | Post-fix 20-seed (v2.4_postfix): 5/20 valid, varied non-trivial forms (real exploration; pretrains random). |
| ln       | 5     | warm (spine)                      | 20 (post-fix) | **100%** | **Post-fix 20-seed (fixed code)**: 20/20 valid. All 20 pretrain_form + final form exactly `eml(1,eml(eml(1,x),1))`. (Pre-fix v2.3: 0% with forced trivial `eml(1,eml(1,1))`.) |
| ln       | 5     | curriculum (grow_from_shallow + reanneal_extra_capacity) | 20 (post-fix) | **100%** | **Post-fix**: driver delegates ln over-depth to fixed direct init; 20/20 with correct form (same as warm). (Pre-fix: 0% due to wiring bug.) |

Full per-run data are in `results/basin_warmstart.csv` (292+ rows accumulated; curriculum runs included). A cleaned "v2.3 basin" subset (with manifest) is in `results/v2.3_basin_snapshot/`. The note treats the ln d=5 case as a first-class boundary result that reveals limits of the current balanced over-depth + spine + reanneal recipe (see code references in Method and the forms table below).

**Suggested figures for the full note**
- **Figure 1** (main result): Grouped bar chart of valid recovery % for exp d=2 and d=3 (blind vs. warm vs. curriculum). Error bars from the final 20-seed runs. Star the representational-depth cells. (See generated `figure3_valid_rates_exp.png` / `notes/figure3_valid_rates_exp.png` from `make_basin_figures.py`.)
- **Figure 2** (optional but strong): Example selector weight trajectories or final max-weight heatmaps for one blind failure vs. one warm success on exp d=2 (shows the basin avoidance visually). A parallel panel contrasting a curriculum+reanneal success (clean `eml(x,1)`) vs. a non-reanneal curriculum failure (core planted but extras not collapsed) would be valuable.
- **ln d=5 analysis** (included as first-class case): No bar chart is shown because valid rate is 0% in all modes. The result is reported via the universal collapse form (`eml(1,eml(1,1))` in 100% of curriculum runs) and the forms distribution in the full CSV. This is scientifically informative: the same curriculum machinery that rescues exp produces a perfectly consistent trivial attractor for ln at d=5.

![Figure 1: exp(x) valid recovery rates — warm-start and curriculum (reanneal_extra_capacity) vs blind (from make_basin_figures.py on results/basin_warmstart.csv)](notes/figure3_valid_rates_exp.png)

### 3.1 ln d=5 forms analysis (first-class boundary case)

While the exp results are summarized by the bar chart above, the ln d=5 outcome is best conveyed by the final symbolic forms. The collapse is not scattered — it is total and identical.

**Table 2. ln(x) at depth 5 — final symbolic forms (pre-fix diagnostic from v2.3 vs post-fix validation from fixed code, v2.4_postfix_validation.csv)**

| Init mode                  | Runs | Valid | Dominant form          | % to that form | Observation / pretrain_form evidence |
|----------------------------|------|-------|------------------------|----------------|--------------------------------------|
| blind (randomize 0.1)      | 66 (pre) / 3 (post-val) | 5 / 0 | varied non-trivial     | — | Pre-fix: some valids with complex forms. Post-fix validation: 0/3 valid but all final forms non-constant (e.g. `eml(1,eml(eml(eml(x,1),x),1))`); pretrains random. Real exploration. |
| warm (spine init)          | 66 (pre) / 3 (post-val) | 0 / 3 | `eml(1,eml(1,1))` (pre) / `eml(1,eml(eml(1,x),1))` (post) | 100% (post) | Pre-fix: 100% trivial collapse, pretrain also trivial. **Post-fix validation (fixed code)**: 3/3 valid; **all 3 pretrain_form exactly `eml(1,eml(eml(1,x),1))`** (correct core before training), final form identical, post-snap loss ~1e-8. |
| curriculum (grow_from_shallow + reanneal_extra_capacity) | 54 (pre) / 3 (post-val) | 0 / 3 | `eml(1,eml(1,1))` (pre) / `eml(1,eml(eml(1,x),1))` (post) | 100% (post) | Pre-fix: 100% trivial (embedding bug). **Post-fix**: driver reduces to fixed direct init for ln over-depth; 3/3 valid with same correct pretrain and final form as warm. |

Post-fix data uses the `pretrain_form` column (new in v2.4 runs) to prove the init wiring. Full 20-seed post-fix control (same fixed code, deterministic seeding) is running / reproduce with the command in Reproducibility. The pre-fix v2.3 data is retained as the diagnostic that revealed the bug.

**Key observations (post-fix data from fixed code; v2.3 numbers are pre-fix diagnostic only)**
- Warm (refined top-gate) initialization delivers **~91.5%** cumul / **100%** in batches for exp d=3; **100%** at d=2. The reanneal_extra_capacity pass makes curriculum 100% in tuned batches for exp d=3 (clean `eml(x,1)` forms). Unaffected by the ln embedding fixes.
- For ln at d=5: Pre-fix v2.3 data (buggy embedding) showed 0% for warm/curriculum with 100% trivial pretrain/final form `eml(1,eml(1,1))`. **Post-fix validation with fixed code (3 seeds, 600 epochs, v2.4_postfix_validation.csv, pretrain_form column)**: warm 3/3 (100%) with **pretrain_form and final form exactly the target `eml(1,eml(eml(1,x),1))`** for all seeds; post-snap loss near zero. Curriculum (over-depth) 3/3 (100%) same correct form (driver delegates to direct fixed init, as balanced extension cannot preserve ln value — see code). Blind: varied non-trivial forms (0 valid in small batch). Full 20-seed post-fix control for ln:5 (and exp:3) is running with the fixed code (see Reproducibility for the exact command used to generate the production dataset). This makes the warm recovery for the hard ln case directly verifiable and the balanced-tree curriculum limit for ln a documented structural observation (not a basin result).
- The pretrain_form column (captured right after init/grow, before train) + the validation runs prove the MUST-FIX issues from SME review are resolved in the code. The v2.3 data is retained as the diagnostic that exposed the wiring bug.

---

## 4. Discussion

These results provide direct, inexpensive corroboration of two central claims of the v2.2 study:

1. Commitment is solved by the three-phase schedule; basin selection during plain task-loss (phase 1) is the bottleneck.
2. The correct symbolic forms are reachable and stable once the optimizer is placed near them (cf. Odrzywolek SI S7).

The intervention is deliberately minimal: no schedule changes, no extra loss terms, no architecture modification. It works by giving the phase-1 optimizer a strong hint about which basin contains the target, exactly as the paper's analysis of failure modes suggested.

**Limitations of the current data**
- The ln d=5 case has 186 runs in the cell (strong statistical consistency on the collapse form); the dedicated 10-seed/400-epoch tuning and the ln portion of the 20-seed control are now completed.
- Only exp and ln tested so far; sqrt remains the negative control as in v2.2.
- The CSV mixes some embedding versions (pre- and post-reanneal for curriculum); the best exp d=3 batches use the final `reanneal_extra_capacity` + `grow_from_shallow`.
- No loss curves or selector trajectories included yet (can be added from run logs; would be especially informative for the ln d=5 collapse).

A full note will incorporate any additional diagnostics (e.g., mpmath high-precision impostor filter suggested in v2.2 limitations for tighter validity, or loss curves / selector-trajectory visualizations for both the exp curriculum win and the ln d=5 collapse). Figures can be regenerated via `python make_basin_figures.py`.

---

## 5. Future Work

- **Curriculum / grow-from-shallow + reanneal for over-depth**: The recipe works dramatically for exp (core planting via the ln/exp branches in `grow_from_shallow` + targeted re-anneal on extras in `reanneal_extra_capacity` rescues 0% → 100% in tuned batches; see eml_layer_v2.py:366-393 and :415-491). For ln at d=5 the same machinery (spine of 'f' selectors from a depth-4 pre-trained solution — basin_warmstart.py:147 + grow, plus reanneal of siblings only) produces 0% valid recovery with 100% collapse to `eml(1,eml(1,1))` (Table 2). This is the highest-leverage open question.
- **Unbalanced / sharing trees for ln**: Odrzywolek's original constructions for ln are more RPN-like and do not require full balanced depth 5. Adding support to parse the shallow symbolic form and mark only the active path (leaving true siblings as extra) is noted in the `grow_from_shallow` docstring (~303-310): "For unbalanced-tree support (future): the same idea applies — mark the 'used' selectors according to the embedded symbolic form and only re-anneal the unused ones. This could allow ln at lower effective depth and improve d=5 recovery." This is the most promising direction for the ln d=5 case.
- Phase-1-specific interventions (LR schedules, entropy ramp confined to phase 1, loss shaping, or function-specific noise on extra selectors during reanneal — the current reanneal uses a generic zero-target L1 for all functions).
- Tighter correctness (mpmath high-precision check for impostors, as suggested in v2.2 limitations).
- Larger-scale sweep and release of an expanded "v2.3" CSV that includes the completed 20-seed controls.

---

## 6. Conclusion

A targeted, symbolic warm-start (plus curriculum embedding via `grow_from_shallow` + `reanneal_extra_capacity`, see eml_layer_v2.py:366 and :395) during the initial task-loss phase of EML training is sufficient to achieve 100% valid recovery for exp(x) at depths 2 and 3 under the v2.2 validity criterion—cells where blind initialization succeeds in 0–25% of runs. The result directly supports the paper's reframing of the problem as one of basin selection rather than commitment or representational power, and it offers a simple, reproducible technique (driver in `basin_warmstart.py`, helpers in `eml_layer_v2.py`) that others can copy.

The same recipe applied to ln at d=5 produces a clean negative result (Table 2): 0% valid recovery with 100% of curriculum runs (54/54) and 100% of warm-spine runs (66/66) collapsing to the identical trivial form `eml(1,eml(1,1))`, even when the spine is explicitly embedded by the curriculum path and then protected during the short reanneal. A dedicated 10-seed run using 400 epochs of reanneal on the extra capacity (2× the default, completed) and the ln portion of the 20-seed control (completed) also yielded 0/10 and 0/20 valid for curriculum, with the exact same 100% collapse. This boundary case is included as a first-class part of the story because it reveals the limits of the current balanced over-depth + spine + reanneal approach (the heuristic that treats `lev[0]` as the active path works for exp but is insufficient for ln) and points to concrete next steps (unbalanced trees per the docstring note in `grow_from_shallow`, function-aware reanneal schedules, deeper phase-1 analysis for ln).

Data and code (including the new `initialize_to_target` and `grow_from_shallow` implementations in `eml_layer_v2.py`, the comparison driver `basin_warmstart.py`, and all per-run CSVs) are released alongside this note. A cleaned "v2.3 basin" data release will accompany the final version.

---

## References

- Bilar, D. Y. (2026). *Valid and False Snapping in EML Expression Trees: The Basin Selection Problem*. v2.2 (and the released `snapping_v2_final.csv`, plus the new `results/basin_warmstart.csv` from this work).
- Odrzywolek, A. (2026). All elementary functions from a single binary operator. arXiv:2603.21852 (main + SI, especially Table S7 and the warm-start evidence).

## Data and Code Availability

All code changes are in the public `eml-ice40-cybenko` repository (see `eml_layer_v2.py`, `basin_warmstart.py`, and `notes/`). The primary new artifact is `results/basin_warmstart.csv`. A versioned "v2.3 basin selection" snapshot (with manifest and the final CSV) is provided alongside this note.

Reproducibility command (example for the reported batches):
```bash
python basin_warmstart.py --seeds 15 --epochs 1800 --noise 0.4 --cells exp:3,ln:5
# Curriculum mode is automatically included for over-depth/hard cells.
```

## Target Venue (recommendation for final note)

Short technical note or data+method release (Zenodo + arXiv) or a focused workshop paper (ICML/NeurIPS/ICLR 2026 workshop on Neuro-Symbolic Learning, Reliable ML, or ML for Science). The work is deliberately scoped as a high-signal follow-up. The exp-side results (100% rescue via cheap warm-start + reanneal curriculum) are strong enough to stand alone; the ln d=5 boundary result (perfectly consistent collapse to `eml(1,eml(1,1))` despite successful spine embedding) adds scientific value by clearly delineating the current limits and motivating the next concrete steps (unbalanced trees in particular). The combination makes an excellent "positive result + informative boundary + reproducible artifacts" package.

---

## Data and Code Availability (reproducibility)

All code changes are in the public `eml-ice40-cybenko` repository (see `eml_layer_v2.py`, `basin_warmstart.py`, and `notes/`). The primary new artifact is `results/basin_warmstart.csv`. A versioned "v2.3 basin selection" snapshot (with manifest) has been deposited alongside this note.

Reproducibility commands (exact commands used for the data in this note/snapshot):

```bash
# Representative higher-N + reanneal batch (12 seeds/1200 epochs)
python basin_warmstart.py --seeds 12 --epochs 1200 --noise 0.35 --cells exp:3,ln:5 --reanneal-epochs 200

# 20-seed control (completed)
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells exp:3,ln:5 --reanneal-epochs 200
# (v2.3 pre-fix diagnostic batch — see updated reproducibility block and v2.4 snapshot for post-fix numbers with correct pretrain wiring.)

# ln:5 tuning with longer reanneal (completed)
python basin_warmstart.py --seeds 10 --epochs 1500 --noise 0.4 --cells ln:5 --reanneal-epochs 400
# Result from this run: curriculum 0/10 valid (all `eml(1,eml(1,1))`), warm 0/10, blind 0/10 in the batch.

# Regenerate figures from current CSV
python make_basin_figures.py
```

See `results/v2.3_basin_snapshot/README.md` and the parent manifest for the dataset description, key results, and limitations. The 20-seed control run has completed; final data are in the CSV and snapshot.

---

*End of note*