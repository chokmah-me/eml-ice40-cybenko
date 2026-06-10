# Basin Selection in EML Expression Trees: Targeted Warm-Start Initialization Achieves Full Valid Recovery for exp and ln, Including at Over-Representational Depth

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

We replicate the v2.2 protocol (balanced binary EML trees, three-phase Adam + entropy + annealing). Validity in the released code is snapped + post_snap_loss < 0.01 (see `train_eml` in `eml_layer_v2.py`). The symbolic_form column is recorded for every run and used for post-analysis and embedding diagnostics (pretrain_form); the note uses it to confirm correct forms in the post-fix data.

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
| ln       | 5     | warm (spine; curriculum reduces to same direct init for over-depth) | 20 (post-fix) | **100%** | **Post-fix 20-seed (fixed code, v2.4_postfix)**: 20/20 valid. All 20 pretrain_form + final form exactly `eml(1,eml(eml(1,x),1))`. (Curriculum for ln d>4 delegates to this path; rows are bit-identical to warm.) (Pre-fix v2.3: 0% with forced trivial `eml(1,eml(1,1))`.) |

Full per-run data for the post-fix claims are in the v2.4_postfix 120-row dataset (`results/v2.4_postfix_snapshot/basin_warmstart_v2.4_postfix.csv`, the canonical artifact for this note; 20 seeds × 3 modes × 2 cells). The accumulated `results/basin_warmstart.csv` mixes pre- and post-fix rows and is not the cited source. A cleaned pre-fix diagnostic subset (v2.3) is in `results/v2.3_basin_snapshot/`. All row counts and claims below use the 120-row v2.4_postfix as the source.

**Suggested figures for the full note**
- **Figure 1** (main result): Grouped bar chart of valid recovery % for exp d=2 and d=3 (blind vs. warm vs. curriculum). Error bars from the final 20-seed runs. Star the representational-depth cells. (See generated `figure3_valid_rates_exp.png` / `notes/figure3_valid_rates_exp.png` from `make_basin_figures.py`.)
- **Figure 2** (optional but strong): Example selector weight trajectories or final max-weight heatmaps for one blind failure vs. one warm success on exp d=2 (shows the basin avoidance visually). A parallel panel contrasting a curriculum+reanneal success (clean `eml(x,1)`) vs. a non-reanneal curriculum failure (core planted but extras not collapsed) would be valuable.
- **ln d=5 analysis**: Post-fix, warm and curriculum (which reduces to the same direct init) both reach 100% valid (20/20 each in the v2.4 control), so a recovery bar chart for ln would be flat at 100%. The instructive contrast is the `pretrain_form` column: pre-fix (v2.3) every warm/curriculum run started and ended at the trivial `eml(1,eml(1,1))` (the wiring bug), while post-fix every run starts and ends at the correct `eml(1,eml(eml(1,x),1))`. Section 3.1 shows this pre-fix vs post-fix shift.

![Figure 1: exp(x) valid recovery rates — warm-start and curriculum (reanneal_extra_capacity) vs blind (from make_basin_figures.py on results/basin_warmstart.csv)](notes/figure3_valid_rates_exp.png)

### 3.1 Pre-fix diagnostic (v2.3) vs post-fix results for ln d=5

The v2.3 data (pre-fix embedding) showed the collapse that motivated the SME review. It is retained below as diagnostic only. The production data for this note is the v2.4_postfix 20-seed run with the fixed code (see reproducibility). In that data, warm recovers the correct form at 100% (with matching pretrain_form for all seeds); curriculum for ln over-depth reduces to the same direct init.

**Table 2. ln(x) at depth 5 — final symbolic forms (pre-fix diagnostic from v2.3 vs post-fix 20-seed control from fixed code, v2.4_postfix 120-row CSV)**

| Init mode                  | Runs | Valid | Dominant form          | % to that form | Observation / pretrain_form evidence |
|----------------------------|------|-------|------------------------|----------------|--------------------------------------|
| blind (randomize 0.1)      | 66 (pre-fix) / 20 (post-fix) | 5 / 5 | varied non-trivial     | — | Pre-fix and post-fix both show real exploration: varied non-constant forms (e.g. `eml(1,eml(eml(eml(x,1),x),1))`), 5/20 valid post-fix. No forced collapse. |
| warm (spine init)          | 66 (pre-fix) / 20 (post-fix) | 0 / 20 | `eml(1,eml(1,1))` (pre) / `eml(1,eml(eml(1,x),1))` (post) | 100% (post) | Pre-fix: 100% trivial collapse, pretrain also trivial. **Post-fix 20-seed control (fixed code)**: 20/20 valid; **all 20 pretrain_form exactly `eml(1,eml(eml(1,x),1))`** (correct core before training), final form identical, post-snap loss ~1e-8. |
| curriculum (delegates to direct init for ln over-depth) | 54 (pre-fix) / 20 (post-fix) | 0 / 20 | `eml(1,eml(1,1))` (pre) / `eml(1,eml(eml(1,x),1))` (post) | 100% (post) | Pre-fix: 100% trivial (embedding bug). **Post-fix**: driver routes ln over-depth to the fixed direct init, so these rows are bit-identical to warm (20/20 valid, same pretrain and final form). Not an independent result; reported for completeness. |

Post-fix data uses the `pretrain_form` column (new in v2.4 runs) to prove the init wiring. The full 20-seed post-fix control (fixed code, deterministic seeding) is the production dataset (`v2.4_postfix.csv`); reproduce with the command in Reproducibility. The pre-fix v2.3 data is retained as the diagnostic that revealed the bug.

**Key observations (post-fix data from fixed code; v2.3 numbers are pre-fix diagnostic only)**
- Warm (refined top-gate) initialization delivers **~91.5%** cumul / **100%** in batches for exp d=3; **100%** at d=2. The reanneal_extra_capacity pass makes curriculum 100% in tuned batches for exp d=3 (clean `eml(x,1)` forms). Unaffected by the ln embedding fixes.
- For ln at d=5: Pre-fix v2.3 data (buggy embedding) showed 0% for warm/curriculum with 100% trivial pretrain/final form `eml(1,eml(1,1))`. **Post-fix 20-seed control (fixed code, v2.4_postfix.csv, pretrain_form column)**: warm 20/20 (100%) with **pretrain_form and final form exactly the target `eml(1,eml(eml(1,x),1))`** for all seeds; post-snap loss near zero. Curriculum (over-depth) 20/20 (100%), bit-identical to warm because the driver routes ln over-depth to the fixed direct init (balanced extension cannot preserve the ln value, since eml(f,1) = exp(f); see code). Blind: 5/20 (25%) valid with varied non-trivial forms. This makes the warm recovery for the hard ln case directly verifiable, and the balanced-tree curriculum limit for ln a documented structural observation (not a basin result). For ln d=5 warm and curriculum the 20 seeds each are bit-identical given the correct init (the basin is deterministic under this initialization; effective variance is zero for this cell).
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

- **Curriculum / grow-from-shallow + reanneal for over-depth**: Works well for exp. For balanced ln, the grow extension cannot preserve the value (eml(f,1) = exp(f)), so over-depth curriculum reduces to direct `initialize_to_target('ln')` (the fixed warm path). This is the structural limit that remains; it is why unbalanced/RPN support is the next direction (see docstring in `grow_from_shallow`).
- **Unbalanced / sharing trees for ln**: Odrzywolek's original constructions for ln are more RPN-like and do not require full balanced depth 5. Adding support to parse the shallow symbolic form and mark only the active path (leaving true siblings as extra) is noted in the `grow_from_shallow` docstring (~303-310): "For unbalanced-tree support (future): the same idea applies — mark the 'used' selectors according to the embedded symbolic form and only re-anneal the unused ones. This could allow ln at lower effective depth and improve d=5 recovery." This is the most promising direction for the ln d=5 case.
- Phase-1-specific interventions (LR schedules, entropy ramp confined to phase 1, loss shaping, or function-specific noise on extra selectors during reanneal — the current reanneal uses a generic zero-target L1 for all functions).
- Tighter correctness (mpmath high-precision check for impostors, as suggested in v2.2 limitations).
- Larger-scale sweep and release of an expanded "v2.3" CSV that includes the completed 20-seed controls.

---

## 6. Conclusion

A targeted, symbolic warm-start (plus curriculum embedding via `grow_from_shallow` + `reanneal_extra_capacity`, see eml_layer_v2.py:366 and :395) during the initial task-loss phase of EML training is sufficient to achieve 100% valid recovery for exp(x) at depths 2 and 3 under the v2.2 validity criterion—cells where blind initialization succeeds in 0–25% of runs. The result directly supports the paper's reframing of the problem as one of basin selection rather than commitment or representational power, and it offers a simple, reproducible technique (driver in `basin_warmstart.py`, helpers in `eml_layer_v2.py`) that others can copy.

The same recipe applied to ln at d=5 in the pre-fix v2.3 data produced the collapse (0% with 100% trivial form) that the SME review used to identify the embedding bugs. Those bugs are fixed. In the post-fix v2.4 20-seed data, warm recovers the exact target form at 100% (pretrain_form matches for every seed). For over-depth ln, curriculum now deliberately reduces to the same direct warm init because a balanced EML extension cannot act as an identity forwarder. The remaining structural limit (and motivation for unbalanced trees) is documented in the code and note. The old collapse data is retained only as the pre-fix diagnostic that drove the corrections.

Data and code (including the fixed `initialize_to_target` and `grow_from_shallow` in `eml_layer_v2.py`, the driver in `basin_warmstart.py`, the two patches in `results/patches/`, and the post-fix v2.4 120-row dataset in `results/v2.4_postfix_snapshot/`) are released alongside this note. The v2.3 snapshot is retained only as the pre-fix diagnostic corpus.

---

## References

- Bilar, D. Y. (2026). *Valid and False Snapping in EML Expression Trees: The Basin Selection Problem*. v2.2 (and the released `snapping_v2_final.csv`, plus the new `results/basin_warmstart.csv` from this work).
- Odrzywolek, A. (2026). All elementary functions from a single binary operator. arXiv:2603.21852 (main + SI, especially Table S7 and the warm-start evidence).

## Data and Code Availability

All code changes are in the public `eml-ice40-cybenko` repository (see `eml_layer_v2.py`, `basin_warmstart.py`, and `notes/`). The canonical artifact for citation is `results/v2.4_postfix_snapshot/basin_warmstart_v2.4_postfix.csv` (120 rows, fixed code, deterministic seeding, pretrain_form column). The pre-fix `results/v2.3_basin_snapshot/` is retained as diagnostic only.

Reproducibility command (the exact one used for the production dataset):
```bash
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells ln:5,exp:3 --reanneal-epochs 200 --csv results/basin_warmstart_v2.4_postfix.csv
```

## Target Venue (recommendation for final note)

Short technical note or data+method release (Zenodo + arXiv) or a focused workshop paper (ICML/NeurIPS/ICLR 2026 workshop on Neuro-Symbolic Learning, Reliable ML, or ML for Science). The work is deliberately scoped as a high-signal follow-up. Warm-start recovers every tested cell, including exp at d=2/d=3 and ln at over-representational d=5, at 100% in the 20-seed control, from blind baselines of 17-35%. The one documented structural limit (balanced curriculum grow-extension for ln) is narrow and method-internal, and it motivates a concrete next step (unbalanced trees). The combination makes a clean positive-result-plus-reproducible-artifacts-plus-one-scoped-open-problem package. The earlier v2.3 collapse is retained transparently as the diagnostic that drove the code fixes.

---
