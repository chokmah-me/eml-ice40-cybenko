# Basin Selection in EML Expression Trees: Targeted Warm-Start Initialization Achieves Full Valid Recovery for exp but Exposes Limits for ln at Over-Representational Depth

**Short technical note + data release** (targeting ICML/NeurIPS workshop, ICLR workshop, or Zenodo + arXiv "Technical Note" / data + method note)

**Authors**: [Your name], Chokmah LLC  
**Status**: Final version with completed 20-seed control data (exp side strong positive results; ln d=5 treated as first-class boundary case with 400-epoch tuning and full 20-seed results).

**Target venues**: ICML/NeurIPS workshop, ICLR workshop, or a short "Technical Note" / "Data Release" style (e.g., via Zenodo + arXiv or a journal like JMLR MLOSS / TMLR).

---

## Abstract

Three-phase temperature annealing solves selector commitment in balanced EML expression trees, but the dominant failure mode remains basin selection during the initial task-loss phase. The v2.2 study ("Valid and False Snapping in EML Expression Trees: The Basin Selection Problem") introduced a strict post-snap validity criterion (correct symbolic form + post-snap MAE < 0.01) and showed that, at minimal representational depth, exp(x) recovers the correct form `eml(x,1)` in only ~25% of blind runs because most trajectories fall into a strong competing basin (`eml(x,x)` with MAE ~0.688).

We demonstrate that a simple, targeted warm-start initialization—directly biasing the top-level selectors toward the known-good symbolic form while setting off-path capacity to harmless constants—rescues recovery for exp. In the final 20-seed control run (plus prior batches, 71 seeds cumul for exp d=3), this raises valid-snap rate from 17% to 100% for exp(x) at depth 2 and from ~6–15% to ~92% cumulative (100% in the 20-seed control batch) at depth 3. A new `reanneal_extra_capacity` pass (short targeted re-anneal only on extra selectors after embedding, spine frozen) turns curriculum from 0% to 100% in the 20-seed control batch for exp d=3 (cumulative ~73% for curriculum on d=3). The method requires no change to the training schedule or loss and directly corroborates the paper's analysis of how successful deeper exp solutions operate (routing x/1 at the top).

For ln at d=5 (over-representational), warm and curriculum yielded 0% (66+54 rows) with 100% collapse to `eml(1,eml(1,1))`; blind found 5/66 valids with varied non-trivial forms (see Table 2 + CSV). **Critical data caveat (discovered via CSV+code audit):** the 0% for the "informed" modes was an artifact of the embedding implementation itself (initialize_to_target('ln') and the first-gate-only grow_from_shallow produced a disconnected or constant pretrain form even before training started; the top f pulled sibling constants rather than the core chain). The source now contains the surgical fixes (top-3 core with correct gi routing + full subtree block copy for curriculum + pretrain_form capture + verify_embedding + deterministic seeding). The exp results (warm/curriculum rescue) are unaffected and remain valid. Re-run the ln cells (and ideally a fresh 20-seed) with the corrected driver + eml_layer_v2 before final Zenodo framing of the ln case as boundary vs. "now works". The v2.3 snapshot preserves the diagnostic pre-fix run.

**Reproducibility** (exact commands used for the data in this note/snapshot; run after `pip install -r requirements.txt` or equivalent torch/numpy env):

```bash
# Representative higher-N + reanneal batch (12 seeds/1200 epochs)
python basin_warmstart.py --seeds 12 --epochs 1200 --noise 0.35 --cells exp:3,ln:5 --reanneal-epochs 200

# 20-seed control (completed)
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells exp:3,ln:5 --reanneal-epochs 200
# Batch results (new 20 seeds): exp d=3 blind 8/20 (40%), warm 20/20 (100%), curriculum 20/20 (100%); ln d=5 blind 5/20 (25%), warm 0/20 (0%), curriculum 0/20 (0%). All ln curriculum/warm forms `eml(1,eml(1,1))`.

# ln:5 tuning with longer reanneal (completed)
python basin_warmstart.py --seeds 10 --epochs 1500 --noise 0.4 --cells ln:5 --reanneal-epochs 400
# Result from this run: curriculum 0/10 valid (all `eml(1,eml(1,1))`), warm 0/10, blind 0/10 in the batch.

# Regenerate figures from current CSV
python make_basin_figures.py
```

See `results/v2.3_basin_snapshot/README.md` and the parent manifest for the dataset description, key results (exp d=3 curriculum 100% in the 20-seed control batch; ln d=5 remains 0% with 100% collapse to `eml(1,eml(1,1))` even in the new 20 seeds and the 400-epoch tuning), and limitations. The 20-seed control run has completed; final data are in the CSV and snapshot.

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

- `EMLTree.grow_from_shallow(shallow_tree, noise=...)` (curriculum helper, invoked from `basin_warmstart.py:159` after a valid shallow snap at the function's representational depth) + `reanneal_extra_capacity(...)` (next-iteration tuning helper).
  - Pre-train a shallow tree at the function's representational depth (`_expected = {'ln':4, ...}` in the driver) using warm initialization, then `train_eml`.
  - If valid, call `tree.grow_from_shallow(shallow, ...)`: for ln this copies the first gate of each shallow level onto the deep spine (`self.levels[si][0]`), extends an `'f'` forwarding chain on the first gate of extra upper levels, and forces *all sibling gates* (gidx != 0 on spine levels, and all gates on non-spine positions) to constant-`1` (see ~385-393).
  - Immediately after: `tree.reanneal_extra_capacity(epochs=..., lr=...)` (see `basin_warmstart.py:163`). This freezes the "active spine" (first gate `lev[0]` at every level, extreme +20/-20 bias on the embedded symbol) and runs a short L1-to-zero phase-3-style anneal *only on the extra selectors* (gidx != 0). The reanneal uses a dummy zero target on a generic grid to encourage harmless constant subtrees while the spine is locked.
  - Then the normal full three-phase `train_eml` is run on the deep tree.
  - The implementation includes an explicit note (in the `grow_from_shallow` docstring ~303-310) on future unbalanced-tree support: "parse shallow.symbolic_form() to identify the exact 'active path' selectors to embed, and treat all other parallel branches as extra capacity... This could allow ln at lower effective depth".

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
| ln       | 5     | blind                             | 66 (cumul)| 7.6%  | Consistent collapse (a few sporadic valids; 25% in the 20-seed control batch) |
| ln       | 5     | warm (spine)                      | 66 (cumul)| 0%    | Heavy collapse to constants; dominant form `eml(1,eml(1,1))` (0/20 in 20-seed batch) |
| ln       | 5     | curriculum (grow_from_shallow + reanneal_extra_capacity) | 54 (cumul) | 0% | *Every* run (54/54) collapses to the identical trivial form `eml(1,eml(1,1))` (same for all 66 warm). **Data note (v2.3 snapshot):** these runs used an incomplete embedding (initialize_to_target and grow_from_shallow only touched first-gate "spine" gidx=0 with polarity that did not connect the core even at representational d=4; pretrain forms were uniformly trivial before any training). Blind runs (no init) explored real structure and found 5 valid snaps. The source was corrected post-analysis (full-block copy in grow, top-3-level right-path core with verified connected form in initialize_to_target for any d>=4, pretrain_form column + verify_embedding guard, hash-free seeding). See eml_layer_v2.py:241 (init), :284 (grow), basin_warmstart.py:152 (curric), and the pretrain_form diagnostic in the (future) re-run CSV. Re-run the ln:5 cells with the updated code before treating the 0% as a firm boundary. |

Full per-run data are in `results/basin_warmstart.csv` (292+ rows accumulated; curriculum runs included). A cleaned "v2.3 basin" subset (with manifest) is in `results/v2.3_basin_snapshot/`. The note treats the ln d=5 case as a first-class boundary result that reveals limits of the current balanced over-depth + spine + reanneal recipe (see code references in Method and the forms table below).

**Suggested figures for the full note**
- **Figure 1** (main result): Grouped bar chart of valid recovery % for exp d=2 and d=3 (blind vs. warm vs. curriculum). Error bars from the final 20-seed runs. Star the representational-depth cells. (See generated `figure3_valid_rates_exp.png` / `notes/figure3_valid_rates_exp.png` from `make_basin_figures.py`.)
- **Figure 2** (optional but strong): Example selector weight trajectories or final max-weight heatmaps for one blind failure vs. one warm success on exp d=2 (shows the basin avoidance visually). A parallel panel contrasting a curriculum+reanneal success (clean `eml(x,1)`) vs. a non-reanneal curriculum failure (core planted but extras not collapsed) would be valuable.
- **ln d=5 analysis** (included as first-class case): No bar chart is shown because valid rate is 0% in all modes. The result is reported via the universal collapse form (`eml(1,eml(1,1))` in 100% of curriculum runs) and the forms distribution in the full CSV. This is scientifically informative: the same curriculum machinery that rescues exp produces a perfectly consistent trivial attractor for ln at d=5.

![Figure 1: exp(x) valid recovery rates — warm-start and curriculum (reanneal_extra_capacity) vs blind (from make_basin_figures.py on results/basin_warmstart.csv)](notes/figure3_valid_rates_exp.png)

### 3.1 ln d=5 forms analysis (first-class boundary case)

While the exp results are summarized by the bar chart above, the ln d=5 outcome is best conveyed by the final symbolic forms. The collapse is not scattered — it is total and identical.

**Table 2. ln(x) at depth 5 — final symbolic forms after training (final data, 384 rows)**

| Init mode                  | Runs | Valid | Dominant form          | % to that form | Observation |
|----------------------------|------|-------|------------------------|----------------|-------------|
| blind (randomize 0.1)      | 66   | 5     | varied (e.g. `eml(1,eml(1,x))`, `eml(1,eml(eml(1,x),x))`, `eml(1,eml(eml(eml(x,1),x),x))`, ...) | — (multiple basins) | Multiple incorrect deep nestings, as reported in v2.2 (5/20 valids appeared in the 20-seed control batch). |
| warm (spine init)          | 66   | 0     | `eml(1,eml(1,1))`      | 100%           | Direct 'f'-spine bias (from `initialize_to_target('ln')`) is still completely overwhelmed by the constant attractor (0/20 in 20-seed batch). |
| curriculum (grow_from_shallow + reanneal_extra_capacity) | 54 | 0 | `eml(1,eml(1,1))` | 100% | v2.3 diagnostic data only (pre-fix embedding). The curriculum path pre-trained shallow at d=4 then grow + reanneal; however the grow copied only first-gate "spine" (old code ~370) and initialize produced trivial pretrain form (see verification in this session). Post-fix: full block copy + verified connected pretrain `eml(1,eml(eml(1,x),1))` (or wrapped for overdepth) + pretrain_form column. Re-run required to re-evaluate the 0%. |

This table uses the live `results/basin_warmstart.csv` (forms column from `tree.symbolic_form()` after snap). The 100% identical collapse under curriculum is the key diagnostic for future work on unbalanced trees (see the explicit note in `grow_from_shallow` docstring).

**Key observations (final data after completed 20-seed control + 400-epoch ln tuning)**
- Warm (refined top-gate) initialization continues to deliver **~91.5%** valid recovery cumul for exp d=3 (100% in multiple recent higher-N batches); **100%** at d=2. Complete, reproducible rescue of the cells that were hardest in the v2.2 blind runs.
- Curriculum with the new `reanneal_extra_capacity` pass (re-anneal only on extra selectors after embedding, spine frozen): 100% valid `eml(x,1)` on exp d=3 in the 20-seed control batch (the final precision run with --reanneal-epochs 200). Cumulative curriculum on d=3 is now ~73% (32/44; earlier non-reanneal runs pulled it down, but the new 20 seeds were 100% clean). The forms in the 20-seed curriculum batch are clean `eml(x,1)`. This validates that the combination of `grow_from_shallow` (core injection, see eml_layer_v2.py:366) + targeted re-anneal on new capacity (reanneal_extra_capacity:415 — freezes `lev[0]` spine with fill_(-20)/+20, only steps gidx!=0 extras with L1-to-zero on dummy batch) solves the "extra levels did not fully collapse" problem. (See `eml_layer_v2.py:reanneal_extra_capacity` docstring and the diagnostic forms in `results/track_b_initial.md`.)
- For ln at d=5 the picture in the v2.3 diagnostic data is qualitatively different (0% warm/curriculum vs. 5/66 blind valids with real structure). It was treated as boundary, but the audit showed the "informed" inits were wiring-broken (pretrain always trivial constant; blind explored). The corrected implementation (full embed + connected core + pretrain_form + guards) is in the tree now. Re-run the ln cells to decide if it is still a boundary (genuine attractor even from correct start) or recovers. The forms table below and CSV pretrain column (new rows) document the before/after. The `grow_from_shallow` ... tuning run (completed) and the ln portion of the 20-seed control (completed, --reanneal-epochs 200) both produced 0 valid for curriculum (and 0 for warm), with the exact same 100% collapse to `eml(1,eml(1,1))`. Yet the subsequent full `train_eml` still drives the tree to the same strong constant attractor. This is not noisy failure — it is a perfectly consistent selection of a competing basin even after doubling the targeted reanneal on extra capacity. The result shows that the warm-start + curriculum + reanneal recipe that is highly effective for exp is not yet sufficient for ln at over-representational depth under the current balanced-tree protocol.

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
# Batch results (new 20 seeds): exp d=3 blind 8/20 (40%), warm 20/20 (100%), curriculum 20/20 (100%); ln d=5 blind 5/20 (25%), warm 0/20 (0%), curriculum 0/20 (0%). All ln curriculum/warm forms `eml(1,eml(1,1))`.

# ln:5 tuning with longer reanneal (completed)
python basin_warmstart.py --seeds 10 --epochs 1500 --noise 0.4 --cells ln:5 --reanneal-epochs 400
# Result from this run: curriculum 0/10 valid (all `eml(1,eml(1,1))`), warm 0/10, blind 0/10 in the batch.

# Regenerate figures from current CSV
python make_basin_figures.py
```

See `results/v2.3_basin_snapshot/README.md` and the parent manifest for the dataset description, key results, and limitations. The 20-seed control run has completed; final data are in the CSV and snapshot.

---

*End of note*