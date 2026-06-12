# Roadmap: Expansions for eml-ice40-cybenko (post-v2.2)

**Status**: Living document. Last updated: 2026-04 (right after v2.2 release).  
**Scope**: What we delivered, the broader historical vision, and prioritized expansion tracks that build directly on the released artifacts.  
**Audience**: The author, future collaborators, or anyone who wants to understand where this project can go next.

---

## Executive Summary

**v2.2 delivered** a focused, high-quality methodological paper + reproducible artifacts:

> "Valid and False Snapping in EML Expression Trees: The Basin Selection Problem" (Bilar, Chokmah LLC, April 2026).

Three-phase temperature annealing (Adam → entropy penalty → annealing) **solves the commitment problem** across all tested conditions: every selector reaches a simplex vertex. It does **not** solve the *basin selection* problem. The paper introduces the minimum viable reporting standard — **valid_snap** (snapped *and* correct symbolic form with post-snap MAE < 0.01) vs **false_snap** — and shows via 240 deterministic runs (plus cross-checks against Odrzywolek SI) that recovery depends on the relationship between tree depth and a function's *representational depth*, modulated by basin geometry. Extra depth can be an escape hatch for some targets (exp) and a liability for others (ln). Pre-snap loss and raw snappability hide the dominant failure mode.

**The repo name and history** (`prior_versions/dev/`) reveal a larger original vision: EML operator (Odrzywolek 2026) + Cybenko universality + real iCE40 FPGA deployment ("ice40-cybenko"), with off-device training → snap → fixed-point → Verilog → synthesis, plus a multi-paper publication strategy (snapping mechanism, hardware Cybenko validation, formal verification for governance/autonomous systems).

**v2.2 was a deliberate, strong first release** that narrowed scope to produce clean, citable artifacts (the CSV is now an empirical oracle; the validity criterion is a reusable contribution). The broader vision is not abandoned — it is now better grounded.

This roadmap organizes **six expansion tracks** (A–F) that leverage the released `eml_layer_v2.py`, `experiment_v2.py`, `snapping_v2_final.csv`, paper framing, and historical material. It includes decision frameworks, concrete immediate actions, and maintenance guidance.

---

## Current Foundation (What v2.2 Gives Us)

Leverage these artifacts heavily; they are the highest-signal output so far.

- **`eml_layer_v2.py`**: Clean, Odrzywolek-param-count-exact balanced binary `EMLTree` + `InputSelector` (temperature, `snap()`, `symbol()`, entropy, `snappability()`), `EMLGate`, and `train_eml()` with the exact 3-phase protocol + post-snap validity logic. Self-test for param counts and basic recovery.
- **`experiment_v2.py`**: Config-driven, checkpointed driver (20 seeds × 3 functions × 4 depths, 2000 epochs, per-function LRs). Produces the rich CSV.
- **`snapping_v2_final.csv`** (238 non-NaN runs): `function,depth,seed,snapped,final_loss,snappability,nan_epoch,converged,symbolic_form,expected_depth,post_snap_loss,valid_snap`. The released data is the empirical baseline for all future basin work.
- **Paper + figures + Odrzywolek SI data** (`dyb-2026m-elm-basin.md` / PDF, `figure*.png`, `odrzywolek_si_table_s5.csv`, `make_figure2.py`): Precise framing (commitment solved, basin is the issue; representational depth matters; over-depth effects differ by basin geometry), limitations (one optimizer/loss/architecture, N=20, untuned schedule), and forward pointers (mpmath impostor checks for tighter validity, ablation on escape mechanisms).
- **Archival**: `CITATION.cff`, `.zenodo.json`, Zenodo DOI integration, MIT + CC-BY licensing.
- **Historical material** (read-only reference in `prior_versions/dev/`): `fixed_point_converter.py` (Q-format, LUT exp/ln, `EMLWeightConverter`, RTL stub + bitstream init generation), `INTEGRATION.md` (full PyTorch → iCE40 workflow), `NOVELTY_ANALYSIS.md`, `RESEARCH_QUESTIONS.md`, `PAPER_PRIORITY_MATRIX.txt`, `EXECUTIVE_SUMMARY.md`, older `eml_layer.py` + `test_eml_poc.py`, etc. These document the original 3-paper + hardware ambition and contain reusable design/quantization ideas.

**Key v2.2 insights that open doors**:
- Phase 1 (plain Adam) is the critical basin-selection phase; phases 2–3 only commit to the nearest vertex.
- Warm-start / curriculum / better initialization during phase 1 are high-leverage (Odrzywolek SI S7 already shows 100% recovery from noised-correct inits at d=5/6).
- Over-depth is not uniformly bad (helps exp(x) escape `eml(x,x)` at d=4).
- The validity filter (post-snap MAE < 0.01 + correct form) cleanly separates signal from the 0.688 MAE `eml(x,x)` competitor and similar basins.

---

## Expansion Tracks

### Track A: Analysis & Usability Extensions (Lowest barrier, highest immediate leverage)
**Why now**: The CSV + validity machinery already exist. Mining them + running cheap sensitivity sweeps costs almost nothing and produces immediate insights, figures, or a short data note / blog / supp material.

**Scope ideas**:
- Deeper CSV mining: basin composition (which false forms dominate per cell), symbolic_form clustering, correlation of pre-snap loss vs post-snap validity, snappability trajectories if logs are kept.
- Sensitivity: entropy_coeff, anneal schedule (final T, ramp), optimizer (AdamW, SGD+momentum), loss (MAE vs MSE), LR, more seeds (the paper notes N=20 is coarse near edges), gradient clipping effects.
- Tooling: selector weight heatmaps over epochs, "basin escape" visualizations, a tiny CLI or notebook to browse runs by (function, depth, valid_snap).
- "Failure catalog": document the common false forms and their post-snap losses.

**Outputs**: New analysis scripts/notebooks under `analysis/`, additional figures or tables, optional short note or v2 appendix.
**Effort**: Days (quick mining) to 1–2 weeks (full sensitivity + polished viz).
**Quick wins**: One script that reproduces the per-cell valid rates + dominant false form for exp d=2/3/4; a small "post-v2 insights" section or separate note.
**Prereqs**: None beyond current Python env + the CSV.

### Track B: Basin Selection Improvements (Direct scientific follow-on)

**Recent progress (this session)**: 
- Added `EMLTree.initialize_to_target(func, noise=...)` (plus `bias_to_symbol` and `bias_selector` helpers) in `eml_layer_v2.py`. This is the central reusable primitive for the track — it implements the "known-good symbolic + noise" warm-start that gave Odrzywolek SI Table S7 its 100% high-depth recovery.
- Created `basin_warmstart.py` (focused blind vs warm comparison on the hard cells the v2.2 paper called out: exp d=2/3 and ln d=5). The script is already launched (background) with 8 seeds / 1200 epochs and will write `results/basin_warmstart.csv`.
- Smoke validation (no training): exp d=2 top gate is now perfectly biased to `eml(x,1)` (max weight 1.0 on x and 1); ln d=4/5 correctly biases the outer `eml(1, f)` routing. Full numbers coming from the running experiment.
- See `results/track_b_initial.md` for the exact launch command, smoke output, and the immediate next actions for this track.

Once the background run finishes you will have a direct head-to-head table on exactly the cells where the paper showed the basin problem is worst. This is the highest-leverage, paper-aligned direction in the whole roadmap.
**Why now**: The paper's core claim is that *basin selection* (not commitment) is the bottleneck. v2 gives the measurement framework and baseline rates. Improving recovery in the hard cells (exp d=2, ln d=5, etc.) is the natural next experiment.

**Scope ideas**:
- Warm-start / curriculum: initialize from shallower valid solutions (or subexpressions), layer-wise training, or Odrzywolek-style noised-correct inits (his S7 result).
- Phase-1 interventions: different inits (the `randomize(scale)` is there), curriculum on data or depth, entropy schedule search during phase 1 only.
- Architectural: unbalanced / sharing trees (closer to Odrzywolek's ln RPN construction that needs only depth 3 in his notation); limited DAG sharing.
- More targets + tighter correctness: functions with closer competing basins; mpmath 128-digit re-evaluation for impostors (explicitly called out in the paper's limitations/impostor discussion as future work when tightening the 0.01 threshold).
- Over-depth as technique: systematic study of when extra capacity helps vs hurts (document the exp(x) escape pattern).

**Outputs**: Extended experiment driver or configs, new larger "v2.3" CSV release with higher recovery rates in key cells, possible short "basin escape methods" workshop paper or note.
**Effort**: 2–6 weeks (highly parallelizable with A).
**Leverage**: The `train_eml` post-snap validity logic, `symbolic_form`, and the released CSV as ground truth.
**Prereqs**: v2 code (already excellent).

### Track C: Hardware Realization (Revive the "ice40" in the repo name)

**Progress (2026-06-11)**: Stage 1 delivered. `hardware/` package: bit-exact fixed-point model (`fixed_point.py`, range-reduced ln replacing the old non-synthesizable log-indexed LUT), netlist extraction with constant folding from snapped v2 trees (`converter.py`), Verilog + LUT-hex + testbench emission (`verilog_gen.py`), end-to-end PoC (`run_poc.py`). Results on the canonical recovered forms: exp d=2 `eml(x,1)` Q8.8 MAE 0.0027; ln d=4 `eml(1,eml(eml(1,x),1))` Q10.12 MAE 0.0003 / max 0.0011 — both under the paper's 0.01 threshold pointwise at Q10.12. RTL in `hardware/rtl/`, report in `results/hardware_poc_report.md`, doc in `hardware/HARDWARE.md`.

**Progress (2026-06-11, stage 2)**: CSV-driven generalization done — `hardware/form_parser.py` parses any `symbolic_form` into a netlist; `python -m hardware.run_csv` evaluates and emits RTL for every unique valid form in the canonical CSVs (report: `results/hardware_csv_forms_report.md`). Finding: all 152 valid rows collapse to the two canonical forms, so the emitted RTL covers 100% of released valid solutions; parsed netlists match tree-extracted numerics bit-for-bit. Remaining: HDL sim cross-check + yosys/nextpnr synthesis (blocked: no toolchain installed — iverilog/verilator/yosys/nextpnr all absent), pipelined RTL variant.
**Why now**: Depth-2 (and some d=4) cells now have well-characterized high valid_snap rates and exact recovered forms (`eml(x,1)`, the ln 7-gate form, etc.). These are perfect targets for quantization and RTL generation. The original pipeline code exists in `prior_versions/dev/`.

**Scope**:
- Adapt `fixed_point_converter.py` concepts (`FixedPoint` Q8.8/Q10.8, LUT-based exp/ln with interpolation, `EMLWeightConverter`) to the v2 `InputSelector` (`.logits`) and `EMLTree` structure. Start with depth 2 (lowest LUT budget, highest reliability in several cells).
- Generate clean Verilog emission (EML gate + balanced tree) or high-quality stubs + resource estimates.
- End-to-end on real valid snaps from the CSV: snap → extract weights → quantize → fixed-point eval error vs double → (optional) yosys/nextpnr synthesis + LUT/timing report.
- Document the exact open-source iCE40 flow (Project Trellis, nextpnr, iceprog) and any constant-baking requirements.
- Measure quantization error on the *actual* recovered symbolic expressions (not just random weights).

**Outputs**: `hardware/` (or `fpga/`) directory with adapted converter, generators, example RTL for a depth-2 valid case from the data, error report + resource numbers, updated `INTEGRATION.md` or new `HARDWARE.md`.
**Effort**: 2–8 weeks (estimation + stubs first; full bitstream depends on local toolchain). Can be staged.
**Risk/dependency**: External Lattice toolchain (yosys + nextpnr + iceprog) not present by default; start simulation/estimation-first.
**Historical note**: The old converter already produces RTL stubs and bitstream init data; the main work is porting the weight extraction to v2's module layout and tying it to real snapped forms from the CSV.

### Track D: Scaling & Universality (Cybenko / Forests)
**Why now**: The original vision included empirical validation that iCE40 (or EML in general) can act as a learnable universal approximator via forests of shallow trees (Cybenko-style width scaling). v2's over-depth observations already hint at width-vs-depth trade-offs.

**Scope**:
- Implement "EML forests" (ensembles of independent shallow trees, e.g., depth 2) on top of the v2 layer.
- Width-scaling experiments (5, 10, 20, 50 trees) on ln/exp/sin/tan/etc.; measure error vs width and compare to 1/√W.
- Resource modeling (even without full synthesis): LUT cost linear in width? Interaction with the validity criterion (each tree snaps independently?).
- Tie results back to the basin paper (when is depth helpful vs when is width + combination better?).

**Outputs**: Forest trainer + scaling plots + resource model; possible DAC/ICCAD/FPL-style paper or workshop version ("Hardware-Efficient Universal Approximation with EML Forests").
**Effort**: 4–10 weeks (can run in parallel with parts of B).
**Alignment**: Directly continues the pre-v2 "Paper 2" thread with better empirical grounding from v2.

### Track E: Verification, Trust & Governance (Highest differentiation)
**Why now**: The snapped outputs are now *exact symbolic expressions* (human-readable closed forms) rather than opaque weights. This is the perfect input for formal or semi-formal methods. The old vision called this the "differentiator" (governance/autonomous systems angle, provably correct math units, reduced HITL arguments).

**Scope**:
- On the released valid snapped forms: prove (within ε) equivalence to the target, or equivalence between the floating-point and a fixed-point model.
- Tools: Lean 4 (full proofs — the original heavy-hitter plan), or lighter starters (interval arithmetic, symbolic execution, SMT, or even just high-precision numeric + the mpmath filter the paper already suggests).
- End-to-end pipeline story: raw data/telemetry → valid-snapped expression → (optional) hardware model → machine-checked certificate.
- Expand the impostor discussion into a practical method.
- Governance framing (interpretable + provable on embedded hardware for autonomous systems).

**Outputs**: Proof scripts / small framework + case studies on the actual released valid snaps (e.g., ln at d=4, exp at d=4), paper in FM/EMSOFT/CCS or a systems/governance venue.
**Effort**: 8–16+ weeks (Lean learning curve if new; can start lighter-weight and escalate).
**Unique value**: Directly addresses the original "your differentiator" while standing on the clean v2 validity criterion and released exact forms. Few others are working at the ML + formal methods + embedded intersection with a governance story.

### Track F: Packaging, Usability & Community
**Why now**: The core is small, clean, and already has a self-test. Making it easy for others (or future self) to extend the basin experiments or plug in new basin-escape methods multiplies impact.

**Scope**:
- Minimal package layout (`pyproject.toml` or `setup.py`, `pip install -e .`) exposing `EMLTree`, `train_eml`, validity helpers, CSV loaders.
- Examples + a small interactive browser (CLI or lightweight web UI) over the 240 runs.
- Improved experiment harness (YAML/JSON configs, easy parallel seeds, automatic high-prec post-processing hooks).
- Notebooks walking through one full run, basin analysis, or "add your own warm-start".
- Docs and contribution notes.

**Outputs**: Installable package, `examples/`, `notebooks/`, better CLI, optional small demo app.
**Effort**: 1–4 weeks.
**Benefit**: Lowers the barrier for Track A/B work by others and makes the validity criterion easy to adopt as a standard.

**Cross-cutting concerns** (apply across tracks):
- More seeds / bigger sweeps (paper acknowledges the limitation).
- Multivariate inputs (natural once basin methods mature).
- Reproducibility (pin torch/numpy, optional Docker for hardware toolchain).
- Licensing / citation hygiene on new releases (update `CITATION.cff`, Zenodo on tagged releases).

---

## Prioritization & Decision Framework

**Leverage order heuristic**:
1. A (analysis) + B (basin improvements) first — they use the v2 artifacts 100% and can produce quick publishable or citable follow-ups.
2. C (hardware) early enough to realize the repo name and original pipeline promise; depth-2 PoC is low-risk.
3. D (forests/Cybenko) in parallel with B/C.
4. E (verification/governance) as the long-term differentiator (can start light-weight while doing others).
5. F (packaging) as soon as any track generates reusable extensions.

**Scenario-based starting points** (put these in the doc so it supports discussion):

- **"Fast follow-on pubs / data releases leveraging v2"** → Start with A + B (warm-starts + sensitivity). Target: short workshop/note + v2.3 CSV in weeks.
- **"Realize the ice40-cybenko name and original vision"** → Prioritize C (depth-2 valid case from the CSV) in parallel with A. Produce hardware PoC note + resource numbers.
- **"Governance / autonomous systems differentiator is core"** → Emphasize E (start with lightweight verification on the released forms) + C (provable on embedded). A/B as supporting science.
- **"Broad exploration, keep options open"** → A + small slice of B + F (packaging) so future work is easier.
- **"Time-constrained (part-time)"** → A (quick wins) + one focused B experiment (e.g., one warm-start strategy on the hard cells) + light C stubs.

**Your domain signal** (from historical notes): autonomous systems, governance, defense tech, HITL arguments → E + C have the highest long-term alignment and uniqueness.

---

## Immediate Next Actions (Runnable This Week)

1. Read (or re-read) the v2 paper limitations + conclusion and the key historical docs (`NOVELTY_ANALYSIS.md`, `INTEGRATION.md`, `fixed_point_converter.py` top-level classes).
2. Run the quick foundation validation (see below) and a CSV rate summary (see `make_figure2.py` or equivalent).
3. Pick one Track A quick win: write a small script that, for every (function, depth) cell, prints the set of `symbolic_form` values among the false snaps and their post-snap losses. Commit it under `analysis/`.
4. Decide (and note in an issue or this doc): primary goal for the next 4–8 weeks (see scenarios above).
5. If leaning hardware (C): sketch the minimal port of the old converter to v2's `InputSelector.logits` for depth 2; generate one RTL stub from a known valid snap in the CSV.
6. If leaning basin work (B): prototype one warm-start strategy (e.g., take a valid d=2 solution and grow the tree) and measure recovery lift on exp d=2 or ln d=5.
7. Update this `ROADMAP.md` with your decision + any new quick actions.

**Validation commands (foundation is healthy)**:

```bash
# Quick module sanity (fast)
python -c "
from eml_layer_v2 import EMLTree
import torch
t = EMLTree(depth=2)
print('Depth-2 params:', t.num_params())
y = t(torch.linspace(-1,1,8))
print('Forward OK')
"

# CSV rates (authoritative numbers also in the paper)
python -c '
import csv
from collections import defaultdict
by = defaultdict(list)
for r in csv.DictReader(open("snapping_v2_final.csv")):
    by[(r["function"], int(r["depth"]))].append(r)
for fn in ["exp","ln","sqrt"]:
    for d in [2,3,4,5]:
        rs = by.get((fn,d), [])
        if rs:
            n = len(rs); nv = sum(r["valid_snap"]=="1" for r in rs)
            nn = sum(int(r["nan_epoch"]) <= 0 for r in rs)
            print(f"{fn} d={d}: {nv}/{nn} valid ({100*nv/nn:.0f}% of non-NaN)")
'
```

---

## Historical Context & Sources

- Pre-v2 ambition (April 23 2026 dev phase): Full EML + Cybenko + iCE40 pipeline + 3 papers (snapping analysis, hardware universality, formal verification/governance). See `prior_versions/dev/EXECUTIVE_SUMMARY.md`, `INDEX.md`, `PAPER_PRIORITY_MATRIX.txt`, `RESEARCH_QUESTIONS.md`, `NOVELTY_ANALYSIS.md`, `INTEGRATION.md`, `PROJECT_MANIFEST.txt`.
- What changed for v2: Deliberate focus on a clean, citable empirical/methodological contribution with strict validity criterion and Odrzywolek SI cross-check. Hardware, forests, and formal work were archived rather than abandoned.
- The v2 paper partially realizes the old "Paper 1" spirit but reframes it around basin selection + validity (stronger, more precise contribution than the early gradient-attenuation/leaf-position hypothesis).
- All historical material remains valuable for Tracks C and E and for institutional memory.

Do not edit files under `prior_versions/` — treat them as an archive.

---

## Maintenance

- Update this file on every significant new artifact (new CSV release, hardware PoC, major analysis, new paper draft) or at least every 3 months / on the next tagged release.
- Add a short "Roadmap & Future Directions" pointer (or badge) in the root `README.md` pointing here.
- When a track produces a new release or paper, add a dated entry under a "Progress" section here and update the top-level status.
- Keep the "Immediate Next Actions" section fresh; move completed items to a "Recently completed" note.
- Version lightly: "Roadmap for the post-v2.2 era (building on the basin selection framing)".

---

**This document turns the rich (but previously scattered) history and the strong v2.2 foundation into a shared, discussable artifact.** Pick a track or scenario, add your decision above, and start executing. The released CSV and validity machinery are excellent; the original hardware + governance vision is still compelling and now has better empirical grounding than it did in April.

Go build the next piece.