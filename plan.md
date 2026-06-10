# Track B Plan: Basin Selection Warm-Start + Curriculum (feature/track-b-basin-warmstart-curriculum)

This plan captures the focused iteration for the short technical note / data release on v2.2 basin selection improvements (warm-start init + curriculum/grow_from_shallow + reanneal_extra_capacity).

See `results/track_b_initial.md` for full experimental log and `notes/basin_selection_warmstart_note_skeleton.md` for the primary deliverable draft.

## Status (as of latest continue)
- Core implementation complete: `initialize_to_target`, `grow_from_shallow`, `reanneal_extra_capacity` (spine frozen, extra selectors only, phase-3 style short anneal) in eml_layer_v2.py.
- Driver `basin_warmstart.py` with --reanneal-epochs/--reanneal-lr, curriculum mode auto for over-depth.
- 12-seed/ reanneal batch integrated: exp d=3 curriculum 12/12 (100%) clean forms in tuned batch; cumul 50%.
- Warm strong: exp d=3 ~88% cumul, d=2 100%.
- ln d=5: 0% (outstanding; longer reanneal + unbalanced note in code).
- 234 rows in results/basin_warmstart.csv at start of this continue.
- Figure script + PNG (figure3_valid_rates_exp.png) ready; copied to notes/.
- v2.3 snapshot dir populated (multiple dated CSVs + README + manifest).
- Skeleton heavily polished (repro cmds, table, observations, limitations, tuning section).

## Saved next-steps (verbatim 5-item list from user)
1. Curriculum tuning pass (the nudge above is a first cut; we can iterate on re-anneal-only-on-new-capacity, different noise schedules for extra levels, or unbalanced-tree support).
2. Final 20-seed control runs (the script + manifest already document the exact commands; the CSV will just grow).
3. Figure generation (script is there and was run; drop the PNG into the skeleton or a real note).
4. Cutting the "v2.3 basin" data snapshot (manifest is written; we can cp the current CSV to a dated basin_warmstart_v2.3.csv + add a tiny README in the snapshot dir whenever you say).
5. Polishing the skeleton into a real short note (it is already very close — just needs the final numbers/figures + one pass on the abstract/discussion once the 20-seed data lands).

## Execution in this iteration
- Item 1: Tuning flags wired (`--reanneal-epochs 200/400`, lr), unbalanced-tree support note added to grow_from_shallow docstring in eml_layer_v2.py. The reanneal_extra_capacity (first-cut "nudge") delivered the 0%→100% (batch) jump for exp d=3 curriculum. ln:5 400-epoch variant launched as tuning.
- Item 2: 20-seed control + ln tuning launched in background (see commands below and in track log). CSV appends live. Integrate exact rates by re-running `python analyze_basin_rates.py` (or the one in basin_warmstart summary) post-completion.
- Item 3: `make_basin_figures.py` re-run; PNG dropped to notes/figure3_valid_rates_exp.png (and root). Skeleton updated to reference/include it.
- Item 4: Snapshot refreshed multiple times (20260609/10 + continue-time copy); README + manifest capture state + "next iteration note" on 20-seed. Can cp fresh dated copy + tiny update README when 20-seed done.
- Item 5: Skeleton updated with precise current rates (from 234-row analysis), reanneal diagnostic, full reproducibility section (exact CLIs including 20-seed), limitations note on N and versions, tuning section. One more pass on abstract/discussion + numbers/figures once 20-seed lands. (plan: turn skeleton into the real note file or leave as-is + commit as the draft.)
- Additional: analyzer helper, plan.md (this), log updates, README/ROADMAP/changelog bumps, commits to feature branch.

## Key commands (for repro + future)
```bash
# 20-seed final control (item 2)
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells exp:3,ln:5 --reanneal-epochs 200

# ln:5 curriculum tuning (longer re-anneal, item 1)
python basin_warmstart.py --seeds 10 --epochs 1500 --noise 0.4 --cells ln:5 --reanneal-epochs 400

# Figures (item 3)
python make_basin_figures.py

# Analysis (post-run)
python analyze_basin_rates.py

# Snapshot refresh example (item 4)
cp results/basin_warmstart.csv results/v2.3_basin_snapshot/basin_warmstart_v2.3_$(date +%Y%m%d).csv
# (edit the snapshot/README.md date + row count + key results blurb)
```

## Background task monitoring (this session)
- 20-seed main: task 019eb042-9cf8-7402-add6-6af0918c979c (log in .grok session terminal/)
- ln tuning: task 019eb042-9f97-7221-94f1-1cbdbaff7b77
Use `get_command_or_subagent_output --task_id=...` (or kill if needed). Progress printed every 5 runs inside the script; CSV grows incrementally (flush per row).

## When 20-seed complete (item 2+5)
- Re-run analyzer + make_basin_figures.py
- Update skeleton Table 1 + key observations + abstract with final N=20 rates + forms.
- Refresh snapshot (new dated CSV + README update).
- One pass abstract/discussion.
- Update track_b_initial.md "final rates" section.
- Commit + (optional) tag or note in main README.

## Notes on curriculum (item 1 ongoing)
- Reanneal helper: freezes spine (extreme negative bias on embedded selectors), Adam only on extra/non-spine logits, short T 1.0→final, then normal train.
- Future: unbalanced (parse symbolic_form of shallow to mark only active path, leave siblings extra); different per-level noise or reanneal epochs schedules.
- Current docstring in eml_layer_v2.py:grow_from_shallow has the note.

See prior commits on this branch for implementation history. This plan saved per user request to preserve the 5-item list + execution status.
