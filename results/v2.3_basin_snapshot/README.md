# v2.3 Basin Selection Data Snapshot

This directory contains the data snapshot for the Track B short note iteration (post curriculum re-anneal tuning pass).

**Snapshot date**: 2026-06-10 (iteration after feature branch + first 4 hypercompetent follow-ups)

**Source (live at time of snapshot)**: ../basin_warmstart.csv (162 rows accumulated across all Track B batches, including the 12-seed/1500-epoch + curriculum batch and the representative tuned-curriculum run).

**Included in this snapshot**:
- The full accumulated CSV at the moment of the "next iteration" (curriculum tuning + 20-seed control run launch).
- Key cells: exp d=2, exp d=3, ln d=5.
- Modes: blind, warm (refined top-gate), curriculum (grow_from_shallow + reanneal_extra_capacity).

**Key results visible in the data (as of snapshot creation)**:
- exp d=2 warm: 100%
- exp d=3 warm: 100% in multiple recent higher-N batches (85%+ cumulative)
- exp d=3 curriculum: 0% valid in the dedicated batch, but forms demonstrate successful core embedding (inner `eml(x,1)`)
- ln d=5: 0% across modes (curriculum still collapses; the re-anneal + spine is a first cut)

**Reproducibility (documented in the note skeleton and manifest)**:
See `notes/basin_selection_warmstart_note_skeleton.md` and the parent `results/basin_warmstart_v2.3_manifest.md` (or the live CSV header comments) for the exact commands used to produce the batches in this snapshot.

**Next iteration note**: This snapshot captures the state after the re-anneal tuning pass. A follow-up 20-seed control run (launched in background as part of this iteration) will grow the live CSV; when it completes the snapshot can be refreshed or a "final" v2.3 subset cut.

The note skeleton is the primary consumer of this data.

For the full experimental log and diagnostics, see `results/track_b_initial.md`.
