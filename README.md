# eml-ice40-cybenko

[![DOI](https://zenodo.org/badge/1220042425.svg)](https://doi.org/10.5281/zenodo.19736075)


Code and data for:

**Valid and False Snapping in EML Expression Trees: The Basin Selection Problem**
Daniyel Yaacov Bilar, Chokmah LLC, April 2026.

An empirical study of symbolic recovery in balanced-binary EML expression
trees trained with three-phase annealing (Adam, entropy penalty,
temperature anneal). The paper distinguishes *valid snaps* (vertex
commitment + correct symbolic form, post-snap MAE < 0.01) from *false
snaps* (vertex commitment, wrong form) across 240 runs over three target
functions (exp, ln, sqrt) and four tree depths.

## Contents

```
dyb-2026m-elm-basin.pdf       rendered paper (add after export)
figure1_heatmap_v2.png    Figure 1
snapping_v2_final.csv     full 240-run results
eml_layer_v2.py           EML layer + tree implementation
experiment_v2.py          training driver (2000 epochs, 3 phases)
CITATION.cff              citation metadata
LICENSE                   code license (MIT recommended)
.zenodo.json              Zenodo record metadata
```

## CSV columns

`snapping_v2_final.csv` has one row per run (240 total, 238 non-NaN).

| column | meaning |
|---|---|
| function | target function: exp, ln, sqrt |
| depth | tree depth (2-5) |
| seed | deterministic RNG seed |
| snapped | 1 if every selector max-weight >= 0.9 |
| final_loss | pre-snap loss (best task MAE during training) |
| snappability | mean max-weight across selectors (1.000 everywhere) |
| nan_epoch | epoch at which training diverged to NaN (-1 if none) |
| converged | 1 if training completed without NaN |
| symbolic_form | argmax-snapped expression string |
| expected_depth | minimum representational depth for this target |
| post_snap_loss | MAE of the snapped expression evaluated on the grid |
| valid_snap | 1 if snapped AND post_snap_loss < 0.01 |

**Use post_snap_loss and valid_snap, not final_loss, as the correctness
signal.** The paper (Sec. 2.5) explains why pre-snap loss systematically
understates the basin-selection failure mode.

## Reproducing

```
python experiment_v2.py
```

Deterministic per (function, depth, seed). Two exp/d=5 runs (seeds 0, 2)
diverge to NaN; this is reported in the CSV via `nan_epoch` and
`converged`.

## Citing

See `CITATION.cff`. The archival DOI is minted by Zenodo on release and
will appear as a badge at the top of this README once the first release
is published.

## License

Code released under MIT. See `LICENSE`.
Paper and figures released under CC BY 4.0 (see Zenodo record metadata).
