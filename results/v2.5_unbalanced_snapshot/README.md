# v2.5 unbalanced-curriculum snapshot (2026-06-11)

`basin_warmstart_v2.5_unbalanced.csv` — 120 rows, 20 seeds x {ln d=5, exp d=3}
x {blind, warm, curriculum}, produced by:

```bash
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells ln:5,exp:3 \
    --reanneal-epochs 200 --csv results/basin_warmstart_v2.5_unbalanced.csv
```

with the v2.5 top-aligned `grow_from_shallow` (commit "v2.5: top-aligned
grow_from_shallow unlocks real ln over-depth curriculum") and the driver no
longer delegating ln over-depth curriculum to direct init.

## Results

| cell    | blind | warm  | curriculum |
|---------|-------|-------|------------|
| ln d=5  | 5/20 (25%) | 20/20 (100%) | **20/20 (100%)** |
| exp d=3 | 7/20 (35%) | 20/20 (100%) | 20/20 (100%) |

All ln valid forms: `eml(1,eml(eml(1,x),1))`. All exp valid forms: `eml(x,1)`.

## What changed vs v2.4

In v2.4 the ln d=5 curriculum mode *delegated* to the direct warm init because
a balanced EML gate cannot forward identity (extending a shallow ln solution
upward embeds exp(ln(x))). v2.5 embeds the trained shallow d=4 solution
TOP-ALIGNED in the d=5 tree: balanced trees have identical widths at
top-aligned levels, child 'f' references line up, the ln value is preserved
exactly, and the new bottom levels dangle as constant-biased extra capacity
(the unbalanced/RPN "inactive branch" property realized inside the balanced
tree). The v2.4 "structural limit" on ln over-depth curriculum is removed.

Audit: the driver's fallback (direct init when shallow pre-training fails)
was checked by replaying the shallow phase for all 20 effective seeds — all
20 shallow d=4 pre-trainings were valid, so all 20 curriculum rows used the
real grow-from-shallow path.

Blind and warm rows are statistically consistent with v2.4_postfix
(seeding identical for blind/warm; exp d=3 blind 7/20 both runs, ln d=5
blind 5/20 both runs).
