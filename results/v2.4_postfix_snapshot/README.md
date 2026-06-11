# v2.4 Post-Fix Full Dataset (20-seed control with corrected embedding code)

**Generated after SME review feedback and code fixes** (initialize_to_target ln core, grow delegation for ln over-depth, bias guard, deterministic seeding, pretrain_form column).

**Exact command used (the one that produced the data referenced in the note):**
python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.4 --cells ln:5,exp:3 --reanneal-epochs 200 --csv results/basin_warmstart_v2.4_postfix.csv

**Results (from this run):**
- ln d=5 blind: 5/20 valid (25%)
- ln d=5 warm: 20/20 valid (100%), all pretrain_form and final form = ml(1,eml(eml(1,x),1))
- ln d=5 curriculum: 20/20 valid (100%), same correct form (driver reduces over-depth ln to fixed direct init)
- exp d=3 blind: 7/20 valid (35%)
- exp d=3 warm: 20/20 valid (100%), all ml(x,1)
- exp d=3 curriculum: 20/20 valid (100%), all ml(x,1)

This CSV (120 rows) is the production post-fix dataset. It includes the pretrain_form column proving the embedding fix (warm/curriculum ln all started with the correct core).

Old v2.3 data is pre-fix diagnostic only (showed the bug).

See main note for full interpretation, tables, and the followup patches.
