"""
Figure 2 for dyb-2026m-elm-basin v2.1.

Grouped bar chart comparing blind-recovery rates:
  - Odrzywolek SI Table S5 (target depth = tree depth)
  - This work, exp(x), representational depth 2
  - This work, ln(x), representational depth 4

Data sources:
  - snapping_v2_final.csv        (this work; 240 runs)
  - odrzywolek_si_table_s5.csv   (transcribed verbatim from Odrzywolek 2026 SI Table S5)
"""
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Our rates: pull from CSV to avoid typos
import os
here = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here, 'snapping_v2_final.csv') if os.path.exists(
    os.path.join(here, 'snapping_v2_final.csv')
) else '/mnt/project/snapping_v2_final.csv'

by_cell = defaultdict(list)
with open(csv_path) as f:
    for r in csv.DictReader(f):
        by_cell[(r['function'], int(r['depth']))].append(r)

def rate(fn, d):
    rs = by_cell[(fn, d)]
    n_non_nan = sum(1 for r in rs if int(r['nan_epoch']) <= 0)
    n_valid = sum(1 for r in rs if r['valid_snap'] == '1')
    return 100.0 * n_valid / n_non_nan if n_non_nan else 0.0

depths = [2, 3, 4, 5]

# Odrzywolek SI Table S5: load from companion CSV transcribed from the SI
odr_csv_path = os.path.join(here, 'odrzywolek_si_table_s5.csv') if os.path.exists(
    os.path.join(here, 'odrzywolek_si_table_s5.csv')
) else '/home/claude/odrzywolek_si_table_s5.csv'
odr_by_depth = {}
with open(odr_csv_path) as f:
    for r in csv.DictReader(f):
        odr_by_depth[int(r['depth'])] = float(r['rate_percent'])
odr = [odr_by_depth[d] for d in depths]
ours_exp = [rate('exp', d) for d in depths]
ours_ln  = [rate('ln',  d) for d in depths]

# Sanity print
print("Odrzywolek d=2..5:", [f"{x:.1f}" for x in odr])
print("Ours exp d=2..5:  ", [f"{x:.1f}" for x in ours_exp])
print("Ours ln  d=2..5:  ", [f"{x:.1f}" for x in ours_ln])

# Plot
fig, ax = plt.subplots(figsize=(9.5, 5.2))

x = np.arange(len(depths))
bar_w = 0.27

# Muted, colorblind-safe palette (Okabe-Ito inspired)
c_odr = '#999999'    # grey for baseline comparison
c_exp = '#E69F00'    # orange
c_ln  = '#0072B2'    # blue

b1 = ax.bar(x - bar_w, odr,      bar_w, label='Odrzywolek SI Table S5 (target depth = tree depth)',
            color=c_odr, edgecolor='black', linewidth=0.6)
b2 = ax.bar(x,         ours_exp, bar_w, label='This work: exp(x), representational depth 2',
            color=c_exp, edgecolor='black', linewidth=0.6)
b3 = ax.bar(x + bar_w, ours_ln,  bar_w, label='This work: ln(x), representational depth 4',
            color=c_ln,  edgecolor='black', linewidth=0.6)

# Numeric labels on bars
def label(bars, vals):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
label(b1, odr)
label(b2, ours_exp)
label(b3, ours_ln)

# Representational-depth markers on the two "ours" series
# exp has representational depth 2 -> star on b2 at d=2
# ln  has representational depth 4 -> star on b3 at d=4
star_kw = dict(marker='*', s=220, color='black', zorder=5, edgecolor='white', linewidth=0.8)
# exp at d=2: star sits above the 25% bar
ax.scatter([x[0]],          [ours_exp[0] + 8], **star_kw)
# ln at d=4: star sits above the 90% bar (which is near the top)
ax.scatter([x[2] + bar_w],  [ours_ln[2] + 8],  **star_kw)

# Axes
ax.set_xticks(x)
ax.set_xticklabels([f'd={d}' for d in depths])
ax.set_xlabel('Tree depth', fontsize=11)
ax.set_ylabel('Blind-recovery rate (%)', fontsize=11)
ax.set_ylim(0, 115)
ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)
ax.set_title('Blind-recovery rate: this work vs Odrzywolek SI Table S5', fontsize=12)

# Legend
star_proxy = mpatches.Patch(color='none', label='\u2605  target at representational depth')
handles, labels = ax.get_legend_handles_labels()
handles.append(star_proxy)
labels.append('\u2605  target at representational depth')
# Use real stars in legend via a scatter proxy
from matplotlib.lines import Line2D
star_line = Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markeredgecolor='white', markersize=14, label='target at representational depth')
handles = handles[:3] + [star_line]
labels  = labels[:3]  + ['target at representational depth']
ax.legend(handles, labels, loc='upper center', fontsize=9, framealpha=0.95, ncol=2,
          bbox_to_anchor=(0.5, -0.18))

# Caption-style footnote
fig.text(0.5, -0.08,
         "Grouped bars show blind-initialization success rates. Odrzywolek's targets are nested EML self-compositions "
         "whose representational depth grows with tree depth, so tree depth and target complexity co-vary. "
         "Our targets are fixed at representational depth 2 (exp) and 4 (ln); stars mark the matched-depth cells. "
         "Denominators: Odrzywolek 32, 64, 64, 448; ours 20 except exp d=5 (18 after NaN).",
         ha='center', va='top', fontsize=8.5, wrap=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.30)

out = '/home/claude/figure2_rate_comparison.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f"Wrote {out}")
