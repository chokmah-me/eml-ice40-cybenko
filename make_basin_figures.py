"""
make_basin_figures.py
Generate figures for the Track B basin-selection note (v2.3 data).

Adapts the style and palette from make_figure2.py (muted, colorblind-safe Okabe-Ito inspired).
Produces:
  - figure3_valid_rates.png : grouped bars for exp d=2 and d=3 (blind vs warm vs curriculum)
  - (optional) figure4_ln_d5.png if we add more ln data later

Usage (after running basin_warmstart.py to populate the CSV):
  python make_basin_figures.py

Data source: results/basin_warmstart.csv (accumulated across all Track B batches).
"""

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / 'results' / 'basin_warmstart.csv'

def load_rates():
    by_cell = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r['nan_epoch'] != '-1':
                continue
            key = (r['function'], int(r['depth']), r['init_mode'])
            by_cell[key].append(r)

    def rate(func, d, mode):
        rs = by_cell.get((func, d, mode), [])
        if not rs:
            return 0.0
        n = len(rs)
        v = sum(1 for r in rs if r['valid_snap'] == '1')
        return 100.0 * v / n if n else 0.0

    return {
        'exp': {
            2: {'blind': rate('exp', 2, 'blind'), 'warm': rate('exp', 2, 'warm')},
            3: {'blind': rate('exp', 3, 'blind'), 'warm': rate('exp', 3, 'warm'),
                'curriculum': rate('exp', 3, 'curriculum')},
        },
        'ln': {
            5: {'blind': rate('ln', 5, 'blind'), 'warm': rate('ln', 5, 'warm'),
                'curriculum': rate('ln', 5, 'curriculum')},
        }
    }

def main():
    rates = load_rates()

    # Color palette (same spirit as make_figure2.py)
    c_blind = '#999999'
    c_warm = '#E69F00'      # orange
    c_curr = '#0072B2'      # blue

    # Figure 3: exp d=2 + d=3
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    # Groups: d=2 (2 bars: blind, warm), d=3 (3 bars: blind, warm, curriculum)
    x = np.arange(2)  # d=2, d=3
    bar_w = 0.22

    # d=2
    ax.bar(x[0] - bar_w, rates['exp'][2]['blind'], bar_w, label='blind (randomize)', color=c_blind, edgecolor='black', linewidth=0.6)
    ax.bar(x[0],          rates['exp'][2]['warm'],  bar_w, label='warm (refined top-gate)', color=c_warm, edgecolor='black', linewidth=0.6)

    # d=3
    ax.bar(x[1] - bar_w, rates['exp'][3]['blind'], bar_w, color=c_blind, edgecolor='black', linewidth=0.6)
    ax.bar(x[1],          rates['exp'][3]['warm'],  bar_w, color=c_warm, edgecolor='black', linewidth=0.6)
    ax.bar(x[1] + bar_w, rates['exp'][3]['curriculum'], bar_w, label='curriculum (grow_from_shallow)', color=c_curr, edgecolor='black', linewidth=0.6)

    # Labels on bars
    for i, (xx, vals) in enumerate(zip([x[0]-bar_w, x[0]], [rates['exp'][2]['blind'], rates['exp'][2]['warm']])):
        ax.text(xx, vals + 2, f'{vals:.0f}%', ha='center', va='bottom', fontsize=9)
    for i, (xx, vals) in enumerate(zip([x[1]-bar_w, x[1], x[1]+bar_w],
                                       [rates['exp'][3]['blind'], rates['exp'][3]['warm'], rates['exp'][3]['curriculum']])):
        ax.text(xx, vals + 2, f'{vals:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(['d=2 (representational)', 'd=3 (over-depth)'])
    ax.set_ylabel('Valid recovery rate (%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title('exp(x): valid recovery with warm-start and curriculum (Track B)', fontsize=12)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

    fig.text(0.5, -0.02,
             "Blind = original randomize(0.1). Warm = direct top-gate x/1 bias (refined per v2.2 analysis of valid deeper solutions). "
             "Curriculum = grow_from_shallow after shallow warm pre-train. Data: results/basin_warmstart.csv (accumulated Track B runs).",
             ha='center', va='top', fontsize=8, wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out = HERE / 'figure3_valid_rates_exp.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"Wrote {out}")

    # Simple text summary for ln d=5 (all 0% so far)
    print("\nln d=5 (all modes 0% in current data — curriculum still collapsing to constants).")

if __name__ == '__main__':
    main()
