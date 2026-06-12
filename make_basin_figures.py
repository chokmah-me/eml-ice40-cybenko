"""
make_basin_figures.py
Generate the figure for the basin-selection warm-start/curriculum note.

Adapts the style and palette from make_figure2.py (muted, colorblind-safe Okabe-Ito inspired).
Produces:
  - figure3_valid_rates_exp.png (root + copy in notes/): grouped bars for
    exp d=3 and ln d=5 (blind vs warm vs curriculum), from the canonical
    v2.5 dataset (results/basin_warmstart_v2.5_unbalanced.csv, 120 rows,
    20 seeds x 2 cells x 3 modes; blind/warm rows match v2.4_postfix
    run-for-run).

Usage:
  python make_basin_figures.py
"""

import csv
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / 'results' / 'basin_warmstart_v2.5_unbalanced.csv'

CELLS = [('exp', 3), ('ln', 5)]
MODES = ['blind', 'warm', 'curriculum']


def load_rates():
    by_cell = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r['nan_epoch'] != '-1':
                continue
            by_cell[(r['function'], int(r['depth']), r['init_mode'])].append(r)

    rates, counts = {}, {}
    for func, d in CELLS:
        for mode in MODES:
            rs = by_cell.get((func, d, mode), [])
            n = len(rs)
            v = sum(1 for r in rs if r['valid_snap'] == '1')
            rates[(func, d, mode)] = 100.0 * v / n if n else 0.0
            counts[(func, d, mode)] = (v, n)
    return rates, counts


def main():
    rates, counts = load_rates()

    c_mode = {'blind': '#999999', 'warm': '#E69F00', 'curriculum': '#0072B2'}
    label = {'blind': 'blind (randomize)',
             'warm': 'warm (initialize_to_target)',
             'curriculum': 'curriculum (grow_from_shallow + reanneal)'}

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(CELLS))
    bar_w = 0.22

    for j, mode in enumerate(MODES):
        offs = (j - 1) * bar_w
        for i, (func, d) in enumerate(CELLS):
            r = rates[(func, d, mode)]
            ax.bar(x[i] + offs, r, bar_w, color=c_mode[mode],
                   edgecolor='black', linewidth=0.6,
                   label=label[mode] if i == 0 else None)
            v, n = counts[(func, d, mode)]
            ax.text(x[i] + offs, r + 2, f'{v}/{n}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(['exp(x), d=3 (over-depth)', 'ln(x), d=5 (over-depth)'])
    ax.set_ylabel('Valid recovery rate (%)', fontsize=11)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title('Valid recovery: blind vs warm-start vs top-aligned curriculum (20 seeds/cell)',
                 fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=9, framealpha=0.95)

    fig.text(0.5, -0.10,
             "Validity: snapped + post-snap MAE < 0.01. "
             "Data: results/basin_warmstart_v2.5_unbalanced.csv (deterministic seeding; "
             "blind/warm rows match v2.4_postfix run-for-run).",
             ha='center', va='top', fontsize=8, wrap=True)

    plt.tight_layout()
    out = HERE / 'figure3_valid_rates_exp.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"Wrote {out}")
    shutil.copyfile(out, HERE / 'notes' / 'figure3_valid_rates_exp.png')
    print(f"Copied to {HERE / 'notes' / 'figure3_valid_rates_exp.png'}")


if __name__ == '__main__':
    main()
