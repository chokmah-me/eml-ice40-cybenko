#!/usr/bin/env python3
"""
Track B: Basin Selection Improvements - Warm-start / Noised-Correct Init Experiments
====================================================================================

Direct follow-on to the v2.2 paper's core finding: temperature annealing solves
*commitment*, but basin selection remains the bottleneck.

This script compares:
  - "blind": the default randomize(0.1) init used in the released paper (baseline)
  - "warm": initialize_to_target(func, noise=...) — strong bias toward the known
            good symbolic form + controlled noise (in the spirit of Odrzywolek SI
            Table S7, where correct+noise recovered 100% even at d=5/6).

Focus cells (the interesting ones from the paper):
  - exp d=2 : 25% valid (dominant eml(x,x) false basin)
  - exp d=3 : 20% valid
  - ln  d=5 : 25% valid (drop from the 90% peak at representational depth 4)
  - ln  d=4 : 90% valid (the "easy" peak, for sanity)

Outputs:
  - Console summary table with valid% per (func, depth, init_mode)
  - results/basin_warmstart.csv with per-run details (compatible with the v2 schema + init_mode column)
  - Optional: higher seed counts or more cells via CLI flags.

Usage (quick demo, ~few minutes):
    python basin_warmstart.py --seeds 8 --epochs 1500

For closer-to-paper conditions (slower):
    python basin_warmstart.py --seeds 20 --epochs 2000 --noise 0.5

The warm init is the main new lever for Track B. Future extensions in this track:
  - Curriculum / grow-from-shallow (train d=2 valid → embed into d=3/4/5)
  - Phase-1 only interventions (different inits, loss shaping, etc.)
  - Unbalanced / sharing trees (closer to Odrzywolek's original ln construction)
  - More targets + mpmath high-precision impostor filter (as the paper itself suggests)
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from eml_layer_v2 import EMLTree, train_eml  # make_data is defined locally below for self-containment

# Local copy of data maker (keeps the script standalone-ish)
def make_data(func, samples=128):
    if func == 'exp':
        x = torch.linspace(-2.0, 2.0, samples)
        y = torch.exp(x)
    elif func == 'ln':
        x = torch.linspace(0.1, 10.0, samples)
        y = torch.log(x)
    elif func == 'sqrt':
        x = torch.linspace(0.1, 10.0, samples)
        y = torch.sqrt(x)
    else:
        raise ValueError(func)
    return x, y


def run_key(func, depth, seed, mode):
    return f'{func}_{depth}_{seed}_{mode}'


def main():
    parser = argparse.ArgumentParser(description="Track B basin-escape warm-start experiments")
    parser.add_argument('--seeds', type=int, default=10, help='Seeds per (func,depth,mode) cell')
    parser.add_argument('--epochs', type=int, default=1500, help='Training epochs')
    parser.add_argument('--noise', type=float, default=0.45, help='Noise std for warm init')
    parser.add_argument('--cells', type=str, default='exp:2,exp:3,ln:4,ln:5',
                        help='Comma list of func:depth to test, e.g. exp:2,ln:5')
    parser.add_argument('--csv', type=str, default='results/basin_warmstart.csv')
    parser.add_argument('--quick', action='store_true', help='Force small seeds/epochs for fast demo')
    args = parser.parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 6)
        args.epochs = min(args.epochs, 800)

    Path('results').mkdir(exist_ok=True)

    # Cells to run
    cells = []
    for item in args.cells.split(','):
        f, d = item.split(':')
        cells.append((f, int(d)))

    modes = ['blind', 'warm']
    # Curriculum is especially powerful for over-depth / hard cells like ln d=5.
    # We include it explicitly for the cells where direct warm was insufficient.
    if any(d >= 4 or (f == 'exp' and d >= 3) for f, d in cells):
        if 'curriculum' not in modes:
            modes.append('curriculum')

    fieldnames = ['function', 'depth', 'seed', 'init_mode', 'snapped', 'final_loss',
                  'snappability', 'nan_epoch', 'converged', 'symbolic_form',
                  'expected_depth', 'post_snap_loss', 'valid_snap']

    csv_path = Path(args.csv)
    write_header = not csv_path.exists()

    total_runs = len(cells) * len(modes) * args.seeds
    done = 0
    t0 = time.time()

    print('='*70)
    print('TRACK B: Basin Selection - Warm-start (noised-correct) vs Blind')
    print(f'Cells: {cells}   Modes: {modes}   Seeds/cell: {args.seeds}   Epochs: {args.epochs}')
    print(f'Warm noise: {args.noise}')
    print('='*70)

    with open(csv_path, 'a', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for func, depth in cells:
            x_data, y_data = make_data(func)
            lr = {2: 0.01, 3: 0.001, 4: 0.001, 5: 0.0005}.get(depth, 0.001)
            exp_d = {'exp': 2, 'ln': 4, 'sqrt': 9}.get(func, depth)

            for mode in modes:
                valid = 0
                snapped = 0
                nans = 0
                forms = []

                for s in range(args.seeds):
                    torch.manual_seed(s * 17 + depth * 11 + hash((func, mode)) % 101)

                    tree = EMLTree(depth=depth)

                    if mode == 'blind':
                        tree.randomize(0.1)
                    elif mode == 'curriculum':
                        # Pre-train a shallow tree at the function's representational depth
                        # using the best known warm init, then grow its structure into the target depth.
                        _expected = {'exp': 2, 'ln': 4, 'sqrt': 9}
                        shallow_depth = _expected.get(func, 2)
                        if shallow_depth >= depth:
                            tree.initialize_to_target(func, noise=args.noise)
                        else:
                            shallow = EMLTree(depth=shallow_depth)
                            shallow.initialize_to_target(func, noise=args.noise * 0.8)
                            m_sh = train_eml(shallow, x_data, y_data,
                                             epochs=max(600, args.epochs // 2),
                                             lr=lr)
                            if m_sh.get('valid_snap', 0):
                                shallow.snap_all()
                                tree.grow_from_shallow(shallow, noise=args.noise * 0.7)
                            else:
                                # Fallback to best direct warm
                                tree.initialize_to_target(func, noise=args.noise)
                    else:
                        # 'warm' and any future modes
                        tree.initialize_to_target(func, noise=args.noise)

                    m = train_eml(tree, x_data, y_data,
                                  epochs=args.epochs, lr=lr)

                    sym = tree.symbolic_form() if m['snapped'] else ''

                    row = {
                        'function': func,
                        'depth': depth,
                        'seed': s,
                        'init_mode': mode,
                        'snapped': m['snapped'],
                        'final_loss': m['final_loss'],
                        'snappability': m['snappability'],
                        'nan_epoch': m['nan_epoch'],
                        'converged': m['converged'],
                        'symbolic_form': sym,
                        'expected_depth': exp_d,
                        'post_snap_loss': m['post_snap_loss'],
                        'valid_snap': m['valid_snap'],
                    }
                    writer.writerow(row)
                    csvf.flush()

                    if m['nan_epoch'] >= 0:
                        nans += 1
                    else:
                        if m['snapped']:
                            snapped += 1
                            forms.append(sym)
                        if m['valid_snap']:
                            valid += 1

                    done += 1
                    if done % 5 == 0:
                        elapsed = time.time() - t0
                        print(f'  progress: {done}/{total_runs}  ({elapsed/60:.1f}m elapsed)')

                # per cell summary
                vrate = 100.0 * valid / max(1, args.seeds - nans) if (args.seeds - nans) > 0 else 0
                srate = 100.0 * snapped / max(1, args.seeds - nans) if (args.seeds - nans) > 0 else 0
                unique_forms = sorted(set(forms))[:4]
                print(f'{func} d={depth}  mode={mode:5s} : valid={valid}/{args.seeds-nans} ({vrate:.0f}%) '
                      f'snapped={snapped}  nans={nans}  forms~{unique_forms}')

    # Final nice table from the written file
    print('\n' + '='*78)
    print('TRACK B RESULTS SUMMARY (from this run)')
    print(f"{'func':<6} {'d':<3} {'mode':<6} {'valid%':>7} {'snap%':>7} {'nans':>5}")
    print('-'*78)

    # Read back what we just wrote for this run's cells
    rows = []
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f)
                if (r['function'], int(r['depth'])) in [(c[0], c[1]) for c in cells]]

    for func, depth in cells:
        for mode in modes:
            dr = [r for r in rows
                  if r['function'] == func and int(r['depth']) == depth and r['init_mode'] == mode
                  and r['nan_epoch'] == '-1']
            if not dr:
                continue
            v = 100 * np.mean([int(r['valid_snap']) for r in dr])
            s = 100 * np.mean([int(r['snapped']) for r in dr])
            n = sum(1 for r in rows if r['function']==func and int(r['depth'])==depth and r['init_mode']==mode and r['nan_epoch']!='-1')
            print(f"{func:<6} {depth:<3} {mode:<6} {v:>6.0f}% {s:>6.0f}% {n:>5}")

    print('\nCSV written to', csv_path)
    print('Next ideas (Track B): curriculum grow-from-shallow, phase-1 interventions, unbalanced trees, mpmath filter.')


if __name__ == '__main__':
    main()
