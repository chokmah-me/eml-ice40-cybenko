#!/usr/bin/env python3
"""
Snapping Curve Experiment v2 - Correct Architecture
=====================================================
Replicates and extends Odrzywolek Section 4.3 using corrected EML architecture.

Target functions and their minimum recoverable depth (our notation):
  exp(x) = eml(x,1)               -> depth 2 (level 1), K=3
  ln(x)  = eml(1,eml(eml(1,x),1)) -> depth 4 (level 3), K=7 in balanced tree
  sqrt(x): K>=35                   -> depth 9+, out of scope here

Experiment: for each (function, depth, seed), train and measure snapping.
Depths tested: 2, 3, 4, 5.
Seeds: 20 per combination.
Epochs: 2000.

Validity criterion is strict post-snap: after snapping all selectors to
their argmax vertex, the snapped tree's MAE on the training grid must be
below 1e-2. `post_snap_loss` and `valid_snap` are recorded per run.

Run from project root (needs eml_layer_v2.py in same dir):
  python experiment_v2.py

Checkpointed: safe to interrupt and resume.
"""

import numpy as np
import torch
import torch.nn as nn
import csv, json, time, sys
from pathlib import Path
from datetime import datetime

from eml_layer_v2 import EMLTree, train_eml

# ============================================================================
# CONFIG
# ============================================================================

CONFIG = {
    'num_seeds': 20,
    'epochs': 2000,
    'depths': [2, 3, 4, 5],
    'functions': ['exp', 'ln', 'sqrt'],
    'learning_rates': {2: 0.01, 3: 0.001, 4: 0.001, 5: 0.0005},
    'snap_threshold': 0.9,
    'csv_file': 'results/snapping_v2.csv',
    'checkpoint_file': 'results/snapping_v2_checkpoint.json',
    'log_file': 'results/snapping_v2.log',
}

# Expected snap depths in OUR balanced binary tree:
# exp(x) = eml(x,1):                 our depth 2 (Odrzywolek level 1)
# ln(x)  = eml(1,eml(eml(1,x),1)):  our depth 4 (needs 3 gate levels)
# sqrt: K>=35, depth 9+ (not tested here)
# Note: Odrzywolek's "level 2" for ln uses unbalanced tree (shares subtrees).
# Our balanced binary tree requires one extra level for the same formula.
EXPECTED_DEPTH = {'exp': 2, 'ln': 4, 'sqrt': 9}

# ============================================================================
# DATA
# ============================================================================

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

# ============================================================================
# CHECKPOINT
# ============================================================================

def load_ckpt():
    p = Path(CONFIG['checkpoint_file'])
    return json.load(open(p)) if p.exists() else {'completed': []}

def save_ckpt(completed):
    with open(CONFIG['checkpoint_file'], 'w') as f:
        json.dump({'completed': completed}, f)

def run_key(func, depth, seed):
    return f'{func}_{depth}_{seed}'

# ============================================================================
# LOGGING
# ============================================================================

def log(msg, logfile):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(logfile, 'a') as f:
        f.write(line + '\n')

# ============================================================================
# MAIN
# ============================================================================

def run():
    Path('results').mkdir(exist_ok=True)
    logfile = CONFIG['log_file']
    ckpt = load_ckpt()
    completed = ckpt['completed']

    csv_path = Path(CONFIG['csv_file'])
    fieldnames = ['function', 'depth', 'seed', 'snapped', 'final_loss',
                  'snappability', 'nan_epoch', 'converged', 'symbolic_form',
                  'expected_depth', 'post_snap_loss', 'valid_snap']
    write_header = not csv_path.exists()

    total = len(CONFIG['functions']) * len(CONFIG['depths']) * CONFIG['num_seeds']
    done = len(completed)
    t0 = time.time()

    log('Snapping Curve v2 - Correct Architecture', logfile)
    log(f"Functions: {CONFIG['functions']}", logfile)
    log(f"Depths: {CONFIG['depths']}  Seeds: {CONFIG['num_seeds']}  Epochs: {CONFIG['epochs']}", logfile)
    log(f"Total: {total}  Already done: {done}", logfile)

    with open(csv_path, 'a', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for func in CONFIG['functions']:
            x_data, y_data = make_data(func)
            log(f'\nFunction: {func}  (expect snap at depth {EXPECTED_DEPTH[func]})', logfile)

            for depth in CONFIG['depths']:
                lr = CONFIG['learning_rates'][depth]
                valid_count = 0
                snapped_count = 0
                nan_count = 0
                symbolic_forms = []

                for seed in range(CONFIG['num_seeds']):
                    key = run_key(func, depth, seed)
                    if key in completed:
                        done += 1
                        continue

                    torch.manual_seed(seed * 13 + depth * 7 + hash(func) % 97)
                    tree = EMLTree(depth=depth)
                    tree.randomize(0.1)

                    m = train_eml(tree, x_data, y_data,
                                  epochs=CONFIG['epochs'], lr=lr)

                    sym = ''
                    if m['snapped']:
                        # train_eml already snaps the tree before returning,
                        # so tree.symbolic_form() reflects the snapped state.
                        sym = tree.symbolic_form()
                        symbolic_forms.append(sym)

                    row = {
                        'function': func,
                        'depth': depth,
                        'seed': seed,
                        'snapped': m['snapped'],
                        'final_loss': m['final_loss'],
                        'snappability': m['snappability'],
                        'nan_epoch': m['nan_epoch'],
                        'converged': m['converged'],
                        'symbolic_form': sym,
                        'expected_depth': EXPECTED_DEPTH[func],
                        'post_snap_loss': m['post_snap_loss'],
                        'valid_snap': m['valid_snap'],
                    }
                    writer.writerow(row)
                    csvf.flush()

                    snapped_count += m['snapped']
                    valid_count += m['valid_snap']
                    if m['nan_epoch'] >= 0:
                        nan_count += 1

                    completed.append(key)
                    save_ckpt(completed)
                    done += 1

                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                log(
                    f'  {func} d{depth}: valid={valid_count}/{CONFIG["num_seeds"]}'
                    f'  snap={snapped_count}/{CONFIG["num_seeds"]}'
                    f'  nan={nan_count}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m',
                    logfile
                )
                if symbolic_forms:
                    unique = list(set(symbolic_forms))
                    log(f'    forms: {unique[:3]}', logfile)

    log('\nDone.', logfile)
    print_summary(csv_path)


def print_summary(csv_path):
    rows = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    print('\n' + '='*78)
    print('SUMMARY')
    print(f"{'Func':<7} {'D':<4} {'ExpD':<6} {'Valid%':<8} {'Snap%':<8} "
          f"{'PreLoss':<10} {'PostLoss':<10} {'NaN'}")
    print('-'*78)
    for func in CONFIG['functions']:
        for d in CONFIG['depths']:
            dr = [r for r in rows if r['function']==func and r['depth']==str(d)]
            good = [r for r in dr if r['nan_epoch']=='-1']
            if not good:
                continue
            valid_pct = 100 * np.mean([int(r['valid_snap']) for r in good])
            snap_pct = 100 * np.mean([int(r['snapped']) for r in good])
            pre_losses = [float(r['final_loss']) for r in good if r['final_loss'] != 'nan']
            post_losses = [float(r['post_snap_loss']) for r in good if r['post_snap_loss'] != 'nan']
            nan_n = sum(1 for r in dr if r['nan_epoch'] != '-1')
            exp_d = EXPECTED_DEPTH[func]
            marker = '<--' if int(d) == exp_d else ''
            print(f"{func:<7} {d:<4} {exp_d:<6} {valid_pct:<8.0f} {snap_pct:<8.0f} "
                  f"{np.mean(pre_losses) if pre_losses else float('nan'):<10.4f} "
                  f"{np.mean(post_losses) if post_losses else float('nan'):<10.4f} "
                  f"{nan_n}  {marker}")
        print()


if __name__ == '__main__':
    run()
