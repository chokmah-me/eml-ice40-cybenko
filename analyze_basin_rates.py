#!/usr/bin/env python3
"""Pure stdlib analyzer for basin_warmstart.csv rates. Run: python analyze_basin_rates.py"""
import csv
from collections import defaultdict

CSV = 'results/basin_warmstart.csv'

def main():
    with open(CSV, newline='') as f:
        rows = list(csv.DictReader(f))
    print(f'Total rows in {CSV}: {len(rows)}')
    if not rows:
        return

    # Filter to completed (nan_epoch == -1 or no nans)
    completed = [r for r in rows if r.get('nan_epoch', '-1') == '-1']
    print(f'Completed (no nan): {len(completed)}')

    # By (mode, cell, depth) but cell often 'exp:3' or func/depth separate?
    # Prefer function + depth + init_mode
    groups = defaultdict(list)
    for r in completed:
        func = r.get('function') or r.get('cell', '?').split(':')[0]
        try:
            depth = int(r.get('depth') or r.get('cell', ':0').split(':')[-1] if ':' in r.get('cell','') else r.get('depth', 0))
        except:
            depth = r.get('depth', '?')
        mode = r.get('init_mode', r.get('mode', '?'))
        key = (func, str(depth), mode)
        groups[key].append(r)

    print('\n=== Valid rates by (func, depth, init_mode) ===')
    print(f"{'func':<6} {'d':<3} {'mode':<12} {'valid':>6} {'total':>6} {'rate':>7}  {'notes'}")
    for key in sorted(groups):
        func, d, mode = key
        rs = groups[key]
        n = len(rs)
        v = sum(1 for r in rs if r.get('valid_snap','0') == '1' or r.get('valid','').lower() in ('1','true','yes'))
        rate = (100.0 * v / n) if n else 0.0
        note = ''
        # peek one form
        forms = [r.get('symbolic_form','') for r in rs if r.get('symbolic_form')]
        if forms:
            uniq = set(forms)
            note = f"forms~{len(uniq)} e.g. {list(uniq)[0][:40]}"
        print(f"{func:<6} {d:<3} {mode:<12} {v:>6} {n:>6} {rate:>6.1f}%  {note}")

    print('\n=== Curriculum exp:3 forms detail (last 8) ===')
    curr = [r for r in completed if (r.get('function')=='exp' or 'exp' in r.get('cell','')) and str(r.get('depth'))=='3' and r.get('init_mode')=='curriculum']
    for r in curr[-8:]:
        print(r.get('seed'), r.get('valid_snap'), r.get('symbolic_form'))

if __name__ == '__main__':
    main()
