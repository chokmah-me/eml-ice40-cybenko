"""
Train tiny scalar MLPs that approximate exp/ln, for the symbolic-vs-MLP study.

One MLP per (function, hidden width H, depth D) cell. Depth D = number of hidden
ReLU layers; the network is 1 -> H -> ... -> H -> 1 (D hidden layers, so D+1
Linear layers total). Float weights are saved to results/mlp/<func>_h<H>_d<D>.npz
as w0,b0,w1,b1,... plus metadata; the fixed-point golden model and the RTL
emitter both consume exactly this layer list.

Run from repo root:
    python -m mlp.train_mlp                  # full sweep
    python -m mlp.train_mlp --func exp --h 8 --depth 1   # one cell
"""

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "results", "mlp")

# (name, function, domain) -- same scalar targets the snapped EML units compute.
FUNCS = {
    "exp": (math.exp, (-2.0, 2.0)),
    "ln": (math.log, (0.1, 10.0)),
}
HIDDENS = [4, 8, 16, 32]
DEPTHS = [1, 2]
# Multi-seed best-of-N: train SEEDS independent inits per cell, keep the best by
# float-grid max-error for the RTL/area numbers, and report mean +/- std so the
# accuracy axis reflects model capacity rather than a single seed's training luck.
SEEDS = list(range(8))
EPOCHS = 4000
N_TRAIN = 1024
SUMMARY_CSV = os.path.join(OUT_DIR, "train_summary.csv")


def build_mlp(depth: int, hidden: int) -> nn.Sequential:
    layers = [nn.Linear(1, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers += [nn.Linear(hidden, 1)]
    return nn.Sequential(*layers)


def _train_one_seed(func: str, hidden: int, depth: int, seed: int):
    """Train a single (cell, seed) and return (layers, mae, maxerr) on the grid."""
    fn, (lo, hi) = FUNCS[func]
    torch.manual_seed(seed)
    np.random.seed(seed)

    xs = torch.linspace(lo, hi, N_TRAIN).unsqueeze(1)
    ys = torch.tensor([[fn(x.item())] for x in xs], dtype=torch.float32)

    net = build_mlp(depth, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(net(xs), ys)
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        pred = net(xs)
        mae = (pred - ys).abs().mean().item()
        maxerr = (pred - ys).abs().max().item()

    layers = []  # flatten to a plain layer list: [(W0, b0), (W1, b1), ...]
    for m in net:
        if isinstance(m, nn.Linear):
            layers.append((m.weight.detach().numpy().astype(np.float64),
                           m.bias.detach().numpy().astype(np.float64)))
    return layers, mae, maxerr


def train_cell(func: str, hidden: int, depth: int, verbose: bool = True) -> dict:
    fn, (lo, hi) = FUNCS[func]

    # Best-of-N: train every seed, select the one with the lowest float-grid
    # max-error for the saved weights; keep the full per-seed spread for reporting.
    best = None  # (maxerr, seed, layers, mae)
    maxerrs, maes = [], []
    for seed in SEEDS:
        layers, mae, maxerr = _train_one_seed(func, hidden, depth, seed)
        maxerrs.append(maxerr)
        maes.append(mae)
        if best is None or maxerr < best[0]:
            best = (maxerr, seed, layers, mae)
    best_maxerr, seed_best, layers, best_mae = best
    maxerr_mean = float(np.mean(maxerrs))
    maxerr_std = float(np.std(maxerrs))
    mae_mean = float(np.mean(maes))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{func}_h{hidden}_d{depth}.npz")
    arrs = {}
    for i, (w, b) in enumerate(layers):
        arrs[f"w{i}"] = w
        arrs[f"b{i}"] = b
    np.savez(path, n_layers=len(layers), func=func, hidden=hidden, depth=depth,
             domain=np.array([lo, hi]), float_mae=best_mae, float_maxerr=best_maxerr,
             seed_best=seed_best, n_seeds=len(SEEDS), maxerr_mean=maxerr_mean,
             maxerr_std=maxerr_std, mae_mean=mae_mean,
             maxerr_all=np.array(maxerrs), **arrs)
    if verbose:
        print(f"{func} h{hidden} d{depth}: best maxerr {best_maxerr:.5f} "
              f"(seed {seed_best}), mean {maxerr_mean:.5f} +/- {maxerr_std:.5f} "
              f"over {len(SEEDS)} seeds -> {os.path.relpath(path)}")
    return {"func": func, "hidden": hidden, "depth": depth,
            "float_mae": best_mae, "float_maxerr": best_maxerr, "seed_best": seed_best,
            "maxerr_mean": maxerr_mean, "maxerr_std": maxerr_std, "path": path}


def main():
    ap = argparse.ArgumentParser(description="Train MLP baselines for the symbolic-vs-MLP study")
    ap.add_argument("--func", choices=list(FUNCS), help="single function (default: all)")
    ap.add_argument("--h", type=int, help="single hidden width (default: sweep)")
    ap.add_argument("--depth", type=int, help="single depth (default: sweep)")
    args = ap.parse_args()

    funcs = [args.func] if args.func else list(FUNCS)
    hiddens = [args.h] if args.h else HIDDENS
    depths = [args.depth] if args.depth else DEPTHS
    rows = []
    for func in funcs:
        for d in depths:
            for h in hiddens:
                rows.append(train_cell(func, h, d))

    # Per-cell summary (best-of-N + spread) for the note's mean+/-std table.
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["func", "hidden", "depth", "seed_best", "float_maxerr_best",
                    "maxerr_mean", "maxerr_std"])
        for r in rows:
            w.writerow([r["func"], r["hidden"], r["depth"], r["seed_best"],
                        round(r["float_maxerr"], 6), round(r["maxerr_mean"], 6),
                        round(r["maxerr_std"], 6)])
    print(f"\nWrote {len(rows)} cells -> {os.path.relpath(SUMMARY_CSV)}")


if __name__ == "__main__":
    main()
