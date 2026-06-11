#!/usr/bin/env python3
"""Quickstart: train one EML tree blind, then warm-started, and compare.

Runs in ~1 minute on CPU. Demonstrates the three core pieces of the package:
  - EMLTree + train_eml (3-phase annealing, valid-snap criterion)
  - the basin-selection problem (blind init often lands in eml(x,x) for exp)
  - initialize_to_target warm-start (recovers the correct form reliably)
"""
import torch
from eml_layer_v2 import EMLTree, train_eml

x = torch.linspace(-2.0, 2.0, 64)
y = torch.exp(x)

print("Target: exp(x) = eml(x,1), depth 2 (representational depth)\n")

for mode in ("blind", "warm"):
    torch.manual_seed(0)
    tree = EMLTree(depth=2)
    if mode == "blind":
        tree.randomize(0.1)
    else:
        tree.initialize_to_target("exp", noise=0.4)
    m = train_eml(tree, x, y, epochs=1500, lr=0.01)
    print(f"{mode:5s}: valid_snap={m['valid_snap']}  "
          f"form={tree.symbolic_form()}  post_snap_MAE={m['post_snap_loss']:.4f}")

print("\nA false snap typically lands in eml(x,x) (MAE ~0.688) — the competing "
      "basin the v2.2 paper identified. The warm start biases phase 1 into the "
      "correct basin; commitment (phases 2-3) was never the problem.")
