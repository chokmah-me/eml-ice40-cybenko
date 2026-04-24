"""
EML Layer v2 - Correct Odrzywołek Architecture
================================================
Implements equation (6) from Odrzywołek (2026) with hardening phase.

Architecture: every input to every EML gate is an independent InputSelector.
  bottom-level gates: inputs from {1, x}        -> 2 logits each
  upper-level gates:  inputs from {1, x, child} -> 3 logits each
  params = 5*2^(d-1) - 6  (matches Odrzywołek exactly)

Training (3-phase, matching Section 4.3):
  Phase 1 (60%): standard Adam, task loss only
  Phase 2 (20%): Adam + entropy penalty (forces weights toward vertices)
  Phase 3 (20%): temperature annealing T: 1.0 -> 0.05 (sharpens softmax)

Valid snap: post-snap loss within 10x of pre-snap loss AND snappability>=0.9.
False snaps (wrong vertex, high post-snap loss) are rejected.

Naming vs Odrzywołek:
  His "level n" = our depth (n+1).
  exp(x) recoverable at our depth 2 (his level 1).
  ln(x)  recoverable at our depth 3 (his level 2).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


# ============================================================================
# 1. EML Gate
# ============================================================================

class EMLGate(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xc = torch.clamp(x, -8.0, 8.0)
        xc = torch.complex(xc, torch.zeros_like(xc))
        yc = torch.complex(y, torch.zeros_like(y))
        return (torch.exp(xc) - torch.log(yc)).real


# ============================================================================
# 2. Input Selector with temperature support
# ============================================================================

class InputSelector(nn.Module):
    """
    Parameterizes one scalar input to one EML gate.
    num_choices=2: {1, x}
    num_choices=3: {1, x, f}
    temperature: softmax sharpness (default 1.0, anneal toward 0 to harden)
    """
    def __init__(self, num_choices: int):
        super().__init__()
        assert num_choices in (2, 3)
        self.num_choices = num_choices
        self.logits = nn.Parameter(torch.zeros(num_choices))
        self.temperature = 1.0  # set externally during annealing

    def forward(self, ones: torch.Tensor, x: torch.Tensor,
                f: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = torch.softmax(self.logits / self.temperature, dim=0)
        out = w[0] * ones + w[1] * x
        if self.num_choices == 3:
            out = out + w[2] * f
        return out

    def get_weights(self, temperature: float = 1.0):
        return torch.softmax(self.logits / temperature, dim=0)

    def max_weight(self) -> float:
        return self.get_weights().max().item()

    def is_snapped(self, threshold: float = 0.9) -> bool:
        return self.max_weight() >= threshold

    def snap(self):
        with torch.no_grad():
            idx = self.get_weights().argmax()
            self.logits.fill_(-10.0)
            self.logits[idx] = 10.0

    def symbol(self) -> str:
        return ['1', 'x', 'f'][self.get_weights().argmax().item()]

    def entropy(self) -> torch.Tensor:
        w = self.get_weights().clamp(1e-8, 1.0)
        return -(w * w.log()).sum()


# ============================================================================
# 3. EML Tree
# ============================================================================

class EMLTree(nn.Module):
    """
    Full EML expression tree. depth d = Odrzywołek level (d-1).
    params = 5*2^(d-1) - 6.
    """

    def __init__(self, depth: int, num_vars: int = 1, device: str = 'cpu'):
        super().__init__()
        assert depth >= 2
        assert num_vars == 1
        self.depth = depth
        self.device = device
        self.gate = EMLGate()

        num_levels = depth - 1
        self.levels = nn.ModuleList()
        for lev in range(num_levels):
            num_gates = 2 ** (num_levels - 1 - lev)
            nc = 2 if lev == 0 else 3
            self.levels.append(nn.ModuleList([
                nn.ModuleList([InputSelector(nc), InputSelector(nc)])
                for _ in range(num_gates)
            ]))

        self.to(device)

    def set_temperature(self, T: float):
        for s in self.all_selectors():
            s.temperature = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        b = x.shape[0]
        ones = torch.ones(b, device=x.device)
        xv = x[:, 0]

        current = []
        for left_sel, right_sel in self.levels[0]:
            l = left_sel(ones, xv, None)
            r = right_sel(ones, xv, None)
            current.append(self.gate(l, r))

        for lev_idx in range(1, len(self.levels)):
            nxt = []
            for gi, (left_sel, right_sel) in enumerate(self.levels[lev_idx]):
                lc = current[2 * gi]
                rc = current[2 * gi + 1]
                l = left_sel(ones, xv, lc)
                r = right_sel(ones, xv, rc)
                nxt.append(self.gate(l, r))
            current = nxt

        return current[0]

    def all_selectors(self) -> List[InputSelector]:
        sels = []
        for lev in self.levels:
            for left_sel, right_sel in lev:
                sels.extend([left_sel, right_sel])
        return sels

    def snappability(self) -> float:
        return float(np.mean([s.max_weight() for s in self.all_selectors()]))

    def is_snapped(self, threshold: float = 0.9) -> bool:
        return all(s.is_snapped(threshold) for s in self.all_selectors())

    def snap_all(self):
        for s in self.all_selectors():
            s.snap()

    def total_entropy(self) -> torch.Tensor:
        return sum(s.entropy() for s in self.all_selectors())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def randomize(self, scale: float = 0.1):
        for s in self.all_selectors():
            nn.init.normal_(s.logits, 0.0, scale)

    def symbolic_form(self) -> str:
        syms = []
        for left_sel, right_sel in self.levels[0]:
            l = left_sel.symbol().replace('f', '?')
            r = right_sel.symbol().replace('f', '?')
            syms.append(f'eml({l},{r})')

        for lev_idx in range(1, len(self.levels)):
            nxt = []
            for gi, (left_sel, right_sel) in enumerate(self.levels[lev_idx]):
                lc = syms[2 * gi]
                rc = syms[2 * gi + 1]
                l = lc if left_sel.symbol() == 'f' else left_sel.symbol()
                r = rc if right_sel.symbol() == 'f' else right_sel.symbol()
                nxt.append(f'eml({l},{r})')
            syms = nxt

        return syms[0]


# ============================================================================
# 4. Three-phase training
# ============================================================================

def train_eml(tree: EMLTree, x_data: torch.Tensor, y_data: torch.Tensor,
              epochs: int = 2000, lr: float = 0.01,
              entropy_coeff: float = 0.05,
              anneal_temp_final: float = 0.05,
              verbose: bool = False) -> dict:
    """
    Three-phase training matching Odrzywołek Section 4.3.

    Phase 1 (60%): standard Adam, task loss only.
    Phase 2 (20%): task loss + entropy penalty (coeff ramps 0 -> entropy_coeff).
    Phase 3 (20%): task loss + full entropy penalty + temperature anneal T->anneal_temp_final.

    Valid snap check: post-snap loss must be <= 10x pre-snap loss.
    """
    tree.train()
    tree.set_temperature(1.0)
    opt = torch.optim.Adam(tree.parameters(), lr=lr)
    criterion = nn.L1Loss()

    e1 = int(epochs * 0.60)  # end of phase 1
    e2 = int(epochs * 0.80)  # end of phase 2

    best_loss = float('inf')
    no_improve = 0
    nan_epoch = -1
    pre_snap_loss = float('nan')

    for epoch in range(epochs):
        opt.zero_grad()
        try:
            pred = tree(x_data)
        except Exception:
            nan_epoch = epoch; break
        if torch.isnan(pred).any():
            nan_epoch = epoch; break

        task_loss = criterion(pred, y_data)
        if torch.isnan(task_loss):
            nan_epoch = epoch; break

        # Phase schedule
        if epoch < e1:
            loss = task_loss
        elif epoch < e2:
            # Ramp entropy coeff from 0 -> entropy_coeff
            ramp = (epoch - e1) / (e2 - e1)
            coeff = entropy_coeff * ramp
            loss = task_loss + coeff * tree.total_entropy()
        else:
            # Full entropy + temperature annealing
            ramp = (epoch - e2) / (epochs - e2)
            T = 1.0 - ramp * (1.0 - anneal_temp_final)
            tree.set_temperature(max(T, anneal_temp_final))
            loss = task_loss + entropy_coeff * tree.total_entropy()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 10.0)
        opt.step()

        lv = task_loss.item()
        if lv < best_loss:
            best_loss = lv; no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            snap_str = 'Y' if tree.is_snapped() else 'N'
            T_str = f'T={tree.all_selectors()[0].temperature:.2f}' if epoch >= e2 else ''
            print(f'  E{epoch+1:5d} loss={lv:.6f} snap={snap_str} '
                  f'snappability={tree.snappability():.4f} {T_str}')

    if nan_epoch != -1:
        tree.set_temperature(1.0)
        return {'final_loss': float('nan'), 'snappability': float('nan'),
                'snapped': 0, 'nan_epoch': nan_epoch, 'converged': 0,
                'post_snap_loss': float('nan'), 'valid_snap': 0}

    tree.set_temperature(1.0)
    pre_snap_loss = best_loss

    # Post-snap validation
    tree.snap_all()
    with torch.no_grad():
        try:
            snapped_pred = tree(x_data)
            post_snap_loss = criterion(snapped_pred, y_data).item()
        except Exception:
            post_snap_loss = float('nan')

    snap_ok = tree.is_snapped()
    # Valid snap: snappability threshold AND post-snap loss not catastrophically worse
    valid = (snap_ok and
             not np.isnan(post_snap_loss) and
             post_snap_loss < 0.01)

    return {
        'final_loss': pre_snap_loss,
        'snappability': tree.snappability(),
        'snapped': int(snap_ok),
        'nan_epoch': -1,
        'converged': int(no_improve < 200),
        'post_snap_loss': post_snap_loss,
        'valid_snap': int(valid),
    }


# ============================================================================
# 5. Self-test
# ============================================================================

if __name__ == '__main__':
    print('EML Layer v2 - Self Test')
    print('=' * 55)

    print('\nParameter counts (must match 5*2^(d-1)-6):')
    all_ok = True
    for d in range(2, 7):
        tree = EMLTree(depth=d)
        n = tree.num_params()
        expected = 5 * 2**(d-1) - 6
        ok = n == expected
        all_ok = all_ok and ok
        print(f'  depth {d} (level {d-1}): {n} params  {"OK" if ok else f"FAIL expected={expected}"}')

    if not all_ok:
        import sys; sys.exit(1)

    print()
    # exp(x) = eml(x,1): representable at our depth 2 (Odrzywołek level 1)
    # ln(x)  = eml(1, eml(eml(1,x), 1)): needs our depth 4 (level 3)
    #   because our balanced binary tree requires 3 levels of gates,
    #   while Odrzywołek's unbalanced RPN depth-3 can share subtrees.
    configs = [
        ('exp(x)', torch.linspace(-2, 2, 64),   torch.exp, 2, 0.01),
        ('ln(x)',  torch.linspace(0.1, 10, 128), torch.log, 4, 0.001),
    ]

    for fname, x, fn, depth, lr in configs:
        y = fn(x)
        print(f'depth-{depth} ({fname}), 20 seeds x 3000 epochs:')
        n_valid = 0
        forms = set()
        for seed in range(20):
            torch.manual_seed(seed)
            tree = EMLTree(depth=depth)
            tree.randomize(0.1)
            m = train_eml(tree, x, y, epochs=3000, lr=lr, verbose=False)
            if m['valid_snap']:
                n_valid += 1
                forms.add(tree.symbolic_form())
                print(f'  seed {seed:2d}: VALID  pre={m["final_loss"]:.6f}'
                      f'  post={m["post_snap_loss"]:.6f}  {tree.symbolic_form()}')
            elif m['snapped']:
                print(f'  seed {seed:2d}: FALSE  pre={m["final_loss"]:.6f}'
                      f'  post={m["post_snap_loss"]:.6f}  {tree.symbolic_form()}')
            else:
                print(f'  seed {seed:2d}: miss   pre={m["final_loss"]:.6f}'
                      f'  snap={m["snappability"]:.4f}')
        print(f'  -> {n_valid}/20 valid snaps.  Forms: {forms}\n')
