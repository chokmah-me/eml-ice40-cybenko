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

Valid snap (see train_eml): snapped (all selectors max_weight>=0.9) AND post_snap_loss < 0.01.
The symbolic_form is recorded by callers for post-analysis and embedding diagnostics
(pretrain_form column); it is *not* part of the automated filter in this implementation.
For the targets here the <0.01 threshold already excludes known competing basins
(e.g. eml(x,x) ~0.688 on exp). See paper for exact wording reconciliation.

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

    def bias_to_symbol(self, sym: str, strength: float = 10.0, noise: float = 0.0):
        """Strongly bias this selector toward one choice (for warm-start / basin experiments)."""
        idx = ['1', 'x', 'f'].index(sym)
        # Guard: bottom-level selectors (num_choices=2, {1,x}) must never receive 'f'.
        # This protects against (a) copy of symbols from a shallow that resolved 'f' at bottom
        # (shouldn't happen for valid ln/exp), or (b) accidental 'f' on nc=2 during custom basin work.
        # Assert surfaces the exact class of bug during smoke / re-runs; remove only if a future
        # unbalanced embedding legitimately needs to forward at the absolute bottom (unlikely).
        assert idx < self.num_choices, f"bias_to_symbol('{sym}') invalid for num_choices={self.num_choices} (bottom-level selectors have no 'f' choice)"
        with torch.no_grad():
            self.logits.fill_(0.0)
            self.logits[idx] = strength
            if noise > 0:
                self.logits += torch.randn_like(self.logits) * noise


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

    def verify_embedding(self, expected: str = None, warn: bool = True) -> str:
        """Return current symbolic_form(); optionally assert/warn it matches expected.
        Use after initialize_to_target or grow_from_shallow to catch wiring bugs before
        expensive training. The pretrain_form column in basin_warmstart runs serves the
        same purpose at data-capture time.
        """
        form = self.symbolic_form()
        if expected is not None and form != expected:
            msg = f"verify_embedding: got {form} != expected {expected}"
            if warn:
                print("WARNING:", msg)
            else:
                raise AssertionError(msg)
        return form

    def initialize_to_target(self, func: str, noise: float = 0.0):
        """
        Warm-start / basin-selection helper (Track B).

        Bias selectors toward the known-good symbolic form for the target function
        (instead of blind random init). This matches the spirit of Odrzywolek SI Table S7
        (correct solution + noise recovers at 100% even at high depth).

        Improved embedding for over-depth (v2 improvement after first Track B runs):
        - For exp at any depth: realize "eml(x,1)" *directly at the top gate* by routing
          original x and constant 1 to the final inputs (this is how the original paper's
          valid deeper exp solutions actually work — they don't compute a bottom eml(x,1)
          and pass it up; they route x all the way to the top and set everything else constant).
          All off-path selectors are biased to '1' (harmless constants).
        - For ln: embed the 3-gate chain using a spine of 'f' selectors, with siblings constant.
        """
        if func == 'exp':
            # Realize eml(x,1) directly at the root (top gate).
            # Top gate = last level, first gate.
            # One input gets direct 'x', the other gets direct '1'.
            # Everything else in the tree is set to constant (both inputs '1').
            top_lev = len(self.levels) - 1
            # Bias the top gate itself
            top_left, top_right = self.levels[top_lev][0]
            top_left.bias_to_symbol('x', strength=9.0, noise=noise)
            top_right.bias_to_symbol('1', strength=9.0, noise=noise)

            # All other selectors (every other gate at every level) → constant '1'
            for lev_idx, lev in enumerate(self.levels):
                for gidx, (left, right) in enumerate(lev):
                    if lev_idx == top_lev and gidx == 0:
                        continue  # already set above
                    left.bias_to_symbol('1', strength=6.0, noise=noise * 0.3)
                    right.bias_to_symbol('1', strength=6.0, noise=noise * 0.3)

        elif func == 'ln':
            # Target: eml(1, eml(eml(1,x),1))  — the 3-gate expression from Odrzywolek S7.
            # For any depth d>=4 (nlev>=3): plant the core at the *top 3 levels* using
            # direct '1'/'x' injection at the core entry (right-half path for polarity match)
            # and 'f' routing that actually connects (right-child pull at each step).
            # All lower levels (below core entry) are harmless constants. This guarantees
            # the pre-training symbolic_form() resolves to the exact target form (connected).
            # See grow_from_shallow for the curriculum (full-subtree embed) path.
            nlev = len(self.levels)
            if nlev < 3:
                self.randomize(0.1)
                return
            core0 = nlev - 3  # entry lev for core "bottom"; widths at top-3 are always 4/2/1
            # Core bottom (gi=2 in 4-wide): direct eml(1, x) injection (ignores its children)
            if core0 >= 0 and len(self.levels[core0]) > 2:
                self.levels[core0][2][0].bias_to_symbol('1', 8.0, noise)
                self.levels[core0][2][1].bias_to_symbol('x', 8.0, noise)
            # Core mid (lev core0+1, gi=1 in 2-wide): f from the core-bottom output (lc=...[2])
            mlev = core0 + 1
            if mlev < nlev and len(self.levels[mlev]) > 1:
                self.levels[mlev][1][0].bias_to_symbol('f', 9.0, noise)
                self.levels[mlev][1][1].bias_to_symbol('1', 8.0, noise)
            # Core top (lev core0+2 == tree top, gi=0): eml(1, f) pulling the mid via rc
            tlev = core0 + 2
            if tlev < nlev and len(self.levels[tlev]) > 0:
                self.levels[tlev][0][0].bias_to_symbol('1', 9.0, noise)
                self.levels[tlev][0][1].bias_to_symbol('f', 9.0, noise)

            # All other selectors (lower levels, unused core gates, siblings) → '1'
            for lev_idx, lev in enumerate(self.levels):
                for gidx, (left, right) in enumerate(lev):
                    if (lev_idx == core0 and gidx == 2) or (lev_idx == mlev and gidx == 1) or (lev_idx == tlev and gidx == 0):
                        continue
                    left.bias_to_symbol('1', strength=5.0, noise=noise * 0.25)
                    right.bias_to_symbol('1', strength=5.0, noise=noise * 0.25)

        else:
            self.randomize(0.1)

    def bias_selector(self, lev: int, gidx: int, side: str, sym: str, strength: float = 8.0, noise: float = 0.0):
        """Low-level hook for custom basin experiments: bias one specific selector."""
        pair = self.levels[lev][gidx]
        sel = pair[0] if side == 'left' else pair[1]
        sel.bias_to_symbol(sym, strength, noise)

    def grow_from_shallow(self, shallow: 'EMLTree', noise: float = 0.1):
        """
        Curriculum / grow-from-shallow helper for Track B.

        Embeds a valid solution from a shallower tree into this deeper tree.
        The shallow tree should typically be at the function's representational depth
        and already valid (snapped to correct form).

        Strategy:
        - For exp: use the proven direct top-gate eml(x,1) bias, plus copy the shallow's
          bottom eml(x,1) unit into one bottom gate of the deep tree (provides a "seed"
          sub-computation that the network can route from if helpful).
        - For ln: embed the shallow's full chain into the bottom `shallow.depth` levels
          along the first spine, then chain 'f' upward through the extra levels, with
          siblings set to constants.
        - All new capacity gets light noise + constant bias.

        This is the natural next lever after pure symbolic warm-start.

        Note on balanced vs unbalanced for ln over-depth:
        A single EML gate cannot act as an identity forwarder: there is no a,b
        (from {1,x,f}) such that exp(a) - ln(b) == f for arbitrary f. Therefore
        eml(f, 1) = exp(f) and eml(1, f) likewise does not preserve the subtree
        value. Adding one (or more) extra balanced levels on top of a valid d=4
        ln solution by 'f'-chaining will embed exp(ln(x)) == x (or a similar
        non-ln function) before any training. 

        For ln curriculum at over-representational depth the driver therefore
        reduces to the direct (fixed) initialize_to_target('ln') path; the
        grow + extension is not used. The full-block copy + reanneal remain
        valuable for exp and for exact-depth cases.

        True deeper ln requires unbalanced / RPN-style sharing (see the note
        below). The current grow assumes balanced binary trees. For unbalanced
        (e.g., Odrzywolek's original ln RPN sharing subtrees, which can use fewer
        levels), extend by parsing shallow.symbolic_form() to identify the exact
        "active path" selectors to embed, and treat all other parallel branches
        as extra capacity for constant bias. This could allow ln at lower effective
        depth and improve d=5 recovery. See prior_versions/dev/ for historical
        plans on unbalanced vs balanced.
        """
        if self.depth <= shallow.depth:
            # No growth needed
            self.initialize_to_target('exp' if shallow.depth == 2 else 'ln', noise)
            return

        # For exp-style shallow (depth==2), do a high-level top-gate bias first (the proven
        # direct eml(x,1) at root) before copying the bottom unit. This is harmless and
        # matches the v2.2 observation that successful deeper exp solutions route x/1 at top.
        # For ln-style (deeper shallow), skip the heuristic init entirely: the subsequent
        # full block copy of the valid shallow will provide the correct structure at the
        # embedded levels. An early initialize_to_target('ln') would plant a top-3 core
        # that then gets (partially) overwritten by the copy, causing the two embeddings
        # to fight (one of the issues identified in post-patch review).
        if shallow.depth == 2:
            try:
                self.initialize_to_target('exp', noise * 0.5)
            except Exception:
                self.randomize(noise)

        # Embed shallow structure
        if shallow.depth == 2:
            # exp-style: copy the single bottom gate's choices into one bottom gate here
            if len(self.levels) > 0 and len(shallow.levels) > 0:
                s_left, s_right = shallow.levels[0][0]
                self.levels[0][0][0].bias_to_symbol(s_left.symbol(), 7.0, noise)
                self.levels[0][0][1].bias_to_symbol(s_right.symbol(), 7.0, noise)

            # For extra levels above the embedded unit, ensure a forwarding path
            # (one spine of 'f' that can carry the bottom computation upward if the
            #  optimizer finds it useful; the top direct x/1 bias is already set above)
            for lev_idx in range(1, len(self.levels)):
                # Bias the first gate on this level to take 'f' from the active child
                # and constant on the other side
                if len(self.levels[lev_idx]) > 0:
                    l, r = self.levels[lev_idx][0]
                    l.bias_to_symbol('f', 6.0, noise * 0.6)
                    r.bias_to_symbol('1', 5.0, noise * 0.4)

            # Next-iteration tuning: for exp curriculum, re-assert a *very strong*
            # direct top-gate x/1 bias after embedding. This prioritizes the proven
            # "route original x at the top" strategy (per v2.2 paper) while still
            # providing the shallow core as a secondary basin seed in lower levels.
            # The extra levels get an extra nudge to constant to encourage collapse.
            top_lev = len(self.levels) - 1
            if top_lev >= 0 and len(self.levels[top_lev]) > 0:
                tl, tr = self.levels[top_lev][0]
                tl.bias_to_symbol('x', 9.5, noise * 0.2)
                tr.bias_to_symbol('1', 9.5, noise * 0.2)
            # Nudge all non-spine gates in extra levels even harder to constant
            for lev_idx in range(1, len(self.levels)):
                for gidx in range(len(self.levels[lev_idx])):
                    if gidx == 0:
                        continue
                    self.levels[lev_idx][gidx][0].bias_to_symbol('1', 7.0, noise * 0.1)
                    self.levels[lev_idx][gidx][1].bias_to_symbol('1', 7.0, noise * 0.1)

        else:
            # ln-style / general curriculum embed: copy the *full* shallow subtree block
            # (all gates at each level, left-aligned) into the deep tree. This preserves
            # any width used by the valid shallow (including right-half paths for ln form).
            # Internal 'f' references remain valid because child 2*gi positions match within block.
            # Then extend a forwarding 'f' on the first extra upper level's g0 (left, since
            # left-align reduction lands in current[0] slot), siblings const.
            # Record metadata so reanneal_extra_capacity can freeze the *entire* embedded
            # block (not just g0) and only sharpen true extras + new-lev siblings.
            n_shallow_lev = len(shallow.levels)
            embedded_widths = []
            for si in range(n_shallow_lev):
                if si < len(self.levels):
                    n_g = len(shallow.levels[si])
                    embedded_widths.append(n_g)
                    for gi in range(n_g):
                        if gi < len(self.levels[si]):
                            s_l, s_r = shallow.levels[si][gi]
                            self.levels[si][gi][0].bias_to_symbol(s_l.symbol(), 7.5, noise)
                            self.levels[si][gi][1].bias_to_symbol(s_r.symbol(), 7.5, noise)
                else:
                    embedded_widths.append(0)

            # Forwarding for extra upper levels (left-align => use left f to pull current[0])
            for lev_idx in range(n_shallow_lev, len(self.levels)):
                if len(self.levels[lev_idx]) > 0:
                    l, r = self.levels[lev_idx][0]
                    l.bias_to_symbol('f', 7.0, noise * 0.5)
                    r.bias_to_symbol('1', 5.0, noise * 0.3)

            # Const everything outside the copied block + outside the extension g0
            for lev_idx, lev in enumerate(self.levels):
                w = embedded_widths[lev_idx] if lev_idx < len(embedded_widths) else 0
                for gidx, (left, right) in enumerate(lev):
                    if lev_idx < n_shallow_lev and gidx < w:
                        continue  # copied embedded
                    if lev_idx >= n_shallow_lev and gidx == 0:
                        continue  # extension forward gate
                    left.bias_to_symbol('1', 5.0, noise * 0.2)
                    right.bias_to_symbol('1', 5.0, noise * 0.2)

            # Metadata for reanneal + debugging (pretrain_form will catch wiring issues)
            self._embedded_nlev = n_shallow_lev
            self._embedded_widths = embedded_widths
            self._embedded_from_shallow_depth = getattr(shallow, 'depth', None)

    def reanneal_extra_capacity(self, epochs: int = 300, lr: float = 0.001, anneal_temp_final: float = 0.05, verbose: bool = False):
        """
        Curriculum tuning helper (next iteration): after grow_from_shallow, run a short
        phase-3-style re-anneal *only on the 'extra' capacity* (non-spine gates).

        This directly addresses the diagnostic from exp d=3 curriculum runs:
        "grow successfully planted the shallow core, but extra levels did not fully collapse."

        Heuristic for "embedded vs extra":
        - The first gate at each level is considered the "active spine" (the embedded shallow solution path).
        - All other gates are "extra" and are the only ones whose logits participate in this short re-anneal.
        - The embedded spine selectors are temporarily frozen (very high bias) so they stay put.

        After the short anneal we restore temperature and leave the tree ready for any remaining main training or evaluation.

        For unbalanced-tree support (future): the same idea applies — mark the "used" selectors according to the embedded symbolic form and only re-anneal the unused ones.
        """
        if not self.levels:
            return

        # Determine embedded region (from grow_from_shallow metadata or legacy single-spine)
        if hasattr(self, '_embedded_widths') and hasattr(self, '_embedded_nlev'):
            emb_ws = self._embedded_widths
            n_emb = self._embedded_nlev
        else:
            emb_ws = [1] * len(self.levels)
            n_emb = len(self.levels)

        # Freeze the embedded block (full widths) + extension forward g0. 
        # Only true extras (embedded siblings + new-lev non-forward) get re-annealed.
        frozen_values = []
        for lev_idx, lev in enumerate(self.levels):
            if len(lev) == 0:
                continue
            w = emb_ws[lev_idx] if lev_idx < len(emb_ws) else 0
            for gidx in range(min(w, len(lev))):
                left, right = lev[gidx]
                for sel in (left, right):
                    sym = sel.symbol()
                    idx = ['1', 'x', 'f'].index(sym)
                    old_logits = sel.logits.detach().clone()
                    frozen_values.append((sel, old_logits))
                    with torch.no_grad():
                        sel.logits.fill_(-20.0)
                        sel.logits[idx] = 20.0
            # Extra levels: freeze their forward g0 too (preserve the link)
            if lev_idx >= n_emb and len(lev) > 0:
                left, right = lev[0]
                for sel in (left, right):
                    # avoid dup if w==1 already covered it
                    if any(id(fv[0]) == id(sel) for fv in frozen_values):
                        continue
                    sym = sel.symbol()
                    idx = ['1', 'x', 'f'].index(sym)
                    old_logits = sel.logits.detach().clone()
                    frozen_values.append((sel, old_logits))
                    with torch.no_grad():
                        sel.logits.fill_(-20.0)
                        sel.logits[idx] = 20.0

        # Now run a short temperature-anneal *only on the extra selectors*.
        original_temps = [s.temperature for lev in self.levels for pair in lev for s in pair]
        extra_selectors = []
        for lev_idx, lev in enumerate(self.levels):
            w = emb_ws[lev_idx] if lev_idx < len(emb_ws) else 0
            for gidx, (left, right) in enumerate(lev):
                if lev_idx < n_emb and gidx < w:
                    continue  # embedded core block (frozen above)
                if lev_idx >= n_emb and gidx == 0:
                    continue  # extension forward gate (keep f-link)
                extra_selectors.extend([left, right])

        if not extra_selectors:
            # Nothing extra to re-anneal (exact depth match)
            # Restore any frozen state just in case
            for sel, old in frozen_values:
                with torch.no_grad():
                    sel.logits.copy_(old)
            return

        # Short re-anneal loop (phase-3 style, task loss only, on extra selectors only)
        # We still need a forward pass, so we do a minimal optimizer step that only touches extra params.
        opt = torch.optim.Adam([s.logits for s in extra_selectors], lr=lr)
        criterion = torch.nn.L1Loss()  # reuse the same loss as train_eml

        # Create a tiny synthetic batch for the re-anneal (use the same grid style as the main experiment)
        # For simplicity we just use a small linspace around the typical domain; the point is to give the extra gates a signal.
        x_re = torch.linspace(-1.0, 1.0, 64)  # generic; the exact function doesn't matter much for collapse pressure
        # We don't have y here, so we use a dummy target that encourages the tree to produce something stable.
        # In practice the re-anneal is mostly about sharpening the extra selectors toward vertices while the spine is frozen.
        # To keep it simple and stable we use a zero target and L1 (encourages the extra subtrees to output near-zero constants).
        y_re = torch.zeros_like(x_re)

        for e in range(epochs):
            opt.zero_grad()
            # Set a linear temperature schedule for the extra selectors only
            ramp = e / max(1, epochs - 1)
            T = 1.0 - ramp * (1.0 - anneal_temp_final)
            for s in extra_selectors:
                s.temperature = max(T, anneal_temp_final)

            pred = self(x_re)
            loss = criterion(pred, y_re)
            loss.backward()
            # Only step the extra logits
            opt.step()

            if verbose and (e + 1) % max(1, epochs // 5) == 0:
                print(f"  reanneal e={e+1}/{epochs} loss={loss.item():.6f} T={T:.2f}")

        # Restore temperatures and unfreeze the spine
        for i, (sel, old) in enumerate(frozen_values):
            with torch.no_grad():
                sel.logits.copy_(old)
        # Restore original temps (they will be reset by the caller anyway)
        idx = 0
        for lev in self.levels:
            for pair in lev:
                for s in pair:
                    s.temperature = original_temps[idx]
                    idx += 1

        if verbose:
            print("reanneal_extra_capacity complete (spine frozen, extra capacity sharpened).")


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

    Valid snap check: snapped (max_weight >=0.9 on every selector) AND post-snap MAE < 0.01.
    (See module docstring and paper for the exact criterion used by the released CSV;
    symbolic form is analyzed offline via the form columns, not enforced here.)
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
