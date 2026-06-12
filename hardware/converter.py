"""
Netlist extraction from a snapped v2 EMLTree.

A snapped tree is pure dataflow: every InputSelector has committed to one of
{'1', 'x', 'f'}, so the soft simplex mixing disappears entirely and each gate
input is a wire from the constant 1.0, the primary input x, or a child gate.
This is what makes the hardware cheap -- no weight storage, no multipliers for
selection; the "weights" become routing.

extract_netlist also constant-folds: a gate whose snapped output cannot depend
on x (both inputs constant or fed by constant gates) is marked const and its
value is baked at generation time, so dangling/off-path gates (the deliberate
extra capacity in over-depth trees) cost zero LUTs in the emitted RTL.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .fixed_point import FixedEML


@dataclass
class Gate:
    idx: int                 # global gate index, bottom level first
    level: int
    pos: int                 # position within level
    left: Tuple[str, int]    # ('const1', -1) | ('x', -1) | ('gate', child_idx)
    right: Tuple[str, int]
    on_path: bool = False    # feeds the tree output
    const_val: Optional[float] = None  # folded float value if x-independent


@dataclass
class Netlist:
    gates: List[Gate] = field(default_factory=list)
    out_idx: int = -1        # index of root gate

    def active_gates(self) -> List[Gate]:
        return [g for g in self.gates if g.on_path and g.const_val is None]


def _src(selector, child_idx: int) -> Tuple[str, int]:
    sym = selector.symbol()
    if sym == '1':
        return ('const1', -1)
    if sym == 'x':
        return ('x', -1)
    return ('gate', child_idx)


def extract_netlist(tree) -> Netlist:
    """Walk a snapped EMLTree's levels and build the gate netlist."""
    assert tree.is_snapped(), "extract_netlist requires a snapped tree (call snap_all() first)"
    net = Netlist()
    idx = 0
    level_idx_of = []  # per level: list of global gate indices
    for lev_i, lev in enumerate(tree.levels):
        ids = []
        for gi, (lsel, rsel) in enumerate(lev):
            if lev_i == 0:
                left, right = _src(lsel, -1), _src(rsel, -1)
                # bottom selectors have no 'f'; _src never returns 'gate' here
            else:
                lc = level_idx_of[lev_i - 1][2 * gi]
                rc = level_idx_of[lev_i - 1][2 * gi + 1]
                left, right = _src(lsel, lc), _src(rsel, rc)
            net.gates.append(Gate(idx=idx, level=lev_i, pos=gi, left=left, right=right))
            ids.append(idx)
            idx += 1
        level_idx_of.append(ids)
    net.out_idx = level_idx_of[-1][0]

    # Mark gates reachable from the output (on the active path)
    stack = [net.out_idx]
    while stack:
        g = net.gates[stack.pop()]
        if g.on_path:
            continue
        g.on_path = True
        for kind, child in (g.left, g.right):
            if kind == 'gate':
                stack.append(child)

    fold_constants(net)
    return net


def fold_constants(net: Netlist):
    """Bake x-independent gates (gates must be in child-before-parent order)."""
    def src_const(kind_child):
        kind, child = kind_child
        if kind == 'const1':
            return 1.0
        if kind == 'x':
            return None
        return net.gates[child].const_val

    for g in net.gates:
        lv, rv = src_const(g.left), src_const(g.right)
        if lv is not None and rv is not None:
            g.const_val = math.exp(min(max(lv, -8.0), 8.0)) - math.log(rv)


def _src_val_float(net: Netlist, kind_child, x: float, cache):
    kind, child = kind_child
    if kind == 'const1':
        return 1.0
    if kind == 'x':
        return x
    return cache[child]


def eval_netlist_float(net: Netlist, x: float) -> float:
    """Float64 reference evaluation of the snapped netlist (with EMLGate's clamp)."""
    cache = {}
    for g in net.gates:
        a = _src_val_float(net, g.left, x, cache)
        b = _src_val_float(net, g.right, x, cache)
        cache[g.idx] = math.exp(min(max(a, -8.0), 8.0)) - math.log(b)
    return cache[net.out_idx]


def eval_netlist_fixed(net: Netlist, fe: FixedEML, x: float) -> float:
    """Bit-exact fixed-point evaluation (only active gates, folded constants baked)."""
    fmt = fe.fmt
    cache = {}
    for g in net.gates:
        if not g.on_path:
            continue
        if g.const_val is not None:
            cache[g.idx] = fmt.quantize(g.const_val)
            continue
        def raw(kind_child):
            kind, child = kind_child
            if kind == 'const1':
                return fe.one
            if kind == 'x':
                return fmt.quantize(x)
            return cache[child]
        cache[g.idx] = fe.eml_fix(raw(g.left), raw(g.right))
    return fmt.dequantize(cache[net.out_idx])
