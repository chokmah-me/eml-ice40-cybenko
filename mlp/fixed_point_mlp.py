"""
Bit-exact integer model of a quantized scalar MLP for iCE40.

This is the MLP analogue of hardware/fixed_point.py::FixedEML: an integer-only
forward pass that mirrors the emitted Verilog (mlp/verilog_gen_mlp.py)
operation-for-operation, so the Python error report *is* the hardware error
report and sim_check only has to confirm RTL == this model.

Arithmetic contract (identical in Python and Verilog), single Q(int).(frac):
  per neuron, over its fan-in:
    acc  = sum_j  w_raw[j] * a_raw[j]          # exact, wide signed accumulator
    pre  = (acc >>> F) + b_raw                 # arithmetic (floor) shift, then bias
    sat  = saturate(pre) to the W-bit range
    out  = max(0, sat)  for hidden layers (ReLU) ; sat for the output layer
Weights/biases are quantized once with FixedPointFormat.quantize (round-to-
nearest, saturating). The input x_raw is already a Q(int).(frac) code.

Python `>>` on negative ints floors, matching Verilog signed `>>>`; both sides
saturate identically, so overflow is well-defined rather than UB.
"""

import os
from typing import List, Tuple

import numpy as np

from hardware.fixed_point import FixedPointFormat


class QuantMLP:
    """Integer MLP: layers quantized to one FixedPointFormat; bit-exact to RTL."""

    def __init__(self, layers_float: List[Tuple[np.ndarray, np.ndarray]],
                 fmt: FixedPointFormat):
        self.fmt = fmt
        self.layers = []  # list of (w_raw [out,in], b_raw [out])
        for w, b in layers_float:
            w_raw = [[fmt.quantize(float(v)) for v in row] for row in w]
            b_raw = [fmt.quantize(float(v)) for v in b]
            self.layers.append((w_raw, b_raw))

    def _neuron(self, w_row: List[int], b: int, a: List[int], relu: bool) -> int:
        f = self.fmt.frac_bits
        acc = sum(wj * aj for wj, aj in zip(w_row, a))
        pre = (acc >> f) + b
        sat = self.fmt.saturate(pre)
        return max(0, sat) if relu else sat

    def forward_raw(self, *x_raw: int) -> int:
        a = list(x_raw)              # one or more input codes (multi-input MLPs)
        n = len(self.layers)
        for li, (w_raw, b_raw) in enumerate(self.layers):
            relu = li < n - 1            # every layer but the output has ReLU
            a = [self._neuron(w_raw[j], b_raw[j], a, relu) for j in range(len(b_raw))]
        return a[0]                       # scalar output

    # convenience -----------------------------------------------------------
    def eval_float(self, x: float) -> float:
        return self.fmt.dequantize(self.forward_raw(self.fmt.quantize(x)))

    @property
    def latency(self) -> int:
        """Pipeline depth = one register stage per Linear layer (matches RTL)."""
        return len(self.layers)


def load_cell(func: str, hidden: int, depth: int):
    """Load a trained cell's float layer list + metadata from results/mlp/."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "results", "mlp",
                        f"{func}_h{hidden}_d{depth}.npz")
    d = np.load(path, allow_pickle=True)
    n = int(d["n_layers"])
    layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
    meta = {"func": str(d["func"]), "hidden": int(d["hidden"]),
            "depth": int(d["depth"]), "domain": tuple(d["domain"].tolist()),
            "float_mae": float(d["float_mae"]), "float_maxerr": float(d["float_maxerr"])}
    return layers, meta


# Q-formats matched to the symbolic units (exp Q8.8, ln Q10.12).
FUNC_FMT = {"exp": FixedPointFormat(8, 8), "ln": FixedPointFormat(10, 12)}


def error_report(func: str, hidden: int, depth: int, npts: int = 2001) -> dict:
    """Fixed-point max-err/MAE vs the true function over the cell's domain."""
    import math
    layers, meta = load_cell(func, hidden, depth)
    fmt = FUNC_FMT[func]
    qm = QuantMLP(layers, fmt)
    lo, hi = meta["domain"]
    true_fn = math.exp if func == "exp" else math.log
    errs = []
    for i in range(npts):
        x = lo + i * (hi - lo) / (npts - 1)
        errs.append(abs(qm.eval_float(x) - true_fn(x)))
    return {"func": func, "hidden": hidden, "depth": depth, "fmt": repr(fmt),
            "fp_mae": float(np.mean(errs)), "fp_maxerr": float(np.max(errs)),
            "float_maxerr": meta["float_maxerr"], "latency": qm.latency}


if __name__ == "__main__":
    from mlp.train_mlp import HIDDENS, DEPTHS, FUNCS
    print(f"{'cell':18} {'fmt':28} {'fp_mae':>9} {'fp_maxerr':>10} {'float_maxerr':>12}")
    for func in FUNCS:
        for d in DEPTHS:
            for h in HIDDENS:
                try:
                    r = error_report(func, h, d)
                except FileNotFoundError:
                    print(f"{func}_h{h}_d{d}: (not trained)")
                    continue
                print(f"{func}_h{h}_d{d:<8} {r['fmt']:28} "
                      f"{r['fp_mae']:9.5f} {r['fp_maxerr']:10.5f} {r['float_maxerr']:12.5f}")
