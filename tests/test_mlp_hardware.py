"""
Tests for the quantized-MLP baseline (symbolic-vs-MLP study).

The arithmetic-contract tests use synthetic weights (no torch / no training), so
they always run and lock QuantMLP's integer semantics -- the same semantics the
emitted Verilog mirrors (verified bit-exact by mlp/sim_check_mlp.py). The
dominance/bounds tests read the trained artifacts in results/mlp/ and are skipped
if the sweep has not been run.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.fixed_point import FixedPointFormat
from mlp.fixed_point_mlp import QuantMLP, FUNC_FMT
from mlp.verilog_gen_mlp import emit_verilog_mlp

Q8_8 = FixedPointFormat(8, 8)
RESULTS_MLP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "mlp")


def L(w, b):
    return (np.array(w, dtype=np.float64), np.array(b, dtype=np.float64))


# ===================================================== arithmetic contract ====

class TestQuantMLPArithmetic:
    def test_single_linear_layer_is_identity(self):
        # one output layer, weight 1.0, bias 0 -> forward_raw is identity (saturated)
        qm = QuantMLP([L([[1.0]], [0.0])], Q8_8)
        for x in (-256, -1, 0, 1, 100, 256):
            assert qm.forward_raw(x) == x

    def test_relu_hidden_clamps_negative(self):
        # hidden ReLU then linear out, both identity weights: negatives -> 0
        qm = QuantMLP([L([[1.0]], [0.0]), L([[1.0]], [0.0])], Q8_8)
        assert qm.forward_raw(256) == 256       # +1.0 passes
        assert qm.forward_raw(-256) == 0        # -1.0 rectified
        assert qm.forward_raw(0) == 0

    def test_output_saturates_both_rails(self):
        # huge positive/negative weight saturates to the format rails (no wrap)
        big = Q8_8.max_val * 4
        qm_hi = QuantMLP([L([[big]], [0.0])], Q8_8)
        qm_lo = QuantMLP([L([[-big]], [0.0])], Q8_8)
        assert qm_hi.forward_raw(256) == Q8_8.raw_max
        assert qm_lo.forward_raw(256) == Q8_8.raw_min

    def test_bias_and_accumulate(self):
        # two inputs summed via a 2->1 layer + bias, checked against the contract
        qm = QuantMLP([L([[1.0]], [0.0]), L([[0.5, 0.5]], [0.25])], Q8_8)
        # layer0 (relu): single input -> [max(0,x)]  -- but in=1,out=1 here; build 1->2
        qm = QuantMLP([L([[1.0], [2.0]], [0.0, 0.0]),   # 1 -> 2, ReLU
                       L([[0.5, 0.5]], [0.25])], Q8_8)    # 2 -> 1, linear
        x = 256  # 1.0
        f = Q8_8.frac_bits
        a0 = [max(0, (Q8_8.quantize(1.0) * x >> f)), max(0, (Q8_8.quantize(2.0) * x >> f))]
        acc = Q8_8.quantize(0.5) * a0[0] + Q8_8.quantize(0.5) * a0[1]
        expect = Q8_8.saturate((acc >> f) + Q8_8.quantize(0.25))
        assert qm.forward_raw(x) == expect

    def test_latency_is_layer_count(self):
        qm = QuantMLP([L([[1.0]], [0.0]), L([[1.0]], [0.0]), L([[1.0]], [0.0])], Q8_8)
        assert qm.latency == 3


# ============================================================ RTL emission ====

class TestMLPEmitter:
    def test_emits_module_and_ports(self, tmp_path):
        qm = QuantMLP([L([[1.0], [2.0]], [0.0, 0.5]), L([[1.0, -1.0]], [0.0])], Q8_8)
        info = emit_verilog_mlp(qm, "mlp_t", str(tmp_path))
        rtl = open(info["rtl"]).read()
        assert "module mlp_t (" in rtl
        assert "input  signed [15:0] x_in" in rtl
        assert "output signed [15:0] y_out" in rtl
        assert info["latency"] == 2
        # ReLU on hidden layer, plain saturate on the output layer
        assert "<= 0) ? 16'sd0 :" in rtl       # ReLU branch present
        assert ">>> 8)" in rtl                  # requant shift by frac_bits

    def test_no_oversized_signed_literal(self, tmp_path):
        # rawmin must not be emitted as 16'sd32768 (unrepresentable); plain -32768
        qm = QuantMLP([L([[1.0]], [0.0])], Q8_8)
        rtl = open(emit_verilog_mlp(qm, "mlp_t2", str(tmp_path))["rtl"]).read()
        assert "16'sd32768" not in rtl
        assert "-32768" in rtl


# ===================================================== trained-cell bounds ====

def _trained(func, h, d):
    return os.path.exists(os.path.join(RESULTS_MLP, f"{func}_h{h}_d{d}.npz"))


@pytest.mark.skipif(not _trained("exp", 32, 2) or not _trained("ln", 32, 2),
                    reason="run `python -m mlp.train_mlp` first")
class TestTrainedCells:
    def test_fp_tracks_float_error(self):
        # quantization should not blow up the float approximation error
        from mlp.fixed_point_mlp import error_report
        for func, h, d in (("exp", 32, 2), ("ln", 32, 2)):
            r = error_report(func, h, d)
            assert r["fp_maxerr"] < r["float_maxerr"] + 0.05

    def test_direct_rom_ties_exp_but_infeasible_for_ln(self):
        # sec 3.5: the direct ROM matches each function's Q-format accuracy floor
        # (so ~= the symbolic unit), but exp fits a few EBR while ln needs >32.
        from mlp.rom_baseline import error_report as rom_err
        rexp, rln = rom_err("exp"), rom_err("ln")
        assert rexp["fp_maxerr"] < 0.02 and rexp["feasible_hx8k"]      # exp ROM fits
        assert rln["n_ebr"] > 32 and not rln["feasible_hx8k"]          # ln ROM does not

    def test_symbolic_dominates_every_mlp(self):
        # Joint Pareto dominance: across the full best-of-8 sweep, no MLP cell is
        # more accurate than the snapped-EML unit of the same function -- so symbolic
        # wins on accuracy outright and the area gap only widens the margin.
        from mlp.fixed_point_mlp import error_report
        from mlp.train_mlp import HIDDENS, DEPTHS
        SYM = {"exp": 0.014979, "ln": 0.001071}
        for func in ("exp", "ln"):
            for d in DEPTHS:
                for h in HIDDENS:
                    if not _trained(func, h, d):
                        continue
                    err = error_report(func, h, d)["fp_maxerr"]
                    assert err > SYM[func], (
                        f"{func} h{h} d{d} fp_maxerr {err} <= symbolic {SYM[func]} "
                        "-- an MLP edged the symbolic unit on accuracy; revisit the claim")


class TestArityTwoInput:
    """The 2-input experiment: symbolic cost is additive, ROM cost multiplicative.

    Toolchain-free (golden models + size formula only); no synthesis needed.
    """

    def test_symbolic_2in_is_native_gate_and_cheap_storage(self):
        # f(x,y) = exp(x) - ln(y) is the raw EML gate: two 257-entry LUT legs,
        # accurate, and stored independently of input wordlength.
        from mlp.symbolic_2in import error_report
        r = error_report()
        assert r["lut_entries"] == 2 * 257            # additive, width-independent
        assert r["fp_maxerr"] < 0.05                  # combined exp+ln Q8.8 error

    def test_rom_2in_explodes_before_matching_symbolic(self):
        # The 2-D ROM is 2^(2*Win) entries. At any size that fits an HX8K it is far
        # less accurate than the symbolic unit; it only reaches symbolic accuracy
        # once it has overflowed every iCE40. That is the arity claim, as data.
        from mlp.rom_baseline import error_report_2d
        from mlp.symbolic_2in import error_report as sym_err
        sym = sym_err()["fp_maxerr"]

        feasible = [error_report_2d(w) for w in (4, 6)]
        for r in feasible:
            assert r["feasible_hx8k"]
            assert r["fp_maxerr"] > sym          # fits, but worse than symbolic

        accurate = error_report_2d(8)
        assert accurate["fp_maxerr"] < 0.05      # finally ~symbolic accuracy...
        assert not accurate["feasible_hx8k"]     # ...but no longer fits any iCE40

    def test_rom_2in_entry_count_is_multiplicative(self):
        from mlp.rom_baseline import rom_spec_2d
        assert rom_spec_2d(6, Q8_8)["n_entries"] == (1 << 6) * (1 << 6)
        assert rom_spec_2d(8, Q8_8)["n_entries"] == (1 << 8) * (1 << 8)
