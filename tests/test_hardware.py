"""
Positive + negative tests for the Track C hardware package.

Numeric thresholds are the measured PoC numbers from results/hardware_poc_report.md,
locked in as regression bounds. The bit-exact Python model is the numeric ground
truth standing in for HDL simulation (no toolchain on this machine), so these
tests are the hardware verification suite until yosys/iverilog are available.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_layer_v2 import EMLTree
from hardware.fixed_point import FixedPointFormat, FixedEML
from hardware.converter import extract_netlist, eval_netlist_float, eval_netlist_fixed
from hardware.form_parser import parse_form
from hardware.verilog_gen import emit_verilog, _hex_lines
from hardware.verilog_gen_pipelined import emit_verilog_pipelined, gate_times, GATE_LATENCY

Q8_8 = FixedPointFormat(8, 8)
Q10_8 = FixedPointFormat(10, 8)
Q10_12 = FixedPointFormat(10, 12)

EXP_FORM = "eml(x,1)"
LN_FORM = "eml(1,eml(eml(1,x),1))"


def grid(a, b, n=501):
    return [a + i * (b - a) / (n - 1) for i in range(n)]


def snapped_tree(func, depth):
    t = EMLTree(depth=depth)
    t.initialize_to_target(func, noise=0.0)
    t.snap_all()
    return t


# ============================================================ fixed-point ====

class TestFixedPointFormat:
    def test_q88_constants(self):
        assert Q8_8.total_bits == 16
        assert Q8_8.scale == 256
        assert Q8_8.raw_max == 32767 and Q8_8.raw_min == -32768
        assert Q8_8.max_val == pytest.approx(127.99609375)

    @pytest.mark.parametrize("fmt", [Q8_8, Q10_12], ids=["Q8.8", "Q10.12"])
    def test_roundtrip_within_half_step(self, fmt):
        for v in [-100.0, -1.234, 0.0, 0.5, 1.0, math.e, 100.0]:
            if fmt.min_val <= v <= fmt.max_val:
                assert abs(fmt.dequantize(fmt.quantize(v)) - v) <= 0.5 / fmt.scale

    def test_quantize_saturates_out_of_range(self):
        assert Q8_8.quantize(1e6) == Q8_8.raw_max
        assert Q8_8.quantize(-1e6) == Q8_8.raw_min

    def test_saturate_bounds(self):
        assert Q8_8.saturate(Q8_8.raw_max + 1) == Q8_8.raw_max
        assert Q8_8.saturate(Q8_8.raw_min - 1) == Q8_8.raw_min

    def test_too_few_frac_bits_rejected(self):
        with pytest.raises(AssertionError):
            FixedPointFormat(8, 3)


class TestFixedEML:
    @pytest.mark.parametrize("fmt,bound", [(Q8_8, 0.016), (Q10_12, 0.005)],
                             ids=["Q8.8", "Q10.12"])
    def test_exp_accuracy(self, fmt, bound):
        fe = FixedEML(fmt)
        worst = max(abs(fmt.dequantize(fe.exp_fix(fmt.quantize(x))) - math.exp(x))
                    for x in grid(-2.0, 2.0))
        assert worst <= bound

    @pytest.mark.parametrize("fmt,bound", [(Q8_8, 0.022), (Q10_12, 0.002)],
                             ids=["Q8.8", "Q10.12"])
    def test_ln_accuracy(self, fmt, bound):
        fe = FixedEML(fmt)
        worst = max(abs(fmt.dequantize(fe.ln_fix(fmt.quantize(x))) - math.log(x))
                    for x in grid(0.1, 10.0))
        assert worst <= bound

    def test_ln_far_above_one_uses_range_reduction(self):
        # x=150 needs k=7 octaves above the mantissa LUT (exercises k*ln2 path).
        # The k*ln2 term amplifies ln2's quantization error by k, so this needs
        # >=12 frac bits to stay tight (at Q10.8 the error is ~7*0.0017 = 0.012).
        fe = FixedEML(Q10_12)
        got = Q10_12.dequantize(fe.ln_fix(Q10_12.quantize(150.0)))
        assert abs(got - math.log(150.0)) < 0.005

    def test_eml_is_exp_minus_ln_saturated(self):
        fe = FixedEML(Q8_8)
        for x in grid(-2.0, 2.0, 101):
            for y in (0.5, 1.0, 5.0):
                a, b = Q8_8.quantize(x), Q8_8.quantize(y)
                assert fe.eml_fix(a, b) == Q8_8.saturate(fe.exp_fix(a) - fe.ln_fix(b))

    def test_deterministic(self):
        f1, f2 = FixedEML(Q10_12), FixedEML(Q10_12)
        assert f1.exp_lut == f2.exp_lut and f1.ln_lut == f2.ln_lut
        for x in grid(-2.0, 2.0, 50):
            r = Q10_12.quantize(x)
            assert f1.exp_fix(r) == f2.exp_fix(r)

    # ---- negative / edge ----

    def test_ln_nonpositive_clamps(self):
        fe = FixedEML(Q8_8)
        ref = fe.ln_fix(1)  # smallest positive code
        assert fe.ln_fix(0) == ref
        assert fe.ln_fix(-1000) == ref
        assert fe.eml_fix(Q8_8.quantize(0.0), 0) == Q8_8.saturate(
            fe.exp_fix(Q8_8.quantize(0.0)) - ref)

    def test_exp_saturates_at_clamp_top(self):
        fe = FixedEML(Q8_8)
        # exp(8) ~ 2981 > 127.996 -> LUT entry already saturated to raw_max
        assert fe.exp_fix(Q8_8.quantize(8.0)) == Q8_8.raw_max
        assert fe.exp_fix(Q8_8.raw_max) == Q8_8.raw_max

    def test_exp_below_clamp_hits_lut_floor(self):
        fe = FixedEML(Q8_8)
        assert fe.exp_fix(Q8_8.raw_min) == fe.exp_lut[0]

    def test_eml_output_saturates_high(self):
        fe = FixedEML(Q8_8)
        # exp(8) - ln(tiny) >> max -> raw_max
        assert fe.eml_fix(Q8_8.quantize(8.0), 1) == Q8_8.raw_max


# ============================================================== netlists ====

class TestNetlistExtraction:
    def test_exp_d2(self):
        net = extract_netlist(snapped_tree('exp', 2))
        assert len(net.gates) == 1
        assert len(net.active_gates()) == 1
        assert net.gates[0].left == ('x', -1) and net.gates[0].right == ('const1', -1)

    def test_ln_d4_structure_and_folding(self):
        net = extract_netlist(snapped_tree('ln', 4))
        assert len(net.gates) == 7
        assert len(net.active_gates()) == 3
        assert net.gates[net.out_idx].on_path
        # off-path eml(1,1) gates fold to exp(1)-ln(1) = e
        consts = [g.const_val for g in net.gates if g.const_val is not None]
        assert consts and all(c == pytest.approx(math.e) for c in consts)

    def test_overdepth_exp_d4_folds_to_one_gate(self):
        net = extract_netlist(snapped_tree('exp', 4))
        assert len(net.active_gates()) == 1
        for x in grid(-2.0, 2.0, 101):
            assert eval_netlist_float(net, x) == pytest.approx(math.exp(x), abs=1e-9)

    def test_ln_d4_float_eval_exact(self):
        net = extract_netlist(snapped_tree('ln', 4))
        for x in grid(0.1, 10.0, 101):
            assert eval_netlist_float(net, x) == pytest.approx(math.log(x), abs=1e-9)

    def test_fixed_eval_regression_bounds(self):
        exp_net = extract_netlist(snapped_tree('exp', 2))
        fe = FixedEML(Q8_8)
        errs = [abs(eval_netlist_fixed(exp_net, fe, x) - math.exp(x))
                for x in grid(-2.0, 2.0)]
        assert sum(errs) / len(errs) <= 0.003 and max(errs) <= 0.016

        ln_net = extract_netlist(snapped_tree('ln', 4))
        fe = FixedEML(Q10_12)
        errs = [abs(eval_netlist_fixed(ln_net, fe, x) - math.log(x))
                for x in grid(0.1, 10.0)]
        assert sum(errs) / len(errs) <= 0.0005 and max(errs) <= 0.0015

    def test_ln_d4_q88_overflow_stays_detectable(self):
        # Documented failure mode: e^e/x reaches ~151.5 > Q8.8 max. If this
        # stops failing, someone changed clamping without widening the format.
        net = extract_netlist(snapped_tree('ln', 4))
        fe = FixedEML(Q8_8)
        worst = max(abs(eval_netlist_fixed(net, fe, x) - math.log(x))
                    for x in grid(0.1, 10.0))
        assert worst > 0.1

    # ---- negative ----

    def test_unsnapped_tree_rejected(self):
        t = EMLTree(depth=2)
        t.randomize(0.1)
        with pytest.raises(AssertionError, match="snapped"):
            extract_netlist(t)


# ============================================================ form parser ====

class TestFormParser:
    @pytest.mark.parametrize("form,n_gates,n_active",
                             [(EXP_FORM, 1, 1), (LN_FORM, 3, 3)])
    def test_canonical_forms_parse(self, form, n_gates, n_active):
        net = parse_form(form)
        assert len(net.gates) == n_gates
        assert len(net.active_gates()) == n_active

    @pytest.mark.parametrize("fmt", [Q8_8, Q10_12], ids=["Q8.8", "Q10.12"])
    @pytest.mark.parametrize("func,depth,form",
                             [('exp', 2, EXP_FORM), ('ln', 4, LN_FORM)])
    def test_parsed_bit_identical_to_tree_extracted(self, fmt, func, depth, form):
        tree_net = extract_netlist(snapped_tree(func, depth))
        parsed_net = parse_form(form)
        fe = FixedEML(fmt)
        lo, hi = (-2.0, 2.0) if func == 'exp' else (0.1, 10.0)
        for x in grid(lo, hi, 201):
            assert eval_netlist_fixed(tree_net, fe, x) == eval_netlist_fixed(parsed_net, fe, x)

    def test_constant_expression_folds_fully(self):
        net = parse_form("eml(1,1)")
        assert net.active_gates() == []
        assert net.gates[net.out_idx].const_val == pytest.approx(math.e)

    # ---- negative ----

    @pytest.mark.parametrize("bad", [
        "eml(x",          # truncated
        "eml(x,1))",      # trailing junk
        "eml(x;1)",       # bad separator
        "y",              # unknown atom
        "foo",            # unknown atom
        "",               # empty
        "x",              # atom only, not a gate expression
        "eml(x,1,1)",     # arity
    ])
    def test_malformed_forms_raise_valueerror(self, bad):
        with pytest.raises(ValueError):
            parse_form(bad)


# ========================================================= verilog emission ====

class TestVerilogEmission:
    def test_hex_lines_twos_complement(self):
        assert _hex_lines([-1], 16).strip() == "FFFF"
        assert _hex_lines([1], 16).strip() == "0001"

    @pytest.mark.parametrize("func,depth,form,fmt",
                             [('exp', 2, EXP_FORM, Q8_8), ('ln', 4, LN_FORM, Q10_12)])
    def test_emission_artifacts(self, tmp_path, func, depth, form, fmt):
        net = parse_form(form)
        fe = FixedEML(fmt)
        info = emit_verilog(net, fe, f"t_{func}", str(tmp_path))

        assert info["active_gates"] == len(net.active_gates())
        assert info["total_gates"] == len(net.gates)
        for key in ("rtl", "tb", "exp_hex", "ln_hex"):
            assert os.path.isfile(info[key])

        width_chars = (fmt.total_bits + 3) // 4
        for key in ("exp_hex", "ln_hex"):
            lines = open(info[key]).read().splitlines()
            assert len(lines) == 257
            assert all(len(l) == width_chars and int(l, 16) >= 0 for l in lines)

        rtl = open(info["rtl"]).read()
        assert f"module t_{func} (" in rtl
        # count instantiations (".W" connection), not the module definition
        assert rtl.count("eml_gate #(.W") == len(net.active_gates())
        assert f"assign y_out = g{net.out_idx}_out;" in rtl

    def test_ln_d4_active_path_references_no_folded_consts(self, tmp_path):
        net = parse_form(LN_FORM)
        rtl = open(emit_verilog(net, FixedEML(Q10_12), "t", str(tmp_path))["rtl"]).read()
        assert "(no folded constants referenced)" in rtl
        assert "const_g" not in rtl

    @pytest.mark.parametrize("form,exp_latency,n_gates",
                             [(EXP_FORM, GATE_LATENCY, 1), (LN_FORM, 3 * GATE_LATENCY, 3)])
    def test_pipelined_emission(self, tmp_path, form, exp_latency, n_gates):
        net = parse_form(form)
        fe = FixedEML(Q10_12)
        info = emit_verilog_pipelined(net, fe, "t_pipe", str(tmp_path))

        assert info["latency"] == exp_latency
        assert gate_times(net)[net.out_idx][1] == exp_latency
        rtl = open(info["rtl"]).read()
        assert rtl.count("eml_gate_p #(.W") == n_gates
        assert f"localparam LATENCY = {exp_latency};" in rtl
        # split-bank ROM images: 256 entries each (A = 0..255, B = 1..256)
        hexes = [f for f in os.listdir(tmp_path) if f.endswith(".hex")]
        assert len(hexes) == 4
        for h in hexes:
            assert len(open(os.path.join(tmp_path, h)).read().splitlines()) == 256

    def test_pipelined_banks_are_shifted_by_one(self, tmp_path):
        fe = FixedEML(Q8_8)
        emit_verilog_pipelined(parse_form(EXP_FORM), fe, "t", str(tmp_path))
        a = open(os.path.join(tmp_path, "t_exp_lut_a.hex")).read().splitlines()
        b = open(os.path.join(tmp_path, "t_exp_lut_b.hex")).read().splitlines()
        assert a[1:] == b[:-1]                       # lutB[i] == lutA[i+1]
        assert int(b[-1], 16) == fe.exp_lut[256] & 0xFFFF  # endpoint entry

    def test_pipelined_rejects_constant_netlist(self, tmp_path):
        with pytest.raises(AssertionError):
            emit_verilog_pipelined(parse_form("eml(1,1)"), FixedEML(Q8_8), "t", str(tmp_path))

    def test_constant_netlist_emits_literal_output(self, tmp_path):
        net = parse_form("eml(1,1)")
        fe = FixedEML(Q8_8)
        rtl = open(emit_verilog(net, fe, "t_const", str(tmp_path))["rtl"]).read()
        assert "eml_gate #(" not in rtl.split("// ------")[-1]  # no instances in top
        assert f"assign y_out = 16'sd{Q8_8.quantize(math.e)};" in rtl
