"""
Bit-exact fixed-point model of the EML gate for iCE40.

This is the v2 port of prior_versions/dev/fixed_point_converter.py with two
hardware-real changes:

1. exp(x): direct-indexed 257-entry LUT over the clamped domain [-8, 8]
   (matching EMLGate's clamp) with 4-bit linear interpolation. The domain
   width 16 = 2^4 makes index extraction a pure bit-slice: for Q m.f input,
   u = raw + 8*2^f spans [0, 2^(f+4)]; index = u >> (f-4), fraction = next 4 bits.

2. ln(y): range reduction instead of the old log-space-indexed LUT (which is
   not synthesizable -- it needs a log to index a log table). We use
   ln(m * 2^k) = ln(m) + k*ln(2), m in [1, 2): a leading-one detector gives k,
   the top 8 mantissa bits index a 257-entry ln LUT, 4 more bits interpolate,
   and k*ln2 is one small signed multiply. Accurate near zero and far above 1,
   which the ln d=4 chain needs (its intermediate exp output reaches ~e^e/x).

All arithmetic below is integer-only and mirrors the emitted Verilog
operation-for-operation, so the Python error report is the hardware error
report (no HDL simulation required for numeric validation).
"""

import math
from typing import List


LUT_BITS = 8          # 2^8 = 256 intervals, 257 stored entries (inclusive endpoint)
INTERP_BITS = 4       # linear interpolation fraction bits
EXP_XMIN, EXP_XMAX = -8.0, 8.0   # matches EMLGate clamp in eml_layer_v2.py
LN2 = math.log(2.0)


class FixedPointFormat:
    """Signed two's-complement Q(int_bits).(frac_bits); int_bits includes sign."""

    def __init__(self, int_bits: int, frac_bits: int):
        assert frac_bits >= LUT_BITS + INTERP_BITS - 8  # need >=4 frac bits for exp slicing
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits
        self.scale = 1 << frac_bits
        self.raw_min = -(1 << (self.total_bits - 1))
        self.raw_max = (1 << (self.total_bits - 1)) - 1
        self.min_val = self.raw_min / self.scale
        self.max_val = self.raw_max / self.scale

    def quantize(self, value: float) -> int:
        raw = int(round(value * self.scale))
        return max(self.raw_min, min(self.raw_max, raw))

    def dequantize(self, raw: int) -> float:
        return raw / self.scale

    def saturate(self, raw: int) -> int:
        return max(self.raw_min, min(self.raw_max, raw))

    def __repr__(self):
        return (f"Q{self.int_bits}.{self.frac_bits} "
                f"({self.total_bits}b, range [{self.min_val:.4f}, {self.max_val:.4f}], "
                f"step {1.0/self.scale:.6f})")


def build_exp_lut(fmt: FixedPointFormat) -> List[int]:
    """257 entries: exp(x) at x = -8 + i*16/256, quantized+saturated to fmt."""
    n = (1 << LUT_BITS) + 1
    return [fmt.quantize(math.exp(EXP_XMIN + i * (EXP_XMAX - EXP_XMIN) / (1 << LUT_BITS)))
            for i in range(n)]


def build_ln_mant_lut(fmt: FixedPointFormat) -> List[int]:
    """257 entries: ln(m) at m = 1 + i/256, m in [1, 2]."""
    n = (1 << LUT_BITS) + 1
    return [fmt.quantize(math.log(1.0 + i / (1 << LUT_BITS))) for i in range(n)]


class FixedEML:
    """Integer-only EML gate eval: eml(x, y) = exp(x) - ln(y), one Q-format throughout."""

    def __init__(self, fmt: FixedPointFormat):
        self.fmt = fmt
        self.exp_lut = build_exp_lut(fmt)
        self.ln_lut = build_ln_mant_lut(fmt)
        self.ln2_fix = fmt.quantize(LN2)
        self.one = fmt.quantize(1.0)
        # use every input bit below the LUT index for interpolation, capped at 8
        self.interp_bits = min(8, fmt.frac_bits - 4)

    def exp_fix(self, raw: int) -> int:
        f = self.fmt.frac_bits
        ib = self.interp_bits
        lo = -8 << f
        hi = 8 << f
        r = max(lo, min(hi, raw))
        u = r - lo                              # [0, 2^(f+4)]
        idx = u >> (f - 4)                      # top LUT_BITS bits (0..256)
        if idx >= (1 << LUT_BITS):
            return self.exp_lut[1 << LUT_BITS]
        fr = (u >> (f - 4 - ib)) & ((1 << ib) - 1)
        y0 = self.exp_lut[idx]
        y1 = self.exp_lut[idx + 1]
        return self.fmt.saturate(y0 + (((y1 - y0) * fr) >> ib))

    def ln_fix(self, raw: int) -> int:
        f = self.fmt.frac_bits
        ib = self.interp_bits
        if raw <= 0:
            raw = 1                              # clamp to smallest positive (ln of <=0 undefined)
        p = raw.bit_length() - 1                 # leading-one position
        k = p - f                                # value = m * 2^k, m in [1,2)
        # top LUT_BITS+ib bits of the mantissa fraction (bits below the leading one)
        nbits = LUT_BITS + ib
        if p >= nbits:
            mant = (raw >> (p - nbits)) & ((1 << nbits) - 1)
        else:
            mant = (raw << (nbits - p)) & ((1 << nbits) - 1)
        idx = mant >> ib
        fr = mant & ((1 << ib) - 1)
        y0 = self.ln_lut[idx]
        y1 = self.ln_lut[idx + 1]
        ln_m = y0 + (((y1 - y0) * fr) >> ib)
        return self.fmt.saturate(ln_m + k * self.ln2_fix)

    def eml_fix(self, a: int, b: int) -> int:
        return self.fmt.saturate(self.exp_fix(a) - self.ln_fix(b))


if __name__ == '__main__':
    for fmt in (FixedPointFormat(8, 8), FixedPointFormat(10, 8)):
        fe = FixedEML(fmt)
        print(fmt)
        worst_e = worst_l = 0.0
        for i in range(2001):
            x = -2.0 + i * 4.0 / 2000
            e = abs(fe.fmt.dequantize(fe.exp_fix(fmt.quantize(x))) - math.exp(x))
            worst_e = max(worst_e, e)
        for i in range(2001):
            x = 0.1 + i * (10.0 - 0.1) / 2000
            l = abs(fe.fmt.dequantize(fe.ln_fix(fmt.quantize(x))) - math.log(x))
            worst_l = max(worst_l, l)
        print(f"  max |exp err| on [-2,2]: {worst_e:.6f}   max |ln err| on [0.1,10]: {worst_l:.6f}")
