"""Quantized-MLP baseline for the symbolic-vs-MLP iCE40 study (Track C note).

Mirrors the hardware/ pipeline: a tiny scalar MLP is trained to approximate the
same scalar functions the snapped EML units compute (exp on [-2,2], ln on
[0.1,10]), quantized to the same Q-formats, emitted to iCE40 RTL, and verified
bit-exact against an integer golden model -- so both function units are measured
by one identical sim/synth/flash flow.
"""
