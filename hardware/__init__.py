"""Track C: hardware realization of snapped EML trees (iCE40 target).

Bit-exact fixed-point model + netlist extraction from v2 EMLTree + Verilog emission.
See HARDWARE.md in this directory.
"""

from .fixed_point import FixedPointFormat, FixedEML
from .converter import extract_netlist, eval_netlist_float, eval_netlist_fixed
