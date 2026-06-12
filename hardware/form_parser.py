"""
Build a hardware Netlist directly from a symbolic_form string.

This makes the Track C flow data-driven: any valid row from the released CSVs
(`symbolic_form` column) can be turned into RTL without reconstructing or
retraining an EMLTree. The grammar is the one emitted by
EMLTree.symbolic_form():

    expr := 'eml(' expr ',' expr ')' | '1' | 'x'

Note that symbolic_form() already substitutes 'f' references and drops
dangling capacity, so parsed netlists contain only the active expression;
constant subexpressions are still folded (e.g. the inner eml(1,1) of a
constant branch, or ln(1) arguments handled at Verilog-gen time).
"""

from .converter import Netlist, Gate, fold_constants


def parse_form(form: str) -> Netlist:
    net = Netlist()
    pos = 0

    def parse(depth: int):
        """Returns ('const1'|'x'|'gate', idx). Gates appended children-first."""
        nonlocal pos
        if form.startswith('eml(', pos):
            pos += 4
            left = parse(depth + 1)
            if pos >= len(form) or form[pos] != ',':
                raise ValueError(f"expected ',' at {pos} in {form!r}")
            pos += 1
            right = parse(depth + 1)
            if pos >= len(form) or form[pos] != ')':
                raise ValueError(f"expected ')' at {pos} in {form!r}")
            pos += 1
            g = Gate(idx=len(net.gates), level=-1, pos=-1,
                     left=left, right=right, on_path=True)
            net.gates.append(g)
            return ('gate', g.idx)
        if form.startswith('1', pos):
            pos += 1
            return ('const1', -1)
        if form.startswith('x', pos):
            pos += 1
            return ('x', -1)
        raise ValueError(f"parse error at {pos} in {form!r}")

    kind, idx = parse(0)
    if pos != len(form):
        raise ValueError(f"trailing input at {pos} in {form!r}")
    if kind != 'gate':
        raise ValueError(f"form is not a gate expression: {form!r}")
    net.out_idx = idx
    fold_constants(net)
    return net


if __name__ == '__main__':
    for f in ('eml(x,1)', 'eml(1,eml(eml(1,x),1))'):
        net = parse_form(f)
        from .converter import eval_netlist_float
        import math
        print(f, '->', len(net.gates), 'gates,',
              len(net.active_gates()), 'active;',
              'f(1.5) =', eval_netlist_float(net, 1.5))
