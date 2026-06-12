"""
Pipelined Verilog emission (Track C stage 4): EBR-friendly, clocked, streaming.

Differences vs the combinational generator (verilog_gen.py):

- ROM reads are registered (`q <= lut[addr]` on posedge), which is the pattern
  yosys needs to infer iCE40 EBR (block RAM only supports synchronous reads --
  the combinational version melted all ROMs into fabric LUTs).
- Interpolation needs lut[idx] and lut[idx+1] in the same cycle, but EBR has a
  single read port, so each LUT is duplicated into two banks: lutA holds
  entries 0..255, lutB holds entries 1..256 (so y1 = lutB[idx], no +1 adder).
  The 257th endpoint lives in lutB[255] / the exp top-clamp constant.
- Each eml_gate is a 3-stage pipeline (ROM read -> interpolate + k*ln2 ->
  subtract/saturate), fully streaming: one new sample per clock.
- Gates are chained with delay matching: a gate's inputs must arrive together,
  so earlier-finishing child outputs get balancing registers and direct 'x'
  inputs at deeper levels tap an x delay line.

The per-stage arithmetic is operation-identical to hardware/fixed_point.py,
so hardware/sim_check.py verifies the pipelined RTL bit-for-bit against the
same Python model (pipelining changes timing, not values).
"""

import os
from .fixed_point import FixedEML, LUT_BITS
from .converter import Netlist
from .verilog_gen import _hex_lines

GATE_LATENCY = 3  # ROM-read reg + interp reg + gate output reg


def gate_times(net: Netlist):
    """Per active gate: (t_in, t_out) in cycles. Gates are child-before-parent."""
    times = {}
    for g in net.gates:
        if not g.on_path or g.const_val is not None:
            continue
        t_in = 0
        for kind, c in (g.left, g.right):
            if kind == 'gate' and net.gates[c].const_val is None:
                t_in = max(t_in, times[c][1])
        times[g.idx] = (t_in, t_in + GATE_LATENCY)
    return times


def emit_verilog_pipelined(net: Netlist, fe: FixedEML, name: str, out_dir: str) -> dict:
    fmt = fe.fmt
    W, F, IB = fmt.total_bits, fmt.frac_bits, fe.interp_bits
    os.makedirs(out_dir, exist_ok=True)

    root = net.gates[net.out_idx]
    assert root.const_val is None, "constant netlist needs no pipeline"

    # Split each 257-entry LUT into the two banks
    hexes = {}
    for tag, lut in (("exp", fe.exp_lut), ("ln", fe.ln_lut)):
        for bank, sl in (("a", lut[0:256]), ("b", lut[1:257])):
            fn = f"{name}_{tag}_lut_{bank}.hex"
            with open(os.path.join(out_dir, fn), "w") as f:
                f.write(_hex_lines(sl, W))
            hexes[f"{tag}{bank}"] = fn

    times = gate_times(net)
    latency = times[net.out_idx][1]
    active = net.active_gates()

    # x delay line: deepest t_in among gates with a direct 'x' input
    max_x_delay = 0
    for g in active:
        for kind, _ in (g.left, g.right):
            if kind == 'x':
                max_x_delay = max(max_x_delay, times[g.idx][0])

    x_pipe = ""
    if max_x_delay > 0:
        taps = "\n".join(f"            x_d[{i}] <= x_d[{i-1}];" for i in range(1, max_x_delay))
        x_pipe = f"""    reg signed [{W-1}:0] x_d [0:{max_x_delay-1}];
    integer xi;
    always @(posedge clk) begin
        x_d[0] <= x_in;
{taps if max_x_delay > 1 else ''}
    end
"""

    def x_tap(delay):
        return "x_in" if delay == 0 else f"x_d[{delay-1}]"

    # Per-gate input expressions, with balancing registers where child outputs
    # arrive earlier than the gate's t_in.
    const_decls = []
    balance_decls = []
    referenced_consts = set()

    def src_expr(g, kind_child):
        kind, child = kind_child
        t_in = times[g.idx][0]
        if kind == 'const1':
            return f"{W}'sd{fe.one}"
        if kind == 'x':
            return x_tap(t_in)
        cg = net.gates[child]
        if cg.const_val is not None:
            referenced_consts.add(child)
            return f"const_g{child}"
        lag = t_in - times[child][1]
        if lag == 0:
            return f"g{child}_out"
        regs = "\n".join(
            f"    always @(posedge clk) g{child}_bal{k} <= "
            + (f"g{child}_bal{k-1};" if k > 0 else f"g{child}_out;")
            for k in range(lag))
        decls = "\n".join(f"    reg signed [{W-1}:0] g{child}_bal{k};" for k in range(lag))
        balance_decls.append(decls + "\n" + regs)
        return f"g{child}_bal{lag-1}"

    gate_insts = []
    for g in active:
        l_expr = src_expr(g, g.left)
        r_expr = src_expr(g, g.right)
        gate_insts.append(f"""    wire signed [{W-1}:0] g{g.idx}_out;
    eml_gate_p #(.W({W}), .F({F})) u_g{g.idx} (
        .clk(clk),
        .a({l_expr}),
        .b({r_expr}),
        .y(g{g.idx}_out)
    );  // t_in={times[g.idx][0]} t_out={times[g.idx][1]}""")

    for ci in sorted(referenced_consts):
        raw = fmt.quantize(net.gates[ci].const_val)
        const_decls.append(
            f"    localparam signed [{W-1}:0] const_g{ci} = {W}'sd{raw};"
            f"  // folded gate {ci} = {net.gates[ci].const_val:.6f}")

    ep_exp = fe.exp_lut[256]

    rtl = f"""// Generated by hardware/verilog_gen_pipelined.py -- do not edit by hand.
// Pipelined snapped EML tree '{name}': {len(active)} gate(s), latency {latency} cycles,
// fully streaming (one sample per clock). Format Q{fmt.int_bits}.{F} ({W} bits).
// ROM reads are registered so yosys synth_ice40 infers EBR (sync read required).

// ------------------------------------------------- exp(a), 2-cycle latency ----
module eml_exp_p #(parameter W = {W}, parameter F = {F}) (
    input clk,
    input  signed [W-1:0] a,
    output signed [W-1:0] y
);
    localparam LUTB = {LUT_BITS}, IB = {IB};
    localparam signed [W-1:0] XMIN = -(8 <<< F), XMAX = (8 <<< F);
    localparam signed [W-1:0] EP = {W}'sd{ep_exp};  // exp(8) endpoint (entry 256)

    reg signed [W-1:0] lutA [0:255];  // entries 0..255
    reg signed [W-1:0] lutB [0:255];  // entries 1..256 (y1 = lutB[idx])
    initial $readmemh("{hexes['expa']}", lutA);
    initial $readmemh("{hexes['expb']}", lutB);

    wire signed [W-1:0] r = (a < XMIN) ? XMIN : (a > XMAX) ? XMAX : a;
    wire [W:0] u = r - XMIN;
    wire [LUTB:0] idx = u >> (F - 4);
    wire [IB-1:0] fr = u >> (F - 4 - IB);
    wire top = (idx >= (1 << LUTB));

    // stage 1: synchronous ROM read (EBR)
    reg signed [W-1:0] qA, qB;
    reg [IB-1:0] fr_r;
    reg top_r;
    always @(posedge clk) begin
        qA <= lutA[idx[LUTB-1:0]];
        qB <= lutB[idx[LUTB-1:0]];
        fr_r <= fr;
        top_r <= top;
    end

    // stage 2: interpolate (+ saturate, mirroring fixed_point.exp_fix)
    wire signed [W+IB:0] d = (qB - qA) * $signed({{1'b0, fr_r}});
    wire signed [W+1:0] s = qA + (d >>> IB);
    localparam signed [W+1:0] PMAX = (1 <<< (W-1)) - 1, PMIN = -(1 <<< (W-1));
    reg signed [W-1:0] y_r;
    always @(posedge clk)
        y_r <= top_r ? EP :
               (s > PMAX) ? PMAX[W-1:0] : (s < PMIN) ? PMIN[W-1:0] : s[W-1:0];
    assign y = y_r;
endmodule

// -------------------------------------------------- ln(b), 2-cycle latency ----
module eml_ln_p #(parameter W = {W}, parameter F = {F}) (
    input clk,
    input  signed [W-1:0] b,
    output signed [W-1:0] y
);
    localparam LUTB = {LUT_BITS}, IB = {IB};
    localparam signed [W-1:0] LN2 = {W}'sd{fe.ln2_fix};

    reg signed [W-1:0] lutA [0:255];  // ln(1 + i/256), entries 0..255
    reg signed [W-1:0] lutB [0:255];  // entries 1..256 (entry 256 = ln(2))
    initial $readmemh("{hexes['lna']}", lutA);
    initial $readmemh("{hexes['lnb']}", lutB);

    wire [W-1:0] r = (b <= 0) ? {W}'d1 : b[W-1:0];

    integer i;
    reg [$clog2(W)-1:0] p;
    always @* begin
        p = 0;
        for (i = 0; i < W; i = i + 1)
            if (r[i]) p = i[$clog2(W)-1:0];
    end

    wire [2*W-1:0] rext = r;
    wire [2*W-1:0] shifted = (rext << W) >> p;
    wire [LUTB+IB-1:0] mant = shifted[W-1 -: LUTB+IB];
    wire [LUTB-1:0] idx = mant[LUTB+IB-1:IB];
    wire [IB-1:0] fr = mant[IB-1:0];
    wire signed [$clog2(W):0] k = $signed({{1'b0, p}}) - F;

    // stage 1: synchronous ROM read (EBR)
    reg signed [W-1:0] qA, qB;
    reg [IB-1:0] fr_r;
    reg signed [$clog2(W):0] k_r;
    always @(posedge clk) begin
        qA <= lutA[idx];
        qB <= lutB[idx];
        fr_r <= fr;
        k_r <= k;
    end

    // stage 2: interpolate + k*ln2 (+ saturate, mirroring fixed_point.ln_fix)
    wire signed [W+IB:0] d = (qB - qA) * $signed({{1'b0, fr_r}});
    wire signed [W-1:0] ln_m = qA + (d >>> IB);
    wire signed [W+6:0] s = ln_m + k_r * LN2;
    localparam signed [W+6:0] PMAX = (1 <<< (W-1)) - 1, PMIN = -(1 <<< (W-1));
    reg signed [W-1:0] y_r;
    always @(posedge clk)
        y_r <= (s > PMAX) ? PMAX[W-1:0] : (s < PMIN) ? PMIN[W-1:0] : s[W-1:0];
    assign y = y_r;
endmodule

// --------------------------------------------- eml gate, 3-cycle latency ----
module eml_gate_p #(parameter W = {W}, parameter F = {F}) (
    input clk,
    input  signed [W-1:0] a,
    input  signed [W-1:0] b,
    output signed [W-1:0] y
);
    wire signed [W-1:0] ea, lb;
    eml_exp_p #(.W(W), .F(F)) u_exp (.clk(clk), .a(a), .y(ea));
    eml_ln_p  #(.W(W), .F(F)) u_ln  (.clk(clk), .b(b), .y(lb));
    wire signed [W:0] s = ea - lb;
    localparam signed [W:0] SMAX = (1 <<< (W-1)) - 1, SMIN = -(1 <<< (W-1));
    reg signed [W-1:0] y_r;
    always @(posedge clk)
        y_r <= (s > SMAX) ? SMAX[W-1:0] : (s < SMIN) ? SMIN[W-1:0] : s[W-1:0];
    assign y = y_r;
endmodule

// ------------------------------------------------------------- top level ----
module {name} (
    input clk,
    input  signed [{W-1}:0] x_in,    // Q{fmt.int_bits}.{F}, one sample per clock
    output signed [{W-1}:0] y_out    // valid LATENCY cycles after its x_in
);
    localparam LATENCY = {latency};
{chr(10).join(const_decls) if const_decls else "    // (no folded constants referenced)"}
{x_pipe}
{chr(10).join(balance_decls)}
{chr(10).join(gate_insts)}

    assign y_out = g{net.out_idx}_out;
endmodule
"""

    tb = f"""// Streaming testbench for {name}: prints x_raw,y_raw CSV (delay-matched).
`timescale 1ns/1ps
module {name}_tb;
    localparam LAT = {latency};
    reg clk = 0;
    always #5 clk = ~clk;
    reg signed [{W-1}:0] x = 0;
    wire signed [{W-1}:0] y;
    {name} dut (.clk(clk), .x_in(x), .y_out(y));

    reg signed [{W-1}:0] xd [0:LAT-1];
    integer j;
    always @(posedge clk) begin
        xd[0] <= x;
        for (j = LAT-1; j > 0; j = j - 1) xd[j] <= xd[j-1];
    end

    integer i;
    initial begin
        for (i = 0; i < 256 + LAT; i = i + 1) begin
            x = $signed((i < 256 ? i : 255) - 128) <<< ({F} - 6);
            @(posedge clk); #1;
            if (i >= LAT - 1 && i <= 255 + LAT - 1)
                $display("%0d,%0d", xd[LAT-1], y);
        end
        $finish;
    end
endmodule
"""

    rtl_path = os.path.join(out_dir, f"{name}.v")
    tb_path = os.path.join(out_dir, f"{name}_tb.v")
    with open(rtl_path, "w") as f:
        f.write(rtl)
    with open(tb_path, "w") as f:
        f.write(tb)

    return {"rtl": rtl_path, "tb": tb_path, "latency": latency,
            "active_gates": len(active), "total_gates": len(net.gates)}
