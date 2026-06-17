// Streaming testbench for mlp_ln_h4_d1_stream: byte-feeds the 256-point sweep,
// reassembles output bytes, prints x_raw,y_raw CSV (same format as other tbs).
`timescale 1ns/1ps
module mlp_ln_h4_d1_stream_tb;
    localparam W = 22, NIN = 3;
    reg clk = 0;
    always #5 clk = ~clk;
    reg [7:0] in_data = 0;
    reg in_valid = 0;
    wire [7:0] out_data;
    wire out_valid;
    mlp_ln_h4_d1_stream dut (.clk(clk), .in_data(in_data), .in_valid(in_valid),
                .out_data(out_data), .out_valid(out_valid));

    reg signed [W-1:0] xq [0:255];
    reg [24-1:0] acc = 0;
    integer acc_n = 0;
    integer recv = 0;
    reg signed [W-1:0] yv;
    always @(posedge clk) begin
        #1;
        if (out_valid) begin
            acc = {out_data, acc[24-1:8]};
            acc_n = acc_n + 1;
            if (acc_n == NIN) begin
                yv = acc[W-1:0];
                $display("%0d,%0d", xq[recv], yv);
                recv = recv + 1;
                acc_n = 0;
                if (recv == 256) $finish;
            end
        end
    end

    integer i, b;
    reg signed [W-1:0] xv;
    reg [24-1:0] xext;
    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            xv = $signed(i - 128) <<< 6;
            xq[i] = xv;
            xext = {{(24-W){xv[W-1]}}, xv};
            for (b = 0; b < NIN; b = b + 1) begin
                @(negedge clk);
                in_data = xext[8*b +: 8];
                in_valid = 1;
            end
        end
        @(negedge clk);
        in_valid = 0;
        repeat (200) @(posedge clk);
        $display("TIMEOUT: only %0d results", recv);
        $finish;
    end
endmodule
