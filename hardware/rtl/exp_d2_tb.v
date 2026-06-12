// Sweep testbench for exp_d2: prints x_raw,y_raw CSV to stdout.
`timescale 1ns/1ps
module exp_d2_tb;
    reg  signed [15:0] x;
    wire signed [15:0] y;
    exp_d2 dut (.x_in(x), .y_out(y));
    integer i;
    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            x = $signed(i - 128) <<< (8 - 6);   // sweep ~[-2, 2)
            #1 $display("%0d,%0d", x, y);
        end
        $finish;
    end
endmodule
