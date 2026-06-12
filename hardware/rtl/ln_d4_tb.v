// Sweep testbench for ln_d4: prints x_raw,y_raw CSV to stdout.
`timescale 1ns/1ps
module ln_d4_tb;
    reg  signed [21:0] x;
    wire signed [21:0] y;
    ln_d4 dut (.x_in(x), .y_out(y));
    integer i;
    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            x = $signed(i - 128) <<< (12 - 6);   // sweep ~[-2, 2)
            #1 $display("%0d,%0d", x, y);
        end
        $finish;
    end
endmodule
