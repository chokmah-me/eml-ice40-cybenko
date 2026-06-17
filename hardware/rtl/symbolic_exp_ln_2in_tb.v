// Diagonal sweep tb for symbolic_exp_ln_2in: prints x_raw,y_raw,out_raw CSV.
`timescale 1ns/1ps
module symbolic_exp_ln_2in_tb;
    reg  signed [15:0] x, y;
    wire signed [15:0] o;
    symbolic_exp_ln_2in dut (.x_in(x), .y_in(y), .y_out(o));
    integer i;
    initial begin
        for (i = 0; i < 256; i = i + 1) begin
            x = $signed(i - 128) <<< (8 - 6);          // ~[-2, 2)
            y = $signed((i >> 1) + 7) <<< (8 - 6);     // ~[0.1, 10), strictly > 0
            #1 $display("%0d,%0d,%0d", x, y, o);
        end
        $finish;
    end
endmodule
