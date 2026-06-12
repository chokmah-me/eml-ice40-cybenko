// Streaming testbench for exp_d2_pipe: prints x_raw,y_raw CSV (delay-matched).
`timescale 1ns/1ps
module exp_d2_pipe_tb;
    localparam LAT = 3;
    reg clk = 0;
    always #5 clk = ~clk;
    reg signed [15:0] x = 0;
    wire signed [15:0] y;
    exp_d2_pipe dut (.clk(clk), .x_in(x), .y_out(y));

    reg signed [15:0] xd [0:LAT-1];
    integer j;
    always @(posedge clk) begin
        xd[0] <= x;
        for (j = LAT-1; j > 0; j = j - 1) xd[j] <= xd[j-1];
    end

    integer i;
    initial begin
        for (i = 0; i < 256 + LAT; i = i + 1) begin
            x = $signed((i < 256 ? i : 255) - 128) <<< (8 - 6);
            @(posedge clk); #1;
            if (i >= LAT - 1 && i <= 255 + LAT - 1)
                $display("%0d,%0d", xd[LAT-1], y);
        end
        $finish;
    end
endmodule
