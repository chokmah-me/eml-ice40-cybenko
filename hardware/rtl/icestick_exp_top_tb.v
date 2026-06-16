// Bench for the iCEstick UART bridge: drives `rx` with UART-framed Q8.8 input
// codes, decodes the `tx` line, and prints "x_raw,y_raw" per sample (same CSV
// format hardware/sim_check.py consumes). Uses a tiny CLKS_PER_BIT so the full
// 256-point sweep simulates quickly; the math is identical to CPB=104.
//
// The tx side is decoded by a free-running background process into a byte FIFO
// (rxq) so the stimulus loop never races the inter-byte gap -- exactly how a
// real host reading a continuous serial stream behaves.
`timescale 1ns/1ps
module icestick_exp_top_tb;
    localparam CPB = 8;          // small bit period for fast sim
    localparam integer NSAMP = 256;
    localparam F = 8;            // Q8.8

    reg clk = 0;
    always #5 clk = ~clk;

    reg  rx = 1'b1;
    wire tx, led;

    icestick_exp_top #(.CPB(CPB)) dut (.clk(clk), .rx(rx), .tx(tx), .led(led));

    // --- UART byte transmit onto rx (8N1, LSB first) ---
    task uart_send(input [7:0] b);
        integer i;
        begin
            rx = 1'b0;
            repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rx = b[i];
                repeat (CPB) @(posedge clk);
            end
            rx = 1'b1;
            repeat (CPB) @(posedge clk);
        end
    endtask

    // --- background tx decoder: pushes each received byte into rxq ---
    reg [7:0] rxq [0:1023];
    integer   rxq_wr = 0;
    integer   rxq_rd = 0;
    reg [7:0] rxbits;
    integer   bi;
    initial begin
        forever begin
            @(negedge tx);                          // start bit
            repeat (CPB + CPB/2) @(posedge clk);    // to middle of bit0
            for (bi = 0; bi < 8; bi = bi + 1) begin
                rxbits[bi] = tx;
                repeat (CPB) @(posedge clk);
            end
            // sample stop bit region, then push
            rxq[rxq_wr] = rxbits;
            rxq_wr = rxq_wr + 1;
            repeat (CPB / 2) @(posedge clk);        // settle into stop bit
        end
    end

    task get_byte(output [7:0] b);
        begin
            while (rxq_rd == rxq_wr) @(posedge clk);  // wait for a byte
            b = rxq[rxq_rd];
            rxq_rd = rxq_rd + 1;
        end
    endtask

    integer s;
    reg signed [15:0] x_raw;
    reg [7:0] lo, hi;
    initial begin
        repeat (10) @(posedge clk);
        for (s = 0; s < NSAMP; s = s + 1) begin
            x_raw = (s - 128) <<< (F - 6);   // same sweep as sim_check / *_tb.v
            uart_send(x_raw[7:0]);           // little-endian, 2 bytes
            uart_send(x_raw[15:8]);
            get_byte(lo);
            get_byte(hi);
            $display("%0d,%0d", x_raw, $signed({hi, lo}));
        end
        $finish;
    end

    initial begin #50000000 $display("TIMEOUT"); $finish; end
endmodule
