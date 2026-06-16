// iCEstick UART bring-up diagnostic #1: FPGA -> host transmit + baud check.
//
// Continuously transmits the byte 0x55 ('U' = 0b01010101, the classic
// alternating-bit baud test) a few times per second via uart_tx. No `rx`
// dependence. If the host reads a clean stream of 'U', the tx pin, baud, and
// 12 MHz clock are all correct. Garbage => baud/clock wrong. Nothing => tx pin
// wrong or design not running. The LED blinks at the send rate.
module icestick_tx_heartbeat #(parameter CPB = 104) (  // 12_000_000 / 115200
    input  wire clk,
    output wire tx,
    output wire led
);
    // Free-running counter: send a byte every 2^16 cycles (~5.5 ms => ~180 B/s,
    // so the host fills a read buffer in well under a second), while the LED
    // blinks from a slow bit (~1.4 s period) so it is visibly toggling.
    reg [23:0] cnt = 24'd0;
    reg        send = 1'b0;
    always @(posedge clk) begin
        cnt  <= cnt + 1'b1;
        send <= (cnt[15:0] == 16'd0);   // one-cycle pulse every 65536 cycles
    end

    wire tx_busy;
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (
        .clk(clk), .start(send & ~tx_busy), .data(8'h55), .tx(tx), .busy(tx_busy)
    );

    assign led = cnt[23];
endmodule
