// iCEstick UART bring-up diagnostic #2: host -> FPGA -> host loopback.
//
// Wires uart_rx straight back into uart_tx -- whatever the host sends on rx is
// echoed on tx. If the host sees its own bytes returned, both UART directions,
// both pins, and the baud are correct (so any remaining failure is inside
// icestick_exp_top's core/FIFO, not the link). A received byte while tx is busy
// is dropped (host should send one byte at a time for this test). LED toggles
// on each received byte.
module icestick_echo #(parameter CPB = 104) (  // 12_000_000 / 115200
    input  wire clk,
    input  wire rx,
    output wire tx,
    output wire led
);
    wire [7:0] rx_byte;
    wire       rx_valid;
    uart_rx #(.CLKS_PER_BIT(CPB)) u_rx (
        .clk(clk), .rx(rx), .data(rx_byte), .valid(rx_valid)
    );

    wire tx_busy;
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (
        .clk(clk), .start(rx_valid & ~tx_busy), .data(rx_byte), .tx(tx), .busy(tx_busy)
    );

    reg led_r = 1'b0;
    always @(posedge clk) if (rx_valid) led_r <= ~led_r;
    assign led = led_r;
endmodule
