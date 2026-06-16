// 8N1 UART receiver. Default CLKS_PER_BIT = 12_000_000 / 115200 = 104
// (iCEstick 12 MHz oscillator, 115200 baud). Registers self-init to a safe
// idle state via iCE40 configuration -- no external reset needed.
module uart_rx #(parameter CLKS_PER_BIT = 104) (
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data  = 8'd0,
    output reg        valid = 1'b0
);
    localparam IDLE = 3'd0, START = 3'd1, DATA = 3'd2, STOP = 3'd3;
    reg [2:0]  state   = IDLE;
    reg [12:0] clk_cnt = 13'd0;
    reg [2:0]  bit_idx = 3'd0;

    // double-flop synchronizer for the asynchronous rx line
    reg rx_d0 = 1'b1, rx_d1 = 1'b1;
    always @(posedge clk) begin rx_d0 <= rx; rx_d1 <= rx_d0; end

    always @(posedge clk) begin
        valid <= 1'b0;
        case (state)
            IDLE: begin
                clk_cnt <= 13'd0;
                bit_idx <= 3'd0;
                if (rx_d1 == 1'b0) state <= START;  // start bit edge
            end
            START: begin
                // sample at mid-bit to confirm a real start bit
                if (clk_cnt == (CLKS_PER_BIT - 1) / 2) begin
                    if (rx_d1 == 1'b0) begin clk_cnt <= 13'd0; state <= DATA; end
                    else               state <= IDLE;  // false start
                end else clk_cnt <= clk_cnt + 1'b1;
            end
            DATA: begin
                if (clk_cnt < CLKS_PER_BIT - 1) clk_cnt <= clk_cnt + 1'b1;
                else begin
                    clk_cnt        <= 13'd0;
                    data[bit_idx]  <= rx_d1;          // LSB first
                    if (bit_idx < 3'd7) bit_idx <= bit_idx + 1'b1;
                    else begin bit_idx <= 3'd0; state <= STOP; end
                end
            end
            STOP: begin
                if (clk_cnt < CLKS_PER_BIT - 1) clk_cnt <= clk_cnt + 1'b1;
                else begin valid <= 1'b1; clk_cnt <= 13'd0; state <= IDLE; end
            end
            default: state <= IDLE;
        endcase
    end
endmodule
