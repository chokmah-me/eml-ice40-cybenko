// 8N1 UART transmitter. Default CLKS_PER_BIT = 12_000_000 / 115200 = 104.
// Assert `start` for one cycle with `data` valid; `busy` is high until the
// stop bit completes. Registers self-init via iCE40 configuration.
module uart_tx #(parameter CLKS_PER_BIT = 104) (
    input  wire       clk,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx   = 1'b1,
    output reg        busy = 1'b0
);
    localparam IDLE = 2'd0, START = 2'd1, DATA = 2'd2, STOP = 2'd3;
    reg [1:0]  state   = IDLE;
    reg [12:0] clk_cnt = 13'd0;
    reg [2:0]  bit_idx = 3'd0;
    reg [7:0]  data_r  = 8'd0;

    always @(posedge clk) begin
        case (state)
            IDLE: begin
                tx      <= 1'b1;
                clk_cnt <= 13'd0;
                bit_idx <= 3'd0;
                busy    <= 1'b0;
                if (start) begin data_r <= data; busy <= 1'b1; state <= START; end
            end
            START: begin
                tx <= 1'b0;  // start bit
                if (clk_cnt < CLKS_PER_BIT - 1) clk_cnt <= clk_cnt + 1'b1;
                else begin clk_cnt <= 13'd0; state <= DATA; end
            end
            DATA: begin
                tx <= data_r[bit_idx];  // LSB first
                if (clk_cnt < CLKS_PER_BIT - 1) clk_cnt <= clk_cnt + 1'b1;
                else begin
                    clk_cnt <= 13'd0;
                    if (bit_idx < 3'd7) bit_idx <= bit_idx + 1'b1;
                    else begin bit_idx <= 3'd0; state <= STOP; end
                end
            end
            STOP: begin
                tx <= 1'b1;  // stop bit
                if (clk_cnt < CLKS_PER_BIT - 1) clk_cnt <= clk_cnt + 1'b1;
                else begin busy <= 1'b0; state <= IDLE; end
            end
            default: state <= IDLE;
        endcase
    end
endmodule
