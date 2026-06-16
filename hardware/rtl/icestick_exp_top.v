// iCEstick (iCE40-HX1K) board top for the exp_d2 EML core.
//
// Bridges the FT2232H USB-serial port (channel B) to the byte-serial streaming
// core `exp_d2_pipe_stream`. Host protocol (matches the stream wrapper):
//   * send 2 bytes little-endian per x sample (Q8.8 code)
//   * receive 2 bytes little-endian per result (Q8.8 code, sign-extended)
//
// The core emits its 2 output bytes on consecutive 12 MHz cycles -- far faster
// than the UART can transmit -- so they are captured into a small FIFO and fed
// to uart_tx one byte at a time. The center green LED toggles on each result.
//
// Pin/clock: 12 MHz oscillator, 115200 baud => CLKS_PER_BIT = 104.
module icestick_exp_top #(parameter CPB = 104) (  // 12_000_000 / 115200
    input  wire clk,
    input  wire rx,
    output wire tx,
    output wire led
);

    // ---- host -> core: UART byte directly drives the deserializer ----
    wire [7:0] rx_byte;
    wire       rx_valid;
    uart_rx #(.CLKS_PER_BIT(CPB)) u_rx (
        .clk(clk), .rx(rx), .data(rx_byte), .valid(rx_valid)
    );

    wire [7:0] core_out;
    wire       core_out_valid;
    exp_d2_pipe_stream u_core (
        .clk(clk),
        .in_data(rx_byte), .in_valid(rx_valid),
        .out_data(core_out), .out_valid(core_out_valid)
    );

    // ---- core -> host: depth-4 byte FIFO absorbs the back-to-back output ----
    reg [7:0] fifo [0:3];
    reg [1:0] wr_ptr = 2'd0;
    reg [1:0] rd_ptr = 2'd0;
    always @(posedge clk)
        if (core_out_valid) begin fifo[wr_ptr] <= core_out; wr_ptr <= wr_ptr + 1'b1; end

    wire       tx_busy;
    reg        tx_start = 1'b0;
    reg  [7:0] tx_data  = 8'd0;
    always @(posedge clk) begin
        tx_start <= 1'b0;
        if (!tx_busy && !tx_start && (rd_ptr != wr_ptr)) begin
            tx_data  <= fifo[rd_ptr];
            tx_start <= 1'b1;
            rd_ptr   <= rd_ptr + 1'b1;
        end
    end
    uart_tx #(.CLKS_PER_BIT(CPB)) u_tx (
        .clk(clk), .start(tx_start), .data(tx_data), .tx(tx), .busy(tx_busy)
    );

    // ---- activity LED ----
    reg led_r = 1'b0;
    always @(posedge clk) if (core_out_valid) led_r <= ~led_r;
    assign led = led_r;
endmodule
