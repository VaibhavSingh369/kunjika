// Phase B / Step 1 - clean audio capture in ESP-IDF
//
// Records REC_SECONDS of audio into a buffer, prints diagnostics, then dumps
// the samples over UART0 as a framed binary packet with a checksum.
//
// Capture and transport are deliberately separated: the recording finishes
// before a single byte is sent, so a slow or noisy serial link can never
// corrupt the audio. If the checksum passes and the audio is still bad, the
// problem is on the mic side. If the checksum fails, it is the link.
//
// Binary data goes out through uart_write_bytes(), NOT printf/fwrite. The
// stdout path in ESP-IDF converts '\n' to "\r\n" by default, which inserts
// extra bytes into PCM data and destroys sample alignment.

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#define SAMPLE_RATE         16000
#define REC_SECONDS         5
#define REC_SAMPLES         (SAMPLE_RATE * REC_SECONDS)
#define DMA_FRAMES          256
#define STARTUP_DISCARD_MS  300   // INMP441 settles ~250 ms after clocks start
#define GAIN_SHIFT          16    // measured with suggested_shift, then frozen

#define PIN_BCLK  GPIO_NUM_4
#define PIN_WS    GPIO_NUM_5
#define PIN_DIN   GPIO_NUM_6

static i2s_chan_handle_t rx_chan;
static int32_t *dma_buf;

typedef struct {
    int32_t peak_raw;
    double  dc;
    double  rms_dbfs;
    int     clipped;
} rec_stats_t;

static void i2s_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num  = 6;
    chan_cfg.dma_frame_num = DMA_FRAMES;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_BCLK,
            .ws   = PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = PIN_DIN,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;   // L/R pin tied to GND

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
}

static size_t read_block(void)
{
    size_t bytes = 0;
    ESP_ERROR_CHECK(i2s_channel_read(rx_chan, dma_buf,
                                     DMA_FRAMES * sizeof(int32_t),
                                     &bytes, portMAX_DELAY));
    return bytes / sizeof(int32_t);
}

// Throws away stale DMA data (left over from the previous dump) and the
// mic's startup transient.
static void discard_ms(int ms)
{
    int remaining = SAMPLE_RATE * ms / 1000;
    while (remaining > 0) remaining -= (int)read_block();
}

static void record(int16_t *pcm, rec_stats_t *st)
{
    int64_t sum = 0;
    int32_t peak = 0;
    int clipped = 0;
    size_t got = 0;

    discard_ms(STARTUP_DISCARD_MS);

    while (got < REC_SAMPLES) {
        size_t n = read_block();
        for (size_t i = 0; i < n && got < REC_SAMPLES; i++) {
            int32_t raw = dma_buf[i];   // 24-bit sample, left-justified in 32
            int32_t a = (raw == INT32_MIN) ? INT32_MAX : (raw < 0 ? -raw : raw);
            if (a > peak) peak = a;

            int32_t v = raw >> GAIN_SHIFT;
            if (v >  32767) { v =  32767; clipped++; }
            if (v < -32768) { v = -32768; clipped++; }
            pcm[got++] = (int16_t)v;
            sum += v;
        }
    }

    double dc = (double)sum / REC_SAMPLES;
    double acc = 0.0;
    for (size_t i = 0; i < REC_SAMPLES; i++) {
        double d = (double)pcm[i] - dc;
        acc += d * d;
    }

    st->peak_raw = peak;
    st->dc       = dc;
    st->rms_dbfs = 20.0 * log10(sqrt(acc / REC_SAMPLES) / 32768.0 + 1e-12);
    st->clipped  = clipped;
}

// Smallest shift that puts the loudest sample at or below half scale,
// leaving ~6 dB of headroom for louder speakers.
static int suggest_shift(int32_t peak_raw)
{
    for (int s = 8; s <= 24; s++) {
        if ((peak_raw >> s) <= 16384) return s;
    }
    return 24;
}

static void dump(const int16_t *pcm)
{
    const uint8_t *b = (const uint8_t *)pcm;
    uint32_t len = REC_SAMPLES * sizeof(int16_t);
    uint32_t csum = 0;
    for (uint32_t i = 0; i < len; i++) csum += b[i];

    esp_log_level_set("*", ESP_LOG_NONE);        // nothing may interleave
    fflush(stdout);
    uart_wait_tx_done(UART_NUM_0, pdMS_TO_TICKS(500));

    uart_write_bytes(UART_NUM_0, "KWSWAV01", 8);
    uart_write_bytes(UART_NUM_0, &len, sizeof(len));
    uart_write_bytes(UART_NUM_0, b, len);
    uart_write_bytes(UART_NUM_0, &csum, sizeof(csum));
    uart_wait_tx_done(UART_NUM_0, portMAX_DELAY);

    esp_log_level_set("*", ESP_LOG_INFO);
}

void app_main(void)
{
    // tx_buffer_size = 0 -> uart_write_bytes blocks until bytes are in the FIFO
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0));

    dma_buf = heap_caps_malloc(DMA_FRAMES * sizeof(int32_t), MALLOC_CAP_INTERNAL);

    // Debug-only capture buffer, 160 KB. Lives in PSRAM so it does not touch
    // the 256 KB internal budget. The real pipeline never holds 5 s of audio.
    int16_t *pcm = heap_caps_malloc(REC_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);

    if (!dma_buf || !pcm) {
        printf("ALLOC FAILED - is PSRAM enabled? (CONFIG_SPIRAM=y, octal mode)\n");
        return;
    }

    i2s_init();
    printf("\nStep 1 ready. GAIN_SHIFT=%d, %d Hz, %d s per take\n",
           GAIN_SHIFT, SAMPLE_RATE, REC_SECONDS);

    for (int take = 0; ; take++) {
        printf("\n--- take %d ---\n", take);
        for (int c = 3; c > 0; c--) {
            printf("%d...\n", c);
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
        printf("GO - recording %d s\n", REC_SECONDS);

        rec_stats_t st;
        record(pcm, &st);

        printf("STATS take=%d peak_raw=%" PRId32 " suggested_shift=%d "
               "dc=%.1f rms=%.1f dBFS clipped=%d\n",
               take, st.peak_raw, suggest_shift(st.peak_raw),
               st.dc, st.rms_dbfs, st.clipped);

        if (st.peak_raw < (1 << 20))
            printf("WARN: very low signal. If you spoke, check L/R pin and slot mask\n");
        if (st.clipped > 0)
            printf("WARN: %d clipped samples. Increase GAIN_SHIFT\n", st.clipped);

        dump(pcm);
        printf("dump complete\n");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
