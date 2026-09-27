// Phase B / Step 5e - detection + streaming, final
//
//   INMP441 --I2S--> 20 ms hops --> kws_engine (log-mel, rolling 1 s window,
//   model every 100 ms, smoothing, threshold) --> LED + serial
//
// Two tasks:
//   kws_audio  core 1  reads the mic and runs detection. Nothing else runs on
//                      core 1, so core 1's load IS the keyword spotter's load.
//   kws_stats  core 0  every 5 s prints real CPU use per core (from FreeRTOS
//                      run-time stats, so it includes I2S interrupts and DMA
//                      handling), audio level, scores, dropped audio and RAM.

#include <cmath>
#include <cstdio>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "led_strip.h"

#include "kws_config.h"
#include "kws_engine.h"
#include "net.h"
#include "stream.h"

// ---- hardware ---------------------------------------------------------------
#define PIN_BCLK            GPIO_NUM_4
#define PIN_WS              GPIO_NUM_5
#define PIN_DIN             GPIO_NUM_6
#define PIN_LED             38     // onboard RGB LED on DevKitC-1 v1.1 (v1.0: 48)
#define GAIN_SHIFT          16     // frozen in Step 1; do not change
#define STARTUP_DISCARD_MS  300    // INMP441 settles after clocks start

// ---- behaviour --------------------------------------------------------------
#define STATS_PERIOD_MS     5000
#define LED_ON_MS           600
#define TRACE_MIN_SCORE     0.40f  // print smoothed scores above this; 2.0 = off
// -----------------------------------------------------------------------------

static i2s_chan_handle_t s_rx;
static led_strip_handle_t s_led;
static volatile uint32_t s_overflows;   // audio the DMA had to drop

static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static struct {
    uint32_t hops;
    uint32_t inferences;
    uint32_t detections;
    uint32_t short_reads;
    int64_t  busy_us;        // time spent converting + detecting
    float    max_smoothed;
    int64_t  sum;            // for audio level
    int64_t  sumsq;
    uint32_t n;
} s_stats;

// ---------------------------------------------------------------- LED

static void led_set(uint32_t r, uint32_t g, uint32_t b)
{
    if (!s_led) return;
    led_strip_set_pixel(s_led, 0, r, g, b);
    led_strip_refresh(s_led);
}

static void led_init(void)
{
    led_strip_config_t strip = {};
    strip.strip_gpio_num = PIN_LED;
    strip.max_leds = 1;
    strip.led_model = LED_MODEL_WS2812;
    // WS2812 byte order is G, R, B. Set directly: the header's GRB macro uses
    // C compound-literal syntax that is not reliably valid C++.
    strip.color_component_format.format.g_pos = 0;
    strip.color_component_format.format.r_pos = 1;
    strip.color_component_format.format.b_pos = 2;
    strip.color_component_format.format.w_pos = 3;
    strip.color_component_format.format.bytes_per_color = 1;
    strip.color_component_format.format.num_components = 3;

    led_strip_rmt_config_t rmt = {};
    rmt.clk_src = RMT_CLK_SRC_DEFAULT;
    rmt.resolution_hz = 10 * 1000 * 1000;

    if (led_strip_new_rmt_device(&strip, &rmt, &s_led) != ESP_OK) {
        printf("LED init failed on GPIO %d - continuing without LED\n", PIN_LED);
        s_led = NULL;
        return;
    }
    led_set(0, 30, 0);                      // green blink = LED works
    vTaskDelay(pdMS_TO_TICKS(300));
    led_set(0, 0, 0);
}

// ---------------------------------------------------------------- I2S

static bool IRAM_ATTR on_recv_overflow(i2s_chan_handle_t, i2s_event_data_t *, void *)
{
    s_overflows = s_overflows + 1;
    return false;
}

static void i2s_init(void)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num  = 6;                   // 6 x 20 ms = 120 ms of slack
    cc.dma_frame_num = KWS_HOP_SAMPLES;     // one DMA buffer = one hop
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &s_rx));

    i2s_std_config_t sc = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(16000),
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
    sc.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &sc));

    i2s_event_callbacks_t cbs = {};
    cbs.on_recv_q_ovf = on_recv_overflow;
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(s_rx, &cbs, NULL));

    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

// ---------------------------------------------------------------- tasks

static void audio_task(void *)
{
    static int32_t raw[KWS_HOP_SAMPLES];
    static int16_t pcm[KWS_HOP_SAMPLES];
    kws_result_t r;
    int64_t led_off_at = 0;
    size_t got = 0;

    for (int i = 0; i < STARTUP_DISCARD_MS / 20; i++) {
        i2s_channel_read(s_rx, raw, sizeof(raw), &got, portMAX_DELAY);
    }
    printf("listening...\n");

    for (;;) {
        got = 0;
        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got, portMAX_DELAY) != ESP_OK ||
            got != sizeof(raw)) {
            portENTER_CRITICAL(&s_mux);
            s_stats.short_reads++;
            portEXIT_CRITICAL(&s_mux);
            continue;
        }

        int64_t t0 = esp_timer_get_time();
        int64_t sum = 0, sumsq = 0;
        for (int i = 0; i < KWS_HOP_SAMPLES; i++) {
            int32_t v = raw[i] >> GAIN_SHIFT;
            if (v > 32767)  v = 32767;
            if (v < -32768) v = -32768;
            pcm[i] = (int16_t)v;
            sum += v;
            sumsq += (int64_t)v * v;
        }
        kws_push_hop(pcm, &r);
        stream_push_hop(pcm, t0, kws_last_frame_energy());   // keep the last ~2 s
        if (r.detected) stream_on_detect(t0);                 // start a session
        int64_t t1 = esp_timer_get_time();

        if (r.detected) {
            printf("DETECTED  t=%.2f s  score %.2f\n", t1 / 1e6, r.smoothed);
            led_set(0, 0, 40);
            led_off_at = t1 + (int64_t)LED_ON_MS * 1000;
        } else if (r.inferred && r.smoothed >= TRACE_MIN_SCORE) {
            printf("  score %.2f\n", r.smoothed);
        }
        if (led_off_at && t1 >= led_off_at) {
            led_set(0, 0, 0);
            led_off_at = 0;
        }

        portENTER_CRITICAL(&s_mux);
        s_stats.hops++;
        s_stats.busy_us += t1 - t0;
        if (r.inferred) {
            s_stats.inferences++;
            if (r.smoothed > s_stats.max_smoothed) s_stats.max_smoothed = r.smoothed;
        }
        if (r.detected) s_stats.detections++;
        s_stats.sum += sum;
        s_stats.sumsq += sumsq;
        s_stats.n += KWS_HOP_SAMPLES;
        portEXIT_CRITICAL(&s_mux);
    }
}

static void stats_task(void *)
{
    static TaskStatus_t tasks[40];
    using rt_t = decltype(tasks[0].ulRunTimeCounter);
    rt_t prev_idle[2] = {0, 0}, prev_idle_all = 0, prev_total = 0;
    bool have_prev = false;
    bool stacks_shown = false;
    uint32_t prev_ovf = s_overflows;
    int64_t t_boot = esp_timer_get_time();

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(STATS_PERIOD_MS));

        portENTER_CRITICAL(&s_mux);
        auto st = s_stats;
        memset(&s_stats, 0, sizeof(s_stats));
        portEXIT_CRITICAL(&s_mux);
        uint32_t ovf = s_overflows;

        // CPU per core from the idle tasks' run time
        rt_t total = 0;
        UBaseType_t n = uxTaskGetSystemState(tasks, 40, &total);
        rt_t idle[2] = {0, 0}, idle_all = 0;
        bool per_core = true;
        for (UBaseType_t i = 0; i < n; i++) {
            const char *nm = tasks[i].pcTaskName;
            if (strncmp(nm, "IDLE", 4) != 0) continue;
            idle_all += tasks[i].ulRunTimeCounter;
            char c = nm[strlen(nm) - 1];
            if (c == '0')      idle[0] += tasks[i].ulRunTimeCounter;
            else if (c == '1') idle[1] += tasks[i].ulRunTimeCounter;
            else               per_core = false;
        }

        double secs = (esp_timer_get_time() - t_boot) / 1e6;
        double level = -120.0;
        if (st.n) {
            double mean = (double)st.sum / st.n;
            double var = (double)st.sumsq / st.n - mean * mean;
            if (var > 0) level = 20.0 * log10(sqrt(var) / 32768.0);
        }
        double work = 100.0 * st.busy_us / (STATS_PERIOD_MS * 1000.0);

        if (have_prev && total != prev_total) {
            double dt = (double)(rt_t)(total - prev_total);
            if (per_core) {
                double c0 = 100.0 * (1.0 - (double)(rt_t)(idle[0] - prev_idle[0]) / dt);
                double c1 = 100.0 * (1.0 - (double)(rt_t)(idle[1] - prev_idle[1]) / dt);
                printf("[%5.0f s] CPU core1 (kws) %4.1f%%  core0 %4.1f%%  avg %4.1f%%",
                       secs, c1, c0, (c0 + c1) / 2);
            } else {
                double avg = 100.0 * (1.0 - (double)(rt_t)(idle_all - prev_idle_all) / (2 * dt));
                printf("[%5.0f s] CPU avg of 2 cores %4.1f%%", secs, avg);
            }
            printf(" | kws work %4.1f%%\n", work);
            printf("          %lu inferences, max score %.2f, %lu detections | mic %5.1f dBFS"
                   " | dropped audio %lu, short reads %lu\n",
                   (unsigned long)st.inferences, st.max_smoothed,
                   (unsigned long)st.detections, level,
                   (unsigned long)(ovf - prev_ovf), (unsigned long)st.short_reads);
            stream_stats_t ss;
            stream_get_stats(&ss);
            if (ss.sessions || ss.skipped)
                printf("          stream: %lu sessions (last %.1f s, %.1f KB), %lu detections not streamed%s\n",
                       (unsigned long)ss.sessions, ss.last_seconds, ss.last_kb, (unsigned long)ss.skipped,
                       ss.lost_hops ? " - ENCODER LOST AUDIO" : "");
            net_status_t ns;
            net_get_status(&ns);
            if (ns.connected)
                printf("          net: '%s', round trip %.1f ms (best %.1f ms), reconnects %lu\n",
                       ns.ssid, ns.rtt_ms, ns.rtt_best_ms, (unsigned long)ns.reconnects);
            else
                printf("          net: %s\n", ns.wifi_up ? "Wi-Fi up, server not connected"
                                                          : "Wi-Fi not connected");
            uint32_t mh, mz, mb;
            stream_mic_health(&mh, &mz, &mb);
            if (mh && (mz + mb) * 10 > mh)
                printf("          MIC FAULT: %u%% digital silence, %u%% bit-pattern garbage - check the INMP441 wiring\n",
                       (unsigned)(100 * mz / mh), (unsigned)(100 * mb / mh));
            if (!stacks_shown && secs > 30) {
                // one-off: unused stack per task, to size the stacks from data
                stacks_shown = true;
                printf("          stack headroom (bytes unused):");
                for (UBaseType_t i = 0; i < n; i++)
                    printf(" %s=%u", tasks[i].pcTaskName, (unsigned)tasks[i].usStackHighWaterMark);
                printf("\n");
            }
            size_t total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
            printf("          heap (internal RAM): %u KB in use, peak %u KB, of %u KB | free %u KB\n",
                   (unsigned)((total - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)) / 1024),
                   (unsigned)((total - heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL)) / 1024),
                   (unsigned)(total / 1024),
                   (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
        }
        prev_idle[0] = idle[0];
        prev_idle[1] = idle[1];
        prev_idle_all = idle_all;
        prev_total = total;
        prev_ovf = ovf;
        have_prev = true;
    }
}

// ---------------------------------------------------------------- main

extern "C" void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(300));
    printf("\n=== Step 5e: detection + streaming to the ASR server ===\n");

    if (!kws_init()) {
        printf("engine init failed - see message above\n");
        return;
    }
#ifdef KWS_MODEL_NAME
    printf("model %s, ", KWS_MODEL_NAME);
#endif
    printf("%u bytes, input quant %.5f / %d, arena used %u of %u bytes\n",
           kws_model_size(), KWS_IN_SCALE, KWS_IN_ZERO_PT,
           kws_arena_used(), (unsigned)KWS_ARENA_SIZE);
#ifdef KWS_MEL_LO_HZ
    printf("features: 40 log-mel bands, %.0f-%.0f Hz\n", KWS_MEL_LO_HZ, KWS_MEL_HI_HZ);
#endif
    printf("inference every %d ms, smoothing %d, threshold %.2f, "
           "lockout %d ms + re-arm below %.2f\n",
           KWS_INFER_EVERY * 20, KWS_SMOOTH_N, KWS_THRESHOLD, KWS_LOCKOUT_MS,
           KWS_REARM_BELOW);
    printf("scores above %.2f are printed for tuning\n", TRACE_MIN_SCORE);

    led_init();
    i2s_init();
    net_start();          // Wi-Fi + always-open server connection
    stream_init();        // Step 5b: stream the audio after each detection

    xTaskCreatePinnedToCore(audio_task, "kws_audio", 4096, NULL, 10, NULL, 1);
    xTaskCreatePinnedToCore(stats_task, "kws_stats", 3072, NULL, 2, NULL, 0);
}
