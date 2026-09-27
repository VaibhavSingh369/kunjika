// stream - after a detection, send the audio to the server.
//
// Memory design: the audio task hands each 20 ms hop over through a small PCM
// staging ring; an encoder task on core 0 compresses every hop (IMA-ADPCM,
// 4x smaller) into a 2.56 s history, remembering the encoder state at the start
// of each hop so a session can begin at any hop. A session then sends
// already-compressed hops. The encoder never touches the network, so a slow
// send can never make it lose audio.
//
// 1.6 s history: 51 KB as PCM, 12.5 KB compressed.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"

#include "adpcm.h"
#include "net.h"
#include "stream.h"

#define SR               16000
#define HOP              320
#define HOP_BYTES        (HOP / 2)
#define STAGE_HOPS       6               // PCM hand-over, 120 ms of slack for the encoder
#define RING_HOPS        80              // compressed history, 1.6 s (1 s pre-roll + margin)
#define PREROLL_HOPS     50              // 1 s before detection: covers the keyword
#define MIN_AFTER_US     1000000         // never end sooner than 1 s after detection
#define MAX_AFTER_US     6000000         // never longer than 6 s after detection
#define SILENCE_HOPS     40              // 0.8 s of silence ends the command
#define SPEECH_DB        10.0f           // "speech" = this far above the noise floor
#define FLOOR_PERCENTILE 20              // noise floor = 20th percentile of recent energies
#define KW_END_SEARCH    60              // look back up to 1.2 s for the keyword end
#define PKT_BYTES        1400            // pack frames into TCP writes of up to ~1 MSS

static int16_t       s_stage[STAGE_HOPS][HOP];
static uint8_t       s_adpcm[RING_HOPS][HOP_BYTES];
static adpcm_state_t s_state[RING_HOPS];        // encoder state at the start of each hop
static int64_t       s_hop_t[RING_HOPS];        // time each hop's audio ended
static float         s_hop_e[RING_HOPS];        // speech-band energy of each hop
static uint32_t      s_hops;                    // hops handed over (release/acquire)
static uint32_t      s_encoded;                 // hops compressed (release/acquire)
static uint32_t      s_lost;                    // hops the encoder could not keep up with
static volatile uint32_t s_mic_hops, s_mic_zero, s_mic_bad;   // since the last health read

static TaskHandle_t s_enc_task, s_task;
static volatile bool s_pending, s_active;
static uint32_t s_session, s_start;
static int64_t  s_t_det, s_t_kw_end, s_t_first;
static float    s_thr;
static stream_stats_t s_stats;

static void put_u32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put_u64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }

// ------------------------------------------------------------------ audio task side

void stream_push_hop(const int16_t *pcm, int64_t t_hop_end_us, float energy)
{
    uint32_t h = s_hops;
    memcpy(s_stage[h % STAGE_HOPS], pcm, sizeof(s_stage[0]));
    s_hop_t[h % RING_HOPS] = t_hop_end_us;
    s_hop_e[h % RING_HOPS] = energy;
    __atomic_store_n(&s_hops, h + 1, __ATOMIC_RELEASE);
    if (s_enc_task) xTaskNotifyGive(s_enc_task);
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

// The quietest single hop is far below the room's typical noise, so a floor
// taken from the minimum made ordinary noise look like speech and sessions
// ran to the 6 s limit. A low percentile is robust to both dips and speech.
static float noise_floor(void)
{
    static float tmp[RING_HOPS];
    uint32_t n = s_hops < RING_HOPS ? s_hops : RING_HOPS, m = 0;
    for (uint32_t i = 0; i < n; i++)
        if (s_hop_e[i] > 0.0f) tmp[m++] = s_hop_e[i];   // zero energy = mic dropout, not room noise
    if (m == 0) return 1e-12f;
    qsort(tmp, m, sizeof(float), cmp_float);
    return tmp[m * FLOOR_PERCENTILE / 100];
}

// The keyword end = the last loud hop before detection, provided at least two
// quiet hops follow it. If speech never paused, the end is not measurable.
static int64_t find_kw_end(float thr)
{
    int quiet = 0;
    for (uint32_t back = 0; back < KW_END_SEARCH && back < s_hops && back < RING_HOPS; back++) {
        uint32_t h = s_hops - 1 - back;
        if (s_hop_e[h % RING_HOPS] > thr) return quiet >= 2 ? s_hop_t[h % RING_HOPS] : 0;
        quiet++;
    }
    return 0;
}

void stream_on_detect(int64_t t_det_us)
{
    net_status_t ns;
    net_get_status(&ns);
    if (s_active || s_pending) { s_stats.skipped++; return; }
    if (!ns.connected) {
        printf("stream: detected, but the server is not connected - nothing sent\n");
        s_stats.skipped++;
        return;
    }
    uint32_t h = s_hops;
    if (h < PREROLL_HOPS) { s_stats.skipped++; return; }

    s_thr = noise_floor() * powf(10.0f, SPEECH_DB / 10.0f);
    s_t_kw_end = find_kw_end(s_thr);
    s_t_det = t_det_us;
    s_start = h - PREROLL_HOPS;
    s_t_first = s_hop_t[(h - 1) % RING_HOPS] - (int64_t)PREROLL_HOPS * HOP * 1000000 / SR;
    s_session++;
    printf("stream: session %lu - keyword end %s\n", (unsigned long)s_session,
           s_t_kw_end ? "measured" : "not measurable (speech ran straight on)");
    __atomic_store_n(&s_pending, true, __ATOMIC_RELEASE);
    xTaskNotifyGive(s_task);
}

// ------------------------------------------------------------------ net task side

void stream_on_first_ack(uint32_t session)
{
    if (session == s_session)
        printf("stream: server is receiving audio (%.0f ms after detection, incl. ack return)\n",
               (esp_timer_get_time() - s_t_det) / 1000.0);
}

// ------------------------------------------------------------------ encoder task (core 0)

static void encoder_task(void *arg)
{
    adpcm_state_t enc = { 0, 0 };
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        uint32_t h = __atomic_load_n(&s_hops, __ATOMIC_ACQUIRE);
        uint32_t e = s_encoded;
        if (h - e > STAGE_HOPS) {                          // fell behind: skip what was overwritten
            s_lost += h - e - STAGE_HOPS;
            e = h - STAGE_HOPS;
        }
        for (; e != h; e++) {
            // Microphone health: a dead INMP441 gives exact zeros; a loose data
            // line gives bit patterns that are never negative. Real audio is
            // never either.
            // A floating line produces runs of 1-bits: 0, 1, 3, 7 ... 511. Checking
            // for that exact shape (not merely "never negative") avoids false
            // alarms from the INMP441's slow drift, which can keep a genuine
            // 20 ms hop on one side of zero.
            const int16_t *pcm = s_stage[e % STAGE_HOPS];
            int zeros = 0, pattern = 0;
            for (int i = 0; i < HOP; i++) {
                int v = pcm[i];
                zeros += v == 0;
                pattern += v >= 0 && (v & (v + 1)) == 0;          // 0 or 2^k - 1
            }
            s_mic_hops++;
            if (zeros == HOP) s_mic_zero++;
            else if (pattern >= HOP * 9 / 10) s_mic_bad++;
            s_state[e % RING_HOPS] = enc;
            adpcm_encode(&enc, pcm, HOP, s_adpcm[e % RING_HOPS]);
            __atomic_store_n(&s_encoded, e + 1, __ATOMIC_RELEASE);
        }
        if (s_active && s_task) xTaskNotifyGive(s_task);
    }
}

// ------------------------------------------------------------------ stream task (core 0)

static void stream_task(void *arg)
{
    // Frames are packed into writes of up to PKT_BYTES: the 1 s pre-roll goes
    // out as ~7 TCP segments instead of 50, which uses far fewer Wi-Fi
    // buffers (peak RAM) and far less CPU. Live audio is sent hop by hop, so
    // packing never adds delay.
    static uint8_t out[PKT_BYTES];

    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (!__atomic_load_n(&s_pending, __ATOMIC_ACQUIRE)) continue;
        s_pending = false;
        s_active = true;

        uint32_t id = s_session, hop = s_start, seq = 0, bytes = 0, writes = 0;
        size_t used = 0;
        out[0] = NET_START; out[1] = 29; out[2] = 0;
        put_u32(out + 3, id);
        put_u64(out + 7, (uint64_t)s_t_det);
        put_u64(out + 15, (uint64_t)s_t_kw_end);
        put_u64(out + 23, (uint64_t)s_t_first);
        out[31] = 1;                                          // codec: IMA-ADPCM
        used = 32;

        bool ok = true;
        int quiet = 0, reason = 1;
        bool done = false;
        while (ok && !done) {
            uint32_t e = __atomic_load_n(&s_encoded, __ATOMIC_ACQUIRE);
            if (e - hop > RING_HOPS - 8) {                   // only if sends stalled for ~1.4 s
                printf("stream: network too slow - session cut\n");
                reason = 2;
                break;
            }
            // pack every hop that is ready (up to the write size)
            while (hop != e && used + 3 + 14 + HOP_BYTES <= PKT_BYTES) {
                const adpcm_state_t *sp = &s_state[hop % RING_HOPS];
                uint8_t *f = out + used;
                f[0] = NET_AUDIO; f[1] = (uint8_t)(14 + HOP_BYTES); f[2] = 0;
                put_u32(f + 3, id);
                put_u32(f + 7, seq);
                f[11] = (uint8_t)(sp->predictor & 0xFF);
                f[12] = (uint8_t)((uint16_t)sp->predictor >> 8);
                f[13] = sp->index;
                f[14] = 0;
                f[15] = (uint8_t)(HOP & 0xFF);
                f[16] = (uint8_t)(HOP >> 8);
                memcpy(f + 17, s_adpcm[hop % RING_HOPS], HOP_BYTES);
                used += 3 + 14 + HOP_BYTES;

                // end of the command: 0.8 s of silence, once 1 s has passed since detection
                int64_t t = s_hop_t[hop % RING_HOPS];
                float energy = s_hop_e[hop % RING_HOPS];
                hop++;
                seq++;
                if (t > s_t_det) {
                    quiet = energy < s_thr ? quiet + 1 : 0;
                    if (t - s_t_det >= MIN_AFTER_US && quiet >= SILENCE_HOPS) { reason = 0; done = true; break; }
                    if (t - s_t_det >= MAX_AFTER_US) { reason = 1; done = true; break; }
                }
            }
            if (used) {
                ok = net_send_raw(out, used);
                bytes += used;
                writes++;
                used = 0;
            }
            if (!done && ok && hop == __atomic_load_n(&s_encoded, __ATOMIC_ACQUIRE))
                ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));    // caught up: wait for audio
        }
        if (!ok) reason = 2;

        uint8_t end[9];
        put_u32(end, id);
        end[4] = (uint8_t)reason;
        put_u32(end + 5, seq * HOP);
        net_send(NET_END, end, sizeof(end));
        bytes += sizeof(end) + 3;

        // Diagnostic for sessions that hit the limit
        char why[80] = "";
        if (reason == 1) {
            double sum = 0;
            int zero = 0;
            for (uint32_t k = 1; k <= SILENCE_HOPS; k++) {
                float en = s_hop_e[(hop - k) % RING_HOPS];
                sum += en;
                zero += en <= 0.0f;
            }
            if (zero > SILENCE_HOPS / 2)
                snprintf(why, sizeof(why), " - the microphone delivered digital silence (check wiring)");
            else
                snprintf(why, sizeof(why), " - last 0.8 s was still %+.0f dB above the silence threshold",
                         10.0 * log10(sum / SILENCE_HOPS / s_thr));
        }
        s_stats.sessions++;
        s_stats.last_kb = bytes / 1024.0f;
        s_stats.last_seconds = seq * HOP / (float)SR;
        printf("stream: session %lu ended (%s) - %.1f s of audio, %.1f KB in %lu writes%s\n",
               (unsigned long)id, reason == 0 ? "silence" : reason == 1 ? "6 s limit" : "connection",
               s_stats.last_seconds, s_stats.last_kb, (unsigned long)writes, why);
        s_active = false;
    }
}

void stream_init(void)
{
    // encoder above stream and net so it always keeps up; all on core 0,
    // leaving core 1 to keyword detection
    xTaskCreatePinnedToCore(encoder_task, "encode", 2048, NULL, 7, &s_enc_task, 0);
    xTaskCreatePinnedToCore(stream_task, "stream", 2560, NULL, 6, &s_task, 0);
}

void stream_get_stats(stream_stats_t *out)
{
    *out = s_stats;
    out->lost_hops = s_lost;
}

void stream_mic_health(uint32_t *hops, uint32_t *zero, uint32_t *bad)
{
    *hops = s_mic_hops; *zero = s_mic_zero; *bad = s_mic_bad;
    s_mic_hops = s_mic_zero = s_mic_bad = 0;
}
