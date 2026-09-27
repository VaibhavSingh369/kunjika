// stream - after a detection, send the audio to the server
//
// The audio task hands every 20 ms hop to stream_push_hop(), which keeps the
// last ~2 s in a ring buffer. On detection, a session starts 1 s in the past
// (so the whole keyword and everything after it is sent), the buffered part
// goes out in one burst, and live audio follows until 0.8 s of silence or 6 s.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void stream_init(void);

// Audio task, every hop: 320 samples, the time the hop's audio ended, and its
// speech-band energy (for finding the end of the keyword and of the command).
void stream_push_hop(const int16_t *pcm, int64_t t_hop_end_us, float energy);

// Audio task, on detection.
void stream_on_detect(int64_t t_det_us);

// net task, when the server confirms it received the first audio.
void stream_on_first_ack(uint32_t session);

typedef struct {
    uint32_t sessions;
    uint32_t skipped;        // detections while offline or already streaming
    float last_kb;
    float last_seconds;
    uint32_t lost_hops;      // audio the encoder could not keep up with (should stay 0)
} stream_stats_t;

void stream_get_stats(stream_stats_t *out);

// Microphone health since the last call: hops seen, hops of digital silence,
// hops that are not audio (never negative: a floating data line). Resets.
void stream_mic_health(uint32_t *hops, uint32_t *zero, uint32_t *bad);

#ifdef __cplusplus
}
#endif
