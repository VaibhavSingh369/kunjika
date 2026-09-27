// kws_engine - streaming keyword detection
//
// Feed audio in 20 ms hops (320 samples at 16 kHz). Every hop adds one
// log-mel frame to a rolling 49-frame (~1 s) window. Every KWS_INFER_EVERY
// hops the model runs on that window, and a smoothed score decides detection.
//
// No hardware code in here: the same file runs on the ESP32 and in the
// host test harness.

#pragma once

#include <stdbool.h>
#include <stdint.h>

// ---- decision tuning --------------------------------------------------------
// These are the knobs you will adjust on real audio.

// Run the model every N hops. 5 hops = 100 ms = 10 inferences/s.
#define KWS_INFER_EVERY       5

// Average this many consecutive model scores before deciding. Smooths out
// one-off spikes; 3 x 100 ms = the word must look right for ~300 ms.
#define KWS_SMOOTH_N          3

// Smoothed score needed to fire. Raise it if you get false activations,
// lower it if real keywords are missed.
#define KWS_THRESHOLD         0.80f

// After a detection, ignore everything for at least this long...
#define KWS_LOCKOUT_MS        1000

// ...AND until the score has dropped below this. A timer alone let one long
// word fire twice when its score was still high as the lockout expired.
#define KWS_REARM_BELOW       0.50f

// Tensor arena for the model. Measured need: ~17 KB with ESP-NN.
#define KWS_ARENA_SIZE        (18 * 1024)   // 16.9 KB used by the Kunjika models
// -----------------------------------------------------------------------------

#define KWS_HOP_SAMPLES       320

typedef struct {
    bool  inferred;    // the model ran on this hop
    float prob;        // that run's keyword probability
    float smoothed;    // mean of the last KWS_SMOOTH_N probabilities
    bool  detected;    // threshold crossed outside the lockout
} kws_result_t;

#ifdef __cplusplus
extern "C" {
#endif

// Loads the model and prepares the feature pipeline. Returns false on error
// (the reason is printed).
bool kws_init(void);

// Push exactly KWS_HOP_SAMPLES new samples (16-bit, 16 kHz, mono).
void kws_push_hop(const int16_t *hop, kws_result_t *result);

// Speech-band energy of the frame computed by the last kws_push_hop()
float kws_last_frame_energy(void);

// Diagnostics
unsigned kws_arena_used(void);
unsigned kws_model_size(void);
const int8_t *kws_debug_input(void);   // last model input; needs -DKWS_DEBUG_INPUT

#ifdef __cplusplus
}
#endif
