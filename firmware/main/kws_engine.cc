// kws_engine - see kws_engine.h
//
// The feature code is the Step 2/3 code (100% int8 parity with TensorFlow),
// reorganised to produce one frame per 20 ms hop instead of 49 at once.
// Any 49 consecutive streamed frames are exactly what TensorFlow computes for
// the corresponding 1 s clip, so the model sees the same input it was
// trained on.

#include "kws_engine.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "esp_dsp.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "kws_config.h"
#include "kws_mel.h"
#include "kws_model.h"
#include "kws_ops.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if KWS_HOP_SAMPLES != KWS_FRAME_HOP
#error "kws_engine.h hop size does not match the model's feature config"
#endif

namespace {

constexpr int kNumBins  = KWS_FFT_SIZE / 2 + 1;
constexpr int kNumFeats = KWS_NUM_FRAMES * KWS_NUM_MEL;
constexpr int kLockoutInferences =
    (KWS_LOCKOUT_MS + KWS_INFER_EVERY * 20 - 1) / (KWS_INFER_EVERY * 20);

// feature pipeline
float s_window[KWS_FRAME_LEN];
alignas(16) float s_fft[2 * KWS_FFT_SIZE];
float s_power[kNumBins];
float s_last_energy = 0.0f;                  // speech-band energy of the last frame
int16_t s_history[KWS_FRAME_LEN];            // last 512 samples

// rolling window of quantized frames; s_head = oldest = next to overwrite
int8_t s_ring[KWS_NUM_FRAMES][KWS_NUM_MEL];
int s_head = 0;
int s_frames_seen = 0;
int s_hops_since_infer = 0;

// decision state
float s_recent[KWS_SMOOTH_N];
int s_recent_count = 0;
int s_recent_pos = 0;
int s_lockout = 0;
bool s_armed = true;

// model
alignas(16) uint8_t s_arena[KWS_ARENA_SIZE];
tflite::MicroInterpreter *s_interp = nullptr;
#ifdef KWS_DEBUG_INPUT
int8_t s_debug_input[kNumFeats];   // TFLM reuses the input buffer during Invoke
#endif
TfLiteTensor *s_in = nullptr;
TfLiteTensor *s_out = nullptr;

// Periodic Hann, identical to tf.signal.hann_window (denominator N).
void window_init()
{
    for (int n = 0; n < KWS_FRAME_LEN; n++) {
        s_window[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / KWS_FRAME_LEN));
    }
}

int8_t quantize(float v)
{
    int q = (int)lrintf(v / KWS_IN_SCALE) + KWS_IN_ZERO_PT;
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

// One frame from the last 512 samples, straight into int8
void frame_to_ring(int8_t *out)
{
    for (int n = 0; n < KWS_FRAME_LEN; n++) {
        s_fft[2 * n]     = ((float)s_history[n] / 32768.0f) * s_window[n];
        s_fft[2 * n + 1] = 0.0f;
    }
    dsps_fft2r_fc32(s_fft, KWS_FFT_SIZE);
    dsps_bit_rev_fc32(s_fft, KWS_FFT_SIZE);
    for (int k = 0; k < kNumBins; k++) {
        float re = s_fft[2 * k], im = s_fft[2 * k + 1];
        s_power[k] = re * re + im * im;
    }
    float energy = 0.0f;
    for (int m = 0; m < KWS_NUM_MEL; m++) {
        const float *w = &kMelWeights[kMelOffset[m]];
        const float *p = &s_power[kMelStart[m]];
        float acc = 0.0f;
        for (int i = 0; i < kMelLen[m]; i++) acc += p[i] * w[i];
        out[m] = quantize(logf(acc + KWS_LOG_FLOOR));
        if (m >= 2) energy += acc;           // skip the lowest bands: breath rumble
    }
    s_last_energy = energy;
}

float dequant(int8_t q)
{
    return ((float)q - KWS_OUT_ZERO_PT) * KWS_OUT_SCALE;
}

float keyword_prob(const int8_t *o)
{
#if KWS_OUT_IS_LOGITS
    float d = dequant(o[KWS_POS_INDEX]) - dequant(o[1 - KWS_POS_INDEX]);
    return 1.0f / (1.0f + expf(-d));
#else
    return dequant(o[KWS_POS_INDEX]);
#endif
}

}  // namespace

extern "C" bool kws_init(void)
{
    // The model runs from flash. A copy in RAM was measured to make no CPU
    // difference (8.9% either way), so it is not worth 16.8 KB of RAM.
    const tflite::Model *model = tflite::GetModel(g_kws_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("kws: model schema %lu, expected %d\n",
               (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroMutableOpResolver<KWS_NUM_OPS> resolver;
    if (!kws_register_ops(resolver)) {
        printf("kws: op registration failed\n");
        return false;
    }

    static tflite::MicroInterpreter interp(model, resolver, s_arena, KWS_ARENA_SIZE);
    if (interp.AllocateTensors() != kTfLiteOk) {
        printf("kws: AllocateTensors failed - increase KWS_ARENA_SIZE\n");
        return false;
    }
    s_interp = &interp;
    s_in = interp.input(0);
    s_out = interp.output(0);

    if (s_in->type != kTfLiteInt8 || s_in->bytes != kNumFeats ||
        s_out->type != kTfLiteInt8 || s_out->bytes != 2) {
        printf("kws: unexpected tensor shapes (in %d, out %d bytes)\n",
               (int)s_in->bytes, (int)s_out->bytes);
        return false;
    }
    if (fabsf(s_in->params.scale - KWS_IN_SCALE) > 1e-6f * KWS_IN_SCALE ||
        s_in->params.zero_point != KWS_IN_ZERO_PT) {
        printf("kws: kws_config.h does not belong to kws_model.h - regenerate both\n");
        return false;
    }

    if (dsps_fft2r_init_fc32(NULL, KWS_FFT_SIZE) != ESP_OK) {
        printf("kws: FFT init failed\n");
        return false;
    }
    window_init();
    memset(s_history, 0, sizeof(s_history));
    return true;
}

extern "C" void kws_push_hop(const int16_t *hop, kws_result_t *r)
{
    r->inferred = false;
    r->detected = false;
    r->prob = 0.0f;
    r->smoothed = 0.0f;

    // slide the 512-sample history by one hop
    memmove(s_history, s_history + KWS_FRAME_HOP,
            (KWS_FRAME_LEN - KWS_FRAME_HOP) * sizeof(int16_t));
    memcpy(s_history + (KWS_FRAME_LEN - KWS_FRAME_HOP), hop,
           KWS_FRAME_HOP * sizeof(int16_t));

    frame_to_ring(s_ring[s_head]);
    s_head = (s_head + 1) % KWS_NUM_FRAMES;
    if (s_frames_seen < KWS_NUM_FRAMES) s_frames_seen++;

    // wait for a full window (the first ~1 s after boot)
    if (s_frames_seen < KWS_NUM_FRAMES) return;
    if (++s_hops_since_infer < KWS_INFER_EVERY) return;
    s_hops_since_infer = 0;

    // oldest frame first, as the model expects
    for (int i = 0; i < KWS_NUM_FRAMES; i++) {
        memcpy(s_in->data.int8 + i * KWS_NUM_MEL,
               s_ring[(s_head + i) % KWS_NUM_FRAMES], KWS_NUM_MEL);
    }
#ifdef KWS_DEBUG_INPUT
    memcpy(s_debug_input, s_in->data.int8, kNumFeats);
#endif
    if (s_interp->Invoke() != kTfLiteOk) {
        printf("kws: Invoke failed\n");
        return;
    }

    float p = keyword_prob(s_out->data.int8);
    s_recent[s_recent_pos] = p;
    s_recent_pos = (s_recent_pos + 1) % KWS_SMOOTH_N;
    if (s_recent_count < KWS_SMOOTH_N) s_recent_count++;

    float sum = 0.0f;
    for (int i = 0; i < s_recent_count; i++) sum += s_recent[i];

    r->inferred = true;
    r->prob = p;
    r->smoothed = sum / s_recent_count;

    if (s_lockout > 0) s_lockout--;
    if (!s_armed && r->smoothed < KWS_REARM_BELOW) s_armed = true;

    if (s_armed && s_lockout == 0 && s_recent_count == KWS_SMOOTH_N &&
        r->smoothed >= KWS_THRESHOLD) {
        r->detected = true;
        s_lockout = kLockoutInferences;
        s_armed = false;
    }
}

extern "C" float kws_last_frame_energy(void)
{
    return s_last_energy;
}

extern "C" unsigned kws_arena_used(void)
{
    return s_interp ? (unsigned)s_interp->arena_used_bytes() : 0;
}

extern "C" unsigned kws_model_size(void)
{
    return g_kws_model_len;
}

extern "C" const int8_t *kws_debug_input(void)
{
#ifdef KWS_DEBUG_INPUT
    return s_debug_input;
#else
    return nullptr;   // build with -DKWS_DEBUG_INPUT to enable
#endif
}
