// IMA-ADPCM encoder: 16-bit PCM -> 4 bits per sample (4x smaller).
// Must match adpcm_decode() in server/kws_server.py (low nibble first).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int16_t predictor;
    uint8_t index;
} adpcm_state_t;

// Encodes n samples (n even) into n/2 bytes, updating the state.
void adpcm_encode(adpcm_state_t *st, const int16_t *in, int n, uint8_t *out);

#ifdef __cplusplus
}
#endif
