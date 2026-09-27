// net - Wi-Fi and the always-open connection to the laptop server
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Frame types (must match kws_server.py)
enum {
    NET_HELLO = 0x01, NET_PING = 0x02, NET_SYNC_REPLY = 0x03,
    NET_START = 0x10, NET_AUDIO = 0x11, NET_END = 0x12,
    NET_PONG = 0x82, NET_SYNC = 0x83, NET_FIRST_ACK = 0x91, NET_TEXT = 0x92,
};

// Starts Wi-Fi and the connection task (core 0). Returns immediately.
void net_start(void);

// Sends one frame. Thread-safe. Returns false if not connected or the send failed.
bool net_send(uint8_t type, const void *payload, uint16_t len);

// Sends bytes that are already framed (several frames packed together). Thread-safe.
bool net_send_raw(const void *buf, size_t len);

typedef struct {
    bool wifi_up;
    bool connected;
    float rtt_ms;          // last ping round trip
    float rtt_best_ms;     // best since connecting
    uint32_t reconnects;
    const char *ssid;      // network in use
} net_status_t;

void net_get_status(net_status_t *out);

#ifdef __cplusplus
}
#endif
