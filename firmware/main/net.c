// net - Wi-Fi and the always-open connection to the laptop server.
//
// The connection is opened at boot and kept open, so a detection never waits
// for a TCP handshake. The task also answers the server's clock-sync
// requests immediately; that is what lets the server measure latency with
// the board's own timestamps.

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

#include "net.h"
#include "net_config.h"
#include "stream.h"

#define GOT_IP_BIT       BIT0
#define PING_PERIOD_US   2000000
#define CONNECT_TIMEOUT_S 3
#define SILENT_US        5000000      // server sends clock sync every 1 s: 5 s of nothing = dead link
#define SEND_TIMEOUT_S   2            // a send may never block longer than this
#define FAILS_PER_NET    2            // failed joins before trying the next network

typedef struct { const char *ssid, *pass, *server_ip; } network_t;
#ifdef WIFI_NETWORKS
static const network_t NETS[] = WIFI_NETWORKS;
#else
static const network_t NETS[] = { { WIFI_SSID, WIFI_PASS, SERVER_IP } };
#endif
#define N_NETS ((int)(sizeof(NETS) / sizeof(NETS[0])))
static volatile int s_net;            // index of the network in use
static int s_fails;

static EventGroupHandle_t s_events;
static SemaphoreHandle_t s_send_lock;
static volatile int s_sock = -1;
static volatile bool s_wifi_up;
static volatile float s_rtt_ms = -1, s_rtt_best_ms = -1;
static volatile uint32_t s_reconnects;

// ------------------------------------------------------------------ Wi-Fi

static const char *reason_text(int r)
{
    static char other[24];
    switch (r) {
    case 2:   return "authentication expired";
    case 8:   return "left the network";
    case 15:
    case 204: return "wrong password (handshake timed out)";
    case 200: return "signal lost";
    case 201: return "network not found - check the name, that it is 2.4 GHz, and range";
    case 202: return "authentication failed - wrong password or security type";
    case 203: return "association failed";
    case 205: return "connection failed";
    case 210: return "found, but its security mode is not supported";
    case 211: return "found, but its security is below WPA2";
    default:  snprintf(other, sizeof(other), "reason %d", r); return other;
    }
}

static void apply_network(void)
{
    const network_t *n = &NETS[s_net];
    wifi_config_t wc;
    memset(&wc, 0, sizeof(wc));
    strncpy((char *)wc.sta.ssid, n->ssid, sizeof(wc.sta.ssid) - 1);
    strncpy((char *)wc.sta.password, n->pass, sizeof(wc.sta.password) - 1);
    // accept WPA2 and WPA3 (phone hotspots often use WPA3 with hash-to-element)
    wc.sta.threshold.authmode = n->pass[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    wc.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
    esp_wifi_set_config(WIFI_IF_STA, &wc);
    printf("net: joining Wi-Fi '%s'...\n", n->ssid);
}

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        int reason = ((wifi_event_sta_disconnected_t *)data)->reason;
        if (s_wifi_up) {
            printf("net: Wi-Fi '%s' lost (%s) - reconnecting\n", NETS[s_net].ssid, reason_text(reason));
            s_fails = 0;
        } else {
            printf("net: cannot join '%s': %s\n", NETS[s_net].ssid, reason_text(reason));
            if (N_NETS > 1 && ++s_fails >= FAILS_PER_NET) {
                s_fails = 0;
                s_net = (s_net + 1) % N_NETS;
                apply_network();
            }
        }
        s_wifi_up = false;
        xEventGroupClearBits(s_events, GOT_IP_BIT);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        printf("net: Wi-Fi '%s' connected, board IP " IPSTR "\n", NETS[s_net].ssid, IP2STR(&e->ip_info.ip));
        s_wifi_up = true;
        s_fails = 0;
        xEventGroupSetBits(s_events, GOT_IP_BIT);
    }
}

static void wifi_init(void)
{
    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        r = nvs_flash_init();
    }
    ESP_ERROR_CHECK(r);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    s_net = 0;
    apply_network();
    ESP_ERROR_CHECK(esp_wifi_start());

    // Power save delays incoming packets until the next beacon (tens to
    // hundreds of ms). Latency matters more here; the stats task shows the
    // CPU cost of keeping the radio awake.
    esp_wifi_set_ps(WIFI_PS_NONE);
}

// ------------------------------------------------------------------ sending

static bool send_all(int sock, const uint8_t *p, size_t n)
{
    while (n) {
        int k = send(sock, p, n, 0);
        if (k <= 0) return false;
        p += k;
        n -= (size_t)k;
    }
    return true;
}

bool net_send(uint8_t type, const void *payload, uint16_t len)
{
    int sock = s_sock;
    if (sock < 0) return false;
    uint8_t head[3] = { type, (uint8_t)(len & 0xFF), (uint8_t)(len >> 8) };
    xSemaphoreTake(s_send_lock, portMAX_DELAY);
    bool ok = send_all(sock, head, 3) && (len == 0 || send_all(sock, (const uint8_t *)payload, len));
    xSemaphoreGive(s_send_lock);
    return ok;
}

bool net_send_raw(const void *buf, size_t len)
{
    int sock = s_sock;
    if (sock < 0) return false;
    xSemaphoreTake(s_send_lock, portMAX_DELAY);
    bool ok = send_all(sock, (const uint8_t *)buf, len);
    xSemaphoreGive(s_send_lock);
    return ok;
}

static void put_u64(uint8_t *p, uint64_t v)
{
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint64_t get_u64(const uint8_t *p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

// ------------------------------------------------------------------ connection

static int connect_server(const char *server_ip)
{
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(SERVER_PORT);
    if (inet_pton(AF_INET, server_ip, &addr.sin_addr) != 1) {
        printf("net: server IP '%s' is not a valid IPv4 address\n", server_ip);
        return -1;
    }
    int s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s < 0) return -1;

    // non-blocking connect with a timeout, so an absent server is noticed fast
    int flags = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, flags | O_NONBLOCK);
    int r = connect(s, (struct sockaddr *)&addr, sizeof(addr));
    if (r != 0 && errno != EINPROGRESS) {
        close(s);
        return -1;
    }
    if (r != 0) {
        fd_set w;
        FD_ZERO(&w);
        FD_SET(s, &w);
        struct timeval tv = { .tv_sec = CONNECT_TIMEOUT_S, .tv_usec = 0 };
        int err = 0;
        socklen_t el = sizeof(err);
        if (select(s + 1, NULL, &w, NULL, &tv) <= 0 ||
            getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &el) != 0 || err != 0) {
            close(s);
            return -1;
        }
    }
    fcntl(s, F_SETFL, flags);                      // back to blocking

    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));   // no batching delay
    struct timeval rt = { .tv_sec = 0, .tv_usec = 100000 };        // recv wakes every 100 ms
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &rt, sizeof(rt));
    struct timeval st = { .tv_sec = SEND_TIMEOUT_S, .tv_usec = 0 };  // a send never hangs
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &st, sizeof(st));
    return s;
}

static void handle_frame(uint8_t type, const uint8_t *p, uint16_t n)
{
    if (type == NET_SYNC && n >= 8) {
        // answer at once: the server's clock sync relies on a quick echo
        uint8_t reply[16];
        memcpy(reply, p, 8);
        put_u64(reply + 8, (uint64_t)esp_timer_get_time());
        net_send(NET_SYNC_REPLY, reply, sizeof(reply));
    } else if (type == NET_PONG && n >= 8) {
        float rtt = (esp_timer_get_time() - (int64_t)get_u64(p)) / 1000.0f;
        s_rtt_ms = rtt;
        if (s_rtt_best_ms < 0 || rtt < s_rtt_best_ms) s_rtt_best_ms = rtt;
    } else if (type == NET_FIRST_ACK && n >= 4) {
        stream_on_first_ack((uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24)));
    } else if (type == NET_TEXT && n >= 4) {
        if (n > 4) printf("SERVER HEARD: \"%.*s\"\n", (int)(n - 4), (const char *)p + 4);
        else       printf("server: session done (no transcript - is faster-whisper installed?)\n");
    }
}

static void net_task(void *arg)
{
    static uint8_t rx[1024];
    bool announced_fail = false;

    for (;;) {
        xEventGroupWaitBits(s_events, GOT_IP_BIT, pdFALSE, pdTRUE, portMAX_DELAY);

        const char *server_ip = NETS[s_net].server_ip;
        int sock = connect_server(server_ip);
        if (sock < 0) {
            if (!announced_fail) {
                printf("net: cannot reach server %s:%d - is kws_server.py running, is the IP "
                       "right, and is the firewall allowing it? (retrying every 2 s)\n",
                       server_ip, SERVER_PORT);
                announced_fail = true;
            }
            vTaskDelay(pdMS_TO_TICKS(2000));
            continue;
        }
        announced_fail = false;
        s_sock = sock;
        s_rtt_best_ms = -1;
        printf("net: connected to server %s:%d\n", server_ip, SERVER_PORT);

        uint8_t hello[4 + 16] = { 1, 0, 0x80, 0x3E };            // protocol 1, 16000 Hz
        strncpy((char *)hello + 4, "esp32-kunjika", 15);
        net_send(NET_HELLO, hello, sizeof(hello));

        size_t have = 0;
        int64_t next_ping = 0;
        int64_t last_rx = esp_timer_get_time();
        for (;;) {
            int64_t now = esp_timer_get_time();
            if (now - last_rx > SILENT_US) {
                // The link can die without either side being told (a "half-open"
                // connection): the server sends clock sync every second, so five
                // seconds of silence means it is gone.
                printf("net: nothing from the server for %d s - connection is dead\n",
                       SILENT_US / 1000000);
                break;
            }
            if (now >= next_ping) {
                uint8_t t[8];
                put_u64(t, (uint64_t)now);
                if (!net_send(NET_PING, t, 8)) break;
                next_ping = now + PING_PERIOD_US;
            }
            int k = recv(sock, rx + have, sizeof(rx) - have, 0);
            if (k == 0) break;                                   // server closed
            if (k < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                break;
            }
            have += (size_t)k;
            last_rx = esp_timer_get_time();
            while (have >= 3) {
                uint16_t n = (uint16_t)(rx[1] | (rx[2] << 8));
                if (n > sizeof(rx) - 3) { have = 0; break; }     // corrupt: resync
                if (have < 3u + n) break;
                handle_frame(rx[0], rx + 3, n);
                memmove(rx, rx + 3 + n, have - 3 - n);
                have -= 3u + n;
            }
        }

        s_sock = -1;
        xSemaphoreTake(s_send_lock, portMAX_DELAY);              // no sender mid-frame
        close(sock);
        xSemaphoreGive(s_send_lock);
        s_reconnects++;
        printf("net: server connection lost - reconnecting\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void net_start(void)
{
    s_events = xEventGroupCreate();
    s_send_lock = xSemaphoreCreateMutex();
    wifi_init();
    xTaskCreatePinnedToCore(net_task, "net", 3072, NULL, 5, NULL, 0);
}

void net_get_status(net_status_t *out)
{
    out->wifi_up = s_wifi_up;
    out->connected = s_sock >= 0;
    out->rtt_ms = s_rtt_ms;
    out->rtt_best_ms = s_rtt_best_ms;
    out->reconnects = s_reconnects;
    out->ssid = NETS[s_net].ssid;
}
