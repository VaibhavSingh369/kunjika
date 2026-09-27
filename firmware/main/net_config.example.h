// Copy this to net_config.h and fill in your networks.
//
// The board tries the networks in order and uses the first one it can join,
// so the same firmware works at home, on a phone hotspot and at the venue.
// Each network has its own server IP: the laptop gets a different address
// on every network (run "ipconfig" on the laptop while connected to it).
#pragma once

#define WIFI_NETWORKS {                                                   \
    { "Home-WiFi-Name",  "home-password",    "192.168.1.6"  },            \
    { "Phone-Hotspot",   "hotspot-password", "10.186.251.5" },            \
}
#define SERVER_PORT  7000

// The older single-network form still works too:
//   #define WIFI_SSID    "Home-WiFi-Name"
//   #define WIFI_PASS    "home-password"
//   #define SERVER_IP    "192.168.1.6"
//   #define SERVER_PORT  7000
