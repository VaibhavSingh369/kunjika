# Firmware (ESP32-S3)

ESP-IDF 6.1 project. Core 1 runs keyword detection only; core 0 runs audio compression, streaming and networking.

## Wiring

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| SCK | GPIO 4 |
| WS | GPIO 5 |
| SD | GPIO 6 |
| L/R | GND (left channel) |

**Solder the INMP441's header pins.** Loose pins cause intermittent digital silence or bit-pattern garbage; the firmware reports both as `MIC FAULT`.

The RGB LED is on GPIO 38 (DevKitC-1 v1.1) or GPIO 48 (v1.0): green at start-up, blue on detection.

## Configure

```bash
cp main/net_config.example.h main/net_config.h
```

Fill in one or more networks. Each network has its own server IP, because the laptop gets a different address on every network (`ipconfig` on Windows). The board tries them in order. It must be a **2.4 GHz** network; WPA2 and WPA3 both work. `net_config.h` is in `.gitignore`, so your password never gets committed.

## Build and flash

```bash
idf.py set-target esp32s3       # first time, and whenever sdkconfig.defaults changes
idf.py build
idf.py -p COM6 flash monitor    # exit the monitor with Ctrl+]
idf.py size                     # static RAM report
```

On Windows, after extracting files from a zip, refresh their timestamps before building (zip times can look older than the last build):

```powershell
Get-ChildItem main\* -Include *.c,*.cc,*.h,CMakeLists.txt | ForEach-Object { $_.LastWriteTime = Get-Date }
```

## Files

| File | Role |
|---|---|
| `main/main.cc` | start-up, audio task (core 1), statistics task, LED |
| `main/kws_engine.cc/.h` | log-mel features and model inference: the listening engine |
| `main/kws_model.h`, `kws_mel.h`, `kws_config.h`, `kws_ops.h` | generated from the trained model by `tools/gen_step4.py` |
| `main/stream.c/.h` | encoder task (ADPCM), 1.6 s compressed history, sessions, microphone health |
| `main/net.c/.h` | Wi-Fi, networks list, always-open TCP, clock-sync replies, 5 s watchdog |
| `main/adpcm.c/.h` | IMA-ADPCM encoder, bit-identical to the server's decoder |
| `sdkconfig.defaults` | build settings, including all the RAM savings (lwIP on core 0, code in flash, Wi-Fi buffers) |

## Using a new model

```bash
python tools/gen_step4.py path/to/model.tflite main
```

Keep `model.features.json` next to the `.tflite` file: the generator reads it, so the firmware's mel filterbank always matches the model, and it refuses to build a mismatched one.

## Reading the log

Every 5 seconds the board prints CPU per core, inferences and detections, microphone level, dropped audio (must be 0), the network and round-trip time, and heap use. Chapter 11.5 of the [documentation](../docs/Kunjika_Project_Documentation.pdf) explains every field.
