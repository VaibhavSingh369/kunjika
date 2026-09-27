# Microphone recorder

A small ESP-IDF project that records the INMP441 through the ESP32-S3 and sends the audio to a PC as WAV files, with a checksum on every transfer. We used it to verify the microphone, measure the sample rate (16,014 Hz), choose the gain shift (16), and record the room silence used in training.

```bash
idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
python tools/rec_capture.py COM6       # close any serial monitor first
```

Each take is saved as `rec_00.wav`, `rec_01.wav`, and so on (`REC_SECONDS` in `main/main.c` sets the length of a take). After each one the script prints whether the audio looks real. A disconnected or floating microphone produces bit patterns that are never negative, and the script flags them.

It uses 115200 baud: on ESP-IDF 6.1 the faster setting is not applied on the board side.
