# Server

Receives the audio the board streams after "Kunjika", measures the latency, and transcribes the command with Whisper.

```bash
pip install -r requirements.txt
python kws_server.py                    # port 7000, English as English, Hindi as Hinglish
python kws_server.py --fast             # experimental 10 s Whisper window (falls back if unsupported)
python kws_server.py --model base       # faster, weaker for Hindi
python kws_server.py --hindi-model DIR  # a converted Hindi Whisper model for Hindi commands
```

Allow Python through the Windows firewall on private networks. Keep the laptop plugged in, with its Wi-Fi adapter on Maximum Performance.

## What it prints

- `>> session N: ASR receiving audio - 71 ms after the keyword ended`: the problem statement's latency, printed the instant the first audio arrives.
- A summary when the session ends: audio length, data rate, the latency split into detection and network, and clock-sync accuracy.
- The transcript a few seconds later. Transcription runs in a separate process and never delays receiving audio.

Every session is saved as a WAV file in `sessions/` and logged to `sessions.csv`.

## How latency is measured

The server synchronizes its clock with the board every second (best round trip of the last 20 exchanges), converts the board's keyword-end timestamp into server time, and subtracts it from the moment the first audio frame arrives. If the clock sync is worse than ±20 ms, it reports no latency for that session rather than a misleading one.

## Files

| File | Role |
|---|---|
| `kws_server.py` | the server: protocol, clock sync, latency, sessions, Whisper |
| `hinglish.py` | Devanagari → Hinglish transliteration with Hindi schwa deletion |
| `fake_board.py` | a pretend board speaking the same protocol, for testing without hardware |
