"""
Fake board - speaks exactly the ESP32's protocol, so the laptop server can be
tested (including Whisper) before the real board is involved.

It connects, answers clock-sync requests, then "detects" the keyword and
streams a WAV the way the firmware does: 1 s of audio from before the
detection in one burst, then live 20 ms frames.

    python fake_board.py                         (synthetic audio)
    python fake_board.py --wav some_16k_mono.wav (your own recording)
    python fake_board.py --server 192.168.1.10   (server on another machine)

Its clock deliberately runs 123.456789 s ahead of the laptop's, so the
server's clock synchronisation is exercised too.
"""

import argparse
import socket
import struct
import threading
import time
import wave

import numpy as np

from kws_server import (HELLO, PING, SYNC_REPLY, START, AUDIO, END, PONG, SYNC, FIRST_ACK,
                        TEXT, CODEC_ADPCM, SAMPLE_RATE, frame, adpcm_encode)

CLOCK_AHEAD_US = 123_456_789
FRAME = 320                                   # 20 ms


def dev_us():
    return time.perf_counter_ns() // 1000 + CLOCK_AHEAD_US


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--wav", default="")
    ap.add_argument("--detect-delay-ms", type=int, default=350,
                    help="simulated time from keyword end to detection")
    ap.add_argument("--sessions", type=int, default=1,
                    help="sessions over the same connection, back to back")
    ap.add_argument("--gap", type=float, default=2.0,
                    help="seconds between the end of one session and the next detection")
    a = ap.parse_args()

    if a.wav:
        with wave.open(a.wav) as w:
            assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    else:
        t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
        audio = (6000 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.int16)

    s = socket.create_connection((a.server, a.port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    lock = threading.Lock()
    got = {}

    def send(ftype, payload=b""):
        with lock:
            s.sendall(frame(ftype, payload))

    def reader():
        buf = b""
        while True:
            try:
                d = s.recv(4096)
            except OSError:
                return
            if not d:
                return
            buf += d
            while len(buf) >= 3:
                ftype, n = struct.unpack_from("<BH", buf)
                if len(buf) < 3 + n:
                    break
                p, buf = buf[3:3 + n], buf[3 + n:]
                if ftype == SYNC:
                    send(SYNC_REPLY, p[:8] + struct.pack("<Q", dev_us()))
                elif ftype == PONG:
                    got["rtt_ms"] = (dev_us() - struct.unpack_from("<Q", p)[0]) / 1000
                elif ftype == FIRST_ACK:
                    got["first_ack"] = dev_us()
                elif ftype == TEXT:
                    got["text"] = p[4:].decode(errors="replace")

    threading.Thread(target=reader, daemon=True).start()
    send(HELLO, struct.pack("<HH", 1, SAMPLE_RATE) + b"fake-board\0")
    send(PING, struct.pack("<Q", dev_us()))
    print("connected; letting the clocks synchronise for 6 s...")
    time.sleep(6)

    acks = []
    for sid in range(1, a.sessions + 1):
        # the keyword "ended" detect-delay-ms ago, and 1 s of audio is buffered
        got.pop("first_ack", None)
        t_det = dev_us()
        t_kw_end = t_det - a.detect_delay_ms * 1000
        t_first = t_det - 1_000_000
        send(START, struct.pack("<IQQQB", sid, t_det, t_kw_end, t_first, CODEC_ADPCM))
        pred, idx, seq, pos = 0, 0, 0, 0
        preroll = SAMPLE_RATE                               # sent as a burst
        while pos < len(audio):
            chunk = audio[pos:pos + FRAME]
            data, pred2, idx2 = adpcm_encode(chunk, pred, idx)
            send(AUDIO, struct.pack("<IIhBxH", sid, seq, pred, idx, len(chunk)) + data)
            pred, idx = pred2, idx2
            pos += len(chunk)
            seq += 1
            if pos > preroll:
                time.sleep(len(chunk) / SAMPLE_RATE)        # live part: real time
        send(END, struct.pack("<IB", sid, 0))
        t_wait = time.time()
        while "first_ack" not in got and time.time() - t_wait < 30:
            time.sleep(0.01)
        acks.append((got.get("first_ack", t_det) - t_det) / 1000)
        print(f"session {sid}: detection -> server acknowledged first audio: {acks[-1]:.1f} ms")
        if sid < a.sessions:
            time.sleep(a.gap)
    t_end = time.time()
    while "text" not in got and time.time() - t_end < 60:
        time.sleep(0.1)
    print(f"device side: round trip {got.get('rtt_ms', float('nan')):.1f} ms, "
          f"worst ack {max(acks):.1f} ms over {len(acks)} sessions")
    print(f"transcript sent back: {got.get('text') or '(empty - is faster-whisper installed?)'!r}")
    s.close()


if __name__ == "__main__":
    main()
