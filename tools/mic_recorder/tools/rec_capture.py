"""
Phase B / Step 1 - host side capture.

Prints the device's text output, and whenever a framed audio packet arrives
(KWSWAV01 | len | pcm | checksum), verifies the checksum and saves a WAV.

    pip install pyserial numpy
    python rec_capture.py /dev/ttyUSB0          (Linux)
    python rec_capture.py COM5                  (Windows)

Close idf.py monitor first - only one program can hold the port.
"""

import struct
import sys
import time
import wave

import numpy as np
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = 115200          # ESP-IDF 6.1: the faster setting is not applied on the board side
RATE = 16000
MAGIC = b"KWSWAV01"
STALL_S = 30

ser = serial.Serial(PORT, BAUD, timeout=0.5)
buf = b""


def need(k):
    """Block until buf holds at least k bytes."""
    global buf
    t0 = time.time()
    while len(buf) < k:
        chunk = ser.read(max(k - len(buf), 1))
        if chunk:
            buf += chunk
            t0 = time.time()
        elif time.time() - t0 > STALL_S:
            raise RuntimeError("transport stalled mid-packet")


def take(k):
    global buf
    need(k)
    out, buf = buf[:k], buf[k:]
    return out


print(f"listening on {PORT} @ {BAUD}")
n = 0
while True:
    buf += ser.read(4096)
    i = buf.find(MAGIC)

    if i < 0:
        # Print text, but hold back a tail in case MAGIC is split across reads
        keep = len(MAGIC) - 1
        if len(buf) > keep:
            sys.stdout.write(buf[:-keep].decode(errors="replace"))
            sys.stdout.flush()
            buf = buf[-keep:]
        continue

    sys.stdout.write(buf[:i].decode(errors="replace"))
    buf = buf[i + len(MAGIC):]

    length = struct.unpack("<I", take(4))[0]
    pcm = take(length)
    csum_dev = struct.unpack("<I", take(4))[0]
    csum_host = sum(pcm) & 0xFFFFFFFF

    fname = f"rec_{n:02d}.wav"
    n += 1
    with wave.open(fname, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)

    a = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    dc = a.mean()
    rms = np.sqrt(np.mean((a - dc) ** 2))
    rms_db = 20 * np.log10(rms / 32768 + 1e-12)
    peak = np.max(np.abs(a))

    ok = "OK" if csum_dev == csum_host else "FAIL  <-- transport problem, not mic"
    print(f"\n[host] saved {fname}  {length} bytes  checksum {ok}")
    print(f"[host] peak={peak:.0f}  dc={dc:.1f}  rms={rms_db:.1f} dBFS")

    # Is this real audio? Sound swings both ways around zero, so among the
    # non-zero samples roughly half should be negative. A disconnected or
    # floating data line gives bit patterns (0, 1, 3, 7 ... 511) that are
    # never negative, and very few distinct values.
    nz = a[a != 0]
    neg_share = float(np.mean(nz < 0)) if len(nz) else 0.0
    distinct = len(np.unique(a))
    problems = []
    if len(nz) < 0.01 * len(a):
        problems.append("almost every sample is exactly 0")
    elif not 0.25 <= neg_share <= 0.75:
        problems.append(f"only {neg_share * 100:.0f}% of samples are negative (real audio ~50%)")
    if a.std() > 10 and distinct < 60:
        problems.append(f"only {distinct} distinct values")
    if problems:
        print("[host] !!! WARNING: this does NOT look like real audio - check the mic connection")
        for p_ in problems:
            print(f"[host]     - {p_}")
    else:
        print(f"[host] audio looks real ({neg_share * 100:.0f}% negative, {distinct} distinct values)")
    print()
