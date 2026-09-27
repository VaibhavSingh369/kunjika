"""
Kunjika streaming server - runs on the laptop.

The ESP32 keeps a TCP connection open to this server. When it detects
"Kunjika", it streams the audio that follows (IMA-ADPCM, 4x smaller than raw
PCM). This server:

  - answers the board's pings, and keeps the board's clock synchronised with
    its own (round-trip method, best of the last 20 exchanges)
  - receives and decodes the stream, saves each session as a WAV
  - measures latency exactly as the problem statement defines it:
        keyword end  ->  the server receiving the audio stream
  - transcribes each session with Whisper (open source) and sends the text back
  - appends every session to sessions.csv for statistics

Usage (any normal Python 3.9+, not the ESP-IDF one):
    pip install faster-whisper numpy         (faster-whisper is optional)
    python kws_server.py                     (listens on port 7000)
    python kws_server.py                        (auto: English as English, Hindi -> Hinglish)
    python kws_server.py --style hinglish       (romanised by the English model)
    python kws_server.py --style english        ("Kunjika, open the gate")
    python kws_server.py --model medium         (more accurate, slower)

Allow Python through the Windows firewall when asked (private networks).
"""

import argparse
import asyncio
import csv
import re
import os
import struct
import time
import wave
from pathlib import Path

import numpy as np

from hinglish import normalise_keyword, to_hinglish

# ---------------------------------------------------------------- protocol
# frame = type (u8) | payload length (u16, little endian) | payload
HELLO, PING, SYNC_REPLY = 0x01, 0x02, 0x03
START, AUDIO, END = 0x10, 0x11, 0x12
PONG, SYNC = 0x82, 0x83
FIRST_ACK, TEXT = 0x91, 0x92

CODEC_PCM16, CODEC_ADPCM = 0, 1
MAX_SYNC_ERROR_MS = 20          # beyond this the clock sync (and so latency) is not trustworthy
LATENCIES = []                  # measured latencies this run, for the running summary
SAMPLE_RATE = 16000
END_REASONS = {0: "silence", 1: "max length", 2: "connection"}


def now_us():
    return time.perf_counter_ns() // 1000


def frame(ftype, payload=b""):
    return struct.pack("<BH", ftype, len(payload)) + payload


# ---------------------------------------------------------------- IMA-ADPCM
STEP = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
        50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
        253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
        1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
        3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
        12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767]
INDEX = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]


def adpcm_decode(data, pred, idx, n):
    """Standard IMA-ADPCM, low nibble first. Returns int16 samples."""
    out = np.empty(n, dtype=np.int16)
    k = 0
    for byte in data:
        for code in (byte & 0x0F, byte >> 4):
            if k >= n:
                break
            step = STEP[idx]
            diff = step >> 3
            if code & 4: diff += step
            if code & 2: diff += step >> 1
            if code & 1: diff += step >> 2
            pred = pred - diff if code & 8 else pred + diff
            pred = max(-32768, min(32767, pred))
            idx = max(0, min(88, idx + INDEX[code]))
            out[k] = pred
            k += 1
    return out


def adpcm_encode(x, pred=0, idx=0):
    """Reference encoder (the firmware has the same one in C). Returns bytes,
    final predictor, final index."""
    codes = []
    for s in x.astype(np.int32):
        step = STEP[idx]
        diff = int(s) - pred
        code = 0
        if diff < 0:
            code = 8
            diff = -diff
        vpdiff = step >> 3
        if diff >= step: code |= 4; diff -= step; vpdiff += step
        step >>= 1
        if diff >= step: code |= 2; diff -= step; vpdiff += step
        step >>= 1
        if diff >= step: code |= 1; vpdiff += step
        pred = pred - vpdiff if code & 8 else pred + vpdiff
        pred = max(-32768, min(32767, pred))
        idx = max(0, min(88, idx + INDEX[code]))
        codes.append(code)
    if len(codes) % 2:
        codes.append(0)
    data = bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2))
    return data, pred, idx


# ---------------------------------------------------------------- recogniser
# Whisper copies the style of its prompt: a few romanised examples make it write
# Hinglish, and they also teach it how "Kunjika" is spelled.
HINGLISH_PROMPT = ("Kunjika, light on karo. Kunjika, fan band karo. Kunjika, gate kholo. "
                   "Kunjika, AC ka temperature kam karo. Kunjika, kya time hua hai?")


HINDI_LIKE = {"hi", "ur", "mr", "ne", "pa", "gu", "bn", "sa"}
LANG_CONFIDENT = 0.85      # below this, try both English and Hindi and keep the more confident
MAX_TOKENS = 48            # longest transcript a command can need
MIN_CONFIDENCE = -1.0      # Whisper's average log-probability; below this the text is a guess
NO_SPEECH = 0.6            # Whisper's own "this segment is silence" probability
HINDI_HINT = 0.15          # Hindi probability this high means "check Hindi too", even if English won
SILENCE_PHRASES = {"you", "thank you", "thanks for watching", "thank you for watching", "bye"}
KEYWORD = "Kunjika"


def _sound(w):
    """Rough phonetic key: 'jeeg' / 'jeega' / 'jiga' all become 'jik' / 'jika'."""
    w = w.lower()
    for a, b in (("ee", "i"), ("oo", "u"), ("gh", "k"), ("g", "k"), ("q", "k"), ("c", "k")):
        w = w.replace(a, b)
    return w


# the cut comes AFTER the keyword, so leftovers are its tail; "kunji" (key) is a real word
KEYWORD_PIECES = ("kunjika", "unjika", "njika", "jika", "ika", "ka")


def strip_keyword(text):
    """Drop a leftover of the keyword at the start of the command. The board
    has already confirmed the keyword, so an opening word that SOUNDS like a
    piece of it ("jeeg", "...jika", "ka") is the tail of the keyword itself."""
    import difflib
    m = re.match(r"\s*([^\s,.!?]+)[\s,.!?]*(.*)", text, re.S)
    if not m:
        return text.strip()
    first = _sound(to_hinglish(m.group(1)))
    for piece in KEYWORD_PIECES:
        # short pieces ("ka", "ika") only when exact: "kya", "kab", "kal" are real words
        # fuzzy only between near-equal lengths: "jeeg"~"jika" yes, "kunji" (a real
        # word, "key") vs "kunjika" no
        fuzzy = len(piece) >= 4 and len(first) >= 3 and abs(len(first) - len(piece)) <= 1 and \
            difflib.SequenceMatcher(None, first, piece).ratio() >= 0.8
        if first == piece or fuzzy:
            return m.group(2).strip()
    return text.strip()


class Recognizer:
    def __init__(self, model, lang, style="hinglish", prompt=None, hindi_model=None, threads=None,
                 fast=False):
        self.style = style
        self.fast = fast
        self.lang = None if lang == "auto" else lang
        self.prompt = prompt
        self.hindi = None
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.model = None
            print("faster-whisper not installed: sessions are saved, not transcribed", flush=True)
            return
        # leave cores free for the network side (see asr_worker_init)
        # about one thread per physical core (logical count / 2) is fastest for
        # CTranslate2, and it leaves the rest of the machine responsive
        threads = threads or max(1, (os.cpu_count() or 2) // 2)
        print(f"loading Whisper '{model}' on {threads} CPU threads (first run downloads it)...", flush=True)
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        if hindi_model:
            print(f"loading Hindi model '{hindi_model}'...", flush=True)
            self.hindi = WhisperModel(hindi_model, device="cpu", compute_type="int8", cpu_threads=threads)
        print("Whisper ready", flush=True)

    def transcribe(self, wav_path, start_s=0.0):
        if self.model is None:
            return None
        if self.style == "auto":
            return self.transcribe_auto(wav_path, start_s)
        if self.style == "hinglish":
            kw = dict(language="en", task="transcribe", initial_prompt=self.prompt or HINGLISH_PROMPT)
        elif self.style == "english":
            kw = dict(language=self.lang, task="translate",
                      initial_prompt=self.prompt or "Kunjika, turn on the light.")
        else:                                              # hindi / native script
            kw = dict(language=self.lang or "hi", task="transcribe", initial_prompt=self.prompt)
        segments, _ = self.model.transcribe(str(wav_path), beam_size=5, **kw)
        return " ".join(s.text.strip() for s in segments).strip(), None

    def transcribe_auto(self, wav_path, start_s=0.0):
        """The board already confirmed the keyword, so Whisper hears only the
        command (audio after the keyword end) and "Kunjika," is written in front.

        No example sentences are given to Whisper: when it is unsure it copies
        them ("prompt leakage"). The language is detected on the command itself;
        if the detection is not confident, the command is transcribed both as
        English and as Hindi, and the one Whisper is more confident in wins."""
        with wave.open(str(wav_path)) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        cmd = audio[int(max(0.0, start_s) * SAMPLE_RATE):]
        if len(cmd) < 0.3 * SAMPLE_RATE:
            return KEYWORD, None
        common = dict(beam_size=5, condition_on_previous_text=False, initial_prompt=None,
                      without_timestamps=True,
                      temperature=0.0,              # no re-decoding "fallbacks" (up to 5 extra passes)
                      max_new_tokens=MAX_TOKENS,    # a command is short: never decode ~450 tokens
                      no_repeat_ngram_size=3)       # stops loops like "aap aap aap aap aap"

        if self.fast:
            common["chunk_length"] = 10

        def run(language, model=None):
            try:
                segs, info = (model or self.model).transcribe(cmd, language=language, **common)
                segs = list(segs)
            except Exception as e:                          # e.g. the engine insists on 30 s
                if "chunk_length" not in common:
                    raise
                print(f"  --fast not supported by this Whisper ({type(e).__name__}); "
                      f"using the normal 30 s window", flush=True)
                self.fast = False
                del common["chunk_length"]
                segs, info = (model or self.model).transcribe(cmd, language=language, **common)
            segs = list(segs)
            # "\ufffd": Whisper sometimes splits a Devanagari character across tokens
            text = " ".join(x.text.strip() for x in segs).replace("\ufffd", "").strip()
            # Whisper "hears" words in silence ("you", "Thank you."): trust its own
            # no-speech estimate, and distrust its favourite silence phrases
            if segs and min(getattr(x, "no_speech_prob", 0.0) for x in segs) > NO_SPEECH:
                text = ""
            elif text.lower().strip(" .!?") in SILENCE_PHRASES and \
                    sum(x.avg_logprob for x in segs) / len(segs) < -0.4:
                text = ""
            words = sum(max(1, len(x.text.split())) for x in segs) or 1
            conf = sum(x.avg_logprob * max(1, len(x.text.split())) for x in segs) / words if segs else -99
            return text, conf, info

        # ONE automatic pass detects the language and transcribes from the same
        # encoder run (every pass costs a full 30 s encoder window, so passes are
        # the thing to minimise). A second pass only when the language is unsure.
        text, conf, info = run(None)
        lang = "hi" if info.language in HINDI_LIKE else info.language
        redo_hi = info.language in HINDI_LIKE and info.language != "hi"   # Urdu etc: wrong script
        # Whisper TRANSLATES Hindi when it decides a clip is English. If Hindi
        # (or a close relative) got real probability, treat the language as unsure.
        probs = dict(getattr(info, "all_language_probs", None) or [])
        p_hindi = sum(probs.get(l, 0.0) for l in HINDI_LIKE)
        hindi_hint = lang == "en" and p_hindi >= HINDI_HINT
        if lang not in ("en", "hi"):
            # Commands are English or Hindi. Any other language (Korean, Georgian...)
            # means Whisper was given noise, not speech: try both, trust neither blindly.
            t_en, c_en, _ = run("en")
            t_hi, c_hi, _ = run("hi", self.hindi)
            text, conf, lang = (t_en, c_en, "en") if c_en >= c_hi else (t_hi, c_hi, "hi")
            redo_hi = False
        elif info.language_probability < LANG_CONFIDENT or hindi_hint:
            other = "en" if lang == "hi" else "hi"
            t2, c2, _ = run(other, self.hindi if other == "hi" else None)
            if c2 > conf:
                text, conf, lang = t2, c2, other
                redo_hi = False
            elif lang == "hi" and (redo_hi or self.hindi is not None):
                text, conf, _ = run("hi", self.hindi)
        elif lang == "hi" and (redo_hi or self.hindi is not None):
            text, conf, _ = run("hi", self.hindi)           # Hindi specialist / Devanagari script
        if not text.strip():
            return KEYWORD, None                            # keyword only, no command heard
        if conf < MIN_CONFIDENCE:
            return f"{KEYWORD}, (not understood)", None
        text = strip_keyword(text)
        if lang == "hi":
            native = text
            return f"{KEYWORD}, {to_hinglish(native)}".rstrip(", "), native
        return f"{KEYWORD}, {text}".rstrip(", "), None


# ---------------------------------------------------------------- ASR worker process
# Whisper runs in a separate process at lower priority with CPU cores left
# free. Run inside the server process it starved the network side: audio
# arrived on time but was read (and timestamped) seconds late, which delayed
# acknowledgements and corrupted the latency measurement.
_REC = None


def asr_worker_init(model, lang, style, prompt, hindi_model, threads, fast=False):
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:
        pass
    global _REC
    _REC = Recognizer(model, lang, style, prompt, hindi_model, threads, fast)


def asr_worker_ready():
    return _REC is not None and _REC.model is not None


def asr_worker_run(wav_path, start_s):
    t0 = time.perf_counter()
    result = _REC.transcribe(wav_path, start_s)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------- connection
class Board:
    def __init__(self, reader, writer, args, recognizer):
        self.r, self.w, self.args, self.rec = reader, writer, args, recognizer
        self.name = "?"
        self.sync = []              # (rtt_us, offset_us) with offset = device - server
        self.session = None
        self.tasks = set()          # background transcriptions

    def clock_offset(self):
        """Offset (device clock - server clock) from the fastest recent exchange."""
        if not self.sync:
            return None, None
        rtt, off = min(self.sync[-20:])
        return off, rtt

    async def send(self, ftype, payload=b""):
        self.w.write(frame(ftype, payload))
        await self.w.drain()

    async def syncer(self):
        while True:
            await self.send(SYNC, struct.pack("<Q", now_us()))
            await asyncio.sleep(1.0)

    async def run(self):
        peer = self.w.get_extra_info("peername")
        print(f"\nboard connected from {peer[0]}")
        sync_task = asyncio.create_task(self.syncer())
        try:
            while True:
                head = await self.r.readexactly(3)
                ftype, n = struct.unpack("<BH", head)
                payload = await self.r.readexactly(n) if n else b""
                t_rx = now_us()
                await self.handle(ftype, payload, t_rx)
        except (asyncio.IncompleteReadError, OSError):     # includes WinError 121
            pass
        finally:
            sync_task.cancel()
            if self.session:
                await self.finish(reason=2)
            print(f"board {self.name} disconnected")

    async def handle(self, ftype, p, t_rx):
        if ftype == HELLO:
            ver, rate = struct.unpack_from("<HH", p)
            self.name = p[4:].split(b"\0")[0].decode(errors="replace") or "esp32"
            print(f"hello from '{self.name}' (protocol {ver}, {rate} Hz)")
        elif ftype == PING:
            await self.send(PONG, p[:8] + struct.pack("<Q", now_us()))
        elif ftype == SYNC_REPLY:
            t_srv_sent, t_dev = struct.unpack("<QQ", p[:16])
            rtt = t_rx - t_srv_sent
            self.sync.append((rtt, t_dev - (t_srv_sent + rtt // 2)))
            if len(self.sync) == 5:
                print(f"clock synchronised (best round trip {min(self.sync)[0] / 1000:.1f} ms)")
            self.sync = self.sync[-60:]
        elif ftype == START:
            sid, t_det, t_kw_end, t_first, codec = struct.unpack_from("<IQQQB", p)
            self.session = {"id": sid, "t_det": t_det, "t_kw_end": t_kw_end,
                            "t_first_sample": t_first, "codec": codec, "chunks": [],
                            "bytes": 0, "t_first_rx": None, "t_start_rx": t_rx}
        elif ftype == AUDIO and self.session:
            sid, seq, pred, idx, n = struct.unpack_from("<IIhBxH", p)
            data = p[14:]
            s = self.session
            if s["t_first_rx"] is None:
                s["t_first_rx"] = t_rx
                await self.send(FIRST_ACK, struct.pack("<IQ", sid, t_rx))
                # say it NOW, not when the session ends: this is the moment the
                # problem statement's latency is about
                off, rtt = self.clock_offset()
                if off is not None and rtt / 2000 <= MAX_SYNC_ERROR_MS:
                    if s["t_kw_end"]:
                        print(f"\n>> session {sid}: ASR receiving audio - "
                              f"{(t_rx - (s['t_kw_end'] - off)) / 1000:.0f} ms after the keyword ended")
                    else:
                        print(f"\n>> session {sid}: ASR receiving audio - "
                              f"{(t_rx - (s['t_det'] - off)) / 1000:.0f} ms after detection")
            s["bytes"] += len(p) + 3
            if s["codec"] == CODEC_ADPCM:
                s["chunks"].append(adpcm_decode(data, pred, idx, n))
            else:
                s["chunks"].append(np.frombuffer(data, dtype="<i2")[:n].copy())
        elif ftype == END and self.session:
            _, reason = struct.unpack_from("<IB", p)
            await self.finish(reason)          # returns at once; transcription runs in the background

    async def finish(self, reason):
        s, self.session = self.session, None
        audio = np.concatenate(s["chunks"]) if s["chunks"] else np.zeros(0, np.int16)
        secs = len(audio) / SAMPLE_RATE
        out_dir = Path(self.args.out)
        out_dir.mkdir(exist_ok=True)
        # board session numbers restart at 1 after a reboot: add the time so
        # earlier recordings are never overwritten
        wav_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_session_{s['id']:04d}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.astype("<i2").tobytes())

        off, rtt = self.clock_offset()
        reliable = rtt is not None and rtt / 2000 <= MAX_SYNC_ERROR_MS
        lat = det = net = None
        if off is not None and s["t_first_rx"] is not None:
            t_det_srv = s["t_det"] - off
            det_to_rx = (s["t_first_rx"] - t_det_srv) / 1000
            if s["t_kw_end"]:
                kw_end_srv = s["t_kw_end"] - off
                lat = (s["t_first_rx"] - kw_end_srv) / 1000
                det = (s["t_det"] - s["t_kw_end"]) / 1000
                net = det_to_rx
            else:
                net = det_to_rx
        kbps = s["bytes"] * 8 / secs / 1000 if secs else 0

        print(f"\nsession {s['id']}: {secs:.1f} s of audio, {s['bytes'] / 1024:.1f} KB "
              f"({kbps:.0f} kbit/s vs 256 raw), ended by {END_REASONS.get(reason, reason)}")
        if not reliable:
            print(f"  clock sync unreliable (+/- {rtt / 2000 if rtt else float('nan'):.0f} ms): the laptop or "
                  f"network stalled - latency NOT measured for this session")
            lat = det = net = None
        if lat is not None:
            print(f"  LATENCY keyword end -> server receiving audio: {lat:.0f} ms "
                  f"(detection {det:.0f} ms + send & network {net:.0f} ms)")
            LATENCIES.append(lat)
            v = np.array(LATENCIES)
            print(f"  so far: {len(v)} measured sessions, median {np.median(v):.0f} ms, "
                  f"90th percentile {np.percentile(v, 90):.0f} ms, worst {v.max():.0f} ms")
        elif net is not None:
            print(f"  keyword end not measurable (speech continued straight on); "
                  f"detection -> server receiving audio: {net:.0f} ms")
        if rtt is not None:
            print(f"  clock sync accuracy: +/- {rtt / 2000:.1f} ms")

        # where the command starts inside the saved audio: at the keyword end, or
        # (speech ran straight on) a little before the detection
        t_cmd = s["t_kw_end"] if s["t_kw_end"] else s["t_det"] - 250_000
        start_s = max(0.0, (t_cmd - s["t_first_sample"]) / 1e6 - 0.05)
        print(f"  saved {wav_path} - transcribing in the background")
        # Do NOT await the transcription here: this coroutine runs inside the
        # connection's reader, and waiting would leave the board's next session
        # (and its pings) unread in the socket for the whole transcription.
        task = asyncio.create_task(self.transcribe_and_reply(
            s, wav_path, start_s, secs, kbps, lat, det, net, rtt))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def transcribe_and_reply(self, s, wav_path, start_s, secs, kbps, lat, det, net, rtt):
        try:
            result, asr_s = await asyncio.get_running_loop().run_in_executor(
                self.rec, asr_worker_run, str(wav_path), start_s)
        except (KeyboardInterrupt, asyncio.CancelledError, RuntimeError, OSError):
            return                          # server shutting down: the WAV is saved anyway
        text, native = result if result is not None else (None, None)
        if text is not None:
            print(f"\nsession {s['id']} transcript (took {asr_s:.1f} s, never delays audio or latency):")
            print(f"  transcript: {text!r}" + (f"   (heard as Hindi: {native})" if native else "")
                  + f"   [command from {start_s:.2f} s of the audio]")
        # always reply, even with empty text: it tells the board the session is done
        try:
            await self.send(TEXT, struct.pack("<I", s["id"]) + (text or "").encode()[:400])
        except ConnectionError:
            pass
        new = not Path(self.args.csv).exists()
        with open(self.args.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "session", "audio_s", "kbit_s", "latency_ms", "detection_ms",
                            "send_network_ms", "sync_accuracy_ms", "transcript", "wav"])
            w.writerow([time.strftime("%H:%M:%S"), s["id"], f"{secs:.2f}", f"{kbps:.0f}",
                        "" if lat is None else f"{lat:.1f}", "" if det is None else f"{det:.1f}",
                        "" if net is None else f"{net:.1f}",
                        "" if rtt is None else f"{rtt / 2000:.1f}", text or "", wav_path.name])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--model", default="small", help="Whisper size: tiny, base, small, medium")
    ap.add_argument("--lang", default="auto", help="spoken language for --style english/hindi")
    ap.add_argument("--style", default="auto", choices=["auto", "hinglish", "english", "hindi"],
                    help="auto: English stays English, Hindi is recognised as Hindi then written "
                         "in Hinglish (most accurate); hinglish: English model writes romanised; "
                         "english: translation; hindi: Devanagari")
    ap.add_argument("--prompt", default=None, help="override the example prompt given to Whisper")
    ap.add_argument("--hindi-model", default=None,
                    help="folder of a converted Hindi Whisper model, used for Hindi commands "
                         "(e.g. whisper-hindi-small-ct2, see README step)")
    ap.add_argument("--fast", action="store_true",
                    help="experimental: 10 s Whisper window instead of 30 s (a command is ~3 s); "
                         "falls back automatically if unsupported")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for Whisper (default: half the logical CPUs)")
    ap.add_argument("--out", default="sessions")
    ap.add_argument("--csv", default="sessions.csv")
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    recognizer = ProcessPoolExecutor(
        max_workers=1, initializer=asr_worker_init,
        initargs=(args.model, args.lang, args.style, args.prompt, args.hindi_model, args.threads,
                  args.fast))
    await asyncio.get_running_loop().run_in_executor(recognizer, asr_worker_ready)   # load now
    print(f"transcript style: {args.style}" + (f", Hindi model: {args.hindi_model}" if args.hindi_model else "")
          + (", --fast (10 s window, experimental)" if args.fast else ""))

    async def on_connect(r, w):
        await Board(r, w, args, recognizer).run()

    server = await asyncio.start_server(on_connect, "0.0.0.0", args.port)
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "?"
    print(f"listening on port {args.port} - put this laptop's IP (probably {ip}) "
          f"in net_config.h as SERVER_IP")
    try:
        async with server:
            await server.serve_forever()
    finally:
        recognizer.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nserver stopped - sessions are in sessions.csv")
