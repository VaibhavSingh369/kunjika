"""
Synthetic training data for Kunjika, with Piper TTS (open source, runs on CPU).

Generates
  positives   "Kunjika" in several spellings, many voices, three speeds
  names       Indian names that sound close to Kunjika, as isolated words
  sentences   the same names inside short Hindi / Hinglish sentences, cut into
              1 s windows (false activations happen in running speech)

Voices are chosen from Piper's official voice list when the script runs:
every Hindi voice (Devanagari text, authentic pronunciation) plus a large
multi-speaker English model (hundreds of distinct vocal tracts).

~15% of voices are held out entirely -> OUT_eval/, never used for training.
The trainer uses them to report how often each name still fires on voices it
has never heard.

Output (16 kHz mono 16-bit, 1 s clips)
  OUT/positives/kunjika/*.wav
  OUT/negatives/names/<name>/*.wav
  OUT/negatives/sentences/<name>/*.wav
  OUT_eval/...            same layout, held-out voices
  OUT/tts_manifest.csv    every file: kind, word, voice, speaker, speed

Colab:
    !pip install -q piper-tts
    !python gen_tts.py --out /content/tts
"""

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
import wave
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SR = 16000
CLIP = SR
SEED = 1337
MIN_COVERAGE = 0.70

KUNJIKA_ROMAN = ["Kunjika", "Koonjika", "Kunjeeka", "Kunjikaa", "Kunnjika"]
KUNJIKA_DEVA = ["कुंजिका", "कुन्जिका"]

# name -> Devanagari
NAMES = {
    "sangeeta": "संगीता", "sujeeta": "सुजीता", "babita": "बबीता", "kavita": "कविता",
    "sunita": "सुनीता", "anita": "अनीता", "geeta": "गीता", "neeta": "नीता",
    "reeta": "रीता", "seeta": "सीता", "savita": "सविता", "lalita": "ललिता",
    "mamta": "ममता", "ranjita": "रंजीता", "manjita": "मंजीता", "poonita": "पूनीता",
    "vinita": "विनीता", "vanita": "वनिता", "namita": "नमिता", "sumita": "सुमिता",
    "shweta": "श्वेता", "chetna": "चेतना", "kanika": "कनिका", "monika": "मोनिका",
    "deepika": "दीपिका", "radhika": "राधिका", "ambika": "अंबिका", "kritika": "कृतिका",
    "mallika": "मल्लिका", "sarika": "सारिका", "nikita": "निकिता", "ankita": "अंकिता",
}
SENT_ROMAN = ["{n} ghar gayi hai", "mera naam {n} hai", "{n} kal aayegi",
              "kya {n} yahan hai", "{n} ko bulao"]
SENT_DEVA = ["{n} घर गई है", "मेरा नाम {n} है", "{n} कल आएगी",
             "क्या {n} यहाँ है", "{n} को बुलाओ"]

SPEEDS = [0.9, 1.0, 1.12]          # Piper length_scale: >1 is slower


# ---------------------------------------------------------------- voices
def voice_list():
    from piper import download_voices as dv
    with urllib.request.urlopen(dv.VOICES_JSON) as r:
        return json.load(r)


def choose_voices(catalog, n_english_single):
    hindi = sorted(k for k, v in catalog.items()
                   if v.get("language", {}).get("family") == "hi")
    multi = sorted((k for k, v in catalog.items()
                    if v.get("language", {}).get("family") == "en"
                    and v.get("num_speakers", 1) > 50),
                   # medium first: "high" models are several times slower on CPU
                   key=lambda k: (catalog[k].get("quality") != "medium",
                                  -catalog[k]["num_speakers"]))[:1]
    single = sorted(k for k, v in catalog.items()
                    if v.get("language", {}).get("family") == "en"
                    and v.get("num_speakers", 1) == 1 and v.get("quality") == "medium")
    rng = np.random.default_rng(SEED)
    single = [str(s) for s in rng.permutation(single)[:n_english_single]]
    return hindi, multi, single


def load_voice(name, voices_dir):
    from piper import PiperVoice
    voices_dir.mkdir(parents=True, exist_ok=True)
    model = voices_dir / f"{name}.onnx"
    if not model.exists():
        subprocess.run([sys.executable, "-m", "piper.download_voices", name,
                        "--download-dir", str(voices_dir)], check=True)
    return PiperVoice.load(str(model))


def synth(voice, text, speaker, speed, rng):
    from piper import SynthesisConfig
    cfg = SynthesisConfig(speaker_id=speaker, length_scale=speed,
                          noise_scale=float(rng.uniform(0.5, 0.8)),
                          noise_w_scale=float(rng.uniform(0.6, 1.0)))
    chunks = list(voice.synthesize(text, syn_config=cfg))
    if not chunks:
        return np.zeros(0, np.float32)
    x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float64)
    sr = chunks[0].sample_rate
    if sr != SR:
        g = np.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    return x


# ---------------------------------------------------------------- audio helpers
def to_int16(x, peak_dbfs):
    x = x / (np.max(np.abs(x)) + 1e-9) * 10 ** (peak_dbfs / 20)
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def speech_band(x):
    x = x.astype(np.float64)
    return x - np.convolve(x, np.ones(200) / 200, mode="same")


def best_second(x):
    """1 s window with the most speech energy, and the share of energy in it."""
    if len(x) <= CLIP:
        pad = CLIP - len(x)
        return np.pad(x, (pad // 2, pad - pad // 2)), 1.0
    e = speech_band(x) ** 2
    c = np.concatenate([[0.0], np.cumsum(e)])
    win = c[CLIP:] - c[:-CLIP]
    i = int(np.argmax(win))
    return x[i:i + CLIP], float(win[i] / max(c[-1], 1e-12))


def windows(x, hop=8000):
    if len(x) < CLIP:
        x = np.pad(x, (0, CLIP - len(x)))
    return [x[i:i + CLIP] for i in range(0, len(x) - CLIP + 1, hop)]


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype("<i2").tobytes())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices-dir", default="piper_voices")
    ap.add_argument("--multi-speakers", type=int, default=120,
                    help="speakers sampled from the multi-speaker English model")
    ap.add_argument("--english-single", type=int, default=6,
                    help="extra single-speaker English voices")
    ap.add_argument("--sentence-speakers", type=int, default=25,
                    help="multi-speaker voices used for sentences (they are long)")
    ap.add_argument("--eval-share", type=float, default=0.15)
    a = ap.parse_args()

    out, out_eval = Path(a.out), Path(str(a.out).rstrip("/") + "_eval")
    rng = np.random.default_rng(SEED)
    catalog = voice_list()
    hindi, multi, single = choose_voices(catalog, a.english_single)
    print(f"Hindi voices          : {hindi or 'none found'}")
    print(f"multi-speaker English : {multi}")
    print(f"single-speaker English: {single}")

    # (voice name, speaker id or None, Hindi?)
    jobs = [(v, None, True) for v in hindi] + [(v, None, False) for v in single]
    for v in multi:
        n = catalog[v]["num_speakers"]
        for s in rng.permutation(n)[:a.multi_speakers]:
            jobs.append((v, int(s), False))

    rows, counts = [], {"positives": 0, "names": 0, "sentences": 0, "dropped": 0}
    loaded = {}
    multi_sentence_used = 0
    for j, (vname, spk, is_hi) in enumerate(jobs):
        if vname not in loaded:
            try:
                loaded[vname] = load_voice(vname, Path(a.voices_dir))
            except Exception as e:                      # keep going without it
                print(f"  cannot load {vname}: {e}")
                loaded[vname] = None
        voice = loaded[vname]
        if voice is None:
            continue
        vid = vname + (f"-s{spk}" if spk is not None else "")
        held_out = (zlib.crc32(vid.encode()) % 1000) / 1000 < a.eval_share
        root = out_eval if held_out else out
        # single voices are few: use every speed; multi-speaker voices: one each
        speeds = SPEEDS if spk is None else [SPEEDS[rng.integers(len(SPEEDS))]]

        def emit(kind, word, text, speed, clip_dir, sentence=False):
            x = synth(voice, text, spk, speed, rng)
            if len(x) == 0:
                return
            x = to_int16(x, float(rng.uniform(-12, -3)))
            parts = windows(x) if sentence else [best_second(x)]
            for k, item in enumerate(parts):
                clip, cov = (item, 1.0) if sentence else item
                if kind == "positives" and cov < MIN_COVERAGE:
                    counts["dropped"] += 1
                    continue
                name = f"{vid}_{speed}_{len(rows):06d}" + (f"_w{k}" if sentence else "") + ".wav"
                path = root / clip_dir / name
                write_wav(path, clip)
                rows.append({"path": str(path), "set": "eval" if held_out else "train",
                             "kind": kind, "word": word, "voice": vname,
                             "speaker": "" if spk is None else spk, "speed": speed})
                counts[kind] += 1

        for speed in speeds:
            for text in (KUNJIKA_DEVA if is_hi else KUNJIKA_ROMAN):
                emit("positives", "kunjika", text, speed, Path("positives/kunjika"))
            for word, deva in NAMES.items():
                emit("names", word, deva if is_hi else word.capitalize(), speed,
                     Path("negatives/names") / word)

        # sentences are long: every Hindi / single voice, but only the first
        # --sentence-speakers speakers of the multi-speaker model
        do_sentences = spk is None or multi_sentence_used < a.sentence_speakers
        if spk is not None and do_sentences:
            multi_sentence_used += 1
        if do_sentences:
            for word, deva in NAMES.items():
                tmpl = SENT_DEVA if is_hi else SENT_ROMAN
                t = tmpl[rng.integers(len(tmpl))]
                emit("sentences", word, t.format(n=deva if is_hi else word.capitalize()),
                     1.0, Path("negatives/sentences") / word, sentence=True)

        if (j + 1) % 20 == 0 or j == len(jobs) - 1:
            print(f"  {j + 1}/{len(jobs)} voices | positives {counts['positives']}, "
                  f"names {counts['names']}, sentence windows {counts['sentences']}")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tts_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)
    n_eval = sum(r["set"] == "eval" for r in rows)
    print(f"\ndone: {len(rows)} clips ({n_eval} from held-out voices in {out_eval})")
    print(f"      positives {counts['positives']} (dropped {counts['dropped']} too long), "
          f"names {counts['names']}, sentence windows {counts['sentences']}")


if __name__ == "__main__":
    main()
