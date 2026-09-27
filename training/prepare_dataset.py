"""
Kunjika dataset preparation.

Turns the raw collection folder - any mix of WAV / AAC / MP3 / M4A, any sample
rate, mono or stereo, organised in condition folders or flat with names like
"laoud_3" - into one clean dataset:

  OUT/<speaker>/<condition>/<speaker>_<condition>_<n>.wav   16 kHz mono 16-bit
  OUT/_podcast/podcast_train.wav   first ~2/3 of the podcast (training only)
  OUT/_podcast/podcast_test.wav    last  ~1/3 (held out: false-activation test)
  OUT/manifest.csv                 one row per output file
  OUT/report.txt                   counts per speaker/condition + every warning

Keyword conditions (normal, fast, slow, quiet, loud, lombard) and negatives are
cut to the 1 s window with the most SPEECH-band energy (breath rumble ignored).
freespeech and silence are kept whole; the training script slices them.

Usage (Colab has ffmpeg built in):
    python prepare_dataset.py RAW_DIR OUT_DIR [--test-fraction 0.33]
"""

import argparse
import csv
import difflib
import re
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
CLIP = SR                      # 1 s
AUDIO_EXT = {".wav", ".aac", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".3gp",
             ".amr", ".wma", ".webm"}

KEYWORD_CONDS = ["normal", "fast", "slow", "quiet", "loud", "lombard"]
NEG_CONDS = ["negatives"]
LONG_CONDS = ["freespeech", "silence"]
ALL_CONDS = KEYWORD_CONDS + NEG_CONDS + LONG_CONDS

# spellings seen in the wild -> canonical name
ALIASES = {
    "normal": "normal", "norm": "normal", "nomal": "normal",
    "fast": "fast", "quick": "fast",
    "slow": "slow",
    "quiet": "quiet", "quite": "quiet", "quit": "quiet", "soft": "quiet",
    "whisper": "quiet",
    "loud": "loud", "laoud": "loud", "lound": "loud", "lowd": "loud",
    "lombard": "lombard", "noisy": "lombard", "noise": "lombard",
    "negative": "negatives", "negatives": "negatives", "neg": "negatives",
    "confusable": "negatives", "confusables": "negatives",
    "freespeech": "freespeech", "free": "freespeech", "freespeach": "freespeech",
    "speech": "freespeech", "conversation": "freespeech",
    "silence": "silence", "silent": "silence", "room": "silence",
}
PODCAST_NAMES = {"podcast", "podcastaudio", "podcasts"}


# ------------------------------------------------------------------ helpers
def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def canonical_condition(name):
    """'Normal', 'quite', 'laoud_3', 'free_speech' -> canonical, or None."""
    key = letters(re.sub(r"[\s_\-\(\)\d]+$", "", name))
    if key in ALIASES:
        return ALIASES[key]
    close = difflib.get_close_matches(key, ALIASES.keys(), n=1, cutoff=0.75)
    return ALIASES[close[0]] if close else None


def speaker_id(folder_name):
    s = folder_name.lower()
    s = re.sub(r"[_\-\s]*kunjika[_\-\s]*", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s or "unknown"


def decode(path):
    """Any audio file -> int16 mono at 16 kHz, via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip()[:200])
    return np.frombuffer(r.stdout, dtype="<i2").copy()


def probe_rate(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=sample_rate,channels",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.stdout.strip().replace(",", "Hz x") + "ch" if r.returncode == 0 else "?"


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype("<i2").tobytes())


def speech_band(x):
    """Remove everything below ~80 Hz (breath, rumble) with a moving average."""
    x = x.astype(np.float64)
    return x - np.convolve(x, np.ones(200) / 200, mode="same")


def dbfs(v):
    return 20 * np.log10(max(v, 1e-9) / 32768.0)


def best_second(x):
    """Start index of the 1 s window with the most speech-band energy, plus
    how much of the clip's speech energy that window holds."""
    if len(x) <= CLIP:
        return 0, 1.0
    e = speech_band(x) ** 2
    c = np.concatenate([[0.0], np.cumsum(e)])
    win = c[CLIP:] - c[:-CLIP]
    i = int(np.argmax(win))
    return i, float(win[i] / max(c[-1], 1e-9))


# ------------------------------------------------------------------ main work
def process_speaker(spk_dir, out, rows, warnings, counters):
    spk = speaker_id(spk_dir.name)
    files = sorted(p for p in spk_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        warnings.append(f"{spk_dir.name}: no audio files found")
        return
    numbering = defaultdict(int)

    for f in files:
        rel = f.relative_to(spk_dir)
        # condition from the sub-folder if there is one, else from the file name
        cond = canonical_condition(rel.parts[0]) if len(rel.parts) > 1 else None
        if cond is None:
            cond = canonical_condition(f.stem)
        if cond is None:
            warnings.append(f"{spk}: cannot tell condition of '{rel}' - skipped")
            continue

        try:
            x = decode(f)
        except RuntimeError as e:
            warnings.append(f"{spk}: cannot decode '{rel}': {e}")
            continue

        dur = len(x) / SR
        peak = int(np.abs(x.astype(np.int32)).max()) if len(x) else 0
        speech_rms = float(np.sqrt(np.mean(speech_band(x) ** 2))) if len(x) else 0.0
        clipped = float(np.mean(np.abs(x.astype(np.int32)) >= 32767)) if len(x) else 0.0
        flags = []
        if clipped > 0.001:
            flags.append(f"clipped {clipped * 100:.1f}%")
        if f.suffix.lower() in {".aac", ".mp3", ".m4a", ".ogg", ".opus", ".amr", ".3gp"}:
            flags.append("lossy")

        if cond in KEYWORD_CONDS + NEG_CONDS:
            if dur < 0.4:
                warnings.append(f"{spk}/{cond}: '{rel}' only {dur:.2f} s - skipped")
                continue
            if cond != "silence" and dbfs(speech_rms) < -60:
                warnings.append(f"{spk}/{cond}: '{rel}' is near-silent "
                                f"({dbfs(speech_rms):.0f} dBFS) - skipped")
                continue
            start, share = best_second(x)
            clip = x[start:start + CLIP]
            if len(clip) < CLIP:                     # short file: centre it
                pad = CLIP - len(clip)
                clip = np.pad(clip, (pad // 2, pad - pad // 2))
            if share < 0.80 and cond in KEYWORD_CONDS:
                flags.append(f"only {share * 100:.0f}% of speech inside 1 s")
            numbering[cond] += 1
            dst = out / spk / cond / f"{spk}_{cond}_{numbering[cond]:03d}.wav"
            write_wav(dst, clip)
            label = 1 if cond in KEYWORD_CONDS else 0
        else:                                         # freespeech, silence
            numbering[cond] += 1
            dst = out / spk / cond / f"{spk}_{cond}_{numbering[cond]:03d}.wav"
            write_wav(dst, x)
            label = 0

        counters[spk][cond] += 1
        if "lossy" in flags:
            counters[spk]["_lossy"] += 1
        serious = [fl for fl in flags if fl != "lossy"]
        if serious:
            warnings.append(f"{spk}/{cond}: '{rel}' - " + ", ".join(serious))
        rows.append({"speaker": spk, "condition": cond, "label": label,
                     "kind": "long" if cond in LONG_CONDS else "clip",
                     "path": str(dst.relative_to(out)), "source": str(f.name),
                     "duration_s": f"{len(x) / SR:.2f}",
                     "peak_dbfs": f"{dbfs(peak):.1f}",
                     "speech_rms_dbfs": f"{dbfs(speech_rms):.1f}",
                     "flags": "; ".join(flags)})


def process_podcast(pod_dir, out, rows, warnings, test_fraction):
    files = sorted(p for p in pod_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        warnings.append("podcast folder has no audio files")
        return 0.0, 0.0
    parts = []
    for f in files:
        try:
            parts.append(decode(f))
        except RuntimeError as e:
            warnings.append(f"podcast: cannot decode '{f.name}': {e}")
    if not parts:
        return 0.0, 0.0
    x = np.concatenate(parts)
    cut = int(len(x) * (1 - test_fraction))
    cut -= cut % SR
    train, test = x[:cut], x[cut:]
    write_wav(out / "_podcast" / "podcast_train.wav", train)
    write_wav(out / "_podcast" / "podcast_test.wav", test)
    for name, part, use in (("podcast_train.wav", train, "train"),
                            ("podcast_test.wav", test, "test")):
        rows.append({"speaker": "_podcast", "condition": f"podcast_{use}", "label": 0,
                     "kind": "long", "path": f"_podcast/{name}",
                     "source": ", ".join(f.name for f in files),
                     "duration_s": f"{len(part) / SR:.1f}", "peak_dbfs": "",
                     "speech_rms_dbfs": "", "flags": ""})
    return len(train) / SR / 60, len(test) / SR / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--test-fraction", type=float, default=0.33,
                    help="share of the podcast held out for testing (default 0.33)")
    a = ap.parse_args()
    raw, out = Path(a.raw_dir), Path(a.out_dir)
    if not raw.is_dir():
        sys.exit(f"not a folder: {raw}")
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        sys.exit("ffmpeg not found (Colab has it; on Windows install it first)")
    out.mkdir(parents=True, exist_ok=True)

    rows, warnings = [], []
    counters = defaultdict(lambda: defaultdict(int))
    pod_train = pod_test = 0.0
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        if letters(d.name) in PODCAST_NAMES:
            print(f"podcast   : {d.name}")
            pod_train, pod_test = process_podcast(d, out, rows, warnings, a.test_fraction)
        else:
            print(f"speaker   : {d.name} -> {speaker_id(d.name)}")
            process_speaker(d, out, rows, warnings, counters)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)

    # ---------------- report
    lines = []
    head = f"{'speaker':12s}" + "".join(f"{c[:8]:>9s}" for c in ALL_CONDS)
    lines += ["files per speaker and condition", head, "-" * len(head)]
    for spk in sorted(counters):
        lines.append(f"{spk:12s}" + "".join(f"{counters[spk].get(c, 0):9d}" for c in ALL_CONDS))
    pos = sum(counters[s][c] for s in counters for c in KEYWORD_CONDS)
    neg = sum(counters[s]["negatives"] for s in counters)
    free = sum(float(r["duration_s"]) for r in rows if r["condition"] == "freespeech")
    sil = sum(float(r["duration_s"]) for r in rows if r["condition"] == "silence")
    lines += ["",
              f"speakers              : {len(counters)}",
              f"keyword clips (1 s)   : {pos}",
              f"confusable negatives  : {neg}",
              f"free speech           : {free / 60:.1f} min",
              f"silence               : {sil:.0f} s",
              f"podcast train / test  : {pod_train:.1f} / {pod_test:.1f} min",
              ""]
    lossy = {s: counters[s]["_lossy"] for s in counters if counters[s].get("_lossy")}
    if lossy:
        lines.append("lossy-format files (usable; prefer WAV next time): " +
                     ", ".join(f"{s} {n}" for s, n in sorted(lossy.items())))
        lines.append("")
    lines += [f"warnings ({len(warnings)}):"]
    lines += ["  " + w for w in warnings] or ["  none"]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
