"""
Kunjika wake-word training.

Inputs
  --clean   folder produced by prepare_dataset.py (manifest.csv, speakers, _podcast/)
  --room    optional: room-silence WAVs recorded through the INMP441 (16 kHz mono)
  --tts         optional: gen_tts.py output (positives/, negatives/names, negatives/sentences)
  --tts-eval    held-out TTS voices (default: <tts>_eval) - per-name evaluation only
  --names-test  optional: REAL recordings of the names, <person>/<name>/*.<any format>,
                evaluation only - never trained on
  Conversation recordings: put them through prepare_dataset.py as speaker folders
  named conv_train and conv_test (each with a freespeech/ subfolder). conv_train
  is used for training; conv_test is held out and reported separately.
  --sc      Speech Commands folder (downloaded automatically if missing)

Outputs (in --out)
  model.tflite          int8 model for the ESP32
  model.features.json   feature settings; gen_step4.py reads this
  report.txt            accuracy, hit rate and false activations per threshold

The evaluation replays audio through the SAME decision logic as the firmware
(inference every 100 ms, mean of 3 scores, threshold, 1 s lockout, re-arm
below 0.5), so the table it prints is what the board will do.

Colab:
    !pip install -q tensorflow-model-optimization tf_keras
    (Runtime -> Restart session)
    !python train_kunjika.py --clean /content/kunjika_clean --room /content/room_silence
"""

import os

# tensorflow_model_optimization needs Keras 2: switch before importing TF
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import csv
import glob
import json
import pathlib
import re
import tarfile
import urllib.request
import wave
import zlib

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------- features
# Written to model.features.json; gen_step4.py builds the device mel table
# from it, so the firmware always matches the model.
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
FRAME_LEN = 512
FRAME_HOP = 320
NUM_FRAMES = 49
NUM_MEL = 40
FFT_SIZE = 512
MEL_LO_HZ = 60.0      # was 20: skips the breath/air rumble measured on the INMP441
MEL_HI_HZ = 7600.0
LOG_FLOOR = 1e-6

# ---------------------------------------------------------------- data
MIN_COVERAGE = 0.70   # drop keyword clips with < 70% of the word inside 1 s
SC_NEG_TRAIN = 15000  # Speech Commands clips used as generic word negatives
SC_NEG_VAL = 1500
LONG_HOP = 8000       # 0.5 s step when cutting long audio into 1 s windows

# ---------------------------------------------------------------- augmentation
SHIFT_MS = 100
GAIN_DB = (-20.0, 6.0)        # down to -20 dB: quieter, far-field speech
NOISE_PROB = 0.8
SNR_DB = (5.0, 20.0)
NOISE_ONLY_PROB = 0.1
REVERB_PROB = 0.5
RT60 = (0.15, 0.7)            # seconds

# ---------------------------------------------------------------- batch composition
# Each training batch is drawn from these streams in these proportions. v1 put
# the ~70 confusable clips in the same pool as 15,000 Speech Commands words, so
# batches almost never contained one: confusables then fired 67.5% of the time.
# Hard negatives and same-channel speech now get guaranteed shares.
W_POSITIVE = 0.22      # real Kunjika recordings
W_TTS_POSITIVE = 0.08  # synthetic Kunjika: extra voices, but real voices stay dominant
W_CONFUSABLE = 0.04    # real kunji / kunji ka / ... (v2 showed a big share hurts)
W_NAMES = 0.12         # synthetic names (Ranjita, Kanika, ...) alone and in sentences
W_WORDS = 0.18         # Speech Commands words
W_SAME_CHANNEL = 0.16  # speakers' free speech, conv_train, silence: same phones and rooms
W_OTHER_AUDIO = 0.20   # podcast training part + room tone

# ---------------------------------------------------------------- firmware decision logic
INFER_EVERY = 5               # hops -> every 100 ms
SMOOTH_N = 3
LOCKOUT_INFER = 10            # 1000 ms
REARM_BELOW = 0.50
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

SC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


# ================================================================ audio I/O
def read_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz mono 16-bit "
                             f"(got {w.getframerate()} Hz, {w.getnchannels()} ch)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


def decode_any(path):
    """Any audio format -> 16 kHz mono int16 (ffmpeg; Colab has it)."""
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar",
                        str(SAMPLE_RATE), "-f", "s16le", "-"], capture_output=True)
    if r.returncode != 0:
        raise ValueError(r.stderr.decode(errors="replace")[:200])
    return np.frombuffer(r.stdout, dtype="<i2").copy()


def best_second(x):
    """1 s window with the most speech-band energy (breath rumble ignored)."""
    if len(x) <= CLIP_SAMPLES:
        pad = CLIP_SAMPLES - len(x)
        return np.pad(x, (pad // 2, pad - pad // 2))
    xf = x.astype(np.float64)
    e = (xf - np.convolve(xf, np.ones(200) / 200, mode="same")) ** 2
    c = np.concatenate([[0.0], np.cumsum(e)])
    i = int(np.argmax(c[CLIP_SAMPLES:] - c[:-CLIP_SAMPLES]))
    return x[i:i + CLIP_SAMPLES]


def canonical_name(folder, known):
    import difflib
    key = re.sub(r"[^a-z]", "", folder.lower())
    if key in known:
        return key
    m = difflib.get_close_matches(key, known, n=1, cutoff=0.75)
    return m[0] if m else key


def windows(x, hop=LONG_HOP):
    if len(x) < CLIP_SAMPLES:
        x = np.pad(x, (0, CLIP_SAMPLES - len(x)))
    return np.stack([x[i:i + CLIP_SAMPLES]
                     for i in range(0, len(x) - CLIP_SAMPLES + 1, hop)])


# ================================================================ data sources
def read_manifest(clean):
    rows = list(csv.DictReader(open(clean / "manifest.csv")))
    for r in rows:
        m = re.search(r"only (\d+)% of speech", r.get("flags", ""))
        r["coverage"] = int(m.group(1)) / 100 if m else 1.0
        r["full"] = str(clean / r["path"])
    return rows


def choose_split(rows, n_test, n_val, test_arg, val_arg, final):
    spk = sorted({r["speaker"] for r in rows if not r["speaker"].startswith("_")
                  and not r["speaker"].startswith("convtest")})
    if final:
        return spk, [], []
    if test_arg:
        test = [s.strip() for s in test_arg.split(",")]
        val = [s.strip() for s in val_arg.split(",")] if val_arg else []
    else:
        # prefer speakers with a complete set for testing
        def has(s, cond):
            return any(r["speaker"] == s and r["condition"] == cond for r in rows)
        complete = [s for s in spk if has(s, "negatives") and has(s, "freespeech")
                    and not s.startswith("conv")]
        rng = np.random.default_rng(SEED)
        pick = [str(x) for x in rng.permutation(complete)]
        test, val = pick[:n_test], pick[n_test:n_test + n_val]
    for s in test + val:
        if s not in spk:
            raise SystemExit(f"unknown speaker '{s}'; have {spk}")
    train = [s for s in spk if s not in test and s not in val]
    return train, val, test


def speech_commands(sc_dir):
    sc_dir = pathlib.Path(sc_dir)
    if not sc_dir.exists():
        sc_dir.mkdir(parents=True)
        print("downloading Speech Commands v0.02 (~2.3 GB)...")
        tgz = "sc.tar.gz"
        urllib.request.urlretrieve(SC_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(sc_dir)
        os.remove(tgz)
    words = [d for d in sc_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    files = sorted(str(f) for d in words for f in d.glob("*.wav"))
    train, val = [], []
    for f in files:
        bucket = zlib.crc32(os.path.basename(f).split("_")[0].encode()) % 100
        (train if bucket < 80 else val if bucket < 90 else []).append(f)
    rng = np.random.default_rng(SEED)
    train = list(rng.choice(train, min(SC_NEG_TRAIN, len(train)), replace=False))
    val = list(rng.choice(val, min(SC_NEG_VAL, len(val)), replace=False))
    bg = sorted(glob.glob(str(sc_dir / "_background_noise_" / "*.wav")))
    return train, val, bg


def load_noise(bg_files, room_train):
    parts = []
    for f in bg_files:
        try:
            parts.append(read_wav(f).astype(np.float32) / 32768.0)
        except ValueError as e:
            print(f"  skipping noise file: {e}")
    if len(room_train):
        parts.append(room_train.astype(np.float32) / 32768.0)
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


# ================================================================ features
_MEL = None


def mel_matrix():
    """Created once, outside any tf.data graph (init_scope), so every pipeline
    can share it."""
    global _MEL
    if _MEL is None:
        with tf.init_scope():
            _MEL = tf.signal.linear_to_mel_weight_matrix(
                NUM_MEL, FFT_SIZE // 2 + 1, SAMPLE_RATE, MEL_LO_HZ, MEL_HI_HZ)
    return _MEL


def log_mel(wav):
    stft = tf.signal.stft(wav, frame_length=FRAME_LEN, frame_step=FRAME_HOP,
                          fft_length=FFT_SIZE, window_fn=tf.signal.hann_window,
                          pad_end=False)
    return tf.math.log(tf.matmul(tf.abs(stft) ** 2, mel_matrix()) + LOG_FLOOR)


def to_features(wav):
    return tf.reshape(log_mel(wav), [NUM_FRAMES, NUM_MEL, 1])


def load_wave(path):
    wav, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1,
                                 desired_samples=CLIP_SAMPLES)
    return tf.squeeze(wav, -1)


# ================================================================ augmentation
def rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)) + 1e-10)


def shift_wave(wav, shift):
    s = SHIFT_MS * SAMPLE_RATE // 1000
    return tf.pad(wav, [[s, s]])[s - shift: s - shift + CLIP_SAMPLES]


def reverb(wav):
    """Synthetic room: direct sound plus an exponentially decaying tail."""
    n = 8000
    rt60 = tf.random.uniform([], RT60[0], RT60[1])
    t = tf.range(n, dtype=tf.float32) / SAMPLE_RATE
    tail = tf.random.normal([n]) * tf.exp(-6.9 * t / rt60)
    tail = tail / (tf.norm(tail) + 1e-9) * tf.random.uniform([], 0.2, 1.0)
    rir = tf.concat([[1.0], tail[1:]], 0)
    L = 32768
    y = tf.signal.irfft(tf.signal.rfft(tf.pad(wav, [[0, L - CLIP_SAMPLES]])) *
                        tf.signal.rfft(tf.pad(rir, [[0, L - n]])))[:CLIP_SAMPLES]
    return y * rms(wav) / rms(y)


def make_augment(noise, positive, shift=True):
    noise = tf.constant(noise)
    have_noise = noise.shape[0] > CLIP_SAMPLES
    s = SHIFT_MS * SAMPLE_RATE // 1000

    def aug(wav):
        if shift:
            wav = shift_wave(wav, tf.random.uniform([], -s, s + 1, dtype=tf.int32))
        wav = tf.where(tf.random.uniform([]) < REVERB_PROB, reverb(wav), wav)
        wav = wav * tf.pow(10.0, tf.random.uniform([], GAIN_DB[0], GAIN_DB[1]) / 20.0)
        if have_noise:
            off = tf.random.uniform([], 0, noise.shape[0] - CLIP_SAMPLES, dtype=tf.int32)
            seg = noise[off: off + CLIP_SAMPLES]
            snr = tf.random.uniform([], SNR_DB[0], SNR_DB[1])
            mixed = wav + seg * (rms(wav) / rms(seg)) / tf.pow(10.0, snr / 20.0)
            wav = tf.where(tf.random.uniform([]) < NOISE_PROB, mixed, wav)
            if not positive:
                level = tf.pow(10.0, tf.random.uniform([], -50.0, -15.0) / 20.0)
                pure = seg * (level / rms(seg))
                wav = tf.where(tf.random.uniform([]) < NOISE_ONLY_PROB, pure, wav)
        return tf.clip_by_value(wav, -1.0, 1.0)
    return aug


# ================================================================ datasets
def clip_ds(paths, aug):
    ds = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed=SEED).repeat()
    ds = ds.map(load_wave, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)


def window_ds(wins, aug):
    ds = tf.data.Dataset.from_tensor_slices(wins).shuffle(len(wins), seed=SEED).repeat()
    ds = ds.map(lambda w: tf.cast(w, tf.float32) / 32768.0, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)


def train_dataset(parts, noise, batch):
    """parts: (data, is_paths, label, weight). Empty parts are skipped and the
    remaining weights renormalised."""
    streams, weights = [], []
    for data, is_paths, label, weight in parts:
        if len(data) == 0:
            continue
        if is_paths:
            ds = clip_ds(data, make_augment(noise, label == 1))
        else:
            ds = window_ds(data, make_augment(noise, False, shift=False))
        streams.append(ds.map(lambda x, l=label: (to_features(x), l)))
        weights.append(weight)
    w = np.array(weights) / sum(weights)
    ds = tf.data.Dataset.sample_from_datasets(streams, weights=list(w), seed=SEED)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def eval_dataset(paths, labels, wins, batch=128):
    a = tf.data.Dataset.from_tensor_slices((paths, labels)).map(
        lambda p, l: (to_features(load_wave(p)), l), num_parallel_calls=tf.data.AUTOTUNE)
    if len(wins):
        b = tf.data.Dataset.from_tensor_slices(wins).map(
            lambda w: (to_features(tf.cast(w, tf.float32) / 32768.0), 0),
            num_parallel_calls=tf.data.AUTOTUNE)
        a = a.concatenate(b)
    return a.batch(batch).prefetch(tf.data.AUTOTUNE)


# ================================================================ model
def ds_cnn(n_ch=32, n_blocks=3, batch_size=None):
    """The architecture proven on the ESP32 (6.0 ms, 16.4 KB): 4x4/4 patch first
    layer, 3 depthwise-separable blocks, average pool, logits out."""
    L = tf.keras.layers
    inp = L.Input(shape=(NUM_FRAMES, NUM_MEL, 1), batch_size=batch_size)
    x = L.Conv2D(n_ch, (4, 4), strides=(4, 4), padding="valid", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for _ in range(n_blocks):
        x = L.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.Conv2D(n_ch, (1, 1), padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    x = L.AveragePooling2D(pool_size=x.shape[1:3])(x)
    x = L.Reshape((n_ch,))(x)
    x = L.Dropout(0.2)(x)
    return tf.keras.Model(inp, L.Dense(2)(x))


def compile_model(m, lr):
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m


def to_int8(model, rep_ds, path):
    def rep():
        for f, _ in rep_ds.unbatch().batch(1).take(300):
            yield [tf.cast(f, tf.float32)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    blob = c.convert()
    open(path, "wb").write(blob)
    return blob


# ================================================================ device-identical evaluation
class Scorer:
    """Scores audio exactly as the firmware does."""

    def __init__(self, path):
        self.it = tf.lite.Interpreter(
            model_path=str(path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.s, self.z = self.inp["quantization"]
        self.os, self.oz = self.out["quantization"]
        self.ops = sorted({d["op_name"] for d in self.it._get_ops_details()})

    def prob(self, q):
        self.it.set_tensor(self.inp["index"], q.reshape(self.inp["shape"]))
        self.it.invoke()
        y = (self.it.get_tensor(self.out["index"]).reshape(-1).astype(np.float64) - self.oz) * self.os
        if "SOFTMAX" in self.ops:
            return float(y[1])
        return float(1.0 / (1.0 + np.exp(-(y[1] - y[0]))))

    def quantize(self, feats):
        return np.clip(np.round(feats / self.s) + self.z, -128, 127).astype(np.int8)

    def stream(self, x_int16):
        """Scores every 100 ms, framed exactly like the device: 192 zeros in
        front align TF's frames with the firmware's 512-sample history."""
        xp = np.concatenate([np.zeros(FRAME_LEN - FRAME_HOP, np.int16), x_int16])
        f = log_mel(tf.constant(xp.astype(np.float32) / 32768.0)).numpy()
        q = self.quantize(f)
        return [self.prob(q[e - NUM_FRAMES + 1: e + 1])
                for e in range(NUM_FRAMES - 1 + INFER_EVERY - 1, len(q), INFER_EVERY)]


def detections(probs, thr):
    return len(detection_times(probs, thr))


def infer_time(k):
    """Seconds into the audio at which inference k's window ends."""
    e = NUM_FRAMES - 1 + INFER_EVERY - 1 + INFER_EVERY * k
    return (FRAME_HOP * (e + 1) - (FRAME_LEN - FRAME_HOP)) / SAMPLE_RATE


def detection_times(probs, thr):
    """Firmware decision logic: mean of 3, threshold, lockout, re-arm."""
    recent, lock, armed, hits = [], 0, True, []
    for k, p in enumerate(probs):
        recent = (recent + [p])[-SMOOTH_N:]
        sm = sum(recent) / len(recent)
        if lock > 0:
            lock -= 1
        if not armed and sm < REARM_BELOW:
            armed = True
        if armed and lock == 0 and len(recent) == SMOOTH_N and sm >= thr:
            hits.append(infer_time(k))
            lock, armed = LOCKOUT_INFER, False
    return hits


def in_context(clip, pad):
    """A 1 s clip with ~1.2 s of room tone either side, as the mic hears it."""
    return np.concatenate([pad[:19200], clip, pad[19200:38400]]).astype(np.int16)


def evaluate(scorer, test_pos, test_conf, long_sets, pad, pos_conditions=None, listen=None,
             names_eval=None):
    lines = []
    pos_scores = [scorer.stream(in_context(read_wav(p), pad)) for p in test_pos]
    conf_scores = [scorer.stream(in_context(read_wav(p), pad)) for p in test_conf]
    long_scores = {name: (scorer.stream(x), len(x) / SAMPLE_RATE / 3600)
                   for name, x in long_sets.items() if len(x) > CLIP_SAMPLES}

    head = f"{'threshold':>9s} {'hit rate':>10s} {'confusable':>11s}"
    head += "".join(f" {name[:18]:>19s}" for name in long_scores)
    lines += ["device-identical streaming evaluation", "",
              f"hit rate   : {len(pos_scores)} keyword clips from unseen speakers, "
              "each played with room tone around it",
              f"confusable : {len(conf_scores)} confusable words (kunji, kunji ka, ...) "
              "- share that fired",
              "long audio : false activations per hour", "", head]
    for thr in THRESHOLDS:
        hit = np.mean([detections(p, thr) > 0 for p in pos_scores]) if pos_scores else float("nan")
        conf = np.mean([detections(p, thr) > 0 for p in conf_scores]) if conf_scores else float("nan")
        row = f"{thr:9.2f} {hit * 100:9.1f}% {conf * 100:10.1f}%"
        for name, (probs, hours) in long_scores.items():
            n = detections(probs, thr)
            row += f" {n / hours:10.1f}/h ({n:3d})"
        lines.append(row)
    lines += ["", "long audio lengths: " + ", ".join(
        f"{k} {h * 60:.1f} min" for k, (_, h) in long_scores.items())]

    if pos_conditions and pos_scores:
        lines += ["", "hit rate by condition (unseen speakers)",
                  f"{'condition':>10s} {'clips':>6s}" + "".join(f" {t:>7.2f}" for t in (0.7, 0.8, 0.9))]
        for cond in sorted(set(pos_conditions)):
            idx = [i for i, c in enumerate(pos_conditions) if c == cond]
            row = f"{cond:>10s} {len(idx):6d}"
            for t in (0.7, 0.8, 0.9):
                row += f" {np.mean([detections(pos_scores[i], t) > 0 for i in idx]) * 100:6.0f}%"
            lines.append(row)

    if names_eval:
        lines += ["", "names: share of clips that fired (each played with room tone around it)",
                  f"{'name':>10s} {'real':>6s}" + "".join(f" {t:>6.1f}" for t in (0.7, 0.8, 0.9)) +
                  f"   {'TTS':>5s}" + "".join(f" {t:>6.1f}" for t in (0.7, 0.8, 0.9))]
        worst = []
        for name in sorted(names_eval):
            real, tts = names_eval[name]
            rs = [scorer.stream(in_context(x, pad)) for x in real]
            ts = [scorer.stream(in_context(x, pad)) for x in tts]
            row = f"{name:>10s} {len(rs):6d}"
            row += "".join(f" {np.mean([detections(p, t) > 0 for p in rs]) * 100:5.0f}%" if rs else "      -"
                           for t in (0.7, 0.8, 0.9))
            row += f"   {len(ts):5d}"
            row += "".join(f" {np.mean([detections(p, t) > 0 for p in ts]) * 100:5.0f}%" if ts else "      -"
                           for t in (0.7, 0.8, 0.9))
            lines.append(row)
            if rs or ts:
                worst.append((np.mean([detections(p, 0.8) > 0 for p in (rs or ts)]), name))
        worst.sort(reverse=True)
        lines.append("  worst at 0.8: " + ", ".join(f"{n} {v * 100:.0f}%" for v, n in worst[:5]))

    if listen:
        lines += ["", "free-speech false activations at 0.80 - listen at these times to see",
                  "whether the speaker actually said Kunjika or a confusable word:"]
        for name, x in listen:
            t = detection_times(scorer.stream(x), 0.80)
            lines.append(f"  {name}: " + (", ".join(f"{v:.1f} s" for v in t) if t else "none"))
    return lines


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--room", default="")
    ap.add_argument("--tts", default="")
    ap.add_argument("--tts-eval", default="")
    ap.add_argument("--names-test", default="")
    ap.add_argument("--sc", default="speech_commands")
    ap.add_argument("--out", default="kunjika_model")
    ap.add_argument("--test-speakers", default="")
    ap.add_argument("--val-speakers", default="")
    ap.add_argument("--final", action="store_true",
                    help="train on ALL speakers (use after choosing the settings)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    clean, out = pathlib.Path(a.clean), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(clean)
    train_s, val_s, test_s = choose_split(rows, 2, 1, a.test_speakers, a.val_speakers, a.final)
    print(f"speakers  train {train_s}\n          val   {val_s}\n          test  {test_s}")

    def sel(speakers, conds, kind="clip"):
        return [r for r in rows if r["speaker"] in speakers and r["condition"] in conds
                and r["kind"] == kind]

    KW = ["normal", "fast", "slow", "quiet", "loud", "lombard"]
    dropped = [r for r in rows if r["condition"] in KW and r["coverage"] < MIN_COVERAGE]
    print(f"dropping {len(dropped)} keyword clips with < {MIN_COVERAGE:.0%} of the word inside 1 s")

    def positives(spk):
        return [r["full"] for r in sel(spk, KW) if r["coverage"] >= MIN_COVERAGE]

    def long_audio(spk, conds):
        parts = [read_wav(r["full"]) for r in sel(spk, conds, "long")]
        return np.concatenate(parts) if parts else np.zeros(0, np.int16)

    room_files = sorted(glob.glob(os.path.join(a.room, "*.wav"))) if a.room else []
    room = np.concatenate([read_wav(f) for f in room_files]) if room_files else np.zeros(0, np.int16)
    # room tone: 80% for training, the rest only for evaluation context
    room_cut = int(len(room) * 0.8)
    room_train, room_eval = room[:room_cut], room[room_cut:]

    sc_train, sc_val, sc_bg = speech_commands(a.sc)
    noise = load_noise(sc_bg, room_train)
    print(f"noise for augmentation: {len(noise) / SAMPLE_RATE / 60:.1f} min "
          f"({len(sc_bg)} Speech Commands files + {len(room_train) / SAMPLE_RATE:.0f} s of room tone)")

    podcast_train = read_wav(clean / "_podcast" / "podcast_train.wav") \
        if (clean / "_podcast" / "podcast_train.wav").exists() else np.zeros(0, np.int16)
    podcast_test = read_wav(clean / "_podcast" / "podcast_test.wav") \
        if (clean / "_podcast" / "podcast_test.wav").exists() else np.zeros(0, np.int16)

    # ---- training data, as separate streams
    pos_train = positives(train_s)
    conf_train = [r["full"] for r in sel(train_s, ["negatives"])]
    tts_pos, tts_neg = [], []
    if a.tts:
        for f in sorted(pathlib.Path(a.tts).rglob("*.wav")):
            (tts_neg if "negatives" in [p.lower() for p in f.parts] else tts_pos).append(str(f))

    def wins_of(*arrays):
        ok = [windows(x) for x in arrays if len(x) >= CLIP_SAMPLES]
        return np.concatenate(ok) if ok else np.zeros((0, CLIP_SAMPLES), np.int16)

    same_channel = wins_of(long_audio(train_s, ["freespeech"]), long_audio(train_s, ["silence"]))
    other_audio = wins_of(podcast_train, room_train)
    neg_wins = same_channel     # kept for the summary line below
    neg_train = conf_train + sc_train

    # ---- validation data (unaugmented)
    val_pos = positives(val_s)
    val_neg = [r["full"] for r in sel(val_s, ["negatives"])] + sc_val
    val_long = long_audio(val_s, ["freespeech"])
    val_wins = windows(val_long)[::2] if len(val_long) >= CLIP_SAMPLES else np.zeros((0, CLIP_SAMPLES), np.int16)

    if not val_pos:
        # no validation speaker: hold back 10% of training positives for early
        # stopping only (not speaker-disjoint, so never used for reporting)
        k = max(1, len(pos_train) // 10)
        val_pos, pos_train = pos_train[-k:], pos_train[:-k]
        print(f"no validation speaker: using {k} held-back training clips for early stopping")

    summary = [f"train: {len(pos_train)} real + {len(tts_pos)} TTS positives | "
               f"{len(conf_train)} confusables | {len(tts_neg)} TTS names | {len(sc_train)} words | "
               f"{len(same_channel)} same-channel windows | {len(other_audio)} podcast/room windows",
               f"val  : {len(val_pos)} positives, {len(val_neg)} negative clips, {len(val_wins)} windows"]
    print("\n".join(summary))
    if not pos_train:
        raise SystemExit("no training positives")

    train_ds = train_dataset([(pos_train, True, 1, W_POSITIVE),
                              (tts_pos, True, 1, W_TTS_POSITIVE),
                              (conf_train, True, 0, W_CONFUSABLE),
                              (tts_neg, True, 0, W_NAMES),
                              (sc_train, True, 0, W_WORDS),
                              (same_channel, False, 0, W_SAME_CHANNEL),
                              (other_audio, False, 0, W_OTHER_AUDIO)], noise, a.batch)
    val_ds = eval_dataset(val_pos + val_neg, [1] * len(val_pos) + [0] * len(val_neg), val_wins)

    # ---- train
    model = compile_model(ds_cnn(), 1e-3)
    model.fit(train_ds, steps_per_epoch=a.steps, epochs=a.epochs, validation_data=val_ds,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True,
                                                          monitor="val_loss"),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5,
                                                              monitor="val_loss")])
    qat = compile_model(tfmot.quantization.keras.quantize_model(model), 1e-4)
    qat.fit(train_ds, steps_per_epoch=a.steps, epochs=a.qat_epochs, validation_data=val_ds)
    qat1 = tfmot.quantization.keras.quantize_model(ds_cnn(batch_size=1))
    qat1.set_weights(qat.get_weights())

    model_path = out / "model.tflite"
    blob = to_int8(qat1, train_ds, model_path)
    scorer = Scorer(model_path)
    bad = {"MEAN", "SOFTMAX", "SHAPE", "STRIDED_SLICE", "PACK"} & set(scorer.ops)
    if bad:
        raise SystemExit(f"model contains slow/dynamic ops {sorted(bad)}")
    json.dump({"keyword": "kunjika", "sample_rate": SAMPLE_RATE, "frame_len": FRAME_LEN,
               "frame_hop": FRAME_HOP, "num_frames": NUM_FRAMES, "num_mel": NUM_MEL,
               "fft_size": FFT_SIZE, "mel_lo_hz": MEL_LO_HZ, "mel_hi_hz": MEL_HI_HZ,
               "log_floor": LOG_FLOOR},
              open(out / "model.features.json", "w"), indent=2)

    # ---- evaluate like the device
    rng = np.random.default_rng(SEED)
    pad = room_eval if len(room_eval) >= 38400 else \
        (rng.normal(0, 30, 38400)).astype(np.int16)          # quiet noise if no room audio
    long_sets = {"podcast (held out)": podcast_test}
    if test_s:
        long_sets["test spk free speech"] = long_audio(test_s, ["freespeech"])
    conv_test = long_audio(sorted({r["speaker"] for r in rows if r["speaker"].startswith("convtest")}),
                           ["freespeech"])
    if len(conv_test):
        long_sets["conversation (held out)"] = conv_test

    # names: real recordings (test only) and held-out TTS voices
    tts_eval = pathlib.Path(a.tts_eval or (a.tts.rstrip("/") + "_eval" if a.tts else ""))
    known = sorted({p.name for p in (pathlib.Path(a.tts) / "negatives" / "names").iterdir()}) \
        if a.tts and (pathlib.Path(a.tts) / "negatives" / "names").exists() else []
    names_eval = {}
    if a.tts and (tts_eval / "negatives" / "names").exists():
        for d in sorted((tts_eval / "negatives" / "names").iterdir()):
            names_eval.setdefault(d.name, ([], []))[1].extend(
                read_wav(f) for f in sorted(d.glob("*.wav")))
    if a.names_test:
        n_real = 0
        for person in sorted(pathlib.Path(a.names_test).iterdir()):
            if not person.is_dir():
                continue
            for d in sorted(p for p in person.iterdir() if p.is_dir()):
                name = canonical_name(d.name, known)
                for f in sorted(d.iterdir()):
                    try:
                        names_eval.setdefault(name, ([], []))[0].append(best_second(decode_any(f)))
                        n_real += 1
                    except ValueError as e:
                        print(f"  cannot read {f}: {e}")
        print(f"real name recordings for testing: {n_real}")
    report = [f"model: {len(blob)} bytes, ops {scorer.ops}", *summary,
              f"speakers train {train_s} | val {val_s} | test {test_s}", ""]
    test_pos_rows = [r for r in sel(test_s, KW) if r["coverage"] >= MIN_COVERAGE]
    listen = [(f"{r['speaker']} free speech ({os.path.basename(r['source'])})", read_wav(r["full"]))
              for r in sel(test_s, ["freespeech"], "long")]
    report += evaluate(scorer, [r["full"] for r in test_pos_rows],
                       [r["full"] for r in sel(test_s, ["negatives"])], long_sets, pad,
                       pos_conditions=[r["condition"] for r in test_pos_rows], listen=listen,
                       names_eval=names_eval)
    report += ["", "Pick the lowest threshold whose false activations are acceptable, and set",
               "KWS_THRESHOLD in kws_engine.h to it. Then generate headers with:",
               "    python tools/gen_step4.py model.tflite     (model.features.json beside it)"]
    text = "\n".join(report)
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
