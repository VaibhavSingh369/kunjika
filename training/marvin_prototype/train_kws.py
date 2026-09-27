"""
KWS pipeline validation - Google Speech Commands, keyword "marvin".

Purpose: prove the train -> QAT -> int8 -> TFLite -> C array chain end to end
on a known-good dataset BEFORE swapping in Kunjika data.

Swap to Kunjika later by replacing load_dataset() only. Nothing else changes.

Run in Colab.  Runtime: ~15 min on a T4.

    !pip install -q tensorflow-model-optimization tf_keras
    (then Runtime -> Restart session, and run this script)
"""

import os
import glob
import pathlib
import urllib.request
import tarfile

# tensorflow_model_optimization needs Keras 2. Current TensorFlow ships Keras 3
# by default, so switch to the Keras 2 package (tf_keras) BEFORE importing TF.
# If TensorFlow was already imported in this runtime, restart the runtime.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# FEATURE CONSTANTS
# ---------------------------------------------------------------------------
# These MUST be identical in the C implementation. dump_c_header() at the
# bottom writes them out so there is a single source of truth.
# Log-mel is used rather than MFCC: skipping the DCT removes one more place
# where Python and C can silently disagree, and costs nothing in accuracy.

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000          # 1.0 s
FRAME_LEN = 512               # 32 ms, power of 2 so esp-dsp FFT is direct
FRAME_HOP = 320               # 20 ms
NUM_FRAMES = 49               # (16000 - 512) // 320 + 1
NUM_MEL = 40
FFT_SIZE = 512
MEL_LO_HZ = 20.0
MEL_HI_HZ = 7600.0
LOG_FLOOR = 1e-6              # log(x + LOG_FLOOR); C must use the same value

KEYWORD = "marvin"            # <-- change to "kunjika" later

# Training augmentation. Live detection slides a 1 s window past the word
# every 100 ms, so the word is almost never centred - but Speech Commands
# clips are. Without shifts the model learns "word in the middle" and scores
# real speech low (measured: 0.79-0.88 peaks on a live-style sweep).
SHIFT_MS = 100                # random time shift, +/- this much
GAIN_DB = (-10.0, 6.0)        # random loudness change
NOISE_PROB = 0.8              # chance of mixing in background noise
SNR_DB = (5.0, 20.0)          # speech-to-noise ratio when mixing
NOISE_ONLY_PROB = 0.1         # chance a negative becomes pure noise
DATA_URL = ("http://download.tensorflow.org/data/"
            "speech_commands_v0.02.tar.gz")
DATA_DIR = pathlib.Path("speech_commands")


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def download_speech_commands():
    if DATA_DIR.exists():
        print("dataset already present")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tgz = "speech_commands.tar.gz"
    print("downloading Speech Commands v0.02 (~2.3 GB)...")
    urllib.request.urlretrieve(DATA_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)
    os.remove(tgz)


def speaker_id(path):
    """Speech Commands filenames are <speakerhash>_nohash_<n>.wav.

    The speaker hash is what we split on. Splitting on clips instead of
    speakers inflates accuracy by 10-20 points and the inflation is not
    discovered until demo day.
    """
    return os.path.basename(path).split("_")[0]


def load_dataset():
    """Returns (paths, labels, speakers). label 1 = keyword, 0 = not."""
    download_speech_commands()

    pos = glob.glob(str(DATA_DIR / KEYWORD / "*.wav"))
    if not pos:
        raise RuntimeError(f"no clips found for keyword '{KEYWORD}'")

    other_words = [d.name for d in DATA_DIR.iterdir()
                   if d.is_dir()
                   and d.name != KEYWORD
                   and not d.name.startswith("_")]

    neg = []
    for w in other_words:
        neg.extend(glob.glob(str(DATA_DIR / w / "*.wav")))

    # 3:1 negatives:positives. Heavier skew matches deployment but starves
    # the positive class at this dataset size.
    rng = np.random.default_rng(SEED)
    neg = list(rng.choice(neg, size=min(len(neg), len(pos) * 3), replace=False))

    paths = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    speakers = [speaker_id(p) for p in paths]
    print(f"{len(pos)} positives, {len(neg)} negatives, "
          f"{len(set(speakers))} speakers")
    return np.array(paths), np.array(labels), np.array(speakers)


def split_by_speaker(paths, labels, speakers, val_frac=0.15, test_frac=0.15):
    uniq = np.array(sorted(set(speakers)))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)

    n_test = int(len(uniq) * test_frac)
    n_val = int(len(uniq) * val_frac)
    test_spk = set(uniq[:n_test])
    val_spk = set(uniq[n_test:n_test + n_val])

    out = {}
    for name, keep in (("test", lambda s: s in test_spk),
                       ("val", lambda s: s in val_spk),
                       ("train", lambda s: s not in test_spk
                        and s not in val_spk)):
        m = np.array([keep(s) for s in speakers])
        out[name] = (paths[m], labels[m])
        print(f"{name:5s}: {m.sum():6d} clips")
    return out


# ---------------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------------
_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MEL,
    num_spectrogram_bins=FFT_SIZE // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_LO_HZ,
    upper_edge_hertz=MEL_HI_HZ,
)


def log_mel(waveform):
    """waveform: float32 [-1, 1], length CLIP_SAMPLES -> [NUM_FRAMES, NUM_MEL]"""
    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LEN,
        frame_step=FRAME_HOP,
        fft_length=FFT_SIZE,
        window_fn=tf.signal.hann_window,   # C side must use periodic Hann
        pad_end=False,
    )
    power = tf.abs(stft) ** 2
    mel = tf.matmul(power, _mel_matrix)
    return tf.math.log(mel + LOG_FLOOR)


def load_wave(path):
    audio = tf.io.read_file(path)
    wav, _ = tf.audio.decode_wav(audio, desired_channels=1,
                                 desired_samples=CLIP_SAMPLES)
    return tf.squeeze(wav, -1)


def to_features(wav):
    return tf.reshape(log_mel(wav), [NUM_FRAMES, NUM_MEL, 1])


def decode(path, label):
    """Unaugmented: used for validation, test and parity."""
    return to_features(load_wave(path)), label


_BG_NOISE = None


def background_noise():
    """All of Speech Commands' _background_noise_ files, joined end to end.

    Read with plain Python (wave + numpy), NOT tf.audio: if this is first
    called while tf.data is tracing augment(), TensorFlow ops would return
    tensors of unknown length. main() also calls it once up front.
    """
    global _BG_NOISE
    if _BG_NOISE is None:
        import wave
        files = sorted(glob.glob(str(DATA_DIR / "_background_noise_" / "*.wav")))
        parts = []
        for fp in files:
            with wave.open(fp, "rb") as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    continue
                a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
                parts.append(a.astype(np.float32) / 32768.0)
        joined = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        _BG_NOISE = tf.constant(joined)
        print(f"background noise: {len(parts)} files, {len(joined) / SAMPLE_RATE:.0f} s")
    return _BG_NOISE


def shift_wave(wav, shift):
    """Move the clip by `shift` samples, filling with silence."""
    s = SHIFT_MS * SAMPLE_RATE // 1000
    padded = tf.pad(wav, [[s, s]])
    return padded[s - shift: s - shift + CLIP_SAMPLES]


def rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)) + 1e-10)


def augment(wav, label):
    s = SHIFT_MS * SAMPLE_RATE // 1000
    wav = shift_wave(wav, tf.random.uniform([], -s, s + 1, dtype=tf.int32))
    gain_db = tf.random.uniform([], GAIN_DB[0], GAIN_DB[1])
    wav = wav * tf.pow(10.0, gain_db / 20.0)

    bg = background_noise()
    n_bg = tf.shape(bg)[0]
    if bg.shape[0] and bg.shape[0] > CLIP_SAMPLES:
        off = tf.random.uniform([], 0, n_bg - CLIP_SAMPLES, dtype=tf.int32)
        noise = bg[off: off + CLIP_SAMPLES]

        snr_db = tf.random.uniform([], SNR_DB[0], SNR_DB[1])
        scaled = noise * (rms(wav) / rms(noise)) / tf.pow(10.0, snr_db / 20.0)
        mix = tf.random.uniform([]) < NOISE_PROB
        wav = tf.where(mix, wav + scaled, wav)

        # some negatives become pure noise at a random level: teaches the
        # model that fans, traffic and hum are "not the keyword"
        level = tf.pow(10.0, tf.random.uniform([], -50.0, -15.0) / 20.0)
        pure = noise * (level / rms(noise))
        to_pure = tf.logical_and(tf.equal(label, 0),
                                 tf.random.uniform([]) < NOISE_ONLY_PROB)
        wav = tf.where(to_pure, pure, wav)

    return tf.clip_by_value(wav, -1.0, 1.0), label


def make_ds(paths, labels, training=False, batch=64, shifts=None):
    """training=True applies augmentation. shifts=array gives each clip a
    fixed time shift (used to test robustness to word position)."""
    if shifts is None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, l: (load_wave(p), l), num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, shifts))
        ds = ds.map(lambda p, l, sh: (shift_wave(load_wave(p), sh), l),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda w, l: (to_features(w), l), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# MODEL - DS-CNN
# ---------------------------------------------------------------------------
def ds_cnn(n_ch=32, n_blocks=3, batch_size=None):
    """DS-CNN shaped for ESP-NN on the ESP32-S3.

    Every choice below comes from on-device per-layer profiling:
      - First layer: 4x4 filter, stride 4, straight to the 12x10 grid.
        The first conv has ONE input channel. ESP-NN's fast 3x3 path needs
        16+ input channels, so this layer always runs on a general routine
        whose cost scales with the number of OUTPUTS. 3x3/stride 2 + max-pool
        produced 16,000 outputs (~10 ms); non-overlapping 4x4 patches produce
        3,840. It also removes the 25x20x32 intermediate map, halving the arena.
      - AveragePooling2D over the whole map (not GlobalAveragePooling2D):
        same maths, converts to AVERAGE_POOL_2D (60 us) instead of TFLM's
        slow MEAN (3.1 ms).
      - No softmax: raw scores (logits) out; the firmware converts to a
        probability. TFLM's int8 SOFTMAX cost 0.45 ms for two numbers.

    batch_size=1 builds the inference copy used for conversion, so every
    shape is static and no SHAPE / STRIDED_SLICE / PACK ops appear.
    """
    L = tf.keras.layers
    inp = L.Input(shape=(NUM_FRAMES, NUM_MEL, 1), batch_size=batch_size)

    x = L.Conv2D(n_ch, (4, 4), strides=(4, 4), padding="valid",
                 use_bias=False)(inp)                 # -> 12 x 10 x n_ch
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
    out = L.Dense(2)(x)                      # logits, no softmax
    return tf.keras.Model(inp, out)


def compile_model(m, lr=1e-3):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


# ---------------------------------------------------------------------------
# CONVERSION
# ---------------------------------------------------------------------------
def to_int8_tflite(model, rep_ds, path="model.tflite"):
    """Full int8 quantization. rep_ds calibrates activation ranges."""
    def rep_gen():
        for feats, _ in rep_ds.unbatch().batch(1).take(200):
            yield [tf.cast(feats, tf.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    with open(path, "wb") as f:
        f.write(blob)
    print(f"{path}: {len(blob)} bytes ({len(blob)/1024:.1f} KB)")
    return blob


def eval_tflite(path, ds):
    """Accuracy of the quantized model. Must be within ~1-2 pts of float."""
    interp = tf.lite.Interpreter(
        model_path=path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]

    correct = total = 0
    for feats, labels in ds.unbatch():
        q = np.round(feats.numpy() / in_scale + in_zp)
        q = np.clip(q, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out["index"])[0]))
        correct += int(pred == int(labels.numpy()))
        total += 1
    acc = correct / total
    print(f"int8 accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


def check_ops(path):
    """Fail loudly if the model contains ops that are slow or unsupported on
    the ESP32 build. Catches architecture regressions before they reach
    the board."""
    interp = tf.lite.Interpreter(
        model_path=path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    print(f"ops: {ops}")
    bad = {"MEAN", "SOFTMAX", "SHAPE", "STRIDED_SLICE", "PACK"} & set(ops)
    if bad:
        raise RuntimeError(f"model contains slow/dynamic ops {sorted(bad)} - "
                           "check ds_cnn() and the batch_size=1 conversion")
    return ops


def dump_c_array(blob, path="model_data.cc", name="g_model"):
    lines = [f"const unsigned char {name}[] "
             "__attribute__((aligned(16))) = {"]
    for i in range(0, len(blob), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in blob[i:i + 12])
        lines.append("  " + chunk + ",")
    lines.append("};")
    lines.append(f"const unsigned int {name}_len = {len(blob)};")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def dump_c_header(interp_path="model.tflite", path="kws_config.h"):
    """Single source of truth for the C side. Prevents parity drift."""
    interp = tf.lite.Interpreter(model_path=interp_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    with open(path, "w") as f:
        f.write(f"""// generated by train_kws.py - do not edit by hand
#pragma once

#define KWS_SAMPLE_RATE   {SAMPLE_RATE}
#define KWS_CLIP_SAMPLES  {CLIP_SAMPLES}
#define KWS_FRAME_LEN     {FRAME_LEN}
#define KWS_FRAME_HOP     {FRAME_HOP}
#define KWS_NUM_FRAMES    {NUM_FRAMES}
#define KWS_NUM_MEL       {NUM_MEL}
#define KWS_FFT_SIZE      {FFT_SIZE}
#define KWS_MEL_LO_HZ     {MEL_LO_HZ}f
#define KWS_MEL_HI_HZ     {MEL_HI_HZ}f
#define KWS_LOG_FLOOR     {LOG_FLOOR}f

// input quantization: q = round(x / scale) + zero_point
#define KWS_IN_SCALE      {in_scale}f
#define KWS_IN_ZERO_PT    {in_zp}
#define KWS_OUT_SCALE     {out_scale}f
#define KWS_OUT_ZERO_PT   {out_zp}
""")
    print(f"wrote {path}  (in_scale={in_scale}, in_zp={in_zp})")


def dump_parity_vector(paths, path="parity.h"):
    """One clip's features, for the Phase B parity test.

    Compute features for the SAME wav on device and diff against this.
    """
    feat, _ = decode(tf.constant(paths[0]), tf.constant(0))
    flat = feat.numpy().flatten()
    with open(path, "w") as f:
        f.write(f"// features for {os.path.basename(paths[0])}\n")
        f.write(f"// shape [{NUM_FRAMES}][{NUM_MEL}], row-major\n")
        f.write("#pragma once\n\n")
        f.write(f"const float kParityRef[{len(flat)}] = {{\n")
        for i in range(0, len(flat), 8):
            f.write("  " + ", ".join(f"{v:.6f}f" for v in flat[i:i + 8]) + ",\n")
        f.write("};\n")
    print(f"wrote {path}  (reference clip: {os.path.basename(paths[0])})")


# ---------------------------------------------------------------------------
def main():
    paths, labels, speakers = load_dataset()
    sp = split_by_speaker(paths, labels, speakers)

    background_noise()                  # load once, before any tf.data tracing
    train_ds = make_ds(*sp["train"], training=True)
    val_ds = make_ds(*sp["val"])
    test_ds = make_ds(*sp["test"])

    # Same test clips, each moved by a fixed random amount (+/- SHIFT_MS).
    # This is closer to live streaming, where the word is rarely centred.
    s = SHIFT_MS * SAMPLE_RATE // 1000
    shifts = np.random.default_rng(SEED).integers(-s, s + 1, len(sp["test"][0]))
    test_shift_ds = make_ds(*sp["test"], shifts=shifts.astype(np.int32))

    # ---- float training ----
    model = compile_model(ds_cnn())
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=30,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(
                      patience=5, restore_best_weights=True,
                      monitor="val_accuracy"),
                  tf.keras.callbacks.ReduceLROnPlateau(
                      patience=3, factor=0.5, monitor="val_accuracy"),
              ])
    float_acc = model.evaluate(test_ds)[1]
    print(f"float accuracy: {float_acc:.4f}")

    # ---- quantization-aware fine-tune ----
    # QAT rather than plain post-training quantization: at this model size
    # PTQ can cost several points that we cannot spare.
    qat = tfmot.quantization.keras.quantize_model(model)
    compile_model(qat, lr=1e-4)
    qat.fit(train_ds, validation_data=val_ds, epochs=5)

    # Convert from a batch-size-1 copy so all shapes are static
    qat1 = tfmot.quantization.keras.quantize_model(ds_cnn(batch_size=1))
    qat1.set_weights(qat.get_weights())
    blob = to_int8_tflite(qat1, train_ds)
    check_ops("model.tflite")
    int8_acc = eval_tflite("model.tflite", test_ds)
    print("shifted test set (word off-centre, closer to live use):")
    shift_acc = eval_tflite("model.tflite", test_shift_ds)

    drop = float_acc - int8_acc
    print(f"\nfloat {float_acc:.4f} -> int8 {int8_acc:.4f}  (drop {drop:.4f})")
    print(f"int8 on shifted test set: {shift_acc:.4f}  "
          f"(gap to centred {int8_acc - shift_acc:+.4f}; small gap = robust to position)")
    if drop > 0.02:
        print("WARNING: >2 point drop. Fix this here, not on the MCU.")

    dump_c_array(blob)
    dump_c_header()
    dump_parity_vector(sp["test"][0])
    print("\nnext: copy model_data.cc, kws_config.h, parity.h to the firmware")


if __name__ == "__main__":
    main()
