"""
Phase B / Step 4 - model header generator for live detection.

From your int8 model, writes into main/:

  kws_model.h   the .tflite model as a C array
  kws_mel.h     TensorFlow's exact mel filterbank (same as Steps 2-3)
  kws_ops.h     the ops the model uses, as resolver calls
  kws_config.h  feature constants and the model's quantization parameters

Switching keyword later (marvin -> Kunjika) = rerun this with the new model.

Feature constants MUST match train_kws.py.

Usage (from the kws_step4 folder):
    python tools/gen_step4.py path/to/model.tflite

Needs: tensorflow, numpy
"""

import os
import sys

import numpy as np
import tensorflow as tf

# ---- must match train_kws.py ------------------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 512
FRAME_HOP = 320
NUM_FRAMES = 49
NUM_MEL = 40
FFT_SIZE = 512
MEL_LO_HZ = 20.0
MEL_HI_HZ = 7600.0
LOG_FLOOR = 1e-6
POSITIVE_INDEX = 1          # train_kws.py: label 1 = keyword
# -----------------------------------------------------------------------------

NUM_BINS = FFT_SIZE // 2 + 1

OP_MAP = {
    "CONV_2D": "AddConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "FULLY_CONNECTED": "AddFullyConnected",
    "MAX_POOL_2D": "AddMaxPool2D",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "MEAN": "AddMean",
    "SOFTMAX": "AddSoftmax",
    "RESHAPE": "AddReshape",
    "QUANTIZE": "AddQuantize",
    "DEQUANTIZE": "AddDequantize",
    "RELU": "AddRelu",
    "RELU6": "AddRelu6",
    "ADD": "AddAdd",
    "MUL": "AddMul",
    "PAD": "AddPad",
    "SQUEEZE": "AddSqueeze",
    "LOGISTIC": "AddLogistic",
    "CONCATENATION": "AddConcatenation",
}


def write_array(f, ctype, name, values, per_line, fmt, attrs=""):
    f.write(f"static const {ctype} {name}[{len(values)}]{attrs} = {{\n")
    for i in range(0, len(values), per_line):
        f.write("  " + ", ".join(fmt(v) for v in values[i:i + per_line]) + ",\n")
    f.write("};\n\n")


def load_features(tflite_path):
    """Feature settings the model was trained with. train_kunjika.py writes
    <model>.features.json next to the model; the device mel table MUST be
    built from the same values or features silently stop matching."""
    global MEL_LO_HZ, MEL_HI_HZ
    stem = os.path.splitext(tflite_path)[0]
    for cand in (stem + ".features.json",
                 os.path.join(os.path.dirname(tflite_path) or ".", "model.features.json")):
        if os.path.exists(cand):
            import json
            cfg = json.load(open(cand))
            fixed = {"sample_rate": SAMPLE_RATE, "frame_len": FRAME_LEN, "frame_hop": FRAME_HOP,
                     "num_frames": NUM_FRAMES, "num_mel": NUM_MEL, "fft_size": FFT_SIZE}
            for k, v in fixed.items():
                if cfg.get(k) != v:
                    sys.exit(f"{cand}: {k} = {cfg.get(k)}, firmware expects {v}")
            MEL_LO_HZ = float(cfg["mel_lo_hz"])
            MEL_HI_HZ = float(cfg["mel_hi_hz"])
            return cand
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tflite_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "main"
    os.makedirs(out_dir, exist_ok=True)
    src = load_features(tflite_path)
    if src:
        print(f"features: mel {MEL_LO_HZ:g}-{MEL_HI_HZ:g} Hz (from {os.path.basename(src)})")
    else:
        print(f"features: no .features.json beside the model - assuming mel "
              f"{MEL_LO_HZ:g}-{MEL_HI_HZ:g} Hz (correct for the marvin models only)")
    model_bytes = open(tflite_path, "rb").read()
    name = os.path.basename(tflite_path)

    interp = tf.lite.Interpreter(
        model_content=model_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_s, in_zp = inp["quantization"]
    out_s, out_zp = out["quantization"]
    if in_s == 0 or out_s == 0:
        sys.exit("model is not fully int8 - use the int8 model.tflite")
    if tuple(inp["shape"][1:3]) != (NUM_FRAMES, NUM_MEL):
        sys.exit(f"model input {inp['shape']} does not match {NUM_FRAMES}x{NUM_MEL}")
    n_out = int(np.prod(out["shape"]))
    if n_out != 2:
        sys.exit(f"expected 2 outputs (not-keyword, keyword), got {n_out}")

    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    missing = [o for o in ops if o not in OP_MAP]
    if missing:
        sys.exit(f"model uses ops not in OP_MAP: {missing}")
    is_logits = 0 if "SOFTMAX" in ops else 1

    mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL, num_spectrogram_bins=NUM_BINS, sample_rate=SAMPLE_RATE,
        lower_edge_hertz=MEL_LO_HZ, upper_edge_hertz=MEL_HI_HZ).numpy()
    starts, lens, offsets, weights = [], [], [], []
    for m in range(NUM_MEL):
        nz = np.nonzero(mel[:, m])[0]
        lo, hi = int(nz[0]), int(nz[-1])
        starts.append(lo)
        lens.append(hi - lo + 1)
        offsets.append(len(weights))
        weights.extend(mel[lo:hi + 1, m].astype(np.float32).tolist())

    hdr = "// generated by gen_step4.py - do not edit\n#pragma once\n#include <stdint.h>\n\n"
    fmt_f = lambda v: f"{float(np.float32(v)):.9g}f"

    with open(os.path.join(out_dir, "kws_model.h"), "w") as f:
        f.write(hdr + f"// {name}, {len(model_bytes)} bytes\n")
        write_array(f, "unsigned char", "g_kws_model", list(model_bytes), 16,
                    lambda b: f"0x{b:02x}", " __attribute__((aligned(16)))")
        f.write(f"static const unsigned int g_kws_model_len = {len(model_bytes)};\n")

    with open(os.path.join(out_dir, "kws_mel.h"), "w") as f:
        f.write(hdr)
        write_array(f, "uint16_t", "kMelStart", starts, 20, str)
        write_array(f, "uint16_t", "kMelLen", lens, 20, str)
        write_array(f, "uint16_t", "kMelOffset", offsets, 20, str)
        write_array(f, "float", "kMelWeights", weights, 6, fmt_f)

    with open(os.path.join(out_dir, "kws_ops.h"), "w") as f:
        f.write(hdr + f"// ops used by {name}\n#define KWS_NUM_OPS {len(ops)}\n\n")
        f.write("template <typename R>\nstatic bool kws_register_ops(R &r)\n{\n")
        for o in ops:
            f.write(f"    if (r.{OP_MAP[o]}() != kTfLiteOk) return false;   // {o}\n")
        f.write("    return true;\n}\n")

    with open(os.path.join(out_dir, "kws_config.h"), "w") as f:
        f.write(hdr + f"// model: {name}\n\n")
        f.write(f'#define KWS_MODEL_NAME     "{name}"\n')
        f.write(f"#define KWS_MODEL_BYTES    {len(model_bytes)}\n\n")
        f.write(f"#define KWS_SAMPLE_RATE    {SAMPLE_RATE}\n")
        f.write(f"#define KWS_FRAME_LEN      {FRAME_LEN}\n")
        f.write(f"#define KWS_FRAME_HOP      {FRAME_HOP}\n")
        f.write(f"#define KWS_NUM_FRAMES     {NUM_FRAMES}\n")
        f.write(f"#define KWS_NUM_MEL        {NUM_MEL}\n")
        f.write(f"#define KWS_FFT_SIZE       {FFT_SIZE}\n")
        f.write(f"#define KWS_LOG_FLOOR      {LOG_FLOOR}f\n")
        f.write(f"#define KWS_MEL_LO_HZ      {MEL_LO_HZ:.1f}f\n")
        f.write(f"#define KWS_MEL_HI_HZ      {MEL_HI_HZ:.1f}f\n\n")
        f.write(f"#define KWS_IN_SCALE       {in_s:.9g}f\n")
        f.write(f"#define KWS_IN_ZERO_PT     {int(in_zp)}\n")
        f.write(f"#define KWS_OUT_SCALE      {out_s:.9g}f\n")
        f.write(f"#define KWS_OUT_ZERO_PT    {int(out_zp)}\n")
        f.write(f"#define KWS_OUT_IS_LOGITS  {is_logits}\n")
        f.write(f"#define KWS_POS_INDEX      {POSITIVE_INDEX}\n")

    print(f"model : {name}, {len(model_bytes)} bytes, ops {ops}")
    print(f"quant : in {in_s:.6g}/{int(in_zp)}  out {out_s:.6g}/{int(out_zp)}  "
          f"{'logits' if is_logits else 'probabilities'}")
    print(f"wrote kws_model.h, kws_mel.h, kws_ops.h, kws_config.h to {out_dir}/")


if __name__ == "__main__":
    main()
