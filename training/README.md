# Training

Runs in Google Colab (CPU is enough; one training run takes 30 to 45 minutes).

## 1. Prepare the recordings

```bash
python prepare_dataset.py RAW_DIR OUT_DIR
```

`RAW_DIR` has one folder per speaker, each with condition folders (`normal`, `fast`, `slow`, `quiet`, `loud`, `lombard`, `negatives`, `freespeech`, `silence`), plus optionally `podcast_audio/`, `conv_train/freespeech/` and `conv_test/freespeech/`. Any audio format works, and misspelled folder names (`quite`, `laoud`) are recognized. The script converts everything to 16 kHz mono WAV, cuts keyword clips to the 1-second window with the most speech, splits the podcast into train and test parts, and writes `manifest.csv` plus a quality report.

## 2. Synthetic voices (optional)

```bash
pip install piper-tts
python gen_tts.py --out tts
```

Generates "Kunjika" and 32 similar-sounding Indian names (alone and in sentences) with Piper voices, including Hindi ones. About 15% of the voices are held out in `tts_eval/`, for testing only.

## 3. Train

```bash
pip install tensorflow-model-optimization tf_keras
python train_kunjika.py --clean OUT_DIR --room ROOM_SILENCE_DIR --tts tts --names-test NAMES_TEST_DIR --out model_v3
```

This splits the data by speaker, trains with augmentation (time shift, gain, noise, room reverb), applies quantization-aware training, and exports `model.tflite` plus `model.features.json`. It then evaluates **exactly as the firmware would**, printing hit rate and false activations per hour at each threshold, results per condition, per-name results, and the timestamps of any false activations in free speech.

`marvin_prototype/train_kws.py` is the earlier prototype trained on Google Speech Commands ("marvin"), which was used to develop and profile the model architecture.
