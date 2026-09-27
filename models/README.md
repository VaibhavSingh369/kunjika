# Models

`kunjika_v3.tflite` is the current Kunjika model: an int8 DS-CNN, 16,808 bytes. `kunjika_v3.features.json` holds the feature settings it was trained with (mel bands 60–7600 Hz).

The firmware already contains this model (in `firmware/main/kws_model.h`). To regenerate the headers:

```bash
cd firmware
python tools/gen_step4.py ../models/kunjika_v3.tflite main
```

The generator reads the `.features.json` file next to the model, so the two files must stay together.
