# BER Workflow

Use this workflow for new Section 3.1 BER / replication tests unless there is a specific reason to compare decoder variants.

## Fast Default

1. Calibrate each new `.raw` file once.
2. Run BER once with the synced transition decoder.
3. Use calibrated ROI/window/frequency.
4. Skip extra decoder variants unless the result looks wrong.

This avoids rereading and rebining the same large RAW files four different ways.

## Calibration

Run once per RAW capture:

```powershell
.\metavision39_env\Scripts\python.exe .\Event-Driven-Camera-VLC\src\3.1\section3_1_replication_calibrate.py `
  --raw "captures\3.1\YOUR_CAPTURE_FOLDER\YOUR_FILE.raw" `
  --replication_manifest_csv "Event-Driven-Camera-VLC\pru1_pwm_CSK_1000Hz\userspace\YOUR_MANIFEST.csv" `
  --out_prefix YOUR_CALIBRATION_PREFIX
```

The calibration step finds the active transmit window, LED ROI, and observed bit frequency.

## BER Analysis

Use this as the default serious BER run:

```powershell
.\metavision39_env\Scripts\python.exe .\Event-Driven-Camera-VLC\src\3.1\section3_1_replication_analyze.py `
  --input_dir "captures\3.1\YOUR_CAPTURE_FOLDER" `
  --manifest_csv "Event-Driven-Camera-VLC\pru1_pwm_CSK_1000Hz\userspace\YOUR_MANIFEST.csv" `
  --bits 10110010110 `
  --decode_mode synced_transition `
  --bin_us 10 `
  --transition_rate_min_scale 0.9 `
  --transition_rate_max_scale 1.1 `
  --transition_rate_steps 81 `
  --use_calibrated_frequency `
  --out_prefix YOUR_OUTPUT_PREFIX
```

For quick checks, add:

```powershell
--no_plot --transition_rate_steps 21
```

## When To Run Extra Variants

Only run extra variants if the synced-transition calibrated result looks suspicious:

- `synced_activity` if transition decoding fails unexpectedly.
- Manifest-frequency runs if calibrated frequency appears wrong.
- Manual ROI runs if calibration picks the wrong LED region.

## Current Thesis Operating Point

Use this validated condition for Section 3.2 energy work:

- 300 Hz OOK.
- Synced preamble symbol file.
- 50 cm distance.
- f/4 aperture.
- Synced transition decoder.

The 50 cm 100-500 Hz sweep showed clean decoding at 200 Hz and 300 Hz, with degradation at 400-500 Hz.
