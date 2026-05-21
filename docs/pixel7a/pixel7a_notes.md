# Pixel 7a Notes For VLC Energy Comparison

## Why This Is Separate From EVK4

The EVK4 pipeline reads event-camera `.raw` files and decodes a sparse event stream. The Pixel 7a records conventional frame-based video, so its receiver pipeline should be separate.

Use the Pixel 7a as the frame-based comparison device for Section 3.2:

- Compare sensing energy and total energy per bit against EVK4.
- Attempt BER decoding only with a video-specific method.
- Clearly report when the phone video does not preserve enough temporal information for reliable 300 Hz decoding.

## First-Pass Capture Protocol

Match the validated EVK4 condition as closely as practical:

- Transmitter: `s31_synced_replication_300Hz_10s_symbols.txt`
- Modulation: OOK
- Requested bit rate: 300 Hz
- Distance: 50 cm
- Ambient lux: match EVK4 baseline if possible, about 500 lux
- Phone position: fixed on tripod/stand
- Focus/exposure: lock if the camera app allows it
- Zoom: record the value used
- Resolution and frame rate: record the exact video mode
- Duration: record the full 10 s transmission plus a little pre/post margin

Suggested capture path:

```text
captures/3.2/pixel7a_ook_300hz_50cm_YYYYMMDD/
```

Suggested filenames:

```text
pixel7a_ook_300hz_t1.mp4
pixel7a_ook_300hz_t2.mp4
pixel7a_ook_300hz_t3.mp4
```

## Power Notes To Record

For each trial, write down:

- idle phone power with the phone connected to the USB meter, screen/camera state matched as closely as possible
- active recording power during the VLC capture
- sensing window duration
- whether battery was charging, discharging, or held near steady
- video mode, brightness setting, and whether screen stayed on
- any compute/decode power measured later on a laptop or phone

For a first thesis-pass comparison, it is acceptable to enter Pixel 7a sensing power as the phone's active recording power and compute power as zero if decoding is not run on-device. Note that choice in the manifest notes.

## BER / Video Decode Plan

Make new Pixel-specific decoding code only after a representative Pixel video exists.

A first decoder should:

1. Read the Pixel video frames.
2. Crop an ROI around the LED.
3. Build an intensity trace from the ROI.
4. Check whether 300 Hz content is visible in frame intensity, rolling-shutter bands, or another measurable signal.
5. If visible, estimate timing/phase and compare decoded bits against `10110010110`.
6. If not visible, report that the Pixel recording cannot score BER at that condition and keep the energy result separate.

Important limitation: ordinary frame rates may not directly sample every 300 Hz OOK bit. A Pixel video may still show rolling-shutter flicker bands, but that requires a video-specific method and should not be mixed with the EVK4 event decoder.

## Minimum Metadata For Every Pixel Trial

Record these in notes or a sidecar text file:

- Pixel model
- camera app and mode
- resolution and frame rate
- focus/exposure lock state
- zoom value
- ambient lux
- distance and alignment
- transmitter symbol file
- power meter idle and active readings
- capture filename
- expected bit pattern and repeat count

## Device Specs To Cite

Primary/spec sources:

- Google Pixel phone hardware tech specs: https://support.google.com/pixelphone/answer/7158570?hl=en
- Google Pixel 7a one-pager PDF supplied locally: `C:\Users\rabis\Downloads\Google-Pixel-7a-One-Pager-EN-CA.pdf`

Supplemental cross-check sources:

- GSMArena Pixel 7a page: https://www.gsmarena.com/google_pixel_7a-12170.php
- DeviceSpecifications Pixel 7a page: https://www.devicespecifications.com/en/model/0c785ca3
- Manuals+ Pixel 7a user guide page: https://manuals.plus/asin/B0CDFKZ4VQ

Source notes:

- Use Google's hardware specs page and the Google one-pager as the main sources for thesis-cited device specifications.
- The DeviceSpecifications page required human verification during lookup here, so manually verify it in a browser before citing it.
- The Manuals+ page is an independent guide/book page; use it only as supplemental context, not as the authority for hardware values.

Working specs for experiment notes:

- Device: Google Pixel 7a.
- Reference operating system: Android 13.
- Processor: Google Tensor G2.
- Memory and storage: 8 GB LPDDR5 RAM, 128 GB UFS 3.1 storage.
- Battery: 4300 mAh minimum, 4385 mAh typical.
- Display: 6.1 inch OLED, 1080 x 2400, up to 90 Hz.
- Rear wide camera: 64 MP Quad Bayer, f/1.89 aperture, 80 degree field of view, 1/1.73 inch sensor, OIS/EIS.
- Rear ultrawide camera: 13 MP, f/2.2 aperture, 120 degree field of view.
- Video modes: rear 4K at 30/60 fps and 1080p at 30/60 fps; front 4K at 30 fps and 1080p at 30 fps; slow motion up to 240 fps.
- Video formats: HEVC/H.265 and AVC/H.264.
- Relevant sensors and ports: ambient light sensor, USB-C 3.2 Gen 2, Qi wireless charging.

Experiment impact:

- Use the rear wide camera as the first Pixel receiver configuration, since it is the main camera and has OIS/EIS.
- Record the exact video mode and codec for every trial. A 1080p60 or 4K60 setting is the clean first comparison for energy, while 240 fps slow motion is the most plausible mode to test if a Pixel BER decoder is attempted.
- A normal frame-rate video stream does not directly sample a 300 Hz OOK symbol stream at the frame level. If decoding is attempted, the likely path is a Pixel-specific rolling-shutter or brightness-trace decoder, not the EVK4 event RAW decoder.
