# EVK4 Notes for Visible Light Communication

Sources are indexed in [url_index.md](url_index.md). These notes are tuned for this repo's EVK4 visible light communication work: blinking LEDs, event-rate analysis, RAW capture, bias tuning, and Python processing.

## Working Mental Model

An event camera reports changes in log intensity at individual pixels instead of producing full frames at a fixed frame rate. Each event usually carries:

- `x`: pixel column
- `y`: pixel row
- `p`: polarity, commonly ON/OFF or 1/0
- `t`: timestamp, usually microseconds in Metavision event streams

For VLC, a blinking LED should appear as a dense event source at a small image region. The useful signal is often in:

- event count over time,
- polarity-specific count over time,
- timestamp intervals,
- frequency content of the event-rate signal,
- spatial concentration around the LED.

## EVK4 / IMX636 Context

The EVK4 uses the Sony IMX636 event-based sensor family in Prophesee's evaluation kit. Public docs list IMX636 resolution as 1280 x 720 and describe EVK4 as a USB evaluation camera.

Practical implications:

- Aim to isolate the LED in the sensor view before tuning software.
- Use focus, exposure environment, and physical geometry first; then tune biases and ROI.
- Keep a baseline settings JSON for reproducibility.
- Record RAW when experimenting so analysis can be repeated without re-capturing.

## Recommended VLC Workflow

1. Start with default camera settings and record a short RAW baseline.
2. Confirm the LED is spatially visible in Metavision Studio or a simple event-rate plot.
3. Add an ROI around the LED and compare total event rate vs ROI event rate.
4. Sweep one bias at a time and record:
   - command/settings used,
   - LED blink frequency and duty cycle,
   - ambient lighting,
   - lens/focus distance,
   - event rate,
   - bit/frequency decoding quality,
   - noise outside the LED ROI.
5. Use `EventsIterator`, `RawReader`, or a camera stream slicer to convert events into fixed-time bins.
6. Analyze ON/OFF counts separately if the modulation or decoding logic benefits from polarity.
7. Keep RAW files plus a small sidecar metadata file for every capture.

## Biases That Matter Most

Search terms: `bias_diff`, `bias_diff_on`, `bias_diff_off`, `bias_fo`, `bias_hpf`, `bias_refr`.

Bias tuning changes how sensitive the pixels are to changes and how much event noise or event burst behavior appears.

Likely VLC use:

- `bias_diff_on` and `bias_diff_off`: tune ON/OFF threshold sensitivity. Lower thresholds can catch weaker LED changes but may increase noise.
- `bias_diff`: general contrast threshold where available.
- `bias_refr`: refractory behavior. Useful when an LED creates too many repeated events from the same pixels.
- `bias_hpf`: high-pass behavior. Can help reject slow background changes.
- `bias_fo`: source follower / bandwidth related tuning. Worth testing for high-frequency blinking, but change carefully.

Important practice:

- Change one bias at a time.
- Save default and modified settings.
- Record before/after RAW files for every tuning attempt.
- Judge tuning by the decoding target, not only by prettier visualization.

## ROI / RONI

ROI means Region of Interest; RONI means Region of Non-Interest. Pixel selection can reduce output and processing load by keeping the LED area and discarding irrelevant background.

For VLC:

- Use a rectangular ROI around the LED after finding the source location.
- Make the ROI slightly larger than the LED blob to handle small motion or focus blur.
- Compare event-rate spectra inside and outside the ROI.
- Use RONI to mask noisy regions if the LED position is fixed but a background source is problematic.

Search terms: `ROI`, `RONI`, `pixel selection`, `I_ROI`, `Digital Crop`.

## Recording RAW Files

RAW is the safest capture format for experiments because it preserves camera event data and metadata needed for replay/analysis.

Useful capture metadata to store beside a RAW file:

- camera model and serial number,
- Metavision SDK version,
- lens/focus information,
- LED frequency/duty cycle,
- ambient lighting,
- distance/alignment,
- ROI/RONI configuration,
- all bias values,
- recording command,
- capture duration,
- expected bit pattern or symbol rate.

## Python Processing Targets

Search terms in this docs folder:

- `EventsIterator`
- `RawReader`
- `Camera`
- `CameraStreamSlicer`
- `DeviceDiscovery`

For offline RAW analysis:

- Use fixed `delta_t` windows to make a time series.
- Count total events per window and polarity-specific events per window.
- Restrict to LED ROI before binning when possible.
- Apply FFT, autocorrelation, or threshold-based symbol detection on binned counts.

For live camera work:

- Open the camera from the stream API or SDK camera API.
- Slice events by time or count.
- Keep a conservative loop that logs rates and drops data gracefully if the event rate spikes.

## File Formats

Common formats:

- RAW: native recording/replay format; best for repeatable capture.
- DAT: event data file format used by older tools or exported streams.
- HDF5: structured data container, useful for processed datasets.

Event encodings:

- EVT2 / EVT2.1 / EVT3 are event stream encodings used by Prophesee sensors and files.
- For most Python analysis in this repo, prefer SDK readers over hand-parsing binary formats.

## Project-Specific Experiments To Try

- LED frequency sweep with default settings.
- Same sweep with ROI only.
- Sweep `bias_diff_on` and `bias_diff_off` around default values.
- Sweep `bias_refr` when event count saturates.
- Compare indoor light, dark room, and bright background.
- Compare ON-only, OFF-only, and ON+OFF event-rate decoding.
- Record a no-LED baseline for noise estimation.

## Red Flags

- Event rate increases but decoding quality gets worse.
- Background motion dominates the ROI.
- Bias settings are changed without a saved baseline.
- RAW files are kept without metadata, making results hard to reproduce.
- Analysis scripts assume frame-rate behavior instead of event timestamps.

