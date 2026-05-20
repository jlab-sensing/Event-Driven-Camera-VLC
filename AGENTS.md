# Project Instructions

For tasks involving EVK4, Metavision SDK, event-camera data, camera streaming, RAW files, biases, or VLC experiments, first search:

docs/metavision/

Useful commands:
- rg -n "EVK4|event camera|Metavision" docs/metavision
- rg -n "bias_diff|bias_refr|bias_fo|bias_hpf" docs/metavision
- rg -n "RAW|recording|file format" docs/metavision
- rg -n "EventsIterator|Camera|RawReader" docs/metavision

For unrelated tasks, do not force the Metavision docs. Use the normal project files or other sources as needed.

## Default BER Workflow

For new Section 3.1 BER / replication tests, use the fast standard workflow unless the user asks for a broader comparison:

1. Calibrate each new `.raw` capture once with `src/3.1/section3_1_replication_calibrate.py`.
2. Run the BER analysis once with `decode_mode=synced_transition` and `--use_calibrated_frequency`.
3. Use the manifest that matches the transmitter symbol files.
4. Add `--no_plot` for quick sanity checks; make plots only for final/report runs.
5. Do not run every decoder variant by default. Only compare `synced_activity`, manifest frequency, or manual ROI runs if the calibrated synced-transition result looks suspicious.

Current validated thesis condition for moving into Section 3.2:

- 300 Hz OOK, synced preamble, 50 cm distance, f/4 aperture.
- Use the synced transition decoder.
- The 50 cm sweep showed clean 200 Hz and 300 Hz decoding; 400-500 Hz degraded.

More detail: `docs/ber_workflow.md`.
