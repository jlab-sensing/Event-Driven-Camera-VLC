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

## Lab Journal Workflow

When the user says they are done for the day, packing up, or asks for a daily lab-journal entry:

1. Use a Markdown file in `lab_journal/` named with the month and day, for example `5-29.md`.
2. Before creating a new file, check whether that date's file already exists. If it does, write in that existing file instead of making another journal file for the same day.
3. If the existing journal file already contains user-written notes, preserve them. Do not overwrite, erase, or silently rewrite the user's notes.
4. If user-written notes need to be organized, place them under a section named `## My Own Written Notes`, keeping the original meaning and details intact.
5. Add or update assistant-written sections around the user's notes to summarize what was done that day using the actual files, captures, measurements, commands, and decisions from the session.
6. Include important notes, findings, unresolved questions, and the next concrete step.
7. Keep the journal factual and thesis-useful. Do not overstate completion; distinguish measured results from planned or pending work.
