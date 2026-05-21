# Pixel 7a Receiver Notes

This folder is the local home for Google Pixel 7a notes, capture protocols, and receiver-specific decisions.

Use it the same way `docs/metavision/` is used for EVK4 work: keep device settings, capture metadata, power-measurement notes, and decoding assumptions here so Pixel 7a results are reproducible.

## Files

- `pixel7a_notes.md`
  - Practical capture checklist.
  - Energy-measurement plan.
  - BER/video-decoding notes and limitations.

## Current Role In The Thesis

The Pixel 7a is the frame-based comparison receiver for Section 3.2. Its most important first-pass output is a matched energy-per-bit comparison against the EVK4 condition:

- 300 Hz OOK.
- 50 cm distance.
- f/4 EVK4 baseline lens setting for the matched EVK4 condition.
- Matched ambient lux when possible.
- Same transmitter symbol file: `s31_synced_replication_300Hz_10s_symbols.txt`.

Pixel 7a BER decoding should use a separate video pipeline, not the EVK4 RAW event decoder.
