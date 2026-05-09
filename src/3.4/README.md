# Section 3.4 Scaffold

This folder is the starting point for the Section 3.4 color-shift-keying study.

## Goal

Section 3.4 extends the OOK timing, reliability, and energy questions into color-coded VLC. The proposal names three useful paths:

- 4-state RGB ratio CSK using the current PRU `1`, `2`, `3`, `4` symbol mapping
- spatially separated RGB LED regions
- frequency-encoded color channels

The first practical starting point is the 4-state RGB ratio case because `pru1_pwm_CSK_1000Hz/main.c` already maps ASCII symbols into RGB duty-cycle states.

## Files

- `section3_4_csk_generate_symbols.py`
  - Generates PRU symbol files for the current `1`..`4` RGB state mapping.
  - Writes a symbol-generation manifest with requested/actual symbol rate and message length.
- `section3_4_csk_analyze.py`
  - Reads a Section 3.4 trial manifest from `data/3.4/`.
  - Computes symbol error rate (SER), bit error rate (BER), score fraction, optional energy per bit, and a confusion matrix when truth/decoded symbols are present.
  - Writes per-trial CSVs, summary CSVs, and plots into `data/3.4/` and `plots/3.4/`.

## Current Symbol Mapping

The current PRU firmware maps:

- `0` or `o`: off/guard
- `w`: white/all channels on
- `1`: CSK state for bit pair `00`
- `2`: CSK state for bit pair `01`
- `3`: CSK state for bit pair `10`
- `4`: CSK state for bit pair `11`

Those duty cycles are defined in `pru1_pwm_CSK_1000Hz/symbols.h`.

## Generate Starter Symbols

From the repository root:

```powershell
python .\src\3.4\section3_4_csk_generate_symbols.py --symbol_rates_hz 1000 1500 --target_duration_s 3 --out_prefix s34_csk_ratio4
```

By default this writes files into `pru1_pwm_CSK_1000Hz/userspace/`, ready for the BBB userspace loader.

## Analyze Completed Trials

Fill in real trial rows in:

- `data/3.4/s34_csk_starter.csv`

Then run:

```powershell
python .\src\3.4\section3_4_csk_analyze.py --manifest .\data\3.4\s34_csk_starter.csv --out_prefix s34_csk_starter
```

Rows marked `planned` are ignored until real decoded results are available.
