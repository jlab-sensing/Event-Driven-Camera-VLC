# Section 3.3 Scaffold

This folder is the starting point for the Section 3.3 robustness study.

## Goal

Section 3.3 evaluates whether OOK decoding remains reliable under real-world lighting conditions:

- different ambient lux levels
- artificial vs natural lighting
- LOS, partial shadow, reflection, and partial NLOS setups
- changes in distance or receiver angle when useful

The first useful output is a BER-vs-condition table. Event-rate and event-gap metrics are optional and can be computed later from raw files.

## Files

- `section3_3_robustness_analyze.py`
  - Reads a trial manifest CSV from `data/3.3/`.
  - Computes BER, score fraction, and pooled summaries by lighting/visibility condition.
  - Optionally reads `.raw` captures to estimate event rate and long event-gap/dropout-style metrics.
  - Outputs per-trial CSV, summary CSV, and plots into `data/3.3/` and `plots/3.3/`.

## Trial Manifest

Use one row per trial. The empty template lives at:

- `data/3.3/section3_3_robustness_manifest_template.csv`

A starter manifest with the existing 1500 Hz LOS result and planned future rows lives at:

- `data/3.3/s33_robustness_starter.csv`

Key columns:

- `status`: use `planned` for future rows that should be skipped
- `lux`: measured ambient light level
- `lighting_source`: for example `artificial_room`, `natural_window`, or `mixed`
- `visibility_condition`: for example `LOS`, `partial_shadow`, `reflection`, or `partial_NLOS`
- `bits_transmitted`, `bits_scored`, `bit_errors`: BER inputs
- `active_start_s`, `active_end_s`: optional active decode window for raw event metrics
- `capture_file`: optional `.raw` file used when running with `--with_raw_metrics`

## Example

From the repository root:

```powershell
python .\src\3.3\section3_3_robustness_analyze.py --manifest .\data\3.3\s33_robustness_starter.csv --out_prefix s33_robustness_starter
```

If the `.raw` files are available and you want event-gap metrics:

```powershell
python .\src\3.3\section3_3_robustness_analyze.py --manifest .\data\3.3\s33_robustness_starter.csv --out_prefix s33_robustness_starter_raw --with_raw_metrics
```
