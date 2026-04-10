# Section 3.2 Scaffold

This folder is the starting point for the Section 3.2 energy study.

## Files

- `section3_2_energy_analyze.py`
  - Reads a trial manifest CSV from `data/3.2/`.
  - Computes sensing, compute, and total energy.
  - Reports gross and idle-subtracted active `J/bit`.
  - Outputs a per-trial CSV, a pooled summary CSV, and basic plots.

## Trial manifest intent

Use one row per trial. The template lives at:

- `data/3.2/section3_2_energy_manifest_template.csv`

The key columns are:

- `sensor_type`: for example `EVK4` or `Pixel7a`
- `modulation`: for example `OOK`
- `frequency_hz`: requested bit rate for the transmission
- `bits_transmitted`: total transmitted bits in that trial
- `bits_scored`: optional; leave blank to treat all transmitted bits as scored
- `bit_errors`: total decoded bit errors in the scored bit set
- `sensing_window_s`: duration of the camera-power measurement window
- `sensing_power_w`: average camera/system sensing power over that window
- `sensing_idle_power_w`: matched idle baseline with the transmitter off
- `compute_window_s`: duration of the decode/processing power window
- `compute_power_w`: average host compute power over that window
- `compute_idle_power_w`: matched host idle baseline

The analyzer floor-clips idle-subtracted power at zero so small baseline noise does not create negative active energy.

## Example

After filling in at least one real trial row, run from the repository root:

```powershell
python .\src\3.2\section3_2_energy_analyze.py --manifest .\data\3.2\section3_2_energy_manifest_template.csv --out_prefix s32_energy_trialset
```

The analyzer will then emit the summary CSVs and figures into `data/3.2/` and `plots/3.2/`.
