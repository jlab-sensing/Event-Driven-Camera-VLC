"""Derive a shared-200000-cycle receiver-energy comparison from saved measurements.

This script does not create new meter measurements.  It combines (1) prior
active-minus-idle receiver-power readings and (2) the saved 1000-symbol/s CFK
packet-match summaries.  The result is an incremental receiver-sensing energy
estimate, excluding transmitter, BeagleBone, and decode-compute energy.
"""

import csv
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "3.2"
PLOTS = ROOT / "plots" / "3.2"
PIXEL_NOTES = ROOT / "docs" / "pixel7a" / "pixel7a_notes.md"
EVK4_POWER = DATA / "s32_energy_evk4_ook_synced_300hz_50cm_per_trial.csv"
EVK4_MATCHES = [
    DATA / "s32_evk4_packet_match_7_14_500_200000_summary.csv",
    DATA / "s32_7_15_evk4_packet_match_200kcycles_repeat_summary.csv",
]
PIXEL_MATCHES = [
    DATA / "s32_pixel7a_packet_match_7_14_500_200000_widestripe_summary.csv",
    DATA / "s32_7_15_pixeltest5_packet_match_200kcycles_repeat_summary.csv",
    DATA / "s32_7_15_pixeltest6_packet_match_200kcycles_repeat_wide_summary.csv",
]
SUMMARY_OUT = DATA / "s32_shared_200k_cfk_receiver_energy_comparison_summary.csv"
PLOT_OUT = PLOTS / "s32_shared_200k_cfk_receiver_energy_comparison.png"
SYMBOL_RATE_HZ = 1000.0


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def evk4_power_deltas():
    with EVK4_POWER.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [float(row["sensing_power_w"]) - float(row["sensing_idle_power_w"]) for row in rows]


def pixel_power_deltas():
    text = PIXEL_NOTES.read_text(encoding="utf-8")
    matches = re.findall(
        r"T[12]: idle ([0-9.]+) W, active recording ([0-9.]+) W, active-minus-idle ([0-9.]+) W",
        text,
    )
    if len(matches) != 2:
        raise ValueError("Could not locate the two saved Pixel active-minus-idle readings")
    return [float(match[2]) for match in matches]


def weighted_match(files, scored_key, error_key):
    scored = errors = 0
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                scored += int(float(row[scored_key]))
                errors += int(float(row[error_key]))
    return scored, errors, (scored - errors) / scored


def build_receiver(name, deltas, match_files, scored_key, error_key, power_source, match_source):
    symbols_scored, symbol_errors, match_rate = weighted_match(match_files, scored_key, error_key)
    mean_delta = statistics.mean(deltas)
    delta_std = sample_std(deltas)
    energy_per_tx = mean_delta / SYMBOL_RATE_HZ
    energy_per_tx_std = delta_std / SYMBOL_RATE_HZ
    return {
        "receiver": name,
        "power_trials": len(deltas),
        "mean_active_minus_idle_power_w": mean_delta,
        "power_reading_sample_std_w": delta_std,
        "shared_symbol_rate_hz": SYMBOL_RATE_HZ,
        "estimated_incremental_j_per_tx_symbol": energy_per_tx,
        "estimated_j_per_tx_symbol_sample_std": energy_per_tx_std,
        "cfk_captures": len(match_files),
        "symbols_scored": symbols_scored,
        "symbol_errors": symbol_errors,
        "aggregate_symbol_match_rate": match_rate,
        "estimated_incremental_j_per_correct_symbol": energy_per_tx / match_rate,
        "power_source": power_source,
        "packet_match_source": match_source,
        "scope": "Derived receiver sensing estimate; excludes transmitter, BeagleBone, and decode-compute energy",
    }


def write_summary(rows):
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows):
    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["receiver"] for row in rows]
    tx_uj = [row["estimated_incremental_j_per_tx_symbol"] * 1e6 for row in rows]
    correct_uj = [row["estimated_incremental_j_per_correct_symbol"] * 1e6 for row in rows]
    tx_err_uj = [row["estimated_j_per_tx_symbol_sample_std"] * 1e6 for row in rows]
    correct_err_uj = [error / row["aggregate_symbol_match_rate"] for error, row in zip(tx_err_uj, rows)]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.24)
    ax.bar([v - width / 2 for v in x], tx_uj, width, yerr=tx_err_uj, capsize=5,
           color="#2f6db0", label="Per transmitted symbol")
    ax.bar([v + width / 2 for v in x], correct_uj, width, yerr=correct_err_uj, capsize=5,
           color="#d97706", label="Per correctly matched symbol")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Incremental receiver energy (uJ/symbol)")
    ax.set_title("Shared 200000-cycle CFK receiver-energy estimate")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for xi, value in zip([v - width / 2 for v in x], tx_uj):
        ax.text(xi, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for xi, value in zip([v + width / 2 for v in x], correct_uj):
        ax.text(xi, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.text(0.5, 0.06,
             "Derived from prior active-minus-idle receiver power; error bars show sample spread in saved power readings.",
             ha="center", fontsize=8)
    fig.savefig(PLOT_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = [
        build_receiver(
            "EVK4", evk4_power_deltas(), EVK4_MATCHES, "symbols_scored", "symbol_errors",
            "s32_energy_evk4_ook_synced_300hz_50cm_per_trial.csv",
            "s32_evk4_packet_match_7_14_500_200000_summary.csv; s32_7_15_evk4_packet_match_200kcycles_repeat_summary.csv",
        ),
        build_receiver(
            "Pixel 7a", pixel_power_deltas(), PIXEL_MATCHES, "aggregate_symbols_scored", "aggregate_symbol_errors",
            "docs/pixel7a/pixel7a_notes.md",
            "s32_pixel7a_packet_match_7_14_500_200000_widestripe_summary.csv; s32_7_15_pixeltest5_packet_match_200kcycles_repeat_summary.csv; s32_7_15_pixeltest6_packet_match_200kcycles_repeat_wide_summary.csv",
        ),
    ]
    write_summary(rows)
    plot(rows)
    for row in rows:
        print(
            f"{row['receiver']}: {row['estimated_incremental_j_per_tx_symbol']:.6g} J/tx symbol, "
            f"{row['estimated_incremental_j_per_correct_symbol']:.6g} J/correct symbol, "
            f"match={row['aggregate_symbol_match_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
