"""Create the EVK4 receiver power-above-idle comparison figure.

The plot uses the saved receiver-only USB-meter trial files.  It intentionally
shows absolute idle and active receiver power together, while annotating the
small active-minus-idle increase for each tested OOK rate.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "3.2"
OUTPUT_PATH = ROOT / "plots" / "3.2" / "s32_evk4_receiver_power_above_idle.png"

CONDITIONS = (
    {
        "rate_hz": 300,
        "illuminance_lux": 500,
        "status": "Validated",
        "file": "s32_energy_evk4_ook_synced_300hz_50cm_per_trial.csv",
    },
    {
        "rate_hz": 1000,
        "illuminance_lux": 430,
        "status": "High BER",
        "file": "s32_energy_evk4_ook_1000hz_per_trial.csv",
    },
    {
        "rate_hz": 1500,
        "illuminance_lux": 540,
        "status": "High BER",
        "file": "s32_energy_evk4_ook_1500hz_receiver_only_pooled_per_trial.csv",
    },
)


def read_trial_powers(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    active = [float(row["sensing_power_w"]) for row in rows]
    idle = [float(row["sensing_idle_power_w"]) for row in rows]
    return active, idle


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    active_means: list[float] = []
    active_stds: list[float] = []
    idle_means: list[float] = []
    idle_stds: list[float] = []

    for condition in CONDITIONS:
        active, idle = read_trial_powers(DATA_DIR / condition["file"])
        active_means.append(statistics.mean(active))
        active_stds.append(sample_std(active))
        idle_means.append(statistics.mean(idle))
        idle_stds.append(sample_std(idle))

    x = np.arange(len(CONDITIONS))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.2, 5.9), constrained_layout=True)

    axis.bar(
        x - width / 2,
        idle_means,
        width,
        yerr=idle_stds,
        capsize=4,
        color="#a8a8a8",
        edgecolor="#555555",
        linewidth=0.7,
        label="Idle receiver power",
    )
    axis.bar(
        x + width / 2,
        active_means,
        width,
        yerr=active_stds,
        capsize=4,
        color="#2171b5",
        edgecolor="#124a7a",
        linewidth=0.7,
        label="Active receiver power",
    )

    for index, (condition, active, idle) in enumerate(
        zip(CONDITIONS, active_means, idle_means)
    ):
        delta_mw = round((active - idle) * 1000)
        axis.text(
            x[index] + width / 2,
            active + active_stds[index] + 0.013,
            f"+{delta_mw} mW",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )

    axis.set_title("EVK4 Receiver Power Above Idle", fontsize=15, fontweight="bold", pad=12)
    axis.set_ylabel("Receiver supply power (W)")
    axis.set_xticks(
        x,
        [
            f"{condition['rate_hz']} Hz\n{condition['illuminance_lux']} lux\n{condition['status']}"
            for condition in CONDITIONS
        ],
    )
    axis.set_ylim(0, 0.60)
    axis.set_yticks(np.arange(0, 0.61, 0.10))
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=True,
        borderaxespad=0,
    )
    axis.text(
        0.5,
        -0.36,
        "Bars show trial means; error bars show sample standard deviation. "
        "Test illuminance differed across conditions.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
