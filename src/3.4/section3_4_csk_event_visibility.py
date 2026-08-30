"""Summarize the four-state CSK RAW recordings with one easy-to-explain metric.

This is not a CSK decoder and does not calculate BER or symbol error rate.
It reads the already-extracted per-symbol LED-ROI event counts, groups them by
the known repeating state (1, 2, 3, or 4), and asks whether the total counts
show a clear separation.  The output is exploratory event-visibility evidence.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "3.4" / "s34_csk_ratio4_300hz_raw_feasibility_per_symbol.csv"
DATA_DIR = ROOT / "data" / "3.4"
PLOT_DIR = ROOT / "plots" / "3.4"
OUT_PREFIX = "s34_csk_ratio4_300hz_event_visibility"
STATES = ("1", "2", "3", "4")


def read_total_event_counts():
    """Return total LED-ROI event counts grouped by known state and capture."""
    counts = defaultdict(list)
    by_trial = defaultdict(lambda: defaultdict(list))
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state = row["truth_symbol"]
            trial = row["trial_id"]
            count = float(row["total_events"])
            counts[state].append(count)
            by_trial[trial][state].append(count)
    return counts, by_trial


def summarize(counts, by_trial):
    rows = []
    for state in STATES:
        values = np.asarray(counts[state], dtype=float)
        trial_means = {
            trial: float(np.mean(by_trial[trial][state]))
            for trial in sorted(by_trial)
        }
        rows.append({
            "state": state,
            "symbol_intervals": len(values),
            "mean_total_events": float(np.mean(values)),
            "median_total_events": float(np.median(values)),
            "sample_std_total_events": float(np.std(values, ddof=1)),
            "trial_means": trial_means,
        })
    return rows


def write_summary(rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{OUT_PREFIX}_by_state.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "known_state", "symbol_intervals", "mean_total_events",
            "median_total_events", "sample_std_total_events",
            "interpretation",
        ])
        for row in rows:
            means = row["trial_means"]
            writer.writerow([
                row["state"], row["symbol_intervals"],
                f"{row['mean_total_events']:.2f}",
                f"{row['median_total_events']:.2f}",
                f"{row['sample_std_total_events']:.2f}",
                "descriptive LED-ROI event-visibility result; not CSK decoding",
            ])
    return path


def plot_summary(rows):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / f"{OUT_PREFIX}_by_state.png"
    x = np.arange(len(STATES))
    means = np.array([row["mean_total_events"] for row in rows])
    spreads = np.array([row["sample_std_total_events"] for row in rows])

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.errorbar(
        x, means, yerr=spreads, fmt="o", color="#1f77b4", ecolor="#6b7280",
        capsize=5, markersize=7, label="Mean ± sample\nstandard deviation",
    )
    ax.set_xticks(x, [f"State {state}" for state in STATES])
    ax.set_ylabel("Total LED-ROI events per 3.33 ms interval")
    ax.set_title("CSK event visibility: total event count by known state")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.set_ylim(7000, 13200)
    mean_range = max(means) - min(means)
    fig.subplots_adjust(left=0.12, right=0.56, bottom=0.23, top=0.86)
    fig.text(
        0.42, 0.06,
        f"State-mean range = {mean_range:.1f} events; each state has ~2,700-event spread.",
        ha="center", va="center", fontsize=9, color="#374151",
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    counts, by_trial = read_total_event_counts()
    rows = summarize(counts, by_trial)
    summary_path = write_summary(rows)
    plot_path = plot_summary(rows)
    mean_range = max(row["mean_total_events"] for row in rows) - min(row["mean_total_events"] for row in rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {plot_path}")
    print(f"Range of state means: {mean_range:.2f} events")
    for row in rows:
        print(
            f"State {row['state']}: n={row['symbol_intervals']}, "
            f"mean={row['mean_total_events']:.2f}, "
            f"median={row['median_total_events']:.2f}, "
            f"sample SD={row['sample_std_total_events']:.2f}"
        )


if __name__ == "__main__":
    main()
