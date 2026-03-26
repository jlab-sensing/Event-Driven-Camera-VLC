import argparse
import csv
import os
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_utils import repo_root_from_this_file
from latency_analyze import load_timestamps_us


def workspace_root_from_this_file(this_file: str) -> str:
    return os.path.abspath(os.path.join(repo_root_from_this_file(this_file), ".."))


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def finite_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def finite_p95(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95))


def iei_cv(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    mu = float(np.mean(values))
    if mu <= 0:
        return float("nan")
    return float(np.std(values) / mu)


def load_detected_pulses(per_pulse_csv: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    with open(per_pulse_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Per-pulse CSV is missing a header row.")

        required = {"trial", "raw_file", "toggle_index", "toggle_s", "first_event_s", "detected_within_window"}
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Per-pulse CSV is missing required columns: {missing_str}")

        for row in reader:
            if row["detected_within_window"].strip() != "1":
                continue
            if not row["first_event_s"].strip():
                continue

            trial = row["trial"].strip()
            grouped.setdefault(trial, []).append({
                "trial": trial,
                "raw_file": row["raw_file"].strip(),
                "toggle_index": int(row["toggle_index"]),
                "toggle_s": float(row["toggle_s"]),
                "first_event_s": float(row["first_event_s"]),
            })

    if not grouped:
        raise ValueError("No detected pulses were found in the per-pulse CSV.")

    for pulses in grouped.values():
        pulses.sort(key=lambda item: item["toggle_index"])

    return grouped


def flatten_arrays(chunks: Iterable[np.ndarray]) -> np.ndarray:
    non_empty = [chunk for chunk in chunks if chunk.size > 0]
    if not non_empty:
        return np.array([], dtype=np.float64)
    return np.concatenate(non_empty)


def summarize_rows(rows: List[dict]) -> dict:
    counts = np.array([row["n_events_in_window"] for row in rows], dtype=np.float64)
    durations = np.array(
        [row["burst_duration_us"] for row in rows if np.isfinite(row["burst_duration_us"])],
        dtype=np.float64,
    )
    dt_chunks = [row["dt_values_us"] for row in rows]
    all_dt = flatten_arrays(dt_chunks)

    return {
        "n_detected_pulses": len(rows),
        "n_windows_with_two_plus_events": int(sum(row["n_events_in_window"] >= 2 for row in rows)),
        "mean_events_in_window": finite_mean(counts),
        "median_events_in_window": finite_median(counts),
        "mean_burst_duration_us": finite_mean(durations),
        "dt_mean_us": finite_mean(all_dt),
        "dt_median_us": finite_median(all_dt),
        "dt_p95_us": finite_p95(all_dt),
        "iei_cv": iei_cv(all_dt),
        "dt_samples": int(all_dt.size),
    }


def save_overall_dt_histogram(
    dt_values_us: np.ndarray,
    out_path: str,
    title: str,
    bins: int = 60,
) -> None:
    if dt_values_us.size == 0:
        raise ValueError("Cannot plot histogram: no inter-event intervals were computed.")

    median_us = float(np.median(dt_values_us))
    p95_us = float(np.percentile(dt_values_us, 95))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(dt_values_us, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    ax.axvline(median_us, color="#F58518", linestyle="--", linewidth=2, label=f"median = {median_us:.2f} us")
    ax.axvline(p95_us, color="#E45756", linestyle="--", linewidth=2, label=f"p95 = {p95_us:.2f} us")
    ax.set_xlabel("Inter-event interval (us)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    repo_root = repo_root_from_this_file(__file__)
    workspace_root = workspace_root_from_this_file(__file__)

    default_per_pulse_csv = os.path.join(
        repo_root, "data", "latency", "s31_latency_1000Hz_10pulses_per_pulse.csv"
    )
    default_captures_dir = os.path.join(workspace_root, "captures", "section3_1_1000Hz")

    ap = argparse.ArgumentParser(
        description="Analyze Section 3.1 inter-event timing from the existing 1000 Hz baseline captures."
    )
    ap.add_argument(
        "--per_pulse_csv",
        default=default_per_pulse_csv,
        help="Latency per-pulse CSV containing first_event_s for each detected pulse.",
    )
    ap.add_argument(
        "--captures_dir",
        default=default_captures_dir,
        help="Directory containing the Section 3.1 baseline .raw files.",
    )
    ap.add_argument(
        "--window_us",
        type=float,
        default=250.0,
        help="Analysis window after the anchor time, in microseconds.",
    )
    ap.add_argument(
        "--anchor",
        choices=["first_event", "toggle"],
        default="first_event",
        help="Use the first detected event or the scheduled toggle as the start of the analysis window.",
    )
    ap.add_argument(
        "--out_prefix",
        default="s31_interevent_1000Hz",
        help="Prefix for the generated CSV files.",
    )
    ap.add_argument(
        "--hist_bins",
        type=int,
        default=60,
        help="Number of bins for the overall inter-event histogram.",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = ap.parse_args()

    if args.window_us <= 0:
        raise ValueError("--window_us must be > 0")
    if args.hist_bins <= 0:
        raise ValueError("--hist_bins must be > 0")

    if not os.path.exists(args.per_pulse_csv):
        raise FileNotFoundError(args.per_pulse_csv)
    if not os.path.isdir(args.captures_dir):
        raise FileNotFoundError(args.captures_dir)

    pulses_by_trial = load_detected_pulses(args.per_pulse_csv)

    data_dir = os.path.join(repo_root, "data", "inter_event")
    plot_dir = os.path.join(repo_root, "plots", "inter event")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    per_pulse_out = os.path.join(data_dir, f"{args.out_prefix}_per_pulse.csv")
    summary_out = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    hist_out = os.path.join(plot_dir, f"{args.out_prefix}_dt_histogram.png")

    summary_rows: List[dict] = []
    all_rows: List[dict] = []
    window_s = args.window_us * 1e-6

    for trial, pulses in pulses_by_trial.items():
        raw_file = pulses[0]["raw_file"]
        raw_path = os.path.join(args.captures_dir, raw_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Could not find raw file for trial '{trial}': {raw_path}"
            )

        ts_us = load_timestamps_us(raw_path)
        if ts_us.size == 0:
            raise ValueError(f"No events found in {raw_path}")
        ts_rel_s = (ts_us - ts_us[0]).astype(np.float64) * 1e-6

        trial_rows: List[dict] = []
        for pulse in pulses:
            anchor_s = pulse["first_event_s"] if args.anchor == "first_event" else pulse["toggle_s"]
            seg = ts_rel_s[(ts_rel_s >= anchor_s) & (ts_rel_s <= anchor_s + window_s)]

            dt_values_us = np.diff(seg) * 1e6 if seg.size >= 2 else np.array([], dtype=np.float64)
            row = {
                "trial": trial,
                "raw_file": raw_file,
                "toggle_index": pulse["toggle_index"],
                "toggle_s": pulse["toggle_s"],
                "first_event_s": pulse["first_event_s"],
                "anchor": args.anchor,
                "anchor_s": anchor_s,
                "window_us": args.window_us,
                "n_events_in_window": int(seg.size),
                "burst_duration_us": float((seg[-1] - seg[0]) * 1e6) if seg.size >= 2 else float("nan"),
                "dt_mean_us": finite_mean(dt_values_us),
                "dt_median_us": finite_median(dt_values_us),
                "dt_p95_us": finite_p95(dt_values_us),
                "iei_cv": iei_cv(dt_values_us),
                "dt_values_us": dt_values_us,
            }
            trial_rows.append(row)
            all_rows.append(row)

        summary = summarize_rows(trial_rows)
        summary.update({
            "trial": trial,
            "raw_file": raw_file,
            "anchor": args.anchor,
            "window_us": args.window_us,
        })
        summary_rows.append(summary)

    overall = summarize_rows(all_rows)
    overall.update({
        "trial": "OVERALL",
        "raw_file": "",
        "anchor": args.anchor,
        "window_us": args.window_us,
    })
    summary_rows.append(overall)

    with open(per_pulse_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial",
            "raw_file",
            "toggle_index",
            "toggle_s",
            "first_event_s",
            "anchor",
            "anchor_s",
            "window_us",
            "n_events_in_window",
            "burst_duration_us",
            "dt_mean_us",
            "dt_median_us",
            "dt_p95_us",
            "iei_cv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out_row = {key: row[key] for key in fieldnames}
            writer.writerow(out_row)

    with open(summary_out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial",
            "raw_file",
            "anchor",
            "window_us",
            "n_detected_pulses",
            "n_windows_with_two_plus_events",
            "mean_events_in_window",
            "median_events_in_window",
            "mean_burst_duration_us",
            "dt_mean_us",
            "dt_median_us",
            "dt_p95_us",
            "iei_cv",
            "dt_samples",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    overall_dt_values = flatten_arrays([row["dt_values_us"] for row in all_rows])
    if not args.no_plot:
        save_overall_dt_histogram(
            dt_values_us=overall_dt_values,
            out_path=hist_out,
            title="Section 3.1 inter-event interval histogram",
            bins=args.hist_bins,
        )

    print("Saved per-pulse CSV:", per_pulse_out)
    print("Saved summary CSV:", summary_out)
    if not args.no_plot:
        print("Saved plot:", hist_out)

    overall_row = summary_rows[-1]
    print("Overall Section 3.1 inter-event summary")
    print("  anchor:", overall_row["anchor"])
    print("  window_us:", overall_row["window_us"])
    print("  detected pulses:", overall_row["n_detected_pulses"])
    print("  windows with >=2 events:", overall_row["n_windows_with_two_plus_events"])
    print("  mean events per window:", f"{overall_row['mean_events_in_window']:.2f}")
    print("  mean burst duration (us):", f"{overall_row['mean_burst_duration_us']:.2f}")
    print("  dt median (us):", f"{overall_row['dt_median_us']:.2f}")
    print("  dt p95 (us):", f"{overall_row['dt_p95_us']:.2f}")
    print("  iei cv:", f"{overall_row['iei_cv']:.3f}")


if __name__ == "__main__":
    main()
