import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator

from io_utils import repo_root_from_this_file


def load_timestamps_us(raw_path: str) -> np.ndarray:
    """Load all event timestamps (microseconds) from a Metavision .raw file."""
    events = EventsIterator(input_path=raw_path)
    chunks: List[np.ndarray] = []
    for evs in events:
        if evs.size:
            chunks.append(evs["t"].astype(np.int64))

    if not chunks:
        return np.array([], dtype=np.int64)

    ts_us = np.concatenate(chunks)
    # Usually already sorted, but sort defensively if not.
    if ts_us.size > 1 and np.any(np.diff(ts_us) < 0):
        ts_us = np.sort(ts_us)
    return ts_us


def parse_float_tokens(text: str) -> np.ndarray:
    toks = re.split(r"[,\s]+", text.strip())
    vals = [float(t) for t in toks if t]
    arr = np.array(vals, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("No toggle times found.")
    return np.sort(arr)


def load_toggle_times_file(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        return parse_float_tokens(f.read())


def load_toggle_times_csv(path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, List[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("Toggle CSV is missing a header row.")

        has_trial = "trial" in r.fieldnames
        has_raw = "raw_file" in r.fieldnames
        has_toggle = "toggle_s" in r.fieldnames
        if not has_toggle or (not has_trial and not has_raw):
            raise ValueError("Toggle CSV must contain toggle_s and trial or raw_file columns.")

        key_col = "trial" if has_trial else "raw_file"
        for row in r:
            key = row.get(key_col, "").strip()
            toggle_s = row.get("toggle_s", "").strip()
            if not key or not toggle_s:
                continue
            out.setdefault(key, []).append(float(toggle_s))

    if not out:
        raise ValueError("No valid toggle entries found in toggle CSV.")

    return {k: np.sort(np.array(v, dtype=np.float64)) for k, v in out.items()}


def build_toggle_schedule(first_toggle_s: float, pulse_period_s: float, num_pulses: int) -> np.ndarray:
    if num_pulses <= 0:
        raise ValueError("num_pulses must be > 0")
    if pulse_period_s <= 0:
        raise ValueError("pulse_period_s must be > 0")
    return first_toggle_s + np.arange(num_pulses, dtype=np.float64) * pulse_period_s


def compute_latencies_us(
    ts_rel_s: np.ndarray,
    toggle_s: np.ndarray,
    max_latency_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each toggle time, find the first event at/after the toggle.
    Mark as detected only if latency <= max_latency_s.
    """
    first_event_s = np.full(toggle_s.shape, np.nan, dtype=np.float64)
    latency_us = np.full(toggle_s.shape, np.nan, dtype=np.float64)
    detected = np.zeros(toggle_s.shape, dtype=bool)

    if ts_rel_s.size == 0 or toggle_s.size == 0:
        return first_event_s, latency_us, detected

    idx = np.searchsorted(ts_rel_s, toggle_s, side="left")
    for i, j in enumerate(idx):
        if j >= ts_rel_s.size:
            continue
        lat_s = ts_rel_s[j] - toggle_s[i]
        if lat_s < 0 or lat_s > max_latency_s:
            continue
        first_event_s[i] = ts_rel_s[j]
        latency_us[i] = lat_s * 1e6
        detected[i] = True

    return first_event_s, latency_us, detected


def iqr_outlier_mask(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=bool)
    finite_idx = np.where(np.isfinite(values))[0]
    if finite_idx.size < 4:
        return out

    v = values[finite_idx]
    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = q3 - q1
    if iqr <= 0:
        return out

    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    bad_local = (v < lo) | (v > hi)
    out[finite_idx[bad_local]] = True
    return out


def summarize_latency(latency_us: np.ndarray, detected: np.ndarray, iqr_outliers: np.ndarray) -> Dict[str, float]:
    valid = latency_us[np.isfinite(latency_us)]
    n_toggles = int(latency_us.size)
    n_detected = int(np.sum(detected))
    out_count = int(np.sum(iqr_outliers))

    if valid.size == 0:
        return {
            "n_toggles": n_toggles,
            "n_detected": n_detected,
            "detected_frac": float(n_detected / n_toggles) if n_toggles > 0 else float("nan"),
            "latency_mean_us": float("nan"),
            "latency_std_us": float("nan"),
            "latency_median_us": float("nan"),
            "latency_p95_us": float("nan"),
            "latency_min_us": float("nan"),
            "latency_max_us": float("nan"),
            "iqr_outlier_count": out_count,
            "iqr_outlier_frac": float("nan"),
        }

    return {
        "n_toggles": n_toggles,
        "n_detected": n_detected,
        "detected_frac": float(n_detected / n_toggles) if n_toggles > 0 else float("nan"),
        "latency_mean_us": float(np.mean(valid)),
        "latency_std_us": float(np.std(valid)),
        "latency_median_us": float(np.median(valid)),
        "latency_p95_us": float(np.percentile(valid, 95)),
        "latency_min_us": float(np.min(valid)),
        "latency_max_us": float(np.max(valid)),
        "iqr_outlier_count": out_count,
        "iqr_outlier_frac": float(out_count / valid.size) if valid.size > 0 else float("nan"),
    }


def mark_outlier_trials_by_mean(summary_rows: List[Dict[str, float]]) -> Dict[str, float]:
    trial_rows = [r for r in summary_rows if r["trial"] != "OVERALL" and np.isfinite(r["latency_mean_us"])]
    flags: Dict[str, float] = {r["trial"]: 0.0 for r in summary_rows if r["trial"] != "OVERALL"}
    zmap: Dict[str, float] = {r["trial"]: float("nan") for r in summary_rows if r["trial"] != "OVERALL"}

    if len(trial_rows) < 3:
        return zmap

    means = np.array([r["latency_mean_us"] for r in trial_rows], dtype=np.float64)
    med = float(np.median(means))
    mad = float(np.median(np.abs(means - med)))
    if mad <= 0:
        return zmap

    mz = 0.6745 * (means - med) / mad
    for row, z in zip(trial_rows, mz):
        zmap[row["trial"]] = float(z)
        flags[row["trial"]] = float(abs(z) > 3.5)

    for r in summary_rows:
        if r["trial"] == "OVERALL":
            continue
        r["trial_mean_outlier_flag"] = flags[r["trial"]]

    return zmap


def wrap_label_on_underscores(label: str, max_line_len: int = 16) -> str:
    """Wrap long labels at underscores so x-tick text stays readable."""
    parts = label.split("_")
    lines: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) <= max_line_len:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


def infer_pulse_width_token(label: str) -> Optional[str]:
    match = re.search(r"(?:^|_)pw_(\d+(?:p\d+)?)ms(?:_|$)", label)
    if not match:
        return None
    return match.group(1)


def pulse_width_token_to_ms(token: str) -> float:
    return float(token.replace("p", "."))


def infer_trial_suffix(label: str) -> Optional[str]:
    match = re.search(r"(?:^|_)t(\d+)$", label)
    if match:
        return f"t{match.group(1)}"

    match = re.search(r"(?:^|_)trial(\d+)(?:_|$)", label)
    if match:
        return f"trial{match.group(1)}"

    return None


def format_trial_label_for_plot(label: str) -> str:
    pulse_width_token = infer_pulse_width_token(label)
    trial_suffix = infer_trial_suffix(label)
    if pulse_width_token:
        base = f"{pulse_width_token}ms"
        return f"{base}\n{trial_suffix}" if trial_suffix else base
    return wrap_label_on_underscores(label, max_line_len=12)


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def finite_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def build_pulse_width_summary(
    summary_rows: List[Dict[str, float]],
    latencies_by_trial: Dict[str, np.ndarray],
) -> List[Dict[str, float]]:
    grouped_trials: Dict[str, List[Dict[str, float]]] = {}

    for row in summary_rows:
        if row["trial"] == "OVERALL":
            continue

        pulse_width_token = infer_pulse_width_token(str(row["trial"]))
        if pulse_width_token is None:
            pulse_width_token = infer_pulse_width_token(str(row["raw_file"]))
        if pulse_width_token is None:
            continue

        grouped_trials.setdefault(pulse_width_token, []).append(row)

    grouped_rows: List[Dict[str, float]] = []
    for pulse_width_token in sorted(grouped_trials.keys(), key=pulse_width_token_to_ms):
        rows = grouped_trials[pulse_width_token]
        pooled_latencies = [
            latencies_by_trial.get(str(row["trial"]), np.array([], dtype=np.float64))
            for row in rows
        ]
        pooled = (
            np.concatenate([lat for lat in pooled_latencies if lat.size > 0])
            if any(lat.size > 0 for lat in pooled_latencies)
            else np.array([], dtype=np.float64)
        )
        pooled_iqr_outliers = iqr_outlier_mask(pooled)
        pooled_stats = summarize_latency(
            latency_us=pooled,
            detected=np.ones(pooled.shape, dtype=bool),
            iqr_outliers=pooled_iqr_outliers,
        )

        total_toggles = int(sum(int(row["n_toggles"]) for row in rows))
        total_detected = int(sum(int(row["n_detected"]) for row in rows))
        pooled_stats["n_toggles"] = total_toggles
        pooled_stats["n_detected"] = total_detected
        pooled_stats["detected_frac"] = (
            float(total_detected / total_toggles) if total_toggles > 0 else float("nan")
        )

        trial_detected = np.array(
            [float(row["detected_frac"]) for row in rows if np.isfinite(row["detected_frac"])],
            dtype=np.float64,
        )
        trial_mean_latencies = np.array(
            [float(row["latency_mean_us"]) for row in rows if np.isfinite(row["latency_mean_us"])],
            dtype=np.float64,
        )

        grouped_rows.append({
            "pulse_width": f"{pulse_width_token}ms",
            "pulse_width_ms": pulse_width_token_to_ms(pulse_width_token),
            "n_trials": len(rows),
            "trial_detected_frac_mean": finite_mean(trial_detected),
            "trial_detected_frac_std": finite_std(trial_detected),
            "trial_latency_mean_us_mean": finite_mean(trial_mean_latencies),
            "trial_latency_mean_us_std": finite_std(trial_mean_latencies),
            **pooled_stats,
        })

    return grouped_rows


def main():
    ap = argparse.ArgumentParser(
        description="Compute LED-toggle to first-event latency from one or more .raw trials."
    )
    ap.add_argument("--raw_files", nargs="+", required=True, help="List of .raw trial files")
    ap.add_argument("--trial_labels", nargs="*", default=None, help="Optional labels for each raw file")

    ap.add_argument("--toggle_times_file", default=None, help="Text file of toggle times (s), applies to all trials")
    ap.add_argument("--toggle_times_csv", default=None, help="CSV with columns [trial or raw_file],toggle_s")
    ap.add_argument("--first_toggle_s", type=float, default=None, help="Schedule mode: first toggle time in seconds")
    ap.add_argument("--pulse_period_s", type=float, default=None, help="Schedule mode: pulse period in seconds")
    ap.add_argument("--num_pulses", type=int, default=None, help="Schedule mode: number of pulses")

    ap.add_argument("--max_latency_ms", type=float, default=20.0, help="Discard first events later than this (ms)")
    ap.add_argument("--out_prefix", default="latency_1000Hz", help="Prefix for output CSV/PNG names")
    ap.add_argument("--no_plot", action="store_true", help="Do not generate plots")
    args = ap.parse_args()

    if args.trial_labels and len(args.trial_labels) != len(args.raw_files):
        raise ValueError("--trial_labels length must match --raw_files length")

    methods = 0
    methods += int(args.toggle_times_file is not None)
    methods += int(args.toggle_times_csv is not None)
    schedule_ready = (
        args.first_toggle_s is not None and
        args.pulse_period_s is not None and
        args.num_pulses is not None
    )
    methods += int(schedule_ready)
    if methods != 1:
        raise ValueError(
            "Choose exactly one toggle source: --toggle_times_file OR --toggle_times_csv OR schedule args"
        )

    trial_labels = (
        args.trial_labels
        if args.trial_labels
        else [os.path.splitext(os.path.basename(p))[0] for p in args.raw_files]
    )

    raw_files = [os.path.abspath(p) for p in args.raw_files]
    for p in raw_files:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    common_toggles = None
    toggles_by_key: Dict[str, np.ndarray] = {}
    if args.toggle_times_file:
        common_toggles = load_toggle_times_file(args.toggle_times_file)
    elif args.toggle_times_csv:
        toggles_by_key = load_toggle_times_csv(args.toggle_times_csv)
    else:
        common_toggles = build_toggle_schedule(args.first_toggle_s, args.pulse_period_s, args.num_pulses)

    root = repo_root_from_this_file(__file__)
    data_dir = os.path.join(root, "data")
    plot_dir = os.path.join(root, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    per_pulse_path = os.path.join(data_dir, f"{args.out_prefix}_per_pulse.csv")
    summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    pulse_width_summary_path = os.path.join(data_dir, f"{args.out_prefix}_by_pulse_width_summary.csv")
    max_latency_s = args.max_latency_ms / 1000.0

    summary_rows: List[Dict[str, float]] = []
    latencies_by_trial: Dict[str, np.ndarray] = {}

    with open(per_pulse_path, "w", newline="", encoding="utf-8") as fpp:
        w = csv.writer(fpp)
        w.writerow([
            "trial", "raw_file", "toggle_index", "toggle_s", "first_event_s",
            "latency_us", "detected_within_window", "iqr_outlier"
        ])

        for raw_path, trial in zip(raw_files, trial_labels):
            ts_us = load_timestamps_us(raw_path)
            if ts_us.size == 0:
                toggles = common_toggles if common_toggles is not None else np.array([], dtype=np.float64)
                if args.toggle_times_csv:
                    base = os.path.basename(raw_path)
                    toggles = toggles_by_key.get(trial, toggles_by_key.get(base, np.array([], dtype=np.float64)))

                first_event_s = np.full(toggles.shape, np.nan, dtype=np.float64)
                latency_us = np.full(toggles.shape, np.nan, dtype=np.float64)
                detected = np.zeros(toggles.shape, dtype=bool)
            else:
                ts_rel_s = (ts_us - ts_us[0]).astype(np.float64) * 1e-6
                if common_toggles is not None:
                    toggles = common_toggles
                else:
                    base = os.path.basename(raw_path)
                    if trial in toggles_by_key:
                        toggles = toggles_by_key[trial]
                    elif base in toggles_by_key:
                        toggles = toggles_by_key[base]
                    else:
                        raise ValueError(
                            f"No toggle times found for trial '{trial}' or raw_file '{base}' in toggle CSV."
                        )

                first_event_s, latency_us, detected = compute_latencies_us(
                    ts_rel_s=ts_rel_s,
                    toggle_s=toggles,
                    max_latency_s=max_latency_s,
                )

            iqr_outliers = iqr_outlier_mask(latency_us)
            for i in range(toggles.size):
                w.writerow([
                    trial,
                    os.path.basename(raw_path),
                    i,
                    float(toggles[i]),
                    "" if not np.isfinite(first_event_s[i]) else float(first_event_s[i]),
                    "" if not np.isfinite(latency_us[i]) else float(latency_us[i]),
                    int(detected[i]),
                    int(iqr_outliers[i]),
                ])

            stats = summarize_latency(latency_us=latency_us, detected=detected, iqr_outliers=iqr_outliers)
            stats.update({
                "trial": trial,
                "raw_file": os.path.basename(raw_path),
                "trial_mean_outlier_flag": float("nan"),
                "trial_mean_modified_z": float("nan"),
            })
            summary_rows.append(stats)
            latencies_by_trial[trial] = latency_us[np.isfinite(latency_us)]

    trial_mean_z = mark_outlier_trials_by_mean(summary_rows)
    for r in summary_rows:
        if r["trial"] != "OVERALL":
            r["trial_mean_modified_z"] = trial_mean_z.get(r["trial"], float("nan"))

    all_lat = np.concatenate([v for v in latencies_by_trial.values() if v.size > 0]) \
        if any(v.size > 0 for v in latencies_by_trial.values()) else np.array([], dtype=np.float64)
    all_iqr_out = iqr_outlier_mask(all_lat)
    overall = summarize_latency(
        latency_us=all_lat,
        detected=np.ones(all_lat.shape, dtype=bool),
        iqr_outliers=all_iqr_out,
    )
    total_toggles = int(sum(r["n_toggles"] for r in summary_rows))
    total_detected = int(sum(r["n_detected"] for r in summary_rows))
    overall["n_toggles"] = total_toggles
    overall["n_detected"] = total_detected
    overall["detected_frac"] = float(total_detected / total_toggles) if total_toggles > 0 else float("nan")
    overall.update({
        "trial": "OVERALL",
        "raw_file": "",
        "trial_mean_outlier_flag": float("nan"),
        "trial_mean_modified_z": float("nan"),
    })
    summary_rows.append(overall)

    header = [
        "trial", "raw_file",
        "n_toggles", "n_detected", "detected_frac",
        "latency_mean_us", "latency_std_us", "latency_median_us", "latency_p95_us",
        "latency_min_us", "latency_max_us",
        "iqr_outlier_count", "iqr_outlier_frac",
        "trial_mean_outlier_flag", "trial_mean_modified_z",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as fs:
        w = csv.DictWriter(fs, fieldnames=header)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print("Saved per-pulse CSV:", per_pulse_path)
    print("Saved summary CSV:", summary_path)

    pulse_width_summary_rows = build_pulse_width_summary(summary_rows, latencies_by_trial)
    if pulse_width_summary_rows:
        pulse_width_header = [
            "pulse_width", "pulse_width_ms", "n_trials",
            "n_toggles", "n_detected", "detected_frac",
            "trial_detected_frac_mean", "trial_detected_frac_std",
            "trial_latency_mean_us_mean", "trial_latency_mean_us_std",
            "latency_mean_us", "latency_std_us", "latency_median_us", "latency_p95_us",
            "latency_min_us", "latency_max_us",
            "iqr_outlier_count", "iqr_outlier_frac",
        ]
        with open(pulse_width_summary_path, "w", newline="", encoding="utf-8") as fg:
            w = csv.DictWriter(fg, fieldnames=pulse_width_header)
            w.writeheader()
            for row in pulse_width_summary_rows:
                w.writerow(row)
        print("Saved pulse-width summary CSV:", pulse_width_summary_path)

    if not args.no_plot:
        non_empty = [(k, v) for k, v in latencies_by_trial.items() if v.size > 0]
        if non_empty:
            plt.figure()
            for trial, lat in non_empty:
                plt.hist(lat, bins=40, alpha=0.5, label=trial)
            plt.xlabel("latency (us)")
            plt.ylabel("count")
            plt.title("Latency histogram by trial")
            if len(non_empty) > 1:
                plt.legend()
            plt.grid(True)
            hist_path = os.path.join(plot_dir, f"{args.out_prefix}_latency_histogram.png")
            plt.savefig(hist_path, dpi=300)
            print("Saved plot:", hist_path)

            fig_width = max(12.0, 1.6 * len(non_empty))
            plt.figure(figsize=(fig_width, 7))
            labels = [format_trial_label_for_plot(k) for k, _ in non_empty]
            data = [v for _, v in non_empty]
            try:
                plt.boxplot(data, tick_labels=labels, showfliers=True)
            except TypeError:
                plt.boxplot(data, labels=labels, showfliers=True)
            plt.xlabel("trial")
            plt.ylabel("latency (us)")
            plt.title("Latency boxplot by trial")
            plt.grid(True)
            plt.xticks(rotation=25, ha="right", fontsize=9)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.24)
            box_path = os.path.join(plot_dir, f"{args.out_prefix}_latency_boxplot_by_trial.png")
            plt.savefig(box_path, dpi=300)
            print("Saved plot:", box_path)

            if "agg" in plt.get_backend().lower():
                plt.close("all")
            else:
                plt.show()
        else:
            print("No valid latencies to plot.")


if __name__ == "__main__":
    main()
