import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator

from io_utils import repo_root_from_this_file


@dataclass
class LuxMetrics:
    raw_file: str
    label: str
    lux: float
    total_events: int
    duration_s: float
    events_per_s: float
    onset_s_rel: float
    noise_mean: float
    noise_std: float
    signal_mean: float
    signal_std: float
    noise_rate_per_s: float
    signal_rate_per_s: float
    snr_ratio: float
    snr_z: float


def detect_onset_time(
    t_rel_s: np.ndarray,
    y: np.ndarray,
    k_sigma: float = 8.0,
    min_sustain_bins: int = 20,
) -> Optional[float]:
    """Find first sustained rise above a robust threshold."""
    if t_rel_s.size == 0:
        return None

    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med))) + 1e-9
    robust_sigma = 1.4826 * mad
    thr = med + k_sigma * robust_sigma

    above = y > thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_sustain_bins:
            return float(t_rel_s[i - min_sustain_bins + 1])
    return None


def split_noise_signal(
    t_rel_s: np.ndarray,
    y: np.ndarray,
    onset_s: Optional[float],
    noise_window_s: float = 1.0,
) -> dict:
    """Split into pre-onset noise and post-onset signal windows."""
    if t_rel_s.size == 0:
        return {
            "noise_mean": float("nan"),
            "noise_std": float("nan"),
            "signal_mean": float("nan"),
            "signal_std": float("nan"),
        }

    if onset_s is None:
        n = len(t_rel_s)
        cut = max(1, int(0.2 * n))
        noise_mask = np.zeros(n, dtype=bool)
        noise_mask[:cut] = True
        signal_mask = ~noise_mask
    else:
        noise_start = onset_s - noise_window_s
        noise_end = onset_s
        noise_mask = (t_rel_s >= noise_start) & (t_rel_s < noise_end)
        signal_mask = (t_rel_s >= onset_s)
        if not np.any(noise_mask):
            n = len(t_rel_s)
            cut = max(1, int(0.2 * n))
            noise_mask = np.zeros(n, dtype=bool)
            noise_mask[:cut] = True

    noise_vals = y[noise_mask]
    signal_vals = y[signal_mask]
    return {
        "noise_mean": float(np.mean(noise_vals)) if noise_vals.size else float("nan"),
        "noise_std": float(np.std(noise_vals)) if noise_vals.size else float("nan"),
        "signal_mean": float(np.mean(signal_vals)) if signal_vals.size else float("nan"),
        "signal_std": float(np.std(signal_vals)) if signal_vals.size else float("nan"),
    }


def snr_proxy(noise_mean: float, noise_std: float, signal_mean: float) -> Tuple[float, float]:
    ratio = float("nan") if noise_mean <= 0 else float(signal_mean / noise_mean)
    z = float("nan") if noise_std <= 0 else float((signal_mean - noise_mean) / noise_std)
    return ratio, z


def stream_binned_counts(
    raw_path: str,
    bin_ms: float,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """
    Stream events to compute total events, duration, and binned activity
    without storing all timestamps in memory.
    Returns:
      total_events, duration_s, t_rel_s_bins, counts
    """
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0")

    bin_us = max(1, int(round(bin_ms * 1000.0)))
    it = EventsIterator(input_path=raw_path)

    t0_us: Optional[int] = None
    t_last_us: Optional[int] = None
    total_events = 0
    counts = np.zeros(0, dtype=np.int64)

    for evs in it:
        if evs.size == 0:
            continue
        ts = evs["t"].astype(np.int64)

        if t0_us is None:
            t0_us = int(ts[0])

        total_events += int(ts.size)
        t_last_us = int(ts[-1])

        idx = ((ts - t0_us) // bin_us).astype(np.int64)
        max_idx = int(idx.max())
        if counts.size <= max_idx:
            counts = np.pad(counts, (0, max_idx + 1 - counts.size), constant_values=0)

        binc = np.bincount(idx, minlength=max_idx + 1)
        counts[:binc.size] += binc

    if t0_us is None or t_last_us is None:
        return 0, 0.0, np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    duration_s = float((t_last_us - t0_us) * 1e-6)
    t_rel_s = np.arange(counts.size, dtype=np.float64) * (bin_us * 1e-6)
    return total_events, duration_s, t_rel_s, counts


def analyze_one(raw_path: str, label: str, lux: float, bin_ms: float, onset_k_sigma: float) -> LuxMetrics:
    total_events, duration_s, t_rel_s, counts = stream_binned_counts(raw_path=raw_path, bin_ms=bin_ms)
    events_per_s = float(total_events / duration_s) if duration_s > 0 else float("nan")

    if counts.size == 0:
        return LuxMetrics(
            raw_file=os.path.basename(raw_path),
            label=label,
            lux=lux,
            total_events=0,
            duration_s=0.0,
            events_per_s=float("nan"),
            onset_s_rel=float("nan"),
            noise_mean=float("nan"),
            noise_std=float("nan"),
            signal_mean=float("nan"),
            signal_std=float("nan"),
            noise_rate_per_s=float("nan"),
            signal_rate_per_s=float("nan"),
            snr_ratio=float("nan"),
            snr_z=float("nan"),
        )

    onset_s = detect_onset_time(t_rel_s, counts, k_sigma=onset_k_sigma)
    seg = split_noise_signal(t_rel_s, counts, onset_s)
    snr_ratio, snr_z = snr_proxy(seg["noise_mean"], seg["noise_std"], seg["signal_mean"])

    bin_s = bin_ms / 1000.0
    noise_rate_per_s = float(seg["noise_mean"] / bin_s) if np.isfinite(seg["noise_mean"]) else float("nan")
    signal_rate_per_s = float(seg["signal_mean"] / bin_s) if np.isfinite(seg["signal_mean"]) else float("nan")

    return LuxMetrics(
        raw_file=os.path.basename(raw_path),
        label=label,
        lux=float(lux),
        total_events=int(total_events),
        duration_s=duration_s,
        events_per_s=events_per_s,
        onset_s_rel=float(onset_s) if onset_s is not None else float("nan"),
        noise_mean=float(seg["noise_mean"]),
        noise_std=float(seg["noise_std"]),
        signal_mean=float(seg["signal_mean"]),
        signal_std=float(seg["signal_std"]),
        noise_rate_per_s=noise_rate_per_s,
        signal_rate_per_s=signal_rate_per_s,
        snr_ratio=float(snr_ratio),
        snr_z=float(snr_z),
    )


def main():
    ap = argparse.ArgumentParser(description="Analyze 1000 Hz illumination sweep and plot event rate/SNR vs lux.")
    ap.add_argument("--raw_files", nargs="+", required=True, help="List of .raw files (e.g. low/mid/high lux)")
    ap.add_argument("--lux_values", nargs="+", type=float, required=True, help="Measured lux values, same order as raw_files")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels for each run")
    ap.add_argument("--bin_ms", type=float, default=1.0, help="Bin width for activity histogram (ms)")
    ap.add_argument("--onset_k_sigma", type=float, default=8.0, help="Onset threshold = median + k_sigma * robust_sigma")
    ap.add_argument("--out_csv", required=True, help="Output CSV filename (saved into repo data/)")
    ap.add_argument("--plot_prefix", default=None, help="Optional prefix for plot filenames (saved into repo plots/)")
    ap.add_argument("--no_plot", action="store_true", help="Disable plots")
    args = ap.parse_args()

    if len(args.raw_files) != len(args.lux_values):
        raise ValueError("--raw_files and --lux_values must have the same length")

    labels = args.labels if args.labels else [os.path.splitext(os.path.basename(p))[0] for p in args.raw_files]
    if len(labels) != len(args.raw_files):
        raise ValueError("--labels length must match --raw_files length")

    raw_files = [os.path.abspath(p) for p in args.raw_files]
    for p in raw_files:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    rows: List[LuxMetrics] = []
    for raw_path, label, lux in zip(raw_files, labels, args.lux_values):
        m = analyze_one(raw_path=raw_path, label=label, lux=lux, bin_ms=args.bin_ms, onset_k_sigma=args.onset_k_sigma)
        rows.append(m)
        print(
            f"Done {m.raw_file}: lux={m.lux:.2f} events/s={m.events_per_s:.2f} "
            f"snr_z={m.snr_z:.2f} noise_rate/s={m.noise_rate_per_s:.2f}"
        )

    rows.sort(key=lambda r: r.lux)

    root = repo_root_from_this_file(__file__)
    data_dir = os.path.join(root, "data")
    plot_dir = os.path.join(root, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    out_name = args.out_csv
    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"
    out_path = os.path.join(data_dir, out_name)

    header = [
        "raw_file", "label", "lux",
        "total_events", "duration_s", "events_per_s",
        "onset_s_rel",
        "noise_mean", "noise_std", "signal_mean", "signal_std",
        "noise_rate_per_s", "signal_rate_per_s",
        "snr_ratio", "snr_z",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.raw_file, r.label, r.lux,
                r.total_events, r.duration_s, r.events_per_s,
                r.onset_s_rel,
                r.noise_mean, r.noise_std, r.signal_mean, r.signal_std,
                r.noise_rate_per_s, r.signal_rate_per_s,
                r.snr_ratio, r.snr_z,
            ])

    print("Saved summary CSV:", out_path)

    if args.no_plot:
        return

    plot_prefix = args.plot_prefix.strip() if args.plot_prefix else os.path.splitext(out_name)[0]
    lux = np.array([r.lux for r in rows], dtype=float)
    ev_rate = np.array([r.events_per_s for r in rows], dtype=float)
    snr_z = np.array([r.snr_z for r in rows], dtype=float)

    fig1 = plt.figure()
    plt.plot(lux, ev_rate, marker="o")
    plt.xlabel("lux")
    plt.ylabel("events/s")
    plt.title("Event rate vs lux")
    plt.grid(True)
    p1 = os.path.join(plot_dir, f"{plot_prefix}_event_rate_vs_lux.png")
    fig1.savefig(p1, dpi=300)
    print("Saved plot:", p1)

    fig2 = plt.figure()
    plt.plot(lux, snr_z, marker="o")
    plt.xlabel("lux")
    plt.ylabel("SNR z-score proxy")
    plt.title("SNR vs lux")
    plt.grid(True)
    p2 = os.path.join(plot_dir, f"{plot_prefix}_snr_vs_lux.png")
    fig2.savefig(p2, dpi=300)
    print("Saved plot:", p2)

    plt.show()


if __name__ == "__main__":
    main()
