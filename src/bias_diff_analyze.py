import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator

# Reuse your repo output conventions
from io_utils import repo_root_from_this_file


DEFAULT_INPUT_DIR = r"C:\Users\rabis\OneDrive\Documents\School\LAB aka 195\captures\testing_bias_diff_1000Hz"


# ----------------------------
# Core primitives
# ----------------------------
def load_timestamps_us(raw_path: str) -> np.ndarray:
    """Load all event timestamps (microseconds) from a Metavision .raw file."""
    events = EventsIterator(input_path=raw_path)
    chunks = []
    for evs in events:
        # Append each iterator chunk, then combine them into one timestamp array.
        chunks.append(evs["t"])
    return np.concatenate(chunks).astype(np.int64) if chunks else np.array([], dtype=np.int64)


def binned_activity(time_s: np.ndarray, bin_width_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram event count vs time.
    Returns (t_left_edges, counts).
    """
    if time_s.size == 0:
        return np.array([]), np.array([])
    t0, t1 = float(time_s.min()), float(time_s.max())
    bins = np.arange(t0, t1 + bin_width_s, bin_width_s)
    counts, edges = np.histogram(time_s, bins=bins)
    return edges[:-1], counts


def find_peaks_simple(t: np.ndarray, y: np.ndarray, min_height: float, min_distance_s: float) -> np.ndarray:
    """
    Simple peak finder.
    Peak at i if y[i] > y[i-1] and y[i] >= y[i+1] and y[i] >= min_height.
    Enforces min spacing in seconds.
    Returns peak times (seconds).
    """
    if t.size < 3:
        return np.array([])

    candidates = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height:
            candidates.append(i)

    if not candidates:
        return np.array([])

    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if (t[idx] - t[peaks[-1]]) >= min_distance_s:
            peaks.append(idx)

    return t[np.array(peaks, dtype=int)]


def estimate_freq_period_jitter(peak_times_s: np.ndarray) -> Dict[str, float]:
    """Compute frequency and period stats from peak times."""
    if peak_times_s.size < 3:
        return {"freq_hz": np.nan, "period_mean_s": np.nan, "period_std_s": np.nan}
    periods = np.diff(peak_times_s)
    return {
        "freq_hz": float(1.0 / np.mean(periods)),
        "period_mean_s": float(np.mean(periods)),
        "period_std_s": float(np.std(periods)),
    }


def edge_residual_jitter_us(peak_times_s: np.ndarray) -> float:
    """
    A useful 'timing jitter' measure without needing ground-truth phase:
    - Fit a line: peak_time[k] ≈ a*k + b  (i.e., constant period model)
    - Residuals are timing errors vs that model
    - Report std(residuals) in microseconds
    """
    if peak_times_s.size < 5:
        return float("nan")
    k = np.arange(peak_times_s.size)
    a, b = np.polyfit(k, peak_times_s, 1)  # a ~ period, b ~ offset
    pred = a * k + b
    resid_s = peak_times_s - pred
    return float(np.std(resid_s) * 1e6)


# ----------------------------
# Optional OOK decode + BER
# ----------------------------
def decode_ook_from_counts(
    t_bins: np.ndarray,
    counts: np.ndarray,
    bitrate_hz: float,
    start_time_s: float,
    n_bits: int,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Decode bits by integrating counts in each symbol window.
    - bitrate_hz: symbols per second
    - start_time_s: where symbol 0 starts (you can sweep this if needed)
    - threshold: if None, use midpoint between low/high cluster means
    """
    Ts = 1.0 / bitrate_hz
    bits = np.zeros(n_bits, dtype=np.uint8)

    # Build per-symbol integrated counts
    sym_sums = np.zeros(n_bits, dtype=np.float64)
    for i in range(n_bits):
        t0 = start_time_s + i * Ts
        t1 = t0 + Ts
        mask = (t_bins >= t0) & (t_bins < t1)
        sym_sums[i] = float(np.sum(counts[mask]))

    if threshold is None:
        # crude auto-threshold: split by median and take means
        med = np.median(sym_sums)
        low = sym_sums[sym_sums <= med]
        high = sym_sums[sym_sums > med]
        if low.size == 0 or high.size == 0:
            threshold = float(med)
        else:
            threshold = float(0.5 * (np.mean(low) + np.mean(high)))

    bits[:] = (sym_sums >= threshold).astype(np.uint8)
    return bits


def ber(bits_hat: np.ndarray, bits_true: np.ndarray) -> float:
    if bits_hat.size == 0 or bits_true.size == 0 or bits_hat.size != bits_true.size:
        return float("nan")
    return float(np.mean(bits_hat != bits_true))


# ----------------------------
# Bias diff parsing + file list
# ----------------------------
def extract_bias_from_name(filename: str, pattern: str) -> Optional[float]:
    """
    Extract bias_diff from filename using a regex with a capturing group.
    Default pattern expects something like: biasdiff_-12.raw, biasdiff_12.raw, or biasdiff_12.5.raw
    """
    m = re.search(pattern, filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def list_raw_files(input_dir: str) -> List[str]:
    raws = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".raw"):
            raws.append(os.path.join(input_dir, name))
    return sorted(raws)


def load_map_csv(map_csv: str) -> Dict[str, float]:
    """
    CSV columns: raw_file,bias_diff
    raw_file can be basename or relative/absolute path; we match by basename first.
    """
    mapping: Dict[str, float] = {}
    with open(map_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rf = row.get("raw_file", "").strip()
            bd = row.get("bias_diff", "").strip()
            if not rf or not bd:
                continue
            mapping[os.path.basename(rf)] = float(bd)
    return mapping


# ----------------------------
# One-file analysis
# ----------------------------
@dataclass
class FileMetrics:
    raw_file: str
    bias_diff: float
    duration_s: float
    total_events: int
    events_per_s: float
    peaks_detected: int
    freq_hz: float
    period_mean_ms: float
    period_std_ms: float
    edge_jitter_us: float
    ber: float


def analyze_one_file(
    raw_path: str,
    bias_diff: float,
    bin_ms: float,
    peak_k: float,
    min_peak_dist_ms: float,
    expected_freq_hz: Optional[float],
    do_ber: bool,
    bitrate_hz: Optional[float],
    bits_true: Optional[np.ndarray],
    ber_start_time_s: float,
) -> FileMetrics:
    # Start from the raw event timestamps for this one capture.
    ts_us = load_timestamps_us(raw_path)
    total_events = int(ts_us.size)

    if total_events < 2:
        return FileMetrics(
            raw_file=os.path.basename(raw_path),
            bias_diff=bias_diff,
            duration_s=0.0,
            total_events=total_events,
            events_per_s=float("nan"),
            peaks_detected=0,
            freq_hz=float("nan"),
            period_mean_ms=float("nan"),
            period_std_ms=float("nan"),
            edge_jitter_us=float("nan"),
            ber=float("nan"),
        )

    time_s = ts_us * 1e-6
    duration_s = float(time_s.max() - time_s.min())
    events_per_s = float(total_events / duration_s) if duration_s > 0 else float("nan")

    # Turn the event stream into a simple count-vs-time signal.
    t_bins, counts = binned_activity(time_s, bin_ms / 1000.0)

    # Peak threshold: median + k * robust sigma (MAD) is usually better than mean/std
    med = float(np.median(counts))
    mad = float(np.median(np.abs(counts - med))) + 1e-9
    robust_sigma = 1.4826 * mad
    min_height = med + peak_k * robust_sigma

    peak_times_s = find_peaks_simple(
        t_bins, counts,
        min_height=min_height,
        min_distance_s=min_peak_dist_ms / 1000.0
    )

    # Extract timing metrics from the detected peaks.
    fj = estimate_freq_period_jitter(peak_times_s)
    edge_jit = edge_residual_jitter_us(peak_times_s)

    # If you provide expected_freq_hz, you can sanity-check:
    # (we just compute; you can filter later)
    _ = expected_freq_hz  # placeholder to keep interface clear

    out_ber = float("nan")
    if do_ber and bitrate_hz and bits_true is not None and bits_true.size > 0:
        # Decode with symbol integration
        bits_hat = decode_ook_from_counts(
            t_bins=t_bins,
            counts=counts,
            bitrate_hz=bitrate_hz,
            start_time_s=ber_start_time_s,
            n_bits=int(bits_true.size),
            threshold=None
        )
        out_ber = ber(bits_hat, bits_true)

    return FileMetrics(
        raw_file=os.path.basename(raw_path),
        bias_diff=bias_diff,
        duration_s=duration_s,
        total_events=total_events,
        events_per_s=events_per_s,
        peaks_detected=int(peak_times_s.size),
        freq_hz=float(fj["freq_hz"]),
        period_mean_ms=float(fj["period_mean_s"] * 1e3),
        period_std_ms=float(fj["period_std_s"] * 1e3),
        edge_jitter_us=float(edge_jit),
        ber=float(out_ber),
    )


# ----------------------------
# Main sweep driver
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze a bias_diff sweep folder of EVK4 .raw files.")
    ap.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing .raw files for the sweep (default: {DEFAULT_INPUT_DIR})"
    )
    ap.add_argument(
        "--bias_regex",
        default=r"biasdiff_([+-]?[0-9]+(?:\.[0-9]+)?)",
        help="Regex with one capture group for bias_diff from filename (default expects names like biasdiff_-12.raw or biasdiff_12.5.raw)"
    )
    ap.add_argument("--map_csv", default=None, help="Optional mapping CSV with columns raw_file,bias_diff")
    ap.add_argument("--bin_ms", type=float, default=1.0, help="Histogram bin width in ms (default 1.0)")
    ap.add_argument("--peak_k", type=float, default=6.0, help="Peak threshold = median + peak_k*robust_sigma (default 6.0)")
    ap.add_argument("--min_peak_dist_ms", type=float, default=0.5, help="Min time between peaks in ms (default 0.5)")
    ap.add_argument("--expected_freq_hz", type=float, default=None, help="Optional expected frequency for sanity checking")
    ap.add_argument("--out", required=True, help="Output CSV filename (saved into repo data/)")
    ap.add_argument("--plot_prefix", default=None, help="Optional prefix for plot filenames (saved into repo plots/)")
    ap.add_argument("--no_plot", action="store_true", help="Do not generate summary plots")

    # Optional BER decoding
    ap.add_argument("--bitrate_hz", type=float, default=None, help="If provided, attempt OOK decode at this bitrate")
    ap.add_argument("--bits_file", default=None, help="Text file containing 0/1 bits (e.g., 101001...)")
    ap.add_argument("--ber_start_time_s", type=float, default=0.0, help="Symbol 0 start time (seconds, relative to capture)")

    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    # If provided, this CSV overrides filename parsing for the bias value.
    mapping = load_map_csv(args.map_csv) if args.map_csv else {}

    # Load truth bits if BER enabled
    bits_true = None
    do_ber = False
    if args.bitrate_hz and args.bits_file:
        with open(args.bits_file, "r", encoding="utf-8") as f:
            s = "".join(ch for ch in f.read() if ch in "01")
        bits_true = np.array([1 if ch == "1" else 0 for ch in s], dtype=np.uint8)
        do_ber = bits_true.size > 0

    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    rows: List[FileMetrics] = []
    for rp in raw_files:
        base = os.path.basename(rp)

        # Prefer the explicit map first, then fall back to the filename regex.
        bd = mapping.get(base)
        if bd is None:
            bd = extract_bias_from_name(base, args.bias_regex)

        if bd is None:
            print(f"Skipping (could not determine bias_diff): {base}")
            continue

        fm = analyze_one_file(
            raw_path=rp,
            bias_diff=float(bd),
            bin_ms=args.bin_ms,
            peak_k=args.peak_k,
            min_peak_dist_ms=args.min_peak_dist_ms,
            expected_freq_hz=args.expected_freq_hz,
            do_ber=do_ber,
            bitrate_hz=args.bitrate_hz,
            bits_true=bits_true,
            ber_start_time_s=args.ber_start_time_s
        )
        rows.append(fm)
        print(f"Done: {base} (bias_diff={fm.bias_diff}) events/s={fm.events_per_s:.2f} freq={fm.freq_hz:.2f}Hz")

    if not rows:
        raise RuntimeError("No files analyzed. Check naming or --map_csv / --bias_regex.")

    # Sort by bias_diff
    rows.sort(key=lambda r: r.bias_diff)

    # Save summary CSV into repo data/
    root = repo_root_from_this_file(__file__)
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)

    out_name = args.out
    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"
    out_path = os.path.join(out_dir, out_name)

    header = [
        "raw_file", "bias_diff",
        "duration_s", "total_events", "events_per_s",
        "peaks_detected", "freq_hz", "period_mean_ms", "period_std_ms",
        "edge_jitter_us", "ber"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.raw_file, r.bias_diff,
                r.duration_s, r.total_events, r.events_per_s,
                r.peaks_detected, r.freq_hz, r.period_mean_ms, r.period_std_ms,
                r.edge_jitter_us, r.ber
            ])

    print("Saved summary CSV:", out_path)

    # Summary plots
    if not args.no_plot:
        plot_dir = os.path.join(root, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_prefix = args.plot_prefix.strip() if args.plot_prefix else os.path.splitext(out_name)[0]

        # Convert the per-file dataclass rows into numeric arrays for plotting.
        bd = np.array([r.bias_diff for r in rows], dtype=float)
        ber_arr = np.array([r.ber for r in rows], dtype=float)
        jit = np.array([r.edge_jitter_us for r in rows], dtype=float)
        evs = np.array([r.events_per_s for r in rows], dtype=float)

        fig1 = plt.figure()
        plt.plot(bd, evs, marker="o")
        plt.xlabel("bias_diff")
        plt.ylabel("events/s")
        plt.title("Event rate vs bias_diff")
        plt.grid(True)
        plot1_path = os.path.join(plot_dir, f"{plot_prefix}_event_rate_vs_bias_diff.png")
        fig1.savefig(plot1_path, dpi=300)
        print("Saved plot:", plot1_path)

        fig2 = plt.figure()
        plt.plot(bd, jit, marker="o")
        plt.xlabel("bias_diff")
        plt.ylabel("edge residual jitter (us)")
        plt.title("Timing jitter proxy vs bias_diff")
        plt.grid(True)
        plot2_path = os.path.join(plot_dir, f"{plot_prefix}_timing_jitter_vs_bias_diff.png")
        fig2.savefig(plot2_path, dpi=300)
        print("Saved plot:", plot2_path)

        if np.any(np.isfinite(ber_arr)):
            fig3 = plt.figure()
            plt.plot(bd, ber_arr, marker="o")
            plt.xlabel("bias_diff")
            plt.ylabel("BER")
            plt.title("BER vs bias_diff")
            plt.grid(True)
            plot3_path = os.path.join(plot_dir, f"{plot_prefix}_ber_vs_bias_diff.png")
            fig3.savefig(plot3_path, dpi=300)
            print("Saved plot:", plot3_path)

        plt.show()


if __name__ == "__main__":
    main()
