import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator


DEFAULT_INPUT_DIR = r"C:\Users\rabis\OneDrive\Documents\School\LAB aka 195\captures\testing_bias_fo"


# ----------------------------
# Load timestamps
# ----------------------------
def load_timestamps_us(raw_path: str) -> np.ndarray:
    it = EventsIterator(input_path=raw_path)
    chunks = []
    for evs in it:
        if evs.size:
            chunks.append(evs["t"].astype(np.int64))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)


# ----------------------------
# Binned activity + peak finding
# ----------------------------
def binned_activity(time_s: np.ndarray, bin_width_s: float) -> Tuple[np.ndarray, np.ndarray]:
    if time_s.size == 0:
        return np.array([]), np.array([])
    t0, t1 = float(time_s.min()), float(time_s.max())
    edges = np.arange(t0, t1 + bin_width_s, bin_width_s)
    counts, edges = np.histogram(time_s, bins=edges)
    return edges[:-1], counts


def robust_threshold(counts: np.ndarray, k: float) -> float:
    med = float(np.median(counts))
    mad = float(np.median(np.abs(counts - med))) + 1e-9
    robust_sigma = 1.4826 * mad
    return med + k * robust_sigma


def find_peaks_simple(t: np.ndarray, y: np.ndarray, min_height: float, min_distance_s: float) -> np.ndarray:
    """
    Simple local-maximum peak detector with minimum time spacing.
    Returns peak times in seconds.
    """
    if t.size < 3:
        return np.array([])

    candidates = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height:
            candidates.append(i)

    if not candidates:
        return np.array([])

    kept = [candidates[0]]
    for idx in candidates[1:]:
        if (t[idx] - t[kept[-1]]) >= min_distance_s:
            kept.append(idx)

    return t[np.array(kept, dtype=int)]


def peak_indices_from_times(t_bins: np.ndarray, peak_times: np.ndarray) -> np.ndarray:
    # peak_times are from t_bins values, so exact match is usually ok; fall back to nearest
    idxs = []
    for pt in peak_times:
        j = int(np.argmin(np.abs(t_bins - pt)))
        idxs.append(j)
    return np.array(idxs, dtype=int)


# ----------------------------
# Edge sharpness + noise metrics
# ----------------------------
def fwhm_ms(counts: np.ndarray, peak_idx: int, bin_ms: float) -> float:
    """
    Full-width at half-maximum of a peak in the binned count signal.
    Returns width in ms; NaN if not well-defined.
    """
    if peak_idx <= 0 or peak_idx >= len(counts) - 1:
        return float("nan")

    peak_val = float(counts[peak_idx])
    if peak_val <= 0:
        return float("nan")

    half = 0.5 * peak_val

    # walk left
    L = peak_idx
    while L > 0 and counts[L] > half:
        L -= 1

    # walk right
    R = peak_idx
    while R < len(counts) - 1 and counts[R] > half:
        R += 1

    width_bins = max(0, R - L)
    return float(width_bins * bin_ms)


def peak_slope_per_ms(counts: np.ndarray, peak_idx: int, bin_ms: float) -> float:
    """
    A simple "edge steepness" proxy: max discrete derivative near the peak.
    Larger = sharper transition response.
    """
    if peak_idx <= 0 or peak_idx >= len(counts) - 1:
        return float("nan")
    d1 = (counts[peak_idx] - counts[peak_idx - 1]) / bin_ms
    d2 = (counts[peak_idx + 1] - counts[peak_idx]) / bin_ms
    return float(max(d1, -d2))  # rising or falling magnitude


def background_stats(counts: np.ndarray, peak_idxs: np.ndarray, guard_bins: int) -> Tuple[float, float, float]:
    """
    Compute background mean/std and background event rate proxy,
    excluding a window +/- guard_bins around each peak.
    """
    if counts.size == 0:
        return float("nan"), float("nan"), float("nan")

    mask = np.ones(counts.size, dtype=bool)
    for pi in peak_idxs:
        a = max(0, pi - guard_bins)
        b = min(counts.size, pi + guard_bins + 1)
        mask[a:b] = False

    bg = counts[mask]
    if bg.size < 10:
        # too little background to estimate
        return float("nan"), float("nan"), float("nan")

    return float(np.mean(bg)), float(np.std(bg)), float(np.median(bg))


def missed_edge_fraction(peak_times_s: np.ndarray, duration_s: float, led_freq_hz: float, edges_per_cycle: int = 2) -> float:
    if duration_s <= 0:
        return float("nan")
    expected = led_freq_hz * edges_per_cycle * duration_s
    if expected <= 0:
        return float("nan")
    return float(max(0.0, 1.0 - (peak_times_s.size / expected)))


# ----------------------------
# Parsing bias_fo from filename
# ----------------------------
def extract_fo_from_name(filename: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def list_raw_files(input_dir: str) -> List[str]:
    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".raw")
    )


# ----------------------------
# Metrics container
# ----------------------------
@dataclass
class FoMetrics:
    raw_file: str
    bias_fo: float
    duration_s: float
    total_events: int
    events_per_s: float

    peaks_detected: int
    missed_edge_frac: float

    # edge sharpness proxies
    mean_fwhm_ms: float
    mean_peak_slope_per_ms: float
    mean_peak_height: float

    # noise proxies
    bg_mean: float
    bg_std: float
    bg_median: float
    peak_snr: float  # (peak_height - bg_mean) / bg_std


def analyze_one(
    raw_path: str,
    bias_fo: float,
    led_freq_hz: float,
    bin_ms: float,
    peak_k: float,
    min_peak_dist_ms: float,
    guard_ms: float,
    edges_per_cycle: int = 2,
) -> FoMetrics:
    ts_us = load_timestamps_us(raw_path)
    N = int(ts_us.size)

    if N < 2:
        return FoMetrics(
            raw_file=os.path.basename(raw_path),
            bias_fo=bias_fo,
            duration_s=0.0,
            total_events=N,
            events_per_s=float("nan"),
            peaks_detected=0,
            missed_edge_frac=float("nan"),
            mean_fwhm_ms=float("nan"),
            mean_peak_slope_per_ms=float("nan"),
            mean_peak_height=float("nan"),
            bg_mean=float("nan"),
            bg_std=float("nan"),
            bg_median=float("nan"),
            peak_snr=float("nan"),
        )

    time_s = ts_us.astype(np.float64) * 1e-6
    duration_s = float(time_s.max() - time_s.min())
    evps = float(N / duration_s) if duration_s > 0 else float("nan")

    t_bins, counts = binned_activity(time_s, bin_ms / 1000.0)
    if counts.size < 5:
        return FoMetrics(
            raw_file=os.path.basename(raw_path),
            bias_fo=bias_fo,
            duration_s=duration_s,
            total_events=N,
            events_per_s=evps,
            peaks_detected=0,
            missed_edge_frac=float("nan"),
            mean_fwhm_ms=float("nan"),
            mean_peak_slope_per_ms=float("nan"),
            mean_peak_height=float("nan"),
            bg_mean=float("nan"),
            bg_std=float("nan"),
            bg_median=float("nan"),
            peak_snr=float("nan"),
        )

    thr = robust_threshold(counts, peak_k)
    peak_times = find_peaks_simple(
        t_bins, counts,
        min_height=thr,
        min_distance_s=min_peak_dist_ms / 1000.0
    )
    peak_idxs = peak_indices_from_times(t_bins, peak_times)

    # Edge sharpness
    fwhms = [fwhm_ms(counts, int(pi), bin_ms) for pi in peak_idxs]
    slopes = [peak_slope_per_ms(counts, int(pi), bin_ms) for pi in peak_idxs]
    heights = [float(counts[int(pi)]) for pi in peak_idxs]

    mean_fwhm = float(np.nanmean(fwhms)) if len(fwhms) else float("nan")
    mean_slope = float(np.nanmean(slopes)) if len(slopes) else float("nan")
    mean_height = float(np.nanmean(heights)) if len(heights) else float("nan")

    # Background / noise
    guard_bins = int(max(1, round(guard_ms / bin_ms)))
    bg_mean, bg_std, bg_median = background_stats(counts, peak_idxs, guard_bins)

    # "SNR" proxy: peak above bg vs bg variability
    if np.isfinite(mean_height) and np.isfinite(bg_mean) and np.isfinite(bg_std) and bg_std > 0:
        peak_snr = float((mean_height - bg_mean) / bg_std)
    else:
        peak_snr = float("nan")

    missed = missed_edge_fraction(peak_times, duration_s, led_freq_hz, edges_per_cycle)

    return FoMetrics(
        raw_file=os.path.basename(raw_path),
        bias_fo=bias_fo,
        duration_s=duration_s,
        total_events=N,
        events_per_s=evps,
        peaks_detected=int(peak_times.size),
        missed_edge_frac=missed,
        mean_fwhm_ms=mean_fwhm,
        mean_peak_slope_per_ms=mean_slope,
        mean_peak_height=mean_height,
        bg_mean=bg_mean,
        bg_std=bg_std,
        bg_median=bg_median,
        peak_snr=peak_snr,
    )


def main():
    ap = argparse.ArgumentParser(description="Step 4: Sweep bias_fo and quantify edge sharpness vs noise.")
    ap.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder of .raw files for the bias_fo sweep (default: {DEFAULT_INPUT_DIR})"
    )
    ap.add_argument("--led_freq_hz", type=float, required=True, help="LED square-wave frequency (e.g., 1000)")
    ap.add_argument("--edges_per_cycle", type=int, default=2, help="2 for ON+OFF edges of a square wave")

    ap.add_argument("--bin_ms", type=float, default=0.25, help="Bin width for activity histogram in ms (default 0.25)")
    ap.add_argument("--peak_k", type=float, default=6.0, help="Peak threshold = median + peak_k*robust_sigma")
    ap.add_argument("--min_peak_dist_ms", type=float, default=0.3, help="Min spacing between peaks (ms)")
    ap.add_argument("--guard_ms", type=float, default=0.8, help="Exclude +/- guard_ms around peaks for background stats")

    ap.add_argument(
        "--fo_regex",
        default=r"(?:fo|biasfo)[_-]([0-9]+(?:\.[0-9]+)?)",
        help="Regex (one capture group) to parse bias_fo from filename (e.g., fo_20.raw)"
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV filename (written into ../data if it exists)")
    ap.add_argument("--no_plot", action="store_true", help="Disable plots")
    args = ap.parse_args()

    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    rows: List[FoMetrics] = []
    for rp in raw_files:
        base = os.path.basename(rp)
        fo = extract_fo_from_name(base, args.fo_regex)
        if fo is None:
            print(f"Skipping (can't parse bias_fo): {base}")
            continue

        m = analyze_one(
            raw_path=rp,
            bias_fo=float(fo),
            led_freq_hz=args.led_freq_hz,
            bin_ms=args.bin_ms,
            peak_k=args.peak_k,
            min_peak_dist_ms=args.min_peak_dist_ms,
            guard_ms=args.guard_ms,
            edges_per_cycle=args.edges_per_cycle,
        )
        rows.append(m)
        print(f"Done {base}: fo={m.bias_fo} FWHM={m.mean_fwhm_ms:.3f}ms bg_mean={m.bg_mean:.2f} missed={m.missed_edge_frac:.3f}")

    if not rows:
        raise RuntimeError("No files analyzed. Check filenames or --fo_regex.")

    rows.sort(key=lambda r: r.bias_fo)

    # Write CSV (prefer ../data/)
    out_path = args.out_csv
    if not os.path.isabs(out_path):
        here = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(here, "..", "data"))
        os.makedirs(data_dir, exist_ok=True)
        out_path = os.path.join(data_dir, out_path)
    if not out_path.lower().endswith(".csv"):
        out_path += ".csv"

    header = [
        "raw_file", "bias_fo",
        "duration_s", "total_events", "events_per_s",
        "peaks_detected", "missed_edge_frac",
        "mean_fwhm_ms", "mean_peak_slope_per_ms", "mean_peak_height",
        "bg_mean", "bg_std", "bg_median", "peak_snr"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.raw_file, r.bias_fo,
                r.duration_s, r.total_events, r.events_per_s,
                r.peaks_detected, r.missed_edge_frac,
                r.mean_fwhm_ms, r.mean_peak_slope_per_ms, r.mean_peak_height,
                r.bg_mean, r.bg_std, r.bg_median, r.peak_snr
            ])

    print("Saved:", out_path)

    if args.no_plot:
        return

    fo = np.array([r.bias_fo for r in rows], dtype=float)
    fwhm = np.array([r.mean_fwhm_ms for r in rows], dtype=float)
    bgm = np.array([r.bg_mean for r in rows], dtype=float)
    snr = np.array([r.peak_snr for r in rows], dtype=float)
    missed = np.array([r.missed_edge_frac for r in rows], dtype=float)
    evps = np.array([r.events_per_s for r in rows], dtype=float)
    slope = np.array([r.mean_peak_slope_per_ms for r in rows], dtype=float)

    plt.figure()
    plt.plot(fo, fwhm, marker="o")
    plt.xlabel("bias_fo")
    plt.ylabel("mean peak FWHM (ms)  ↓ sharper")
    plt.title("Edge sharpness vs bias_fo")
    plt.grid(True)

    plt.figure()
    plt.plot(fo, slope, marker="o")
    plt.xlabel("bias_fo")
    plt.ylabel("mean peak slope (counts/ms)  ↑ sharper")
    plt.title("Edge steepness vs bias_fo")
    plt.grid(True)

    plt.figure()
    plt.plot(fo, bgm, marker="o")
    plt.xlabel("bias_fo")
    plt.ylabel("background mean (counts/bin)  ↓ less noise")
    plt.title("Background noise vs bias_fo")
    plt.grid(True)

    plt.figure()
    plt.plot(fo, snr, marker="o")
    plt.xlabel("bias_fo")
    plt.ylabel("peak SNR proxy (z-score-ish)  ↑ better")
    plt.title("Peak prominence vs bias_fo")
    plt.grid(True)

    plt.figure()
    plt.plot(fo, missed, marker="o")
    plt.xlabel("bias_fo")
    plt.ylabel("missed edge fraction  ↓ better")
    plt.title("Missed edges vs bias_fo")
    plt.grid(True)

    plt.figure()
    plt.plot(fo, evps, marker="o")
    plt.xlabel("bias_fo")
    plt.ylabel("events/s")
    plt.title("Event rate vs bias_fo")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
