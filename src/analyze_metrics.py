import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from metavision_core.event_io import EventsIterator
from typing import Optional


# HOW TO RUN THIS CODE IN YOUR TERMINAL
# 1. make sure your virtual enviroment is active and your in the lab folder etc. 
# 2. python analyze_metrics.py --raw led_test2.raw --save_csv
#   If the peak detection is too sensitive (too many lines), increase the threshold:
#       python analyze_metrics.py --raw led_test2.raw --peak_k 10 --save_csv
#   If it misses peaks, lower it:
#       python analyze_metrics.py --raw led_test2.raw --peak_k 4 --save_csv     





# ----------------------------
# Core loading + time signal
# ----------------------------
def load_timestamps_us(raw_path: str) -> np.ndarray:
    """Load all event timestamps (microseconds) from a Metavision .raw file."""
    events = EventsIterator(input_path=raw_path)
    ts = []
    for evs in events:
        # evs['t'] is microseconds
        ts.append(evs["t"])
    if not ts:
        return np.array([], dtype=np.int64)
    return np.concatenate(ts).astype(np.int64)


def event_rate_signal(time_s: np.ndarray, bin_width_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_centers, counts_per_bin) using histogram binning."""
    if time_s.size == 0:
        return np.array([]), np.array([])
    t0, t1 = float(time_s.min()), float(time_s.max())
    bins = np.arange(t0, t1 + bin_width_s, bin_width_s)
    counts, edges = np.histogram(time_s, bins=bins)
    t_centers = edges[:-1]  # left edges; fine for plotting and peak timing at bin resolution
    return t_centers, counts


# ----------------------------
# Onset / segmentation
# ----------------------------
def detect_onset_time(t: np.ndarray, y: np.ndarray, k_sigma: float = 8.0, min_sustain_bins: int = 20) -> Optional[float]:
    """
    Detect when activity 'turns on' by finding the first time the signal stays above a threshold.
    Threshold = median + k_sigma * robust_sigma (MAD-based).
    """
    if t.size == 0:
        return None

    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med))) + 1e-9
    robust_sigma = 1.4826 * mad
    thr = med + k_sigma * robust_sigma

    above = y > thr
    # Find first index where above stays True for min_sustain_bins
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_sustain_bins:
            onset_idx = i - min_sustain_bins + 1
            return float(t[onset_idx])
    return None


def split_noise_signal(t: np.ndarray, y: np.ndarray, onset_s: float, noise_window_s: float = 1.0) -> dict:
    """
    Pick a noise window just before onset, and a signal window after onset.
    Returns indices and basic stats windows.
    """
    if onset_s is None:
        # fall back: treat first 20% as noise, last 80% as signal
        n = len(t)
        cut = max(1, int(0.2 * n))
        noise_idx = slice(0, cut)
        signal_idx = slice(cut, n)
    else:
        noise_start = onset_s - noise_window_s
        noise_end = onset_s
        noise_mask = (t >= noise_start) & (t < noise_end)
        signal_mask = (t >= onset_s)  # all after onset
        # If noise window is empty (short recording), fallback to early section
        if not np.any(noise_mask):
            n = len(t)
            cut = max(1, int(0.2 * n))
            noise_mask = np.zeros(n, dtype=bool)
            noise_mask[:cut] = True
        noise_idx = noise_mask
        signal_idx = signal_mask

    noise_vals = y[noise_idx]
    signal_vals = y[signal_idx]
    return {
        "noise_idx": noise_idx,
        "signal_idx": signal_idx,
        "noise_mean": float(np.mean(noise_vals)) if noise_vals.size else float("nan"),
        "noise_std": float(np.std(noise_vals)) if noise_vals.size else float("nan"),
        "signal_mean": float(np.mean(signal_vals)) if signal_vals.size else float("nan"),
        "signal_std": float(np.std(signal_vals)) if signal_vals.size else float("nan"),
    }


# ----------------------------
# Peak / edge detection
# ----------------------------
def find_peaks_simple(t: np.ndarray, y: np.ndarray, min_height: float, min_distance_s: float) -> np.ndarray:
    """
    Simple peak finder:
    - y[i] is a peak if y[i] > y[i-1] and y[i] >= y[i+1] and y[i] >= min_height
    - enforce min distance in seconds between peaks
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

    # enforce min spacing
    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if (t[idx] - t[peaks[-1]]) >= min_distance_s:
            peaks.append(idx)

    return t[np.array(peaks, dtype=int)]


def estimate_frequency_and_jitter(peak_times_s: np.ndarray) -> dict:
    """
    Compute frequency and jitter from peak times.
    Jitter is reported as std dev of period.
    """
    if peak_times_s.size < 3:
        return {"freq_hz": float("nan"), "period_mean_s": float("nan"), "period_std_s": float("nan")}

    periods = np.diff(peak_times_s)
    return {
        "freq_hz": float(1.0 / np.mean(periods)),
        "period_mean_s": float(np.mean(periods)),
        "period_std_s": float(np.std(periods)),
    }


# ----------------------------
# Inter-event timing around edges
# ----------------------------
def inter_event_dt_near_peaks(ts_us: np.ndarray, peak_times_s: np.ndarray, window_s: float = 0.002) -> dict:
    """
    For each peak time, look at events within ±window_s and compute inter-event dt stats.
    This is a proxy for how dense the burst is temporally.
    """
    if ts_us.size == 0 or peak_times_s.size == 0:
        return {"dt_mean_us": float("nan"), "dt_median_us": float("nan"), "dt_p95_us": float("nan")}

    ts_s = ts_us * 1e-6
    dts = []

    for pt in peak_times_s:
        mask = (ts_s >= pt - window_s) & (ts_s <= pt + window_s)
        seg = ts_s[mask]
        if seg.size >= 3:
            seg_sorted = np.sort(seg)
            d = np.diff(seg_sorted) * 1e6  # to microseconds
            dts.append(d)

    if not dts:
        return {"dt_mean_us": float("nan"), "dt_median_us": float("nan"), "dt_p95_us": float("nan")}

    dts = np.concatenate(dts)
    return {
        "dt_mean_us": float(np.mean(dts)),
        "dt_median_us": float(np.median(dts)),
        "dt_p95_us": float(np.percentile(dts, 95)),
    }


# ----------------------------
# SNR proxy (event-rate)
# ----------------------------
def snr_proxy(noise_mean: float, noise_std: float, signal_mean: float) -> dict:
    """
    SNR proxy using event-rate:
    - ratio = signal_mean / noise_mean
    - z = (signal_mean - noise_mean) / noise_std
    """
    ratio = float("nan") if noise_mean <= 0 else float(signal_mean / noise_mean)
    z = float("nan") if noise_std <= 0 else float((signal_mean - noise_mean) / noise_std)
    return {"snr_ratio": ratio, "snr_z": z}


# ----------------------------
# Plot helpers
# ----------------------------
def plot_activity(t: np.ndarray, y: np.ndarray, onset_s: Optional[float], peak_times_s: np.ndarray, title: str):
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Event count per bin")
    plt.title(title)

    if onset_s is not None:
        plt.axvline(onset_s, linestyle="--")
    for pt in peak_times_s[:200]:  # avoid drawing a million lines
        plt.axvline(pt, linestyle=":", linewidth=0.8)

    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to .raw file (e.g. led_test2.raw)")
    ap.add_argument("--bin_ms", type=float, default=1.0, help="Histogram bin width in milliseconds (default 1.0)")
    ap.add_argument("--peak_k", type=float, default=6.0, help="Peak threshold = noise_mean + peak_k*noise_std (default 6.0)")
    ap.add_argument("--min_peak_dist_ms", type=float, default=0.5, help="Min time between peaks in ms (default 0.5)")
    ap.add_argument("--burst_window_ms", type=float, default=2.0, help="Window around peaks for inter-event dt stats (default 2.0)")
    ap.add_argument("--save_csv", action="store_true", help="Save metrics summary as CSV next to the raw file")
    ap.add_argument("--no_plot", action="store_true", help="Do not show plots")
    args = ap.parse_args()

    raw_path = args.raw
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)

    # Load timestamps
    ts_us = load_timestamps_us(raw_path)
    print("Total events:", int(ts_us.size))

    # Build time-domain activity
    time_s = ts_us * 1e-6
    bin_width_s = args.bin_ms / 1000.0
    t_bins, counts = event_rate_signal(time_s, bin_width_s)

    # Detect onset
    onset_s = detect_onset_time(t_bins, counts)
    print("Detected onset (s):", onset_s if onset_s is not None else "None")

    # Noise/signal stats
    seg = split_noise_signal(t_bins, counts, onset_s)
    print(f"Noise mean/std: {seg['noise_mean']:.2f} / {seg['noise_std']:.2f}")
    print(f"Signal mean/std: {seg['signal_mean']:.2f} / {seg['signal_std']:.2f}")

    # Peaks (edges) after onset (or whole signal if onset unknown)
    if onset_s is None:
        t_use, y_use = t_bins, counts
    else:
        mask = t_bins >= onset_s
        t_use, y_use = t_bins[mask], counts[mask]

    # Peak height threshold based on noise stats
    min_height = seg["noise_mean"] + args.peak_k * seg["noise_std"]
    peak_times_s = find_peaks_simple(
        t_use, y_use,
        min_height=min_height,
        min_distance_s=args.min_peak_dist_ms / 1000.0
    )
    print("Detected peaks:", int(peak_times_s.size))

    # Frequency + jitter
    fj = estimate_frequency_and_jitter(peak_times_s)
    print(f"Estimated freq (Hz): {fj['freq_hz']:.3f}")
    print(f"Period mean/std (ms): {fj['period_mean_s']*1e3:.3f} / {fj['period_std_s']*1e3:.3f}")

    # Inter-event dt near peaks (burst density proxy)
    dt_stats = inter_event_dt_near_peaks(ts_us, peak_times_s, window_s=args.burst_window_ms / 1000.0)
    print(f"Inter-event dt near peaks (us) mean/median/p95: "
          f"{dt_stats['dt_mean_us']:.2f} / {dt_stats['dt_median_us']:.2f} / {dt_stats['dt_p95_us']:.2f}")

    # SNR proxy
    snr = snr_proxy(seg["noise_mean"], seg["noise_std"], seg["signal_mean"])
    print(f"SNR proxy ratio (signal/noise): {snr['snr_ratio']:.3f}")
    print(f"SNR proxy z-score: {snr['snr_z']:.3f}")

    # Save CSV summary
    if args.save_csv:
        out_csv = os.path.splitext(raw_path)[0] + "_metrics.csv"
        header = [
            "raw_file", "total_events", "bin_ms", "onset_s",
            "noise_mean", "noise_std", "signal_mean", "signal_std",
            "peaks_detected", "freq_hz", "period_mean_ms", "period_std_ms",
            "dt_mean_us", "dt_median_us", "dt_p95_us",
            "snr_ratio", "snr_z"
        ]
        row = [
            os.path.basename(raw_path), int(ts_us.size), args.bin_ms, onset_s if onset_s is not None else "",
            seg["noise_mean"], seg["noise_std"], seg["signal_mean"], seg["signal_std"],
            int(peak_times_s.size), fj["freq_hz"], fj["period_mean_s"]*1e3, fj["period_std_s"]*1e3,
            dt_stats["dt_mean_us"], dt_stats["dt_median_us"], dt_stats["dt_p95_us"],
            snr["snr_ratio"], snr["snr_z"]
        ]
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            f.write(",".join(map(str, row)) + "\n")
        print("Saved:", out_csv)

    # Plot
    if not args.no_plot:
        title = f"Event activity vs time ({os.path.basename(raw_path)})"
        plot_activity(t_bins, counts, onset_s, peak_times_s, title)


if __name__ == "__main__":
    main()
