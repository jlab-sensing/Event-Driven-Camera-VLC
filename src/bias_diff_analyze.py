import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator

# Reuse your repo output conventions
from io_utils import repo_root_from_this_file


DEFAULT_INPUT_DIR = r"C:\Users\rabis\OneDrive\Documents\School\LAB aka 195\captures\3.1\testing_bias_diff_1000Hz"


# ----------------------------
# ROI and settings helpers
# ----------------------------
def normalize_roi(values: Optional[List[int]]) -> Optional[Tuple[int, int, int, int]]:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("ROI must contain x0 y0 x1 y1.")
    x0, y0, x1, y1 = [int(v) for v in values]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid ROI {values}; expected x1 > x0 and y1 > y0.")
    return x0, y0, x1, y1


def load_roi_from_settings(settings_json: str) -> Optional[Tuple[int, int, int, int]]:
    """Load the first enabled Metavision ROI window from a settings JSON file."""
    with open(settings_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    roi_state = data.get("roi_state", {})
    if not roi_state.get("enabled", False):
        return None

    windows = roi_state.get("window", [])
    if not windows:
        return None

    first = windows[0]
    x0 = int(first["x"])
    y0 = int(first["y"])
    width = int(first["width"])
    height = int(first["height"])
    return normalize_roi([x0, y0, x0 + width, y0 + height])


def roi_label(roi: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if roi is None:
        return -1, -1, -1, -1
    return roi


def build_roi_mask(evs: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if roi is None:
        return None
    if "x" not in evs.dtype.names or "y" not in evs.dtype.names:
        raise ValueError("ROI filtering requires event fields x and y.")
    x0, y0, x1, y1 = roi
    return (evs["x"] >= x0) & (evs["x"] < x1) & (evs["y"] >= y0) & (evs["y"] < y1)


# ----------------------------
# Streaming event binning
# ----------------------------
@dataclass
class EventScan:
    first_t_us: int
    last_t_us: int
    capture_events: int
    analyzed_events: int
    on_events: int


@dataclass
class BinnedEvents:
    t_bins_s: np.ndarray
    total_counts: np.ndarray
    on_counts: np.ndarray
    off_counts: np.ndarray
    duration_s: float
    capture_events: int
    analyzed_events: int
    on_fraction: float


def scan_event_window(raw_path: str, roi: Optional[Tuple[int, int, int, int]]) -> EventScan:
    """
    First pass over the RAW file.

    Finds the timestamp range and counts events after optional ROI filtering without
    keeping the full event stream in memory.
    """
    first_t_us: Optional[int] = None
    last_t_us: Optional[int] = None
    capture_events = 0
    analyzed_events = 0
    on_events = 0

    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue

        capture_events += int(evs.size)
        mask = build_roi_mask(evs, roi)
        selected_count = int(evs.size) if mask is None else int(np.count_nonzero(mask))
        if selected_count == 0:
            continue

        t = evs["t"].astype(np.int64) if mask is None else evs["t"][mask].astype(np.int64)
        if t.size == 0:
            continue

        if first_t_us is None:
            first_t_us = int(t[0])
        last_t_us = int(t[-1])
        analyzed_events += int(t.size)

        if "p" in evs.dtype.names:
            p = evs["p"] if mask is None else evs["p"][mask]
            on_events += int(np.count_nonzero(p > 0))

    if first_t_us is None or last_t_us is None:
        return EventScan(
            first_t_us=0,
            last_t_us=0,
            capture_events=capture_events,
            analyzed_events=0,
            on_events=0,
        )

    return EventScan(
        first_t_us=first_t_us,
        last_t_us=last_t_us,
        capture_events=capture_events,
        analyzed_events=analyzed_events,
        on_events=on_events,
    )


def load_binned_events(
    raw_path: str,
    bin_ms: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> BinnedEvents:
    """
    Stream RAW events into fixed-width time bins.

    The output keeps total, ON, and OFF counts per bin so BER can use an
    event-camera-friendly signal instead of timestamp-only full-frame activity.
    """
    if bin_ms <= 0:
        raise ValueError("bin_ms must be > 0.")

    bin_width_us = max(1, int(round(bin_ms * 1000.0)))
    scan = scan_event_window(raw_path, roi)
    if scan.analyzed_events < 2:
        empty = np.array([], dtype=np.float64)
        return BinnedEvents(
            t_bins_s=empty,
            total_counts=empty,
            on_counts=empty,
            off_counts=empty,
            duration_s=0.0,
            capture_events=scan.capture_events,
            analyzed_events=scan.analyzed_events,
            on_fraction=float("nan"),
        )

    duration_us = max(1, scan.last_t_us - scan.first_t_us + 1)
    n_bins = max(1, int(np.ceil(duration_us / float(bin_width_us))))
    total_counts = np.zeros(n_bins, dtype=np.int64)
    on_counts = np.zeros(n_bins, dtype=np.int64)
    off_counts = np.zeros(n_bins, dtype=np.int64)

    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue

        mask = build_roi_mask(evs, roi)
        if mask is not None and not np.any(mask):
            continue

        t = evs["t"].astype(np.int64) if mask is None else evs["t"][mask].astype(np.int64)
        if t.size == 0:
            continue

        idx = ((t - scan.first_t_us) // bin_width_us).astype(np.int64)
        valid = (idx >= 0) & (idx < n_bins)
        if not np.any(valid):
            continue

        idx = idx[valid]
        total_counts += np.bincount(idx, minlength=n_bins).astype(np.int64)

        if "p" in evs.dtype.names:
            p = evs["p"] if mask is None else evs["p"][mask]
            p = p[valid]
            on_idx = idx[p > 0]
            off_idx = idx[p <= 0]
            if on_idx.size:
                on_counts += np.bincount(on_idx, minlength=n_bins).astype(np.int64)
            if off_idx.size:
                off_counts += np.bincount(off_idx, minlength=n_bins).astype(np.int64)

    t_bins_s = (np.arange(n_bins, dtype=np.float64) * bin_width_us) * 1e-6
    duration_s = float(duration_us * 1e-6)
    on_fraction = float(scan.on_events / scan.analyzed_events) if scan.analyzed_events else float("nan")
    return BinnedEvents(
        t_bins_s=t_bins_s,
        total_counts=total_counts,
        on_counts=on_counts,
        off_counts=off_counts,
        duration_s=duration_s,
        capture_events=scan.capture_events,
        analyzed_events=scan.analyzed_events,
        on_fraction=on_fraction,
    )


def select_signal(binned: BinnedEvents, signal: str) -> np.ndarray:
    """Choose which binned signal drives peak finding and optional BER."""
    if signal == "total":
        return binned.total_counts.astype(np.float64)
    if signal == "on":
        return binned.on_counts.astype(np.float64)
    if signal == "off":
        return binned.off_counts.astype(np.float64)
    if signal == "diff":
        return binned.on_counts.astype(np.float64) - binned.off_counts.astype(np.float64)
    if signal == "absdiff":
        return np.abs(binned.on_counts.astype(np.float64) - binned.off_counts.astype(np.float64))
    raise ValueError(f"Unknown signal: {signal}")


# ----------------------------
# Peak and timing metrics
# ----------------------------
def find_peaks_simple(t: np.ndarray, y: np.ndarray, min_height: float, min_distance_s: float) -> np.ndarray:
    """
    Simple peak finder.
    Peak at i if y[i] > y[i-1] and y[i] >= y[i+1] and y[i] >= min_height.
    Enforces min spacing in seconds.
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
    Fit peak_time[k] ~= a*k + b and report residual timing std in microseconds.
    """
    if peak_times_s.size < 5:
        return float("nan")
    k = np.arange(peak_times_s.size)
    a, b = np.polyfit(k, peak_times_s, 1)
    pred = a * k + b
    resid_s = peak_times_s - pred
    return float(np.std(resid_s) * 1e6)


# ----------------------------
# Optional OOK decode + BER
# ----------------------------
@dataclass
class DecodeResult:
    ber: float
    bits_scored: int
    activity_start_time_s: float
    start_time_s: float
    threshold: float


def moving_average(y: np.ndarray, window_bins: int) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float64)
    if window_bins <= 1:
        return y.astype(np.float64)

    window_bins = min(int(window_bins), int(y.size))
    kernel = np.ones(window_bins, dtype=np.float64) / float(window_bins)
    return np.convolve(y.astype(np.float64), kernel, mode="same")


def detect_activity_start(
    t_bins: np.ndarray,
    counts: np.ndarray,
    bin_width_s: float,
    bitrate_hz: float,
    earliest_start_s: float,
    baseline_s: float,
    threshold_k: float,
    hold_bits: float,
) -> float:
    """
    Estimate when LED transmission begins from sustained event activity.

    The detector uses the quiet early bins as a baseline, then finds the first
    sustained rise above that baseline. It returns earliest_start_s if the
    signal already appears active or no reliable onset is found.
    """
    if t_bins.size == 0 or counts.size == 0:
        return float(earliest_start_s)
    if bin_width_s <= 0:
        raise ValueError("bin_width_s must be > 0.")
    if bitrate_hz <= 0:
        raise ValueError("bitrate_hz must be > 0.")

    earliest_start_s = max(0.0, float(earliest_start_s))
    start_idx = int(np.searchsorted(t_bins, earliest_start_s, side="left"))
    if start_idx >= counts.size:
        return float(earliest_start_s)

    activity = np.abs(counts.astype(np.float64))
    search = activity[start_idx:]
    if search.size == 0:
        return float(earliest_start_s)

    symbol_s = 1.0 / bitrate_hz
    baseline_bins = max(10, int(round(max(0.0, baseline_s) / bin_width_s)))
    baseline_bins = min(int(search.size), baseline_bins)
    early_baseline = search[:baseline_bins]
    quiet_baseline = np.partition(search, baseline_bins - 1)[:baseline_bins]
    baseline = (
        quiet_baseline
        if np.percentile(quiet_baseline, 95.0) < np.percentile(early_baseline, 95.0)
        else early_baseline
    )

    base_med = float(np.median(baseline))
    base_mad = float(np.median(np.abs(baseline - base_med)))
    base_sigma = 1.4826 * base_mad
    base_p95 = float(np.percentile(baseline, 95.0))
    threshold = max(
        base_med + threshold_k * max(base_sigma, 1e-9),
        base_p95 + max(1.0, 0.5 * base_sigma),
    )

    smooth_bins = max(1, int(round((0.5 * symbol_s) / bin_width_s)))
    hold_bins = max(1, int(round((max(hold_bits, 0.0) * symbol_s) / bin_width_s)))
    smoothed = moving_average(search, smooth_bins)
    active = smoothed >= threshold

    run = 0
    for i, is_active in enumerate(active):
        if is_active:
            run += 1
        else:
            run = 0
        if run >= hold_bins:
            onset_idx = max(0, i - run + 1 - (smooth_bins // 2))
            return float(t_bins[start_idx + onset_idx])

    return float(earliest_start_s)


def threshold_from_symbol_sums(sym_sums: np.ndarray) -> float:
    """Crude OOK threshold: midpoint between low/high groups split by median."""
    if sym_sums.size == 0:
        return float("nan")
    med = float(np.median(sym_sums))
    low = sym_sums[sym_sums <= med]
    high = sym_sums[sym_sums > med]
    if low.size == 0 or high.size == 0:
        return med
    return float(0.5 * (np.mean(low) + np.mean(high)))


def decode_ook_from_counts(
    t_bins: np.ndarray,
    counts: np.ndarray,
    bin_width_s: float,
    bitrate_hz: float,
    start_time_s: float,
    bits_true: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Decode OOK bits by integrating event counts over symbol windows.

    This assumes OOK is represented by more event activity in a '1' symbol than
    in a '0' symbol. That is usually best when '1' is a blinking carrier or when
    the transmitted bit pattern has enough transitions.
    """
    if bitrate_hz <= 0:
        raise ValueError("bitrate_hz must be > 0.")
    if t_bins.size == 0 or counts.size == 0 or bits_true.size == 0:
        return np.array([], dtype=np.uint8), float("nan")

    symbol_s = 1.0 / bitrate_hz
    capture_end_s = float(t_bins[-1] + bin_width_s)
    available_s = capture_end_s - start_time_s
    if available_s <= 0:
        return np.array([], dtype=np.uint8), float("nan")

    n_bits = min(int(bits_true.size), int(np.floor(available_s / symbol_s)))
    if n_bits <= 0:
        return np.array([], dtype=np.uint8), float("nan")

    sym_sums = np.zeros(n_bits, dtype=np.float64)
    for i in range(n_bits):
        t0 = start_time_s + i * symbol_s
        t1 = t0 + symbol_s
        left = int(np.searchsorted(t_bins, t0, side="left"))
        right = int(np.searchsorted(t_bins, t1, side="left"))
        sym_sums[i] = float(np.sum(counts[left:right]))

    threshold = threshold_from_symbol_sums(sym_sums)
    bits_hat = (sym_sums >= threshold).astype(np.uint8)
    return bits_hat, threshold


def ber(bits_hat: np.ndarray, bits_true: np.ndarray) -> float:
    if bits_hat.size == 0 or bits_true.size == 0:
        return float("nan")
    n = min(int(bits_hat.size), int(bits_true.size))
    if n <= 0:
        return float("nan")
    return float(np.mean(bits_hat[:n] != bits_true[:n]))


def decode_ook_with_phase_sweep(
    t_bins: np.ndarray,
    counts: np.ndarray,
    bin_width_s: float,
    bitrate_hz: float,
    bits_true: np.ndarray,
    start_time_s: float,
    phase_steps: int,
) -> DecodeResult:
    """
    Try several symbol phases and keep the one with the lowest BER.

    If phase_steps is 1, this behaves like the old fixed ber_start_time_s logic.
    """
    if phase_steps <= 1:
        starts = [float(start_time_s)]
    else:
        symbol_s = 1.0 / bitrate_hz
        starts = [float(start_time_s + phase) for phase in np.linspace(0.0, symbol_s, phase_steps, endpoint=False)]

    best = DecodeResult(
        ber=float("nan"),
        bits_scored=0,
        activity_start_time_s=float(start_time_s),
        start_time_s=float("nan"),
        threshold=float("nan"),
    )
    for candidate_start in starts:
        bits_hat, threshold = decode_ook_from_counts(
            t_bins=t_bins,
            counts=counts,
            bin_width_s=bin_width_s,
            bitrate_hz=bitrate_hz,
            start_time_s=candidate_start,
            bits_true=bits_true,
        )
        score = ber(bits_hat, bits_true)
        if not np.isfinite(score):
            continue
        if not np.isfinite(best.ber) or score < best.ber or (
            score == best.ber and bits_hat.size > best.bits_scored
        ):
            best = DecodeResult(
                ber=float(score),
                bits_scored=int(bits_hat.size),
                activity_start_time_s=float(start_time_s),
                start_time_s=float(candidate_start),
                threshold=float(threshold),
            )
    return best


# ----------------------------
# Bias diff parsing + file list
# ----------------------------
def extract_bias_from_name(filename: str, pattern: str) -> Optional[float]:
    """
    Extract bias_diff from filename using a regex with a capturing group.
    Default pattern expects names like biasdiff_-12.raw or biasdiff_12.5.raw.
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
    capture_events: int
    analyzed_events: int
    events_per_s: float
    on_fraction: float
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    signal: str
    peaks_detected: int
    freq_hz: float
    period_mean_ms: float
    period_std_ms: float
    edge_jitter_us: float
    ber: float
    bits_scored: int
    ber_activity_start_time_s: float
    ber_start_time_s: float
    ber_threshold: float


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
    phase_steps: int,
    ber_start_mode: str,
    activity_start_baseline_s: float,
    activity_start_k: float,
    activity_start_hold_bits: float,
    roi: Optional[Tuple[int, int, int, int]],
    signal: str,
) -> FileMetrics:
    binned = load_binned_events(raw_path=raw_path, bin_ms=bin_ms, roi=roi)
    rx0, ry0, rx1, ry1 = roi_label(roi)

    if binned.analyzed_events < 2:
        return FileMetrics(
            raw_file=os.path.basename(raw_path),
            bias_diff=bias_diff,
            duration_s=0.0,
            capture_events=binned.capture_events,
            analyzed_events=binned.analyzed_events,
            events_per_s=float("nan"),
            on_fraction=float("nan"),
            roi_x0=rx0,
            roi_y0=ry0,
            roi_x1=rx1,
            roi_y1=ry1,
            signal=signal,
            peaks_detected=0,
            freq_hz=float("nan"),
            period_mean_ms=float("nan"),
            period_std_ms=float("nan"),
            edge_jitter_us=float("nan"),
            ber=float("nan"),
            bits_scored=0,
            ber_activity_start_time_s=float("nan"),
            ber_start_time_s=float("nan"),
            ber_threshold=float("nan"),
        )

    counts = select_signal(binned, signal)
    events_per_s = (
        float(binned.analyzed_events / binned.duration_s)
        if binned.duration_s > 0
        else float("nan")
    )

    med = float(np.median(counts))
    mad = float(np.median(np.abs(counts - med))) + 1e-9
    robust_sigma = 1.4826 * mad
    min_height = med + peak_k * robust_sigma

    peak_times_s = find_peaks_simple(
        binned.t_bins_s,
        counts,
        min_height=min_height,
        min_distance_s=min_peak_dist_ms / 1000.0,
    )

    fj = estimate_freq_period_jitter(peak_times_s)
    edge_jit = edge_residual_jitter_us(peak_times_s)
    _ = expected_freq_hz

    decode = DecodeResult(
        ber=float("nan"),
        bits_scored=0,
        activity_start_time_s=float("nan"),
        start_time_s=float("nan"),
        threshold=float("nan"),
    )
    if do_ber and bitrate_hz and bits_true is not None and bits_true.size > 0:
        decode_start_time_s = float(ber_start_time_s)
        if ber_start_mode == "auto":
            decode_start_time_s = detect_activity_start(
                t_bins=binned.t_bins_s,
                counts=counts,
                bin_width_s=bin_ms / 1000.0,
                bitrate_hz=bitrate_hz,
                earliest_start_s=ber_start_time_s,
                baseline_s=activity_start_baseline_s,
                threshold_k=activity_start_k,
                hold_bits=activity_start_hold_bits,
            )
        decode = decode_ook_with_phase_sweep(
            t_bins=binned.t_bins_s,
            counts=counts,
            bin_width_s=bin_ms / 1000.0,
            bitrate_hz=bitrate_hz,
            bits_true=bits_true,
            start_time_s=decode_start_time_s,
            phase_steps=phase_steps,
        )

    return FileMetrics(
        raw_file=os.path.basename(raw_path),
        bias_diff=bias_diff,
        duration_s=binned.duration_s,
        capture_events=binned.capture_events,
        analyzed_events=binned.analyzed_events,
        events_per_s=events_per_s,
        on_fraction=binned.on_fraction,
        roi_x0=rx0,
        roi_y0=ry0,
        roi_x1=rx1,
        roi_y1=ry1,
        signal=signal,
        peaks_detected=int(peak_times_s.size),
        freq_hz=float(fj["freq_hz"]),
        period_mean_ms=float(fj["period_mean_s"] * 1e3),
        period_std_ms=float(fj["period_std_s"] * 1e3),
        edge_jitter_us=float(edge_jit),
        ber=float(decode.ber),
        bits_scored=int(decode.bits_scored),
        ber_activity_start_time_s=float(decode.activity_start_time_s),
        ber_start_time_s=float(decode.start_time_s),
        ber_threshold=float(decode.threshold),
    )


# ----------------------------
# Main sweep driver
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze a bias_diff sweep folder of EVK4 .raw files.")
    ap.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing .raw files for the sweep (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--bias_regex",
        default=r"biasdiff_([+-]?[0-9]+(?:\.[0-9]+)?)",
        help="Regex with one capture group for bias_diff from filename.",
    )
    ap.add_argument("--map_csv", default=None, help="Optional mapping CSV with columns raw_file,bias_diff")
    ap.add_argument("--bin_ms", type=float, default=1.0, help="Histogram bin width in ms (default 1.0)")
    ap.add_argument("--peak_k", type=float, default=6.0, help="Peak threshold = median + peak_k*robust_sigma")
    ap.add_argument("--min_peak_dist_ms", type=float, default=0.5, help="Min time between peaks in ms")
    ap.add_argument("--expected_freq_hz", type=float, default=None, help="Optional expected frequency for sanity checking")
    ap.add_argument("--out", required=True, help="Output CSV filename (saved into repo data/)")
    ap.add_argument("--plot_prefix", default=None, help="Optional prefix for plot filenames (saved into repo plots/)")
    ap.add_argument("--no_plot", action="store_true", help="Do not generate summary plots")

    # Event-camera-specific analysis controls
    ap.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Optional LED ROI, using half-open coordinates [x0,x1), [y0,y1).",
    )
    ap.add_argument(
        "--settings_json",
        default=None,
        help="Optional Metavision settings JSON; first enabled ROI is used when --roi is not provided.",
    )
    ap.add_argument(
        "--signal",
        choices=["total", "on", "off", "diff", "absdiff"],
        default="total",
        help="Binned event signal used for peak metrics and BER.",
    )

    # Optional BER decoding
    ap.add_argument("--bitrate_hz", type=float, default=None, help="If provided, attempt OOK decode at this bitrate")
    ap.add_argument("--bits_file", default=None, help="Text file containing truth bits, e.g. 101001")
    ap.add_argument("--ber_start_time_s", type=float, default=0.0, help="Earliest BER start time to use or search from")
    ap.add_argument(
        "--ber_start_mode",
        choices=["auto", "fixed"],
        default="auto",
        help="auto detects first sustained LED activity; fixed uses --ber_start_time_s directly",
    )
    ap.add_argument("--phase_steps", type=int, default=25, help="Number of symbol phases to test for BER (default 25)")
    ap.add_argument(
        "--activity_start_baseline_s",
        type=float,
        default=0.5,
        help="Seconds from the search start used to estimate quiet/background activity",
    )
    ap.add_argument(
        "--activity_start_k",
        type=float,
        default=8.0,
        help="Robust threshold multiplier for automatic LED activity start detection",
    )
    ap.add_argument(
        "--activity_start_hold_bits",
        type=float,
        default=3.0,
        help="How many bit periods activity must stay high before accepting an automatic start",
    )

    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    roi = normalize_roi(args.roi)
    if roi is None and args.settings_json:
        roi = load_roi_from_settings(args.settings_json)
        if roi is None:
            print(f"No enabled ROI found in settings JSON: {args.settings_json}; using full frame.")

    if roi is None:
        print("Using full-frame events. For LED BER, consider adding --roi X0 Y0 X1 Y1.")
    else:
        print(f"Using ROI x=[{roi[0]},{roi[2]}) y=[{roi[1]},{roi[3]})")

    mapping = load_map_csv(args.map_csv) if args.map_csv else {}

    bits_true = None
    do_ber = False
    if args.bitrate_hz and args.bits_file:
        with open(args.bits_file, "r", encoding="utf-8") as f:
            s = "".join(ch for ch in f.read() if ch in "01")
        bits_true = np.array([1 if ch == "1" else 0 for ch in s], dtype=np.uint8)
        do_ber = bits_true.size > 0
        print(f"Loaded {bits_true.size} truth bits for BER.")
    elif args.bitrate_hz or args.bits_file:
        print("BER disabled: provide both --bitrate_hz and --bits_file.")

    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    rows: List[FileMetrics] = []
    for rp in raw_files:
        base = os.path.basename(rp)

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
            ber_start_time_s=args.ber_start_time_s,
            phase_steps=args.phase_steps,
            ber_start_mode=args.ber_start_mode,
            activity_start_baseline_s=args.activity_start_baseline_s,
            activity_start_k=args.activity_start_k,
            activity_start_hold_bits=args.activity_start_hold_bits,
            roi=roi,
            signal=args.signal,
        )
        rows.append(fm)
        start_msg = (
            f" activity_start={fm.ber_activity_start_time_s:.4f}s"
            if np.isfinite(fm.ber_activity_start_time_s)
            else ""
        )
        print(
            f"Done: {base} bias_diff={fm.bias_diff} "
            f"events/s={fm.events_per_s:.2f} freq={fm.freq_hz:.2f}Hz "
            f"ber={fm.ber if np.isfinite(fm.ber) else 'nan'}{start_msg}"
        )

    if not rows:
        raise RuntimeError("No files analyzed. Check naming or --map_csv / --bias_regex.")

    rows.sort(key=lambda r: r.bias_diff)

    root = repo_root_from_this_file(__file__)
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)

    out_name = args.out
    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"
    out_path = os.path.join(out_dir, out_name)

    header = [
        "raw_file",
        "bias_diff",
        "duration_s",
        "capture_events",
        "analyzed_events",
        "events_per_s",
        "on_fraction",
        "roi_x0",
        "roi_y0",
        "roi_x1",
        "roi_y1",
        "signal",
        "peaks_detected",
        "freq_hz",
        "period_mean_ms",
        "period_std_ms",
        "edge_jitter_us",
        "ber",
        "bits_scored",
        "ber_activity_start_time_s",
        "ber_start_time_s",
        "ber_threshold",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.raw_file,
                r.bias_diff,
                r.duration_s,
                r.capture_events,
                r.analyzed_events,
                r.events_per_s,
                r.on_fraction,
                r.roi_x0,
                r.roi_y0,
                r.roi_x1,
                r.roi_y1,
                r.signal,
                r.peaks_detected,
                r.freq_hz,
                r.period_mean_ms,
                r.period_std_ms,
                r.edge_jitter_us,
                r.ber,
                r.bits_scored,
                r.ber_activity_start_time_s,
                r.ber_start_time_s,
                r.ber_threshold,
            ])

    print("Saved summary CSV:", out_path)

    if not args.no_plot:
        plot_dir = os.path.join(root, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_prefix = args.plot_prefix.strip() if args.plot_prefix else os.path.splitext(out_name)[0]

        bd = np.array([r.bias_diff for r in rows], dtype=float)
        ber_arr = np.array([r.ber for r in rows], dtype=float)
        jit = np.array([r.edge_jitter_us for r in rows], dtype=float)
        evs = np.array([r.events_per_s for r in rows], dtype=float)
        on_frac = np.array([r.on_fraction for r in rows], dtype=float)

        fig1 = plt.figure()
        plt.plot(bd, evs, marker="o")
        plt.xlabel("bias_diff")
        plt.ylabel("analyzed events/s")
        plt.title(f"Event rate vs bias_diff ({args.signal})")
        plt.grid(True)
        plot1_path = os.path.join(plot_dir, f"{plot_prefix}_event_rate_vs_bias_diff.png")
        fig1.savefig(plot1_path, dpi=300)
        print("Saved plot:", plot1_path)

        fig2 = plt.figure()
        plt.plot(bd, jit, marker="o")
        plt.xlabel("bias_diff")
        plt.ylabel("edge residual jitter (us)")
        plt.title(f"Timing jitter proxy vs bias_diff ({args.signal})")
        plt.grid(True)
        plot2_path = os.path.join(plot_dir, f"{plot_prefix}_timing_jitter_vs_bias_diff.png")
        fig2.savefig(plot2_path, dpi=300)
        print("Saved plot:", plot2_path)

        fig3 = plt.figure()
        plt.plot(bd, on_frac, marker="o")
        plt.xlabel("bias_diff")
        plt.ylabel("ON-event fraction")
        plt.title("Polarity balance vs bias_diff")
        plt.grid(True)
        plot3_path = os.path.join(plot_dir, f"{plot_prefix}_on_fraction_vs_bias_diff.png")
        fig3.savefig(plot3_path, dpi=300)
        print("Saved plot:", plot3_path)

        if np.any(np.isfinite(ber_arr)):
            fig4 = plt.figure()
            plt.plot(bd, ber_arr, marker="o")
            plt.xlabel("bias_diff")
            plt.ylabel("BER")
            plt.title(f"BER vs bias_diff ({args.signal}, phase sweep)")
            plt.grid(True)
            plot4_path = os.path.join(plot_dir, f"{plot_prefix}_ber_vs_bias_diff.png")
            fig4.savefig(plot4_path, dpi=300)
            print("Saved plot:", plot4_path)

        plt.show()


if __name__ == "__main__":
    main()
