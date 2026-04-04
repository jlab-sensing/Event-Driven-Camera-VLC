import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator

from io_utils import repo_root_from_this_file


def parse_numeric_token(token: str) -> float:
    return float(token.replace("p", "."))


def extract_frequency_from_name(filename: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, filename)
    if not match:
        return None
    try:
        return parse_numeric_token(match.group(1))
    except Exception:
        return None


@dataclass
class ActivityWindow:
    start_s: float
    end_s: float
    duration_s: float
    threshold: float


@dataclass
class RoiBox:
    x0: int
    y0: int
    x1: int
    y1: int
    peak_block_x: int
    peak_block_y: int
    peak_score: float


@dataclass
class CalibrationSummary:
    raw_file: str
    nominal_frequency_hz: float
    capture_duration_s: float
    active_start_s: float
    active_end_s: float
    active_duration_s: float
    activity_threshold: float
    baseline_start_s: float
    baseline_end_s: float
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    roi_peak_score: float
    roi_events: int
    polarity_on_fraction: float
    transition_bin_us: float
    peak_count: int
    peak_height_threshold: float
    median_peak_dt_ms: float
    estimated_bit_period_ms: float
    estimated_bit_frequency_hz: float


def scan_capture_metadata(raw_path: str) -> Tuple[int, int, int, int]:
    first_t_us: Optional[int] = None
    last_t_us: Optional[int] = None
    max_x = 0
    max_y = 0
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        if first_t_us is None:
            first_t_us = int(t[0])
        last_t_us = int(t[-1])
        max_x = max(max_x, int(evs["x"].max()))
        max_y = max(max_y, int(evs["y"].max()))
    if first_t_us is None or last_t_us is None:
        raise RuntimeError(f"No events found in {raw_path}")
    return first_t_us, last_t_us, max_x, max_y


def accumulate_time_hist(
    raw_path: str,
    start_us: int,
    end_us: int,
    bin_width_us: int,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    n_bins = max(1, int(np.ceil((end_us - start_us) / float(bin_width_us))))
    counts = np.zeros(n_bins, dtype=np.int64)
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        mask = (t >= start_us) & (t < end_us)
        if roi is not None:
            x0, y0, x1, y1 = roi
            mask &= (evs["x"] >= x0) & (evs["x"] < x1) & (evs["y"] >= y0) & (evs["y"] < y1)
        if not np.any(mask):
            continue
        idx = ((t[mask] - start_us) // bin_width_us).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        if idx.size:
            counts += np.bincount(idx, minlength=n_bins).astype(np.int64)
    return counts


def detect_activity_window(
    raw_path: str,
    capture_start_us: int,
    capture_end_us: int,
    bin_width_ms: float,
    threshold_sigma: float,
) -> ActivityWindow:
    bin_width_us = int(round(bin_width_ms * 1000.0))
    counts = accumulate_time_hist(raw_path, capture_start_us, capture_end_us + 1, bin_width_us)
    n_baseline = max(3, len(counts) // 10)
    baseline = counts[:n_baseline].astype(np.float64)
    threshold = float(np.median(baseline) + threshold_sigma * np.std(baseline))
    active = counts > threshold
    if not np.any(active):
        top_idx = int(np.argmax(counts))
        start_s = top_idx * bin_width_us * 1e-6
        end_s = min((top_idx + 1) * bin_width_us * 1e-6, (capture_end_us - capture_start_us) * 1e-6)
        return ActivityWindow(start_s=start_s, end_s=end_s, duration_s=end_s - start_s, threshold=threshold)

    idx = np.where(active)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    best_group = max(groups, key=lambda g: (int(np.sum(counts[g])), int(g.size)))
    start_s = float(best_group[0] * bin_width_us * 1e-6)
    end_s = float(min((best_group[-1] + 1) * bin_width_us * 1e-6, (capture_end_us - capture_start_us) * 1e-6))
    return ActivityWindow(start_s=start_s, end_s=end_s, duration_s=end_s - start_s, threshold=threshold)


def choose_baseline_window(
    activity_window: ActivityWindow,
    capture_duration_s: float,
) -> Tuple[float, float]:
    dur = float(activity_window.duration_s)
    if activity_window.start_s >= dur:
        return float(activity_window.start_s - dur), float(activity_window.start_s)
    if activity_window.end_s + dur <= capture_duration_s:
        return float(activity_window.end_s), float(activity_window.end_s + dur)
    return 0.0, float(min(dur, capture_duration_s))


def accumulate_block_counts(
    raw_path: str,
    start_us: int,
    end_us: int,
    block_px: int,
    shape: Tuple[int, int],
) -> np.ndarray:
    counts = np.zeros(shape, dtype=np.int64)
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        mask = (t >= start_us) & (t < end_us)
        if not np.any(mask):
            continue
        xb = (evs["x"][mask].astype(np.int64) // block_px)
        yb = (evs["y"][mask].astype(np.int64) // block_px)
        np.add.at(counts, (xb, yb), 1)
    return counts


def propose_roi(
    active_counts: np.ndarray,
    baseline_counts: np.ndarray,
    block_px: int,
    roi_blocks: int,
) -> RoiBox:
    score = active_counts.astype(np.float64) - baseline_counts.astype(np.float64)
    peak_x, peak_y = np.unravel_index(int(np.argmax(score)), score.shape)
    half = max(1, roi_blocks // 2)
    bx0 = max(0, peak_x - half)
    by0 = max(0, peak_y - half)
    bx1 = min(score.shape[0], bx0 + roi_blocks)
    by1 = min(score.shape[1], by0 + roi_blocks)
    bx0 = max(0, bx1 - roi_blocks)
    by0 = max(0, by1 - roi_blocks)
    return RoiBox(
        x0=int(bx0 * block_px),
        y0=int(by0 * block_px),
        x1=int(bx1 * block_px),
        y1=int(by1 * block_px),
        peak_block_x=int(peak_x),
        peak_block_y=int(peak_y),
        peak_score=float(score[peak_x, peak_y]),
    )


def load_roi_events(
    raw_path: str,
    start_us: int,
    end_us: int,
    roi: RoiBox,
) -> Tuple[np.ndarray, np.ndarray]:
    t_chunks: List[np.ndarray] = []
    p_chunks: List[np.ndarray] = []
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        mask = (
            (t >= start_us)
            & (t < end_us)
            & (evs["x"] >= roi.x0)
            & (evs["x"] < roi.x1)
            & (evs["y"] >= roi.y0)
            & (evs["y"] < roi.y1)
        )
        if not np.any(mask):
            continue
        t_chunks.append(t[mask])
        p_chunks.append(evs["p"][mask].astype(np.uint8))
    if not t_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)
    return np.concatenate(t_chunks), np.concatenate(p_chunks)


def binned_activity(time_s: np.ndarray, duration_s: float, bin_width_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = max(1, int(np.ceil(duration_s / bin_width_s)))
    edges = np.arange(0.0, (n_bins + 1) * bin_width_s, bin_width_s, dtype=np.float64)
    counts, _ = np.histogram(time_s, bins=edges)
    return edges[:-1], counts.astype(np.float64)


def smooth_counts(y: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return y.copy()
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    return np.convolve(y, kernel, mode="same")


def find_peaks_simple(
    t: np.ndarray,
    y: np.ndarray,
    min_height: float,
    min_distance_s: float,
) -> np.ndarray:
    if t.size < 3:
        return np.array([], dtype=np.float64)

    candidates: List[int] = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height:
            candidates.append(i)
    if not candidates:
        return np.array([], dtype=np.float64)

    kept = [candidates[0]]
    for idx in candidates[1:]:
        if (t[idx] - t[kept[-1]]) >= min_distance_s:
            kept.append(idx)
    return t[np.array(kept, dtype=np.int64)]


def estimate_bit_period_s(
    peak_dt_s: np.ndarray,
    nominal_period_s: float,
) -> float:
    if peak_dt_s.size == 0:
        return float("nan")

    usable = peak_dt_s[np.isfinite(peak_dt_s)]
    usable = usable[(usable > 0.1 * nominal_period_s) & (usable < 12.0 * nominal_period_s)]
    if usable.size == 0:
        return float("nan")

    lo = max(1e-6, 0.25 * nominal_period_s)
    hi = max(lo * 1.1, 4.0 * nominal_period_s)
    candidates = np.linspace(lo, hi, 1200, dtype=np.float64)
    best_period_s = float("nan")
    best_error = float("inf")
    for period_s in candidates:
        ratio = usable / period_s
        nearest = np.clip(np.round(ratio), 1.0, 12.0)
        error = float(np.median(np.abs(ratio - nearest)))
        if error < best_error:
            best_error = error
            best_period_s = float(period_s)
    return best_period_s


def save_activity_plot(
    t_s: np.ndarray,
    counts: np.ndarray,
    activity_window: ActivityWindow,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.plot(t_s, counts, linewidth=1.0)
    ax.axvspan(activity_window.start_s, activity_window.end_s, color="tab:orange", alpha=0.2)
    ax.set_title("Replication Calibration: Full-Capture Activity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Events / bin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_roi_heatmap(score: np.ndarray, roi: RoiBox, block_px: int, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.imshow(score.T, origin="lower", aspect="auto", cmap="inferno")
    rect = plt.Rectangle(
        (roi.x0 / block_px, roi.y0 / block_px),
        max(1, (roi.x1 - roi.x0) / block_px),
        max(1, (roi.y1 - roi.y0) / block_px),
        fill=False,
        edgecolor="cyan",
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.set_title("Active Minus Baseline Block Counts")
    ax.set_xlabel(f"Block X ({block_px}px)")
    ax.set_ylabel(f"Block Y ({block_px}px)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_roi_activity_plot(
    t_s: np.ndarray,
    counts: np.ndarray,
    peaks_s: np.ndarray,
    estimated_period_s: float,
    nominal_period_s: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(t_s, counts, linewidth=1.0, color="tab:blue")
    if peaks_s.size:
        peak_idx = np.searchsorted(t_s, peaks_s, side="left")
        peak_idx = peak_idx[(peak_idx >= 0) & (peak_idx < t_s.size)]
        ax.scatter(t_s[peak_idx], counts[peak_idx], s=12, color="tab:red")
    title = f"ROI Activity Within Active Window (nominal={nominal_period_s*1e3:.3f} ms"
    if np.isfinite(estimated_period_s):
        title += f", estimated={estimated_period_s*1e3:.3f} ms)"
    else:
        title += ")"
    ax.set_title(title)
    ax.set_xlabel("Time Within Active Window (s)")
    ax.set_ylabel("Events / bin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    root = repo_root_from_this_file(__file__)
    default_raw = os.path.abspath(
        os.path.join(root, "..", "captures", "experiment_replication_3_1", "s31_replication_500Hz_t1.raw")
    )
    ap = argparse.ArgumentParser(
        description="Calibrate one Section 3.1 replication capture by estimating the active transmit window, LED ROI, and observed transition timing."
    )
    ap.add_argument("--raw", default=default_raw, help="Path to one replication .raw capture.")
    ap.add_argument(
        "--nominal_frequency_hz",
        type=float,
        default=None,
        help="Nominal OOK bit frequency. If omitted, parse from the raw filename.",
    )
    ap.add_argument(
        "--freq_regex",
        default=r"([0-9]+(?:p[0-9]+|\\.[0-9]+)?)Hz",
        help="Regex with one capture group for extracting frequency from the raw filename.",
    )
    ap.add_argument("--activity_bin_ms", type=float, default=50.0, help="Bin width for full-capture activity search.")
    ap.add_argument(
        "--activity_threshold_sigma",
        type=float,
        default=5.0,
        help="Threshold = median(baseline) + sigma * std(baseline) for selecting the active window.",
    )
    ap.add_argument("--block_px", type=int, default=32, help="Spatial block size in pixels for ROI contrast mapping.")
    ap.add_argument("--roi_blocks", type=int, default=4, help="ROI width/height in coarse blocks.")
    ap.add_argument(
        "--transition_bin_us",
        type=float,
        default=50.0,
        help="Bin width in microseconds for the ROI activity trace used in transition calibration.",
    )
    ap.add_argument(
        "--smooth_bins",
        type=int,
        default=3,
        help="Moving-average kernel size applied to the ROI activity trace before peak finding.",
    )
    ap.add_argument(
        "--peak_quantile",
        type=float,
        default=0.995,
        help="Quantile used as the initial peak-height threshold for transition bursts.",
    )
    ap.add_argument(
        "--peak_min_distance_fraction",
        type=float,
        default=0.35,
        help="Minimum spacing between peaks as a fraction of the nominal bit period.",
    )
    ap.add_argument("--out_prefix", default="s31_replication_calibration", help="Prefix for saved CSV/plots.")
    args = ap.parse_args()

    raw_path = os.path.abspath(args.raw)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)

    nominal_frequency_hz = args.nominal_frequency_hz
    if nominal_frequency_hz is None:
        nominal_frequency_hz = extract_frequency_from_name(os.path.basename(raw_path), args.freq_regex)
    if nominal_frequency_hz is None or nominal_frequency_hz <= 0:
        raise ValueError("Could not determine nominal frequency. Provide --nominal_frequency_hz or rename the raw file.")

    first_t_us, last_t_us, max_x, max_y = scan_capture_metadata(raw_path)
    capture_duration_s = (last_t_us - first_t_us) * 1e-6
    activity_window = detect_activity_window(
        raw_path=raw_path,
        capture_start_us=first_t_us,
        capture_end_us=last_t_us,
        bin_width_ms=args.activity_bin_ms,
        threshold_sigma=args.activity_threshold_sigma,
    )

    baseline_start_s, baseline_end_s = choose_baseline_window(activity_window, capture_duration_s)
    block_shape = (max_x // args.block_px + 1, max_y // args.block_px + 1)
    active_counts = accumulate_block_counts(
        raw_path=raw_path,
        start_us=first_t_us + int(round(activity_window.start_s * 1e6)),
        end_us=first_t_us + int(round(activity_window.end_s * 1e6)),
        block_px=args.block_px,
        shape=block_shape,
    )
    baseline_counts = accumulate_block_counts(
        raw_path=raw_path,
        start_us=first_t_us + int(round(baseline_start_s * 1e6)),
        end_us=first_t_us + int(round(baseline_end_s * 1e6)),
        block_px=args.block_px,
        shape=block_shape,
    )
    roi = propose_roi(active_counts, baseline_counts, args.block_px, args.roi_blocks)

    roi_t_us, roi_p = load_roi_events(
        raw_path=raw_path,
        start_us=first_t_us + int(round(activity_window.start_s * 1e6)),
        end_us=first_t_us + int(round(activity_window.end_s * 1e6)),
        roi=roi,
    )
    if roi_t_us.size < 2:
        raise RuntimeError("No ROI events were found inside the active window.")

    roi_time_s = (roi_t_us - roi_t_us[0]).astype(np.float64) * 1e-6
    nominal_period_s = 1.0 / float(nominal_frequency_hz)
    trace_t_s, trace_counts = binned_activity(
        time_s=roi_time_s,
        duration_s=float(roi_time_s.max()),
        bin_width_s=args.transition_bin_us * 1e-6,
    )
    trace_counts_smoothed = smooth_counts(trace_counts, args.smooth_bins)
    peak_threshold = float(np.quantile(trace_counts_smoothed, args.peak_quantile))
    peaks_s = find_peaks_simple(
        t=trace_t_s,
        y=trace_counts_smoothed,
        min_height=peak_threshold,
        min_distance_s=max(args.transition_bin_us * 1e-6, args.peak_min_distance_fraction * nominal_period_s),
    )
    peak_dt_s = np.diff(peaks_s) if peaks_s.size > 1 else np.array([], dtype=np.float64)
    estimated_period_s = estimate_bit_period_s(peak_dt_s, nominal_period_s)
    estimated_frequency_hz = 1.0 / estimated_period_s if np.isfinite(estimated_period_s) and estimated_period_s > 0 else float("nan")

    data_dir = os.path.join(root, "data", "replication_calibration")
    plot_dir = os.path.join(root, "plots", "replication_calibration")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    raw_stem = os.path.splitext(os.path.basename(raw_path))[0]
    summary_path = os.path.join(data_dir, f"{args.out_prefix}_{raw_stem}_summary.csv")
    full_activity_path = os.path.join(plot_dir, f"{args.out_prefix}_{raw_stem}_capture_activity.png")
    roi_heatmap_path = os.path.join(plot_dir, f"{args.out_prefix}_{raw_stem}_roi_heatmap.png")
    roi_activity_path = os.path.join(plot_dir, f"{args.out_prefix}_{raw_stem}_roi_activity.png")

    coarse_counts = accumulate_time_hist(
        raw_path=raw_path,
        start_us=first_t_us,
        end_us=last_t_us + 1,
        bin_width_us=int(round(args.activity_bin_ms * 1000.0)),
    )
    coarse_t_s = np.arange(coarse_counts.size, dtype=np.float64) * args.activity_bin_ms * 1e-3
    save_activity_plot(coarse_t_s, coarse_counts.astype(np.float64), activity_window, full_activity_path)
    save_roi_heatmap(active_counts.astype(np.float64) - baseline_counts.astype(np.float64), roi, args.block_px, roi_heatmap_path)
    save_roi_activity_plot(
        trace_t_s,
        trace_counts_smoothed,
        peaks_s,
        estimated_period_s,
        nominal_period_s,
        roi_activity_path,
    )

    summary = CalibrationSummary(
        raw_file=os.path.basename(raw_path),
        nominal_frequency_hz=float(nominal_frequency_hz),
        capture_duration_s=float(capture_duration_s),
        active_start_s=float(activity_window.start_s),
        active_end_s=float(activity_window.end_s),
        active_duration_s=float(activity_window.duration_s),
        activity_threshold=float(activity_window.threshold),
        baseline_start_s=float(baseline_start_s),
        baseline_end_s=float(baseline_end_s),
        roi_x0=int(roi.x0),
        roi_y0=int(roi.y0),
        roi_x1=int(roi.x1),
        roi_y1=int(roi.y1),
        roi_peak_score=float(roi.peak_score),
        roi_events=int(roi_t_us.size),
        polarity_on_fraction=float(np.mean(roi_p)) if roi_p.size else float("nan"),
        transition_bin_us=float(args.transition_bin_us),
        peak_count=int(peaks_s.size),
        peak_height_threshold=float(peak_threshold),
        median_peak_dt_ms=float(np.median(peak_dt_s) * 1e3) if peak_dt_s.size else float("nan"),
        estimated_bit_period_ms=float(estimated_period_s * 1e3) if np.isfinite(estimated_period_s) else float("nan"),
        estimated_bit_frequency_hz=float(estimated_frequency_hz),
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.__dict__.keys()))
        writer.writeheader()
        writer.writerow(summary.__dict__)

    print(f"Raw file: {summary.raw_file}")
    print(f"Nominal bit frequency: {summary.nominal_frequency_hz:.3f} Hz")
    print(
        "Active window:"
        f" {summary.active_start_s:.3f}s to {summary.active_end_s:.3f}s"
        f" (duration {summary.active_duration_s:.3f}s)"
    )
    print(
        "Baseline window:"
        f" {summary.baseline_start_s:.3f}s to {summary.baseline_end_s:.3f}s"
    )
    print(
        "ROI:"
        f" x=[{summary.roi_x0},{summary.roi_x1})"
        f" y=[{summary.roi_y0},{summary.roi_y1})"
        f" score={summary.roi_peak_score:.1f}"
    )
    print(f"ROI events: {summary.roi_events}")
    print(f"Detected ROI peaks: {summary.peak_count}")
    if np.isfinite(summary.estimated_bit_frequency_hz):
        print(
            "Estimated transition timing:"
            f" period={summary.estimated_bit_period_ms:.3f} ms"
            f" frequency={summary.estimated_bit_frequency_hz:.3f} Hz"
        )
    else:
        print("Estimated transition timing: not enough peaks to estimate a bit period.")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved plot: {full_activity_path}")
    print(f"Saved plot: {roi_heatmap_path}")
    print(f"Saved plot: {roi_activity_path}")


if __name__ == "__main__":
    main()
