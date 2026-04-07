import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


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
class ManifestEntry:
    requested_frequency_hz: float
    actual_frequency_hz: float
    duration_s: float


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
    expected_frequency_hz: float
    manifest_duration_s: float
    manifest_source: str
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
    peak_estimated_bit_period_ms: float
    peak_estimated_bit_frequency_hz: float
    fft_peak_frequency_hz: float
    fft_peak_period_ms: float
    autocorr_bit_period_ms: float
    autocorr_bit_frequency_hz: float
    frequency_estimator: str
    estimated_bit_period_ms: float
    estimated_bit_frequency_hz: float


def is_valid_frequency_hz(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def relative_frequency_error(value_hz: float, target_hz: float) -> float:
    if not is_valid_frequency_hz(value_hz) or not is_valid_frequency_hz(target_hz):
        return float("inf")
    return float(abs(value_hz - target_hz) / max(abs(target_hz), 1e-9))


def load_manifest_csv(path: str) -> Dict[float, ManifestEntry]:
    manifest: Dict[float, ManifestEntry] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV is missing a header row: {path}")

        required = {"requested_frequency_hz", "actual_frequency_hz", "duration_s"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Manifest CSV {path} is missing required columns: {', '.join(missing)}")

        for row in reader:
            requested_text = str(row.get("requested_frequency_hz", "")).strip()
            actual_text = str(row.get("actual_frequency_hz", "")).strip()
            duration_text = str(row.get("duration_s", "")).strip()
            if not requested_text or not actual_text or not duration_text:
                continue

            requested_frequency_hz = float(requested_text)
            manifest[requested_frequency_hz] = ManifestEntry(
                requested_frequency_hz=requested_frequency_hz,
                actual_frequency_hz=float(actual_text),
                duration_s=float(duration_text),
            )

    return manifest


def lookup_manifest_entry(
    manifest: Dict[float, ManifestEntry],
    requested_frequency_hz: float,
    tolerance_fraction: float = 0.02,
) -> Optional[ManifestEntry]:
    best: Optional[ManifestEntry] = None
    best_error = float("inf")
    for entry in manifest.values():
        err = relative_frequency_error(entry.requested_frequency_hz, requested_frequency_hz)
        if err < best_error:
            best = entry
            best_error = err
    if best is None or best_error > tolerance_fraction:
        return None
    return best


def choose_manifest_entry_for_raw(
    raw_file: str,
    nominal_frequency_hz: float,
    replication_manifest: Dict[float, ManifestEntry],
    alt_manifest: Dict[float, ManifestEntry],
) -> Tuple[Optional[ManifestEntry], str]:
    lower_name = raw_file.lower()
    preferred: List[Tuple[str, Dict[float, ManifestEntry]]] = []
    fallback: List[Tuple[str, Dict[float, ManifestEntry]]] = []

    if lower_name.startswith("s31_replication"):
        preferred.append(("replication_manifest", replication_manifest))
        fallback.append(("alt_manifest", alt_manifest))
    elif lower_name.startswith("s31_cal"):
        preferred.append(("alt_manifest", alt_manifest))
        fallback.append(("replication_manifest", replication_manifest))
    else:
        preferred.extend(
            [
                ("replication_manifest", replication_manifest),
                ("alt_manifest", alt_manifest),
            ]
        )

    for source_name, manifest in preferred + fallback:
        if not manifest:
            continue
        entry = lookup_manifest_entry(manifest, nominal_frequency_hz)
        if entry is not None:
            return entry, source_name

    return None, "none"


def choose_estimated_frequency_hz(
    nominal_frequency_hz: float,
    expected_frequency_hz: float,
    fft_frequency_hz: float,
    autocorr_frequency_hz: float,
    peak_frequency_hz: float,
    max_disagreement_fraction: float = 0.12,
    max_target_error_fraction: float = 0.15,
) -> Tuple[float, str]:
    target_hz = float(expected_frequency_hz if is_valid_frequency_hz(expected_frequency_hz) else nominal_frequency_hz)
    valid_candidates = [
        ("autocorr", float(autocorr_frequency_hz)),
        ("fft", float(fft_frequency_hz)),
        ("peak", float(peak_frequency_hz)),
    ]
    valid_candidates = [(label, freq) for label, freq in valid_candidates if is_valid_frequency_hz(freq)]

    if is_valid_frequency_hz(fft_frequency_hz) and is_valid_frequency_hz(autocorr_frequency_hz):
        disagreement = relative_frequency_error(float(autocorr_frequency_hz), float(fft_frequency_hz))
        if disagreement > max_disagreement_fraction:
            ordered = sorted(valid_candidates, key=lambda item: relative_frequency_error(item[1], target_hz))
            if ordered and relative_frequency_error(ordered[0][1], target_hz) <= max_target_error_fraction:
                return float(ordered[0][1]), f"{ordered[0][0]}_disagreement_resolved"
            fallback_source = "manifest_fallback" if is_valid_frequency_hz(expected_frequency_hz) else "nominal_fallback"
            return float(target_hz), fallback_source

    if valid_candidates:
        ordered = sorted(valid_candidates, key=lambda item: relative_frequency_error(item[1], target_hz))
        best_label, best_frequency_hz = ordered[0]
        if relative_frequency_error(best_frequency_hz, target_hz) <= max_target_error_fraction:
            return float(best_frequency_hz), f"{best_label}_preferred"

    if is_valid_frequency_hz(target_hz):
        fallback_source = "manifest_fallback" if is_valid_frequency_hz(expected_frequency_hz) else "nominal_fallback"
        return float(target_hz), fallback_source

    return float("nan"), "unavailable"


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
    expected_duration_s: float = float("nan"),
) -> ActivityWindow:
    bin_width_us = int(round(bin_width_ms * 1000.0))
    counts = accumulate_time_hist(raw_path, capture_start_us, capture_end_us + 1, bin_width_us)
    capture_duration_s = float((capture_end_us - capture_start_us) * 1e-6)
    n_baseline = max(3, len(counts) // 10)
    baseline = counts[:n_baseline].astype(np.float64)
    threshold = float(np.median(baseline) + threshold_sigma * np.std(baseline))
    active = counts > threshold
    if not np.any(active):
        top_idx = int(np.argmax(counts))
        start_s = top_idx * bin_width_us * 1e-6
        fallback_duration_s = (
            float(expected_duration_s)
            if np.isfinite(expected_duration_s) and expected_duration_s > 0
            else bin_width_us * 1e-6
        )
        end_s = min(start_s + fallback_duration_s, capture_duration_s)
        return ActivityWindow(start_s=start_s, end_s=end_s, duration_s=end_s - start_s, threshold=threshold)

    idx = np.where(active)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    best_group = max(groups, key=lambda g: (int(np.sum(counts[g])), int(g.size)))
    start_s = float(best_group[0] * bin_width_us * 1e-6)
    end_s = float(min((best_group[-1] + 1) * bin_width_us * 1e-6, capture_duration_s))
    if np.isfinite(expected_duration_s) and expected_duration_s > 0:
        end_s = float(min(end_s, start_s + float(expected_duration_s), capture_duration_s))
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


def estimate_fft_frequency_hz(
    y: np.ndarray,
    bin_width_s: float,
    nominal_frequency_hz: float,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> float:
    if y.size < 8 or not np.isfinite(bin_width_s) or bin_width_s <= 0 or nominal_frequency_hz <= 0:
        return float("nan")

    centered = y.astype(np.float64) - float(np.mean(y))
    if not np.any(np.abs(centered) > 0):
        return float("nan")

    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window))
    freqs = np.fft.rfftfreq(centered.size, d=bin_width_s)
    lo = max(1e-9, float(nominal_frequency_hz) * float(min_scale))
    hi = max(lo * 1.1, float(nominal_frequency_hz) * float(max_scale))
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")

    masked_freqs = freqs[mask]
    masked_spec = spectrum[mask]
    # Prefer strong peaks near the nominal bitrate instead of harmonics at ~2x.
    log_ratio = np.log(np.maximum(masked_freqs, 1e-12) / float(nominal_frequency_hz))
    nominal_weight = np.exp(-0.5 * (log_ratio / 0.35) ** 2)
    weighted_spec = masked_spec * nominal_weight
    best_idx = int(np.argmax(weighted_spec))
    return float(masked_freqs[best_idx])


def estimate_autocorr_period_s(
    y: np.ndarray,
    bin_width_s: float,
    nominal_frequency_hz: float,
    coarse_frequency_hz: float,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> float:
    if y.size < 8 or not np.isfinite(bin_width_s) or bin_width_s <= 0 or nominal_frequency_hz <= 0:
        return float("nan")

    centered = y.astype(np.float64) - float(np.mean(y))
    if not np.any(np.abs(centered) > 0):
        return float("nan")

    autocorr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    min_lag = max(1, int(np.floor(1.0 / (float(nominal_frequency_hz) * float(max_scale) * bin_width_s))))
    max_lag = min(
        autocorr.size - 2,
        int(np.ceil(1.0 / (float(nominal_frequency_hz) * float(min_scale) * bin_width_s))),
    )
    if max_lag <= min_lag:
        return float("nan")

    if np.isfinite(coarse_frequency_hz) and coarse_frequency_hz > 0:
        coarse_lag = 1.0 / (float(coarse_frequency_hz) * bin_width_s)
        half_window = max(2.0, 0.5 * coarse_lag)
        min_lag = max(min_lag, int(np.floor(coarse_lag - half_window)))
        max_lag = min(max_lag, int(np.ceil(coarse_lag + half_window)))
        if max_lag <= min_lag:
            return float("nan")

    search = autocorr[min_lag : max_lag + 1]
    best_rel = int(np.argmax(search))
    best_lag = int(min_lag + best_rel)
    refined_lag = float(best_lag)
    if 1 <= best_lag < (autocorr.size - 1):
        y0 = float(autocorr[best_lag - 1])
        y1 = float(autocorr[best_lag])
        y2 = float(autocorr[best_lag + 1])
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            refined_lag += float(np.clip(delta, -1.0, 1.0))

    if refined_lag <= 0:
        return float("nan")
    return float(refined_lag * bin_width_s)


def estimate_trace_frequency(
    counts: np.ndarray,
    bin_width_s: float,
    nominal_frequency_hz: float,
) -> Tuple[float, float, float, float, str]:
    fft_frequency_hz = estimate_fft_frequency_hz(
        y=counts,
        bin_width_s=bin_width_s,
        nominal_frequency_hz=nominal_frequency_hz,
    )
    fft_period_s = 1.0 / fft_frequency_hz if np.isfinite(fft_frequency_hz) and fft_frequency_hz > 0 else float("nan")

    autocorr_period_s = estimate_autocorr_period_s(
        y=counts,
        bin_width_s=bin_width_s,
        nominal_frequency_hz=nominal_frequency_hz,
        coarse_frequency_hz=fft_frequency_hz,
    )
    autocorr_frequency_hz = (
        1.0 / autocorr_period_s if np.isfinite(autocorr_period_s) and autocorr_period_s > 0 else float("nan")
    )

    if np.isfinite(autocorr_frequency_hz) and autocorr_frequency_hz > 0:
        return (
            float(fft_frequency_hz),
            float(fft_period_s),
            float(autocorr_frequency_hz),
            float(autocorr_period_s),
            "fft_autocorr",
        )
    if np.isfinite(fft_frequency_hz) and fft_frequency_hz > 0:
        return (
            float(fft_frequency_hz),
            float(fft_period_s),
            float("nan"),
            float("nan"),
            "fft",
        )
    return float("nan"), float("nan"), float("nan"), float("nan"), "unavailable"


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
    estimator: str,
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
        title += f", estimated={estimated_period_s*1e3:.3f} ms"
        if estimator:
            title += f" via {estimator}"
        title += ")"
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
    root = REPO_ROOT
    default_raw = os.path.abspath(
        os.path.join(root, "..", "captures", "experiment_replication_3_1", "s31_replication_500Hz_t1.raw")
    )
    default_replication_manifest_csv = os.path.join(root, "pru1_pwm_CSK_1000Hz", "userspace", "s31_replication_manifest.csv")
    default_alt_manifest_csv = os.path.join(root, "pru1_pwm_CSK_1000Hz", "userspace", "s31_cal_alt_manifest.csv")
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
    ap.add_argument(
        "--replication_manifest_csv",
        default=default_replication_manifest_csv,
        help="Manifest CSV for the repeated-message replication sweep.",
    )
    ap.add_argument(
        "--alt_manifest_csv",
        default=default_alt_manifest_csv,
        help="Manifest CSV for the alternating-bit calibration captures.",
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

    replication_manifest = (
        load_manifest_csv(args.replication_manifest_csv)
        if args.replication_manifest_csv and os.path.exists(args.replication_manifest_csv)
        else {}
    )
    alt_manifest = (
        load_manifest_csv(args.alt_manifest_csv)
        if args.alt_manifest_csv and os.path.exists(args.alt_manifest_csv)
        else {}
    )
    manifest_entry, manifest_source = choose_manifest_entry_for_raw(
        raw_file=os.path.basename(raw_path),
        nominal_frequency_hz=float(nominal_frequency_hz),
        replication_manifest=replication_manifest,
        alt_manifest=alt_manifest,
    )
    expected_duration_s = float(manifest_entry.duration_s) if manifest_entry is not None else float("nan")
    expected_frequency_hz = (
        float(manifest_entry.actual_frequency_hz) if manifest_entry is not None else float(nominal_frequency_hz)
    )

    first_t_us, last_t_us, max_x, max_y = scan_capture_metadata(raw_path)
    capture_duration_s = (last_t_us - first_t_us) * 1e-6
    activity_window = detect_activity_window(
        raw_path=raw_path,
        capture_start_us=first_t_us,
        capture_end_us=last_t_us,
        bin_width_ms=args.activity_bin_ms,
        threshold_sigma=args.activity_threshold_sigma,
        expected_duration_s=expected_duration_s,
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
    peak_estimated_period_s = estimate_bit_period_s(peak_dt_s, nominal_period_s)
    peak_estimated_frequency_hz = (
        1.0 / peak_estimated_period_s
        if np.isfinite(peak_estimated_period_s) and peak_estimated_period_s > 0
        else float("nan")
    )
    (
        fft_frequency_hz,
        fft_period_s,
        autocorr_frequency_hz,
        autocorr_period_s,
        _trace_frequency_estimator,
    ) = estimate_trace_frequency(
        counts=trace_counts_smoothed,
        bin_width_s=args.transition_bin_us * 1e-6,
        nominal_frequency_hz=float(nominal_frequency_hz),
    )
    estimated_frequency_hz, frequency_estimator = choose_estimated_frequency_hz(
        nominal_frequency_hz=float(nominal_frequency_hz),
        expected_frequency_hz=expected_frequency_hz,
        fft_frequency_hz=float(fft_frequency_hz),
        autocorr_frequency_hz=float(autocorr_frequency_hz),
        peak_frequency_hz=float(peak_estimated_frequency_hz),
    )
    estimated_period_s = (
        1.0 / estimated_frequency_hz if np.isfinite(estimated_frequency_hz) and estimated_frequency_hz > 0 else float("nan")
    )

    data_dir = os.path.join(root, "data", "3.1", "replication_calibration")
    plot_dir = os.path.join(root, "plots", "3.1", "replication_calibration")
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
        frequency_estimator,
        roi_activity_path,
    )

    summary = CalibrationSummary(
        raw_file=os.path.basename(raw_path),
        nominal_frequency_hz=float(nominal_frequency_hz),
        expected_frequency_hz=float(expected_frequency_hz),
        manifest_duration_s=float(expected_duration_s),
        manifest_source=str(manifest_source),
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
        peak_estimated_bit_period_ms=float(peak_estimated_period_s * 1e3) if np.isfinite(peak_estimated_period_s) else float("nan"),
        peak_estimated_bit_frequency_hz=float(peak_estimated_frequency_hz),
        fft_peak_frequency_hz=float(fft_frequency_hz),
        fft_peak_period_ms=float(fft_period_s * 1e3) if np.isfinite(fft_period_s) else float("nan"),
        autocorr_bit_period_ms=float(autocorr_period_s * 1e3) if np.isfinite(autocorr_period_s) else float("nan"),
        autocorr_bit_frequency_hz=float(autocorr_frequency_hz),
        frequency_estimator=str(frequency_estimator),
        estimated_bit_period_ms=float(estimated_period_s * 1e3) if np.isfinite(estimated_period_s) else float("nan"),
        estimated_bit_frequency_hz=float(estimated_frequency_hz),
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.__dict__.keys()))
        writer.writeheader()
        writer.writerow(summary.__dict__)

    print(f"Raw file: {summary.raw_file}")
    print(f"Nominal bit frequency: {summary.nominal_frequency_hz:.3f} Hz")
    if np.isfinite(summary.manifest_duration_s):
        print(
            "Manifest guidance:"
            f" source={summary.manifest_source}"
            f" expected_frequency={summary.expected_frequency_hz:.3f} Hz"
            f" duration={summary.manifest_duration_s:.3f} s"
        )
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
            f" source={summary.frequency_estimator}"
        )
    else:
        print("Estimated transition timing: not enough peaks to estimate a bit period.")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved plot: {full_activity_path}")
    print(f"Saved plot: {roi_heatmap_path}")
    print(f"Saved plot: {roi_activity_path}")


if __name__ == "__main__":
    main()
