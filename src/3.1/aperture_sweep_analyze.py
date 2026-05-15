import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

REPLICATION_DIR = os.path.abspath(os.path.join(THIS_DIR))
if REPLICATION_DIR not in sys.path:
    sys.path.insert(0, REPLICATION_DIR)

from section3_1_replication_analyze import (  # noqa: E402
    ManifestEntry,
    bits_to_string,
    load_transmission_manifest,
    load_truth_bits,
    lookup_manifest_entry,
    search_best_decode,
    search_best_transition_decode,
)


APERTURE_ORDER = {
    "C": 0.0,
    "f16": 16.0,
    "f11": 11.0,
    "f8": 8.0,
    "f5p6": 5.6,
    "f4": 4.0,
    "f2p8": 2.8,
    "f2": 2.0,
}


@dataclass
class CaptureBins:
    t_bins_s: np.ndarray
    total_counts: np.ndarray
    on_counts: np.ndarray
    off_counts: np.ndarray
    capture_duration_s: float
    capture_events: int
    roi_events: int
    on_fraction: float


@dataclass
class ApertureResult:
    aperture: str
    aperture_f_number: float
    raw_file: str
    bias_file: str
    capture_duration_s: float
    analysis_start_s: float
    analysis_end_s: float
    analysis_duration_s: float
    capture_events: int
    roi_events: int
    active_roi_events: int
    roi_event_rate_eps: float
    background_rate_eps: float
    expected_edges: int
    detected_edges: int
    edge_detection_rate: float
    timing_jitter_us: float
    ber: float
    mar: float
    bit_errors: int
    bits_scored: int
    messages_scored: int
    correct_messages: int
    phase_s: float
    threshold: float
    on_fraction: float


def parse_aperture(raw_file: str) -> Tuple[str, float]:
    match = re.search(r"_aperture_([^_]+)_", raw_file)
    if not match:
        raise ValueError(f"Could not parse aperture from filename: {raw_file}")
    aperture = match.group(1)
    if aperture not in APERTURE_ORDER:
        raise ValueError(f"Unknown aperture token {aperture!r} in {raw_file}")
    return aperture, APERTURE_ORDER[aperture]


def list_raw_files(input_dir: str) -> List[str]:
    return sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".raw")
    )


def scan_capture(raw_path: str) -> Tuple[int, int, int]:
    first_t_us: Optional[int] = None
    last_t_us: Optional[int] = None
    capture_events = 0

    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        capture_events += int(evs.size)
        if first_t_us is None:
            first_t_us = int(t[0])
        last_t_us = int(t[-1])

    if first_t_us is None or last_t_us is None:
        return 0, 0, capture_events
    return first_t_us, last_t_us, capture_events


def load_binned_roi_events(
    raw_path: str,
    roi: Tuple[int, int, int, int],
    bin_us: float,
) -> CaptureBins:
    if bin_us <= 0:
        raise ValueError("--bin_us must be > 0")

    capture_start_us, capture_end_us, capture_events = scan_capture(raw_path)
    if capture_events == 0 or capture_end_us <= capture_start_us:
        empty = np.array([], dtype=np.float64)
        return CaptureBins(
            t_bins_s=empty,
            total_counts=empty,
            on_counts=empty,
            off_counts=empty,
            capture_duration_s=0.0,
            capture_events=capture_events,
            roi_events=0,
            on_fraction=float("nan"),
        )

    bin_width_us = max(1, int(round(bin_us)))
    duration_us = capture_end_us - capture_start_us + 1
    n_bins = max(1, int(np.ceil(duration_us / float(bin_width_us))))
    total_counts = np.zeros(n_bins, dtype=np.int64)
    on_counts = np.zeros(n_bins, dtype=np.int64)
    off_counts = np.zeros(n_bins, dtype=np.int64)

    x0, y0, x1, y1 = roi
    roi_events = 0
    on_events = 0

    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue

        mask = (
            (evs["x"] >= x0)
            & (evs["x"] < x1)
            & (evs["y"] >= y0)
            & (evs["y"] < y1)
        )
        if not np.any(mask):
            continue

        t = evs["t"][mask].astype(np.int64)
        idx = ((t - capture_start_us) // bin_width_us).astype(np.int64)
        valid = (idx >= 0) & (idx < n_bins)
        if not np.any(valid):
            continue

        idx = idx[valid]
        roi_events += int(idx.size)
        total_counts += np.bincount(idx, minlength=n_bins).astype(np.int64)

        if "p" in evs.dtype.names:
            p = evs["p"][mask][valid]
            on_idx = idx[p > 0]
            off_idx = idx[p <= 0]
            on_events += int(on_idx.size)
            if on_idx.size:
                on_counts += np.bincount(on_idx, minlength=n_bins).astype(np.int64)
            if off_idx.size:
                off_counts += np.bincount(off_idx, minlength=n_bins).astype(np.int64)

    t_bins_s = np.arange(n_bins, dtype=np.float64) * bin_width_us * 1e-6
    capture_duration_s = duration_us * 1e-6
    on_fraction = float(on_events / roi_events) if roi_events else float("nan")
    return CaptureBins(
        t_bins_s=t_bins_s,
        total_counts=total_counts,
        on_counts=on_counts,
        off_counts=off_counts,
        capture_duration_s=float(capture_duration_s),
        capture_events=int(capture_events),
        roi_events=int(roi_events),
        on_fraction=on_fraction,
    )


def select_signal(bins: CaptureBins, signal: str) -> np.ndarray:
    if signal == "total":
        return bins.total_counts.astype(np.float64)
    if signal == "on":
        return bins.on_counts.astype(np.float64)
    if signal == "off":
        return bins.off_counts.astype(np.float64)
    if signal == "diff":
        return bins.on_counts.astype(np.float64) - bins.off_counts.astype(np.float64)
    if signal == "absdiff":
        return np.abs(bins.on_counts.astype(np.float64) - bins.off_counts.astype(np.float64))
    raise ValueError(f"Unsupported signal: {signal}")


def window_counts(
    t_bins_s: np.ndarray,
    counts: np.ndarray,
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (t_bins_s >= start_s) & (t_bins_s < end_s)
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return t_bins_s[mask] - start_s, counts[mask].astype(np.float64)


def background_rate(
    t_bins_s: np.ndarray,
    counts: np.ndarray,
    analysis_start_s: float,
    bin_s: float,
    fallback_s: float,
) -> float:
    pre_mask = t_bins_s < analysis_start_s
    if np.any(pre_mask):
        pre_counts = counts[pre_mask]
        pre_duration_s = max(bin_s, float(np.count_nonzero(pre_mask)) * bin_s)
        return float(np.sum(pre_counts) / pre_duration_s)

    fallback_bins = max(1, int(round(fallback_s / bin_s)))
    fallback_counts = counts[:fallback_bins]
    fallback_duration_s = max(bin_s, float(fallback_counts.size) * bin_s)
    return float(np.sum(fallback_counts) / fallback_duration_s)


def robust_threshold(values: np.ndarray, k: float) -> float:
    if values.size == 0:
        return float("nan")
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return float(med + k * max(1e-9, 1.4826 * mad))


def find_edge_peaks(
    t_bins_s: np.ndarray,
    counts: np.ndarray,
    min_height: float,
    min_distance_s: float,
) -> np.ndarray:
    if t_bins_s.size < 3:
        return np.array([], dtype=np.float64)

    candidates = []
    for i in range(1, counts.size - 1):
        if counts[i] > counts[i - 1] and counts[i] >= counts[i + 1] and counts[i] >= min_height:
            candidates.append(i)

    if not candidates:
        return np.array([], dtype=np.float64)

    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if (t_bins_s[idx] - t_bins_s[peaks[-1]]) >= min_distance_s:
            peaks.append(idx)
    return t_bins_s[np.array(peaks, dtype=int)]


def edge_residual_jitter_us(peak_times_s: np.ndarray) -> float:
    if peak_times_s.size < 5:
        return float("nan")
    k = np.arange(peak_times_s.size, dtype=np.float64)
    slope, intercept = np.polyfit(k, peak_times_s, 1)
    residual = peak_times_s - (slope * k + intercept)
    return float(np.std(residual) * 1e6)


def expected_transition_count(
    truth_message_bits: np.ndarray,
    message_repeats: int,
    guard_bits: int,
) -> int:
    bits = bits_to_string(truth_message_bits)
    if not bits or message_repeats <= 0:
        return 0

    guard = "0" * max(0, int(guard_bits))
    stream = guard + (bits * int(message_repeats)) + guard
    return sum(1 for a, b in zip(stream, stream[1:]) if a != b)


def find_analysis_window_from_counts(
    t_bins_s: np.ndarray,
    counts: np.ndarray,
    manifest_entry: ManifestEntry,
    message_len_bits: int,
    search_bin_s: float,
) -> Tuple[float, float]:
    if t_bins_s.size == 0:
        return 0.0, 0.0

    actual_frequency_hz = float(manifest_entry.actual_frequency_hz)
    payload_duration_s = float((manifest_entry.message_repeats * message_len_bits) / actual_frequency_hz)
    guard_duration_s = float(manifest_entry.guard_bits / actual_frequency_hz)

    if search_bin_s <= 0:
        raise ValueError("--window_search_bin_us must be > 0")

    capture_duration_s = float(t_bins_s[-1] + (t_bins_s[1] - t_bins_s[0] if t_bins_s.size > 1 else search_bin_s))
    n_bins = max(1, int(np.ceil(capture_duration_s / search_bin_s)))
    edges = np.arange(0.0, (n_bins + 1) * search_bin_s, search_bin_s, dtype=np.float64)
    rebinned, _ = np.histogram(t_bins_s, bins=edges, weights=counts)

    window_bins = max(1, int(np.ceil(payload_duration_s / search_bin_s)))
    if rebinned.size <= window_bins:
        payload_start_s = 0.0
    else:
        cumulative = np.concatenate(([0.0], np.cumsum(rebinned)))
        rolling = cumulative[window_bins:] - cumulative[:-window_bins]
        payload_start_s = float(edges[int(np.argmax(rolling))])
    payload_end_s = float(min(capture_duration_s, payload_start_s + payload_duration_s))

    start_s = max(0.0, payload_start_s - guard_duration_s)
    end_s = min(capture_duration_s, payload_end_s + guard_duration_s)
    return float(start_s), float(end_s)


def analyze_one(
    raw_path: str,
    manifest_entry: ManifestEntry,
    truth_message_bits: np.ndarray,
    roi: Tuple[int, int, int, int],
    bin_us: float,
    signal: str,
    phase_steps: int,
    decode_mode: str,
    transition_rate_min_scale: float,
    transition_rate_max_scale: float,
    transition_rate_steps: int,
    transition_edge_window_fraction: float,
    transition_max_rate_drift_fraction: float,
    edge_peak_k: float,
    edge_min_distance_ms: float,
    background_window_s: float,
    window_search_bin_us: float,
) -> ApertureResult:
    raw_file = os.path.basename(raw_path)
    aperture, aperture_f_number = parse_aperture(raw_file)
    bias_file = os.path.splitext(raw_file)[0] + ".bias"
    bias_path = os.path.join(os.path.dirname(raw_path), bias_file)
    bias_file_label = bias_file if os.path.exists(bias_path) else ""

    bins = load_binned_roi_events(raw_path=raw_path, roi=roi, bin_us=bin_us)
    signal_counts = select_signal(bins, signal)
    bin_s = float(bin_us) * 1e-6

    analysis_start_s, analysis_end_s = find_analysis_window_from_counts(
        t_bins_s=bins.t_bins_s,
        counts=signal_counts,
        manifest_entry=manifest_entry,
        message_len_bits=int(truth_message_bits.size),
        search_bin_s=window_search_bin_us * 1e-6,
    )
    analysis_duration_s = max(0.0, analysis_end_s - analysis_start_s)
    active_t, active_counts = window_counts(
        t_bins_s=bins.t_bins_s,
        counts=signal_counts,
        start_s=analysis_start_s,
        end_s=analysis_end_s,
    )
    _, active_total_counts = window_counts(
        t_bins_s=bins.t_bins_s,
        counts=bins.total_counts.astype(np.float64),
        start_s=analysis_start_s,
        end_s=analysis_end_s,
    )

    active_roi_events = int(np.sum(active_total_counts))
    roi_event_rate = (
        float(active_roi_events / analysis_duration_s)
        if analysis_duration_s > 0
        else float("nan")
    )
    bg_rate = background_rate(
        t_bins_s=bins.t_bins_s,
        counts=bins.total_counts.astype(np.float64),
        analysis_start_s=analysis_start_s,
        bin_s=bin_s,
        fallback_s=background_window_s,
    )

    expected_edges = expected_transition_count(
        truth_message_bits=truth_message_bits,
        message_repeats=int(manifest_entry.message_repeats),
        guard_bits=int(manifest_entry.guard_bits),
    )
    min_height = robust_threshold(active_counts, edge_peak_k)
    peak_times_s = find_edge_peaks(
        t_bins_s=active_t,
        counts=active_counts,
        min_height=min_height,
        min_distance_s=edge_min_distance_ms * 1e-3,
    )
    detected_edges = int(peak_times_s.size)
    edge_detection_rate = (
        float(min(detected_edges, expected_edges) / expected_edges)
        if expected_edges > 0
        else float("nan")
    )
    timing_jitter = edge_residual_jitter_us(peak_times_s)

    best = None
    if active_counts.size > 0 and analysis_duration_s > 0:
        if decode_mode == "transition":
            best = search_best_transition_decode(
                t_bins=active_t,
                counts=active_counts,
                duration_s=analysis_duration_s,
                nominal_rate_hz=float(manifest_entry.actual_frequency_hz),
                truth_message_bits=truth_message_bits,
                phase_steps=phase_steps,
                rate_min_scale=transition_rate_min_scale,
                rate_max_scale=transition_rate_max_scale,
                rate_steps=transition_rate_steps,
                edge_window_fraction=transition_edge_window_fraction,
                expected_message_count=int(manifest_entry.message_repeats),
                max_rate_drift_fraction=transition_max_rate_drift_fraction,
            )
        else:
            best = search_best_decode(
                t_bins=active_t,
                counts=active_counts,
                duration_s=analysis_duration_s,
                symbol_rate_hz=float(manifest_entry.actual_frequency_hz),
                truth_message_bits=truth_message_bits,
                phase_steps=phase_steps,
                expected_message_count=int(manifest_entry.message_repeats),
            )

    if best is None:
        ber = float("nan")
        mar = float("nan")
        bit_errors = 0
        bits_scored = 0
        messages_scored = 0
        correct_messages = 0
        phase_s = float("nan")
        threshold = float("nan")
    else:
        ber = float(best.ber)
        mar = float(best.mar)
        bit_errors = int(best.n_bit_errors)
        bits_scored = int(best.n_symbols_scored)
        messages_scored = int(best.n_messages)
        correct_messages = int(best.n_correct_messages)
        phase_s = float(best.phase_s)
        threshold = float(best.threshold)

    return ApertureResult(
        aperture=aperture,
        aperture_f_number=aperture_f_number,
        raw_file=raw_file,
        bias_file=bias_file_label,
        capture_duration_s=float(bins.capture_duration_s),
        analysis_start_s=float(analysis_start_s),
        analysis_end_s=float(analysis_end_s),
        analysis_duration_s=float(analysis_duration_s),
        capture_events=int(bins.capture_events),
        roi_events=int(bins.roi_events),
        active_roi_events=active_roi_events,
        roi_event_rate_eps=roi_event_rate,
        background_rate_eps=bg_rate,
        expected_edges=int(expected_edges),
        detected_edges=detected_edges,
        edge_detection_rate=edge_detection_rate,
        timing_jitter_us=float(timing_jitter),
        ber=ber,
        mar=mar,
        bit_errors=bit_errors,
        bits_scored=bits_scored,
        messages_scored=messages_scored,
        correct_messages=correct_messages,
        phase_s=phase_s,
        threshold=threshold,
        on_fraction=float(bins.on_fraction),
    )


def save_results(rows: List[ApertureResult], out_path: str) -> None:
    fieldnames = [
        "aperture",
        "aperture_f_number",
        "raw_file",
        "bias_file",
        "capture_duration_s",
        "analysis_start_s",
        "analysis_end_s",
        "analysis_duration_s",
        "capture_events",
        "roi_events",
        "active_roi_events",
        "roi_event_rate_eps",
        "background_rate_eps",
        "expected_edges",
        "detected_edges",
        "edge_detection_rate",
        "timing_jitter_us",
        "ber",
        "mar",
        "bit_errors",
        "bits_scored",
        "messages_scored",
        "correct_messages",
        "phase_s",
        "threshold",
        "on_fraction",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def save_plot(rows: List[ApertureResult], out_path: str) -> None:
    labels = [row.aperture for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    ber = np.array([row.ber for row in rows], dtype=np.float64)
    edge_rate = np.array([row.edge_detection_rate for row in rows], dtype=np.float64)
    roi_rate = np.array([row.roi_event_rate_eps for row in rows], dtype=np.float64)
    bg_rate = np.array([row.background_rate_eps for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(x, ber, marker="o")
    axes[0, 0].set_ylabel("BER")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, edge_rate, marker="o")
    axes[0, 1].set_ylabel("edge detection rate")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, roi_rate, marker="o")
    axes[1, 0].set_ylabel("ROI events/s")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, bg_rate, marker="o")
    axes[1, 1].set_ylabel("background events/s")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Section 3.1 aperture sweep at 1500 Hz")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    default_input_dir = os.path.abspath(
        os.path.join(REPO_ROOT, "..", "captures", "3.1", "aperture_sweep_1500Hz_20260515")
    )
    default_manifest = os.path.join(
        REPO_ROOT,
        "pru1_pwm_CSK_1000Hz",
        "userspace",
        "s31_replication_1500Hz_10s_manifest.csv",
    )
    default_bits = os.path.join(
        REPO_ROOT,
        "pru1_pwm_CSK_1000Hz",
        "userspace",
        "replication_bits_11b.txt",
    )

    parser = argparse.ArgumentParser(description="Analyze one Section 3.1 aperture sweep at 1500 Hz.")
    parser.add_argument("--input_dir", default=default_input_dir)
    parser.add_argument("--manifest_csv", default=default_manifest)
    parser.add_argument("--bits_file", default=default_bits)
    parser.add_argument("--frequency_hz", type=float, default=1500.0)
    parser.add_argument("--roi", nargs=4, type=int, default=[544, 320, 672, 448])
    parser.add_argument("--bin_us", type=float, default=15.0)
    parser.add_argument("--signal", choices=["total", "on", "off", "diff", "absdiff"], default="total")
    parser.add_argument("--decode_mode", choices=["activity", "transition"], default="activity")
    parser.add_argument("--phase_steps", type=int, default=50)
    parser.add_argument("--transition_rate_min_scale", type=float, default=0.9)
    parser.add_argument("--transition_rate_max_scale", type=float, default=1.1)
    parser.add_argument("--transition_rate_steps", type=int, default=41)
    parser.add_argument("--transition_edge_window_fraction", type=float, default=0.4)
    parser.add_argument("--transition_max_rate_drift_fraction", type=float, default=0.10)
    parser.add_argument("--edge_peak_k", type=float, default=6.0)
    parser.add_argument("--edge_min_distance_ms", type=float, default=0.45)
    parser.add_argument("--background_window_s", type=float, default=0.5)
    parser.add_argument("--window_search_bin_us", type=float, default=250.0)
    parser.add_argument("--out_prefix", default="s31_aperture_sweep_1500Hz")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    truth_message_bits = load_truth_bits(args.bits_file, None)
    manifest = load_transmission_manifest(args.manifest_csv)
    manifest_entry = lookup_manifest_entry(manifest, args.frequency_hz)
    if manifest_entry is None:
        raise ValueError(f"No manifest entry for frequency {args.frequency_hz}")

    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    rows: List[ApertureResult] = []
    roi = tuple(args.roi)
    print(f"Using ROI x=[{roi[0]},{roi[2]}) y=[{roi[1]},{roi[3]})")
    print(
        f"Using signal={args.signal}, decode_mode={args.decode_mode}, "
        f"bin_us={args.bin_us}, truth={bits_to_string(truth_message_bits)}"
    )
    for raw_path in raw_files:
        row = analyze_one(
            raw_path=raw_path,
            manifest_entry=manifest_entry,
            truth_message_bits=truth_message_bits,
            roi=roi,
            bin_us=float(args.bin_us),
            signal=args.signal,
            phase_steps=int(args.phase_steps),
            decode_mode=args.decode_mode,
            transition_rate_min_scale=float(args.transition_rate_min_scale),
            transition_rate_max_scale=float(args.transition_rate_max_scale),
            transition_rate_steps=int(args.transition_rate_steps),
            transition_edge_window_fraction=float(args.transition_edge_window_fraction),
            transition_max_rate_drift_fraction=float(args.transition_max_rate_drift_fraction),
            edge_peak_k=float(args.edge_peak_k),
            edge_min_distance_ms=float(args.edge_min_distance_ms),
            background_window_s=float(args.background_window_s),
            window_search_bin_us=float(args.window_search_bin_us),
        )
        rows.append(row)
        print(
            f"Done: {row.raw_file} aperture={row.aperture} "
            f"roi_rate={row.roi_event_rate_eps:.3g}/s bg={row.background_rate_eps:.3g}/s "
            f"edge_rate={row.edge_detection_rate:.3f} jitter={row.timing_jitter_us:.3g}us "
            f"BER={row.ber:.4f}"
        )

    order = {aperture: idx for idx, aperture in enumerate(["C", "f16", "f11", "f8", "f5p6", "f4", "f2p8", "f2"])}
    rows.sort(key=lambda row: order.get(row.aperture, 999))

    data_dir = os.path.join(REPO_ROOT, "data", "3.1", "aperture_sweep")
    plot_dir = os.path.join(REPO_ROOT, "plots", "3.1", "aperture_sweep")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    save_results(rows, out_path)
    print("Saved summary CSV:", out_path)

    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{args.out_prefix}_summary.png")
        save_plot(rows, plot_path)
        print("Saved plot:", plot_path)


if __name__ == "__main__":
    main()
