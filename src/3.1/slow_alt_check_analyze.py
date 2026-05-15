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

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aperture_sweep_analyze import (  # noqa: E402
    background_rate,
    expected_transition_count,
    find_analysis_window_from_counts,
    find_edge_peaks,
    load_binned_roi_events,
    robust_threshold,
    select_signal,
    window_counts,
)
from section3_1_replication_analyze import (  # noqa: E402
    bits_to_string,
    compare_candidates,
    load_transmission_manifest,
    load_truth_bits,
    lookup_manifest_entry,
    relative_frequency_error,
    search_best_decode,
    search_best_transition_decode,
    score_transition_stream,
)


@dataclass
class SlowAltResult:
    frequency_hz: float
    trial: int
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
    decode_mode: str
    decode_rate_hz: float
    decode_rate_error_fraction: float
    ber: float
    mar: float
    bit_errors: int
    bits_scored: int
    messages_scored: int
    correct_messages: int
    phase_s: float
    threshold: float
    on_fraction: float
    matched_edges: int
    false_edges: int
    edge_match_phase_s: float


def parse_name(raw_file: str) -> Tuple[float, int]:
    freq_match = re.search(r"_([0-9]+(?:p[0-9]+|\.[0-9]+)?)Hz_", raw_file)
    trial_match = re.search(r"_t([0-9]+)\.raw$", raw_file)
    if not freq_match:
        raise ValueError(f"Could not parse frequency from filename: {raw_file}")
    frequency_hz = float(freq_match.group(1).replace("p", "."))
    trial = int(trial_match.group(1)) if trial_match else 0
    return frequency_hz, trial


def expected_transition_times(
    truth_message_bits: np.ndarray,
    message_repeats: int,
    guard_bits: int,
    bit_rate_hz: float,
) -> np.ndarray:
    bits = bits_to_string(truth_message_bits)
    if not bits or message_repeats <= 0 or bit_rate_hz <= 0:
        return np.array([], dtype=np.float64)

    bit_stream = ("0" * max(0, int(guard_bits))) + (bits * int(message_repeats)) + ("0" * max(0, int(guard_bits)))
    bit_period_s = 1.0 / float(bit_rate_hz)
    transition_times = [
        i * bit_period_s
        for i, (prev_bit, next_bit) in enumerate(zip(bit_stream, bit_stream[1:]), start=1)
        if prev_bit != next_bit
    ]
    return np.array(transition_times, dtype=np.float64)


def match_expected_transitions(
    expected_times_s: np.ndarray,
    peak_times_s: np.ndarray,
    duration_s: float,
    bit_rate_hz: float,
    phase_steps: int,
    tolerance_fraction: float = 0.45,
) -> Tuple[int, int, float, float, float]:
    if expected_times_s.size == 0:
        return 0, int(peak_times_s.size), float("nan"), float("nan"), 0.0
    if peak_times_s.size == 0 or bit_rate_hz <= 0:
        return 0, 0, 0.0, float("nan"), 0.0

    bit_period_s = 1.0 / float(bit_rate_hz)
    tolerance_s = tolerance_fraction * bit_period_s
    phases = np.linspace(0.0, bit_period_s, max(1, phase_steps), endpoint=False)

    best_matched = 0
    best_false_edges = int(peak_times_s.size)
    best_jitter_us = float("nan")
    best_mean_abs_us = float("inf")
    best_phase_s = 0.0

    for phase_s in phases:
        shifted = expected_times_s + phase_s
        shifted = shifted[(shifted >= 0.0) & (shifted <= duration_s)]
        if shifted.size == 0:
            continue

        peak_idx = 0
        residuals: List[float] = []
        for expected_time_s in shifted:
            while peak_idx < peak_times_s.size and peak_times_s[peak_idx] < expected_time_s - tolerance_s:
                peak_idx += 1
            if peak_idx >= peak_times_s.size:
                break

            candidates = [peak_idx]
            if peak_idx + 1 < peak_times_s.size:
                candidates.append(peak_idx + 1)

            nearest = min(candidates, key=lambda idx: abs(float(peak_times_s[idx] - expected_time_s)))
            residual_s = float(peak_times_s[nearest] - expected_time_s)
            if abs(residual_s) <= tolerance_s:
                residuals.append(residual_s)
                peak_idx = nearest + 1

        matched = len(residuals)
        false_edges = max(0, int(peak_times_s.size) - matched)
        residual_arr = np.array(residuals, dtype=np.float64)
        jitter_us = float(np.std(residual_arr) * 1e6) if residual_arr.size >= 2 else float("nan")
        mean_abs_us = float(np.mean(np.abs(residual_arr)) * 1e6) if residual_arr.size else float("inf")

        better = matched > best_matched
        if matched == best_matched and false_edges < best_false_edges:
            better = True
        if matched == best_matched and false_edges == best_false_edges and mean_abs_us < best_mean_abs_us:
            better = True
        if better:
            best_matched = matched
            best_false_edges = false_edges
            best_jitter_us = jitter_us
            best_mean_abs_us = mean_abs_us
            best_phase_s = float(phase_s)

    detection_rate = float(best_matched / expected_times_s.size)
    return best_matched, best_false_edges, detection_rate, best_jitter_us, best_phase_s


def effective_edge_min_distance_ms(
    explicit_min_distance_ms: float,
    bit_rate_hz: float,
    bit_period_fraction: float,
) -> float:
    if bit_rate_hz <= 0:
        return float(explicit_min_distance_ms)
    bit_period_ms = 1000.0 / float(bit_rate_hz)
    return float(max(explicit_min_distance_ms, bit_period_fraction * bit_period_ms))


def boundary_counts_from_peaks(
    peak_times_s: np.ndarray,
    duration_s: float,
    symbol_rate_hz: float,
    start_time_s: float,
    tolerance_fraction: float,
) -> np.ndarray:
    if symbol_rate_hz <= 0 or duration_s <= start_time_s:
        return np.array([], dtype=np.float64)

    symbol_period_s = 1.0 / float(symbol_rate_hz)
    n_boundaries = int(np.floor((duration_s - start_time_s) / symbol_period_s))
    if n_boundaries <= 0:
        return np.array([], dtype=np.float64)

    tolerance_s = tolerance_fraction * symbol_period_s
    boundaries = start_time_s + np.arange(n_boundaries, dtype=np.float64) * symbol_period_s
    boundary_counts = np.zeros(n_boundaries, dtype=np.float64)

    peak_idx = 0
    for i, boundary_s in enumerate(boundaries):
        while peak_idx < peak_times_s.size and peak_times_s[peak_idx] < boundary_s - tolerance_s:
            peak_idx += 1
        if peak_idx >= peak_times_s.size:
            break
        if abs(float(peak_times_s[peak_idx] - boundary_s)) <= tolerance_s:
            boundary_counts[i] = 1.0
            peak_idx += 1

    return boundary_counts


def search_peak_transition_decode(
    peak_times_s: np.ndarray,
    duration_s: float,
    nominal_rate_hz: float,
    truth_message_bits: np.ndarray,
    phase_steps: int,
    rate_min_scale: float,
    rate_max_scale: float,
    rate_steps: int,
    edge_window_fraction: float,
    expected_message_count: int,
    max_rate_drift_fraction: float,
):
    if phase_steps <= 0:
        raise ValueError("phase_steps must be > 0")
    if nominal_rate_hz <= 0:
        raise ValueError("nominal_rate_hz must be > 0")
    if rate_steps <= 0:
        raise ValueError("rate_steps must be > 0")

    candidate_rates = np.linspace(
        nominal_rate_hz * rate_min_scale,
        nominal_rate_hz * rate_max_scale,
        rate_steps,
        dtype=np.float64,
    )
    if np.isfinite(max_rate_drift_fraction) and max_rate_drift_fraction > 0:
        min_rate_hz = nominal_rate_hz * max(0.0, 1.0 - max_rate_drift_fraction)
        max_rate_hz = nominal_rate_hz * (1.0 + max_rate_drift_fraction)
        candidate_rates = candidate_rates[(candidate_rates >= min_rate_hz) & (candidate_rates <= max_rate_hz)]
        if candidate_rates.size == 0:
            candidate_rates = np.array([float(nominal_rate_hz)], dtype=np.float64)

    best = None
    for decode_rate_hz in candidate_rates:
        symbol_period_s = 1.0 / float(decode_rate_hz)
        phases = np.linspace(0.0, symbol_period_s, phase_steps, endpoint=False)
        for phase_s in phases:
            boundary_counts = boundary_counts_from_peaks(
                peak_times_s=peak_times_s,
                duration_s=duration_s,
                symbol_rate_hz=float(decode_rate_hz),
                start_time_s=float(phase_s),
                tolerance_fraction=edge_window_fraction,
            )
            candidate = score_transition_stream(
                boundary_counts=boundary_counts,
                truth_message_bits=truth_message_bits,
                phase_s=float(phase_s),
                decode_rate_hz=float(decode_rate_hz),
                target_rate_hz=float(nominal_rate_hz),
                expected_message_count=expected_message_count,
            )
            if candidate is not None:
                candidate.rate_error_fraction = relative_frequency_error(float(decode_rate_hz), float(nominal_rate_hz))
            if candidate is not None and compare_candidates(best, candidate):
                best = candidate
    return best


def list_raw_files(input_dir: str) -> List[str]:
    return sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".raw")
    )


def analyze_one(
    raw_path: str,
    manifest: Dict[float, object],
    truth_message_bits: np.ndarray,
    roi: Tuple[int, int, int, int],
    bin_us: float,
    signal: str,
    decode_mode: str,
    phase_steps: int,
    edge_peak_k: float,
    edge_min_distance_ms: float,
    edge_min_distance_fraction: float,
    background_window_s: float,
    window_search_bin_us: float,
    transition_rate_min_scale: float,
    transition_rate_max_scale: float,
    transition_rate_steps: int,
    transition_edge_window_fraction: float,
    transition_max_rate_drift_fraction: float,
) -> SlowAltResult:
    raw_file = os.path.basename(raw_path)
    frequency_hz, trial = parse_name(raw_file)
    manifest_entry = lookup_manifest_entry(manifest, frequency_hz)
    if manifest_entry is None:
        raise ValueError(f"No manifest entry found for {frequency_hz:g} Hz")

    bias_file = os.path.splitext(raw_file)[0] + ".bias"
    bias_path = os.path.join(os.path.dirname(raw_path), bias_file)
    bias_file_label = bias_file if os.path.exists(bias_path) else ""

    bins = load_binned_roi_events(raw_path=raw_path, roi=roi, bin_us=bin_us)
    counts = select_signal(bins, signal)
    bin_s = float(bin_us) * 1e-6

    analysis_start_s, analysis_end_s = find_analysis_window_from_counts(
        t_bins_s=bins.t_bins_s,
        counts=counts,
        manifest_entry=manifest_entry,
        message_len_bits=int(truth_message_bits.size),
        search_bin_s=window_search_bin_us * 1e-6,
    )
    analysis_duration_s = max(0.0, analysis_end_s - analysis_start_s)

    active_t, active_counts = window_counts(
        t_bins_s=bins.t_bins_s,
        counts=counts,
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
    edge_threshold = robust_threshold(active_counts, edge_peak_k)
    actual_edge_min_distance_ms = effective_edge_min_distance_ms(
        explicit_min_distance_ms=edge_min_distance_ms,
        bit_rate_hz=float(manifest_entry.actual_frequency_hz),
        bit_period_fraction=edge_min_distance_fraction,
    )
    peak_times_s = find_edge_peaks(
        t_bins_s=active_t,
        counts=active_counts,
        min_height=edge_threshold,
        min_distance_s=actual_edge_min_distance_ms * 1e-3,
    )
    detected_edges = int(peak_times_s.size)
    expected_times_s = expected_transition_times(
        truth_message_bits=truth_message_bits,
        message_repeats=int(manifest_entry.message_repeats),
        guard_bits=int(manifest_entry.guard_bits),
        bit_rate_hz=float(manifest_entry.actual_frequency_hz),
    )
    matched_edges, false_edges, edge_detection_rate, timing_jitter_us, edge_match_phase_s = match_expected_transitions(
        expected_times_s=expected_times_s,
        peak_times_s=peak_times_s,
        duration_s=analysis_duration_s,
        bit_rate_hz=float(manifest_entry.actual_frequency_hz),
        phase_steps=phase_steps,
    )

    best = None
    if active_counts.size > 0 and analysis_duration_s > 0:
        if decode_mode == "peak_transition":
            best = search_peak_transition_decode(
                peak_times_s=peak_times_s,
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
        elif decode_mode == "transition":
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
        decode_rate_hz = float("nan")
        decode_rate_error_fraction = float("nan")
        ber = float("nan")
        mar = float("nan")
        bit_errors = 0
        bits_scored = 0
        messages_scored = 0
        correct_messages = 0
        phase_s = float("nan")
        threshold = float("nan")
    else:
        decode_rate_hz = float(best.decode_rate_hz)
        decode_rate_error_fraction = float(best.rate_error_fraction)
        ber = float(best.ber)
        mar = float(best.mar)
        bit_errors = int(best.n_bit_errors)
        bits_scored = int(best.n_symbols_scored)
        messages_scored = int(best.n_messages)
        correct_messages = int(best.n_correct_messages)
        phase_s = float(best.phase_s)
        threshold = float(best.threshold)

    return SlowAltResult(
        frequency_hz=float(frequency_hz),
        trial=int(trial),
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
        timing_jitter_us=float(timing_jitter_us),
        decode_mode=decode_mode,
        decode_rate_hz=decode_rate_hz,
        decode_rate_error_fraction=decode_rate_error_fraction,
        ber=ber,
        mar=mar,
        bit_errors=bit_errors,
        bits_scored=bits_scored,
        messages_scored=messages_scored,
        correct_messages=correct_messages,
        phase_s=phase_s,
        threshold=threshold,
        on_fraction=float(bins.on_fraction),
        matched_edges=int(matched_edges),
        false_edges=int(false_edges),
        edge_match_phase_s=float(edge_match_phase_s),
    )


def save_results(rows: List[SlowAltResult], out_path: str) -> None:
    fieldnames = list(SlowAltResult.__dataclass_fields__.keys())
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def save_by_frequency(rows: List[SlowAltResult], out_path: str) -> None:
    fieldnames = [
        "frequency_hz",
        "n_trials",
        "ber_mean",
        "ber_std",
        "mar_mean",
        "edge_detection_rate_mean",
        "timing_jitter_us_mean",
        "matched_edges_mean",
        "false_edges_mean",
        "roi_event_rate_eps_mean",
        "background_rate_eps_mean",
    ]
    grouped: Dict[float, List[SlowAltResult]] = {}
    for row in rows:
        grouped.setdefault(row.frequency_hz, []).append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frequency_hz in sorted(grouped):
            group = grouped[frequency_hz]
            writer.writerow({
                "frequency_hz": frequency_hz,
                "n_trials": len(group),
                "ber_mean": float(np.nanmean([row.ber for row in group])),
                "ber_std": float(np.nanstd([row.ber for row in group])),
                "mar_mean": float(np.nanmean([row.mar for row in group])),
                "edge_detection_rate_mean": float(np.nanmean([row.edge_detection_rate for row in group])),
                "timing_jitter_us_mean": float(np.nanmean([row.timing_jitter_us for row in group])),
                "matched_edges_mean": float(np.nanmean([row.matched_edges for row in group])),
                "false_edges_mean": float(np.nanmean([row.false_edges for row in group])),
                "roi_event_rate_eps_mean": float(np.nanmean([row.roi_event_rate_eps for row in group])),
                "background_rate_eps_mean": float(np.nanmean([row.background_rate_eps for row in group])),
            })


def save_plot(rows: List[SlowAltResult], out_path: str) -> None:
    labels = [f"{row.frequency_hz:g}Hz t{row.trial}" for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    ber = np.array([row.ber for row in rows], dtype=np.float64)
    edge_rate = np.array([row.edge_detection_rate for row in rows], dtype=np.float64)
    jitter = np.array([row.timing_jitter_us for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(x, ber, marker="o")
    axes[0].set_ylabel("BER")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, edge_rate, marker="o")
    axes[1].set_ylabel("edge detection")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(x, jitter, marker="o")
    axes[2].set_ylabel("jitter (us)")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Slow alternating transmitter check")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def values_at_times(t_bins_s: np.ndarray, counts: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    if t_bins_s.size == 0 or counts.size == 0 or times_s.size == 0:
        return np.array([], dtype=np.float64)
    idx = np.searchsorted(t_bins_s, times_s, side="left")
    idx = np.clip(idx, 0, counts.size - 1)
    return counts[idx].astype(np.float64)


def plot_transition_diagnostic(
    raw_path: str,
    row: SlowAltResult,
    manifest: Dict[float, object],
    truth_message_bits: np.ndarray,
    roi: Tuple[int, int, int, int],
    bin_us: float,
    signal: str,
    edge_peak_k: float,
    edge_min_distance_ms: float,
    edge_min_distance_fraction: float,
    out_dir: str,
    zoom_s: float,
) -> List[str]:
    manifest_entry = lookup_manifest_entry(manifest, row.frequency_hz)
    if manifest_entry is None:
        return []

    bins = load_binned_roi_events(raw_path=raw_path, roi=roi, bin_us=bin_us)
    counts = select_signal(bins, signal)
    active_t, active_counts = window_counts(
        t_bins_s=bins.t_bins_s,
        counts=counts,
        start_s=row.analysis_start_s,
        end_s=row.analysis_end_s,
    )
    if active_t.size == 0:
        return []

    edge_threshold = robust_threshold(active_counts, edge_peak_k)
    actual_edge_min_distance_ms = effective_edge_min_distance_ms(
        explicit_min_distance_ms=edge_min_distance_ms,
        bit_rate_hz=float(manifest_entry.actual_frequency_hz),
        bit_period_fraction=edge_min_distance_fraction,
    )
    peak_times_s = find_edge_peaks(
        t_bins_s=active_t,
        counts=active_counts,
        min_height=edge_threshold,
        min_distance_s=actual_edge_min_distance_ms * 1e-3,
    )
    peak_values = values_at_times(active_t, active_counts, peak_times_s)

    expected_times_s = expected_transition_times(
        truth_message_bits=truth_message_bits,
        message_repeats=int(manifest_entry.message_repeats),
        guard_bits=int(manifest_entry.guard_bits),
        bit_rate_hz=float(manifest_entry.actual_frequency_hz),
    )
    expected_times_s = expected_times_s + row.edge_match_phase_s
    expected_times_s = expected_times_s[(expected_times_s >= 0.0) & (expected_times_s <= row.analysis_duration_s)]

    base = os.path.splitext(row.raw_file)[0]
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    def draw(path: str, x_limit: Optional[Tuple[float, float]], title_suffix: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        ax.plot(active_t, active_counts, color="0.25", linewidth=0.5, label=f"ROI {signal} counts")
        for i, t_s in enumerate(expected_times_s):
            if x_limit is not None and (t_s < x_limit[0] or t_s > x_limit[1]):
                continue
            ax.axvline(
                t_s,
                color="#2ca02c",
                linewidth=0.7,
                alpha=0.45,
                label="expected transition" if i == 0 else None,
            )
        if peak_times_s.size:
            mask = np.ones(peak_times_s.size, dtype=bool)
            if x_limit is not None:
                mask = (peak_times_s >= x_limit[0]) & (peak_times_s <= x_limit[1])
            ax.scatter(
                peak_times_s[mask],
                peak_values[mask],
                s=13,
                color="#d62728",
                label="detected peak",
                zorder=3,
            )
        ax.axhline(edge_threshold, color="#1f77b4", linewidth=0.8, linestyle="--", label="peak threshold")
        ax.set_title(
            f"{row.raw_file} {title_suffix}: expected vs detected transitions\n"
            f"matched={row.matched_edges}/{row.expected_edges}, false={row.false_edges}, "
            f"BER={row.ber:.3f}"
        )
        ax.set_xlabel("seconds from selected transmit window start")
        ax.set_ylabel(f"{signal} counts per {bin_us:g} us bin")
        if x_limit is not None:
            ax.set_xlim(*x_limit)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        fig.savefig(path, dpi=300)
        plt.close(fig)

    overview_path = os.path.join(out_dir, f"{base}_transition_diagnostic_overview.png")
    draw(overview_path, None, "overview")
    paths.append(overview_path)

    zoom_end_s = min(max(float(zoom_s), 0.05), row.analysis_duration_s)
    zoom_path = os.path.join(out_dir, f"{base}_transition_diagnostic_zoom.png")
    draw(zoom_path, (0.0, zoom_end_s), f"first {zoom_end_s:g}s")
    paths.append(zoom_path)

    return paths


def main() -> None:
    default_input_dir = os.path.abspath(
        os.path.join(REPO_ROOT, "..", "captures", "3.1", "transmitter_slow_alt_check_20260515")
    )
    default_manifest = os.path.join(
        REPO_ROOT,
        "pru1_pwm_CSK_1000Hz",
        "userspace",
        "s31_slow_alt_check_manifest.csv",
    )
    default_bits = os.path.join(
        REPO_ROOT,
        "pru1_pwm_CSK_1000Hz",
        "userspace",
        "slow_alt_bits_11b.txt",
    )

    parser = argparse.ArgumentParser(description="Analyze slow alternating transmitter sanity captures.")
    parser.add_argument("--input_dir", default=default_input_dir)
    parser.add_argument("--manifest_csv", default=default_manifest)
    parser.add_argument("--bits_file", default=default_bits)
    parser.add_argument("--roi", nargs=4, type=int, default=[544, 320, 672, 448])
    parser.add_argument("--bin_us", type=float, default=25.0)
    parser.add_argument("--signal", choices=["total", "on", "off", "diff", "absdiff"], default="total")
    parser.add_argument("--decode_mode", choices=["activity", "transition", "peak_transition"], default="peak_transition")
    parser.add_argument("--phase_steps", type=int, default=80)
    parser.add_argument("--edge_peak_k", type=float, default=6.0)
    parser.add_argument("--edge_min_distance_ms", type=float, default=0.5)
    parser.add_argument(
        "--edge_min_distance_fraction",
        type=float,
        default=0.55,
        help="Minimum peak spacing as a fraction of the bit period; suppresses ringing/decay double-counts.",
    )
    parser.add_argument("--background_window_s", type=float, default=0.5)
    parser.add_argument("--window_search_bin_us", type=float, default=250.0)
    parser.add_argument("--transition_rate_min_scale", type=float, default=0.9)
    parser.add_argument("--transition_rate_max_scale", type=float, default=1.1)
    parser.add_argument("--transition_rate_steps", type=int, default=41)
    parser.add_argument("--transition_edge_window_fraction", type=float, default=0.4)
    parser.add_argument("--transition_max_rate_drift_fraction", type=float, default=0.10)
    parser.add_argument("--diagnostic_frequency_hz", type=float, default=100.0)
    parser.add_argument("--diagnostic_zoom_s", type=float, default=0.5)
    parser.add_argument("--no_diagnostic_plots", action="store_true")
    parser.add_argument("--out_prefix", default="s31_slow_alt_check")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    truth_message_bits = load_truth_bits(args.bits_file, None)
    manifest = load_transmission_manifest(args.manifest_csv)
    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    roi = tuple(args.roi)
    print(f"Using ROI x=[{roi[0]},{roi[2]}) y=[{roi[1]},{roi[3]})")
    print(
        f"Using signal={args.signal}, decode_mode={args.decode_mode}, "
        f"bin_us={args.bin_us}, truth={bits_to_string(truth_message_bits)}"
    )

    raw_paths_by_name: Dict[str, str] = {}
    rows = []
    for raw_path in raw_files:
        row = analyze_one(
            raw_path=raw_path,
            manifest=manifest,
            truth_message_bits=truth_message_bits,
            roi=roi,
            bin_us=float(args.bin_us),
            signal=args.signal,
            decode_mode=args.decode_mode,
            phase_steps=int(args.phase_steps),
            edge_peak_k=float(args.edge_peak_k),
            edge_min_distance_ms=float(args.edge_min_distance_ms),
            edge_min_distance_fraction=float(args.edge_min_distance_fraction),
            background_window_s=float(args.background_window_s),
            window_search_bin_us=float(args.window_search_bin_us),
            transition_rate_min_scale=float(args.transition_rate_min_scale),
            transition_rate_max_scale=float(args.transition_rate_max_scale),
            transition_rate_steps=int(args.transition_rate_steps),
            transition_edge_window_fraction=float(args.transition_edge_window_fraction),
            transition_max_rate_drift_fraction=float(args.transition_max_rate_drift_fraction),
        )
        rows.append(row)
        raw_paths_by_name[row.raw_file] = raw_path
    rows.sort(key=lambda row: (row.frequency_hz, row.trial))

    for row in rows:
        print(
            f"Done: {row.raw_file} BER={row.ber:.4f} MAR={row.mar:.3f} "
            f"edge_rate={row.edge_detection_rate:.3f} matched={row.matched_edges}/{row.expected_edges} "
            f"false={row.false_edges} jitter={row.timing_jitter_us:.3g}us"
        )

    data_dir = os.path.join(REPO_ROOT, "data", "3.1", "slow_alt_check")
    plot_dir = os.path.join(REPO_ROOT, "plots", "3.1", "slow_alt_check")
    os.makedirs(data_dir, exist_ok=True)
    summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    by_frequency_path = os.path.join(data_dir, f"{args.out_prefix}_by_frequency_summary.csv")
    save_results(rows, summary_path)
    save_by_frequency(rows, by_frequency_path)
    print("Saved summary CSV:", summary_path)
    print("Saved by-frequency CSV:", by_frequency_path)

    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{args.out_prefix}_summary.png")
        save_plot(rows, plot_path)
        print("Saved plot:", plot_path)

        if not args.no_diagnostic_plots:
            diagnostic_dir = os.path.join(plot_dir, "diagnostics")
            diagnostic_rows = [
                row for row in rows if abs(row.frequency_hz - float(args.diagnostic_frequency_hz)) < 1e-6
            ]
            for row in diagnostic_rows:
                raw_path = raw_paths_by_name.get(row.raw_file)
                if not raw_path:
                    continue
                for path in plot_transition_diagnostic(
                    raw_path=raw_path,
                    row=row,
                    manifest=manifest,
                    truth_message_bits=truth_message_bits,
                    roi=roi,
                    bin_us=float(args.bin_us),
                    signal=args.signal,
                    edge_peak_k=float(args.edge_peak_k),
                    edge_min_distance_ms=float(args.edge_min_distance_ms),
                    edge_min_distance_fraction=float(args.edge_min_distance_fraction),
                    out_dir=diagnostic_dir,
                    zoom_s=float(args.diagnostic_zoom_s),
                ):
                    print("Saved diagnostic plot:", path)


if __name__ == "__main__":
    main()
