import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from io_utils import repo_root_from_this_file


DEFAULT_MAIN_C = os.path.join("pru1_pwm_CFK_continuous_1000Hz", "main.c")
DEFAULT_OUT_PREFIX = "s32_evk4_packet_match"
DEFAULT_SYMBOL_RATE_HZ = 2000.0
DEFAULT_ZERO_SYMBOL = 0
DEFAULT_ONE_SYMBOL = 10


@dataclass
class RawInfo:
    path: str
    first_t_us: int
    last_t_us: int
    width: int
    height: int
    events: int

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.last_t_us - self.first_t_us) * 1e-6)


@dataclass
class RoiBox:
    x0: int
    y0: int
    x1: int
    y1: int
    peak_block_x: int = -1
    peak_block_y: int = -1
    peak_score: float = 0.0


@dataclass
class ActivityWindow:
    start_s: float
    end_s: float
    threshold: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class MatchCandidate:
    score: float
    signed_corr: float
    edge_match_rate: float
    signed_edge_accuracy: float
    symbol_match_rate: float
    phase_s: float
    polarity: int
    edge_threshold: float
    symbols_scored: int
    symbol_errors: int
    edges_scored: int
    edge_errors: int
    signed_edges_scored: int
    signed_edge_errors: int
    boundary_times_s: np.ndarray
    packet_positions: np.ndarray
    expected_signed_edges: np.ndarray
    signed_counts: np.ndarray
    total_counts: np.ndarray
    edge_present: np.ndarray
    decoded_bits: np.ndarray
    expected_bits: np.ndarray


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def parse_symbols_from_main_c(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as handle:
        text = strip_c_comments(handle.read())
    match = re.search(
        r"const\s+uint8_t\s+symbols_to_send\s*\[\s*\]\s*=\s*(?P<body>.*?);",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"Could not find const uint8_t symbols_to_send[] in {path}")
    symbols = [int(value) for value in re.findall(r"\b\d+\b", match.group("body"))]
    if not symbols:
        raise RuntimeError(f"Found symbols_to_send[] in {path}, but it contained no numeric symbols.")
    return symbols


def symbols_to_bits(
    symbols: Sequence[int],
    zero_symbol: int,
    one_symbol: int,
    nonzero_is_one: bool,
) -> List[int]:
    bits: List[int] = []
    unexpected: List[int] = []
    for symbol in symbols:
        if symbol == zero_symbol:
            bits.append(0)
        elif symbol == one_symbol:
            bits.append(1)
        elif nonzero_is_one and symbol != zero_symbol:
            bits.append(1)
        else:
            unexpected.append(symbol)
    if unexpected:
        unique = ", ".join(str(value) for value in sorted(set(unexpected)))
        raise RuntimeError(
            f"symbols_to_send[] contains symbols that are not mapped to bits: {unique}. "
            "Pass --nonzero_is_one, or adjust --zero_symbol/--one_symbol."
        )
    return bits


def parse_bits_argument(bits: str) -> List[int]:
    cleaned = re.sub(r"[^01]", "", bits)
    if not cleaned:
        raise ValueError("--bits must contain at least one 0 or 1.")
    return [1 if char == "1" else 0 for char in cleaned]


def scan_raw_info(raw_path: str) -> RawInfo:
    first_t_us: Optional[int] = None
    last_t_us: Optional[int] = None
    max_x = 0
    max_y = 0
    events = 0
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        if first_t_us is None:
            first_t_us = int(t[0])
        last_t_us = int(t[-1])
        max_x = max(max_x, int(evs["x"].max()))
        max_y = max(max_y, int(evs["y"].max()))
        events += int(evs.size)
    if first_t_us is None or last_t_us is None:
        raise RuntimeError(f"No events found in {raw_path}")
    return RawInfo(
        path=raw_path,
        first_t_us=first_t_us,
        last_t_us=last_t_us,
        width=max_x + 1,
        height=max_y + 1,
        events=events,
    )


def accumulate_time_hist(
    raw_path: str,
    start_us: int,
    end_us: int,
    bin_width_us: int,
    roi: Optional[RoiBox] = None,
) -> np.ndarray:
    n_bins = max(1, int(np.ceil((end_us - start_us) / float(bin_width_us))))
    counts = np.zeros(n_bins, dtype=np.int64)
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        mask = (t >= start_us) & (t < end_us)
        if roi is not None:
            mask &= (
                (evs["x"] >= roi.x0)
                & (evs["x"] < roi.x1)
                & (evs["y"] >= roi.y0)
                & (evs["y"] < roi.y1)
            )
        if not np.any(mask):
            continue
        idx = ((t[mask] - start_us) // bin_width_us).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        if idx.size:
            counts += np.bincount(idx, minlength=n_bins).astype(np.int64)
    return counts


def detect_activity_window(
    raw_path: str,
    info: RawInfo,
    bin_width_ms: float,
    threshold_sigma: float,
) -> ActivityWindow:
    bin_width_us = max(1, int(round(bin_width_ms * 1000.0)))
    counts = accumulate_time_hist(raw_path, info.first_t_us, info.last_t_us + 1, bin_width_us)
    n_baseline = max(3, len(counts) // 10)
    baseline = counts[:n_baseline].astype(np.float64)
    threshold = float(np.median(baseline) + threshold_sigma * np.std(baseline))
    active = counts > threshold
    if not np.any(active):
        return ActivityWindow(start_s=0.0, end_s=info.duration_s, threshold=threshold)

    idx = np.where(active)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    best = max(groups, key=lambda group: (int(np.sum(counts[group])), int(group.size)))
    start_s = float(best[0] * bin_width_us * 1e-6)
    end_s = float(min((best[-1] + 1) * bin_width_us * 1e-6, info.duration_s))
    if end_s <= start_s:
        return ActivityWindow(start_s=0.0, end_s=info.duration_s, threshold=threshold)
    return ActivityWindow(start_s=start_s, end_s=end_s, threshold=threshold)


def choose_baseline_window(activity: ActivityWindow, capture_duration_s: float) -> Tuple[float, float]:
    duration = activity.duration_s
    if duration <= 0:
        return 0.0, min(0.1, capture_duration_s)
    if activity.start_s >= duration:
        return activity.start_s - duration, activity.start_s
    if activity.end_s + duration <= capture_duration_s:
        return activity.end_s, activity.end_s + duration
    return 0.0, 0.0


def accumulate_block_counts(
    raw_path: str,
    start_us: int,
    end_us: int,
    block_px: int,
    shape: Tuple[int, int],
) -> np.ndarray:
    counts = np.zeros(shape, dtype=np.int64)
    if end_us <= start_us:
        return counts
    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        mask = (t >= start_us) & (t < end_us)
        if not np.any(mask):
            continue
        xb = (evs["x"][mask].astype(np.int64) // block_px)
        yb = (evs["y"][mask].astype(np.int64) // block_px)
        valid = (xb >= 0) & (xb < shape[0]) & (yb >= 0) & (yb < shape[1])
        if np.any(valid):
            np.add.at(counts, (xb[valid], yb[valid]), 1)
    return counts


def propose_roi(
    active_counts: np.ndarray,
    baseline_counts: np.ndarray,
    block_px: int,
    roi_blocks: int,
    width: int,
    height: int,
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
        x0=max(0, min(width, int(bx0 * block_px))),
        y0=max(0, min(height, int(by0 * block_px))),
        x1=max(0, min(width, int(bx1 * block_px))),
        y1=max(0, min(height, int(by1 * block_px))),
        peak_block_x=int(peak_x),
        peak_block_y=int(peak_y),
        peak_score=float(score[peak_x, peak_y]),
    )


def detect_roi(raw_path: str, info: RawInfo, activity: ActivityWindow, args: argparse.Namespace) -> RoiBox:
    if args.roi is not None:
        x0, y0, x1, y1 = (int(value) for value in args.roi)
        return RoiBox(x0=x0, y0=y0, x1=x1, y1=y1)

    block_px = int(args.block_px)
    shape = (
        max(1, int(math.ceil(info.width / float(block_px)))),
        max(1, int(math.ceil(info.height / float(block_px)))),
    )
    active_start_us = info.first_t_us + int(round(activity.start_s * 1e6))
    active_end_us = info.first_t_us + int(round(activity.end_s * 1e6))
    baseline_start_s, baseline_end_s = choose_baseline_window(activity, info.duration_s)
    baseline_start_us = info.first_t_us + int(round(baseline_start_s * 1e6))
    baseline_end_us = info.first_t_us + int(round(baseline_end_s * 1e6))

    active_counts = accumulate_block_counts(raw_path, active_start_us, active_end_us, block_px, shape)
    baseline_counts = accumulate_block_counts(raw_path, baseline_start_us, baseline_end_us, block_px, shape)
    return propose_roi(
        active_counts,
        baseline_counts,
        block_px=block_px,
        roi_blocks=int(args.roi_blocks),
        width=info.width,
        height=info.height,
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
        p_chunks.append(evs["p"][mask].astype(np.int8))
    if not t_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int8)
    t_all = np.concatenate(t_chunks)
    p_all = np.concatenate(p_chunks)
    order = np.argsort(t_all)
    return t_all[order], p_all[order]


def zscore(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    arr = arr - float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-9:
        return arr
    return arr / std


def expected_signed_edges(bits: Sequence[int]) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.int8)
    prev = np.roll(arr, 1)
    return (arr - prev).astype(np.int8)


def make_histograms(
    t_us: np.ndarray,
    p: np.ndarray,
    start_us: int,
    duration_s: float,
    bin_us: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = max(1, int(math.ceil(duration_s * 1e6 / float(bin_us))))
    local_bin = ((t_us - start_us) // bin_us).astype(np.int64)
    valid = (local_bin >= 0) & (local_bin < n_bins)
    local_bin = local_bin[valid]
    p = p[valid]
    total = np.bincount(local_bin, minlength=n_bins).astype(np.float64)
    signed_weights = np.where(p > 0, 1.0, -1.0)
    signed = np.bincount(local_bin, weights=signed_weights, minlength=n_bins).astype(np.float64)
    return total, signed


def window_sums(cumulative: np.ndarray, center_bins: np.ndarray, window_bins: int) -> np.ndarray:
    left = np.clip(center_bins - window_bins, 0, cumulative.size - 1)
    right = np.clip(center_bins + window_bins + 1, 0, cumulative.size - 1)
    return cumulative[right] - cumulative[left]


def edge_threshold_from_expected(
    signed_counts: np.ndarray,
    expected_edges: np.ndarray,
) -> float:
    magnitudes = np.abs(signed_counts)
    expected_on = expected_edges != 0
    if np.any(expected_on) and np.any(~expected_on):
        edge_med = float(np.median(magnitudes[expected_on]))
        no_edge_med = float(np.median(magnitudes[~expected_on]))
        return max(0.0, 0.5 * (edge_med + no_edge_med))
    return float(np.quantile(magnitudes, 0.6)) if magnitudes.size else 0.0


def decode_bits_from_edges(
    signed_counts: np.ndarray,
    edge_present: np.ndarray,
    packet_positions: np.ndarray,
    bits: Sequence[int],
    polarity: int,
) -> Tuple[np.ndarray, float, int]:
    expected = np.asarray([bits[int(position)] for position in packet_positions], dtype=np.uint8)
    best_decoded: Optional[np.ndarray] = None
    best_match = -1.0
    best_errors = expected.size
    signed = signed_counts * float(polarity)
    for initial in (0, 1):
        prev = int(initial)
        decoded: List[int] = []
        for value, is_edge in zip(signed, edge_present):
            if is_edge:
                bit = 1 if value > 0 else 0
            else:
                bit = prev
            decoded.append(bit)
            prev = bit
        decoded_arr = np.asarray(decoded, dtype=np.uint8)
        good = int(np.sum(decoded_arr == expected))
        match = good / expected.size if expected.size else 0.0
        if match > best_match:
            best_match = match
            best_decoded = decoded_arr
            best_errors = int(expected.size - good)
    if best_decoded is None:
        best_decoded = np.array([], dtype=np.uint8)
    return best_decoded, float(best_match), int(best_errors)


def score_phase(
    total_cum: np.ndarray,
    signed_cum: np.ndarray,
    duration_s: float,
    bin_s: float,
    phase_s: float,
    symbol_period_s: float,
    edge_window_bins: int,
    packet_bits: Sequence[int],
    expected_edges_packet: np.ndarray,
) -> Optional[MatchCandidate]:
    if phase_s >= duration_s:
        return None
    n_symbols = int(math.floor((duration_s - phase_s) / symbol_period_s))
    if n_symbols < len(packet_bits):
        return None

    boundary_times = phase_s + np.arange(n_symbols, dtype=np.float64) * symbol_period_s
    center_bins = np.rint(boundary_times / bin_s).astype(np.int64)
    valid = (center_bins >= 0) & (center_bins < total_cum.size - 1)
    if int(np.sum(valid)) < len(packet_bits):
        return None
    center_bins = center_bins[valid]
    boundary_times = boundary_times[valid]
    symbol_numbers = np.arange(n_symbols, dtype=np.int64)[valid]
    packet_positions = symbol_numbers % len(packet_bits)
    expected_edges = expected_edges_packet[packet_positions].astype(np.float64)
    if float(np.std(expected_edges)) <= 1e-9:
        return None

    total_counts = window_sums(total_cum, center_bins, edge_window_bins)
    signed_counts = window_sums(signed_cum, center_bins, edge_window_bins)
    signed_corr_raw = float(np.mean(zscore(signed_counts) * zscore(expected_edges)))
    polarity = 1 if signed_corr_raw >= 0 else -1
    signed_corr = abs(signed_corr_raw)

    threshold = edge_threshold_from_expected(signed_counts, expected_edges)
    edge_present = np.abs(signed_counts) >= threshold
    expected_present = expected_edges != 0
    edge_good = int(np.sum(edge_present == expected_present))
    edge_match_rate = edge_good / expected_present.size if expected_present.size else 0.0
    edge_errors = int(expected_present.size - edge_good)

    signed_mask = expected_present & edge_present
    if np.any(signed_mask):
        expected_sign = np.sign(expected_edges[signed_mask])
        observed_sign = np.sign(signed_counts[signed_mask] * float(polarity))
        signed_good = int(np.sum(observed_sign == expected_sign))
        signed_accuracy = signed_good / int(np.sum(signed_mask))
        signed_errors = int(np.sum(signed_mask) - signed_good)
    else:
        signed_accuracy = 0.0
        signed_errors = 0

    decoded_bits, symbol_match_rate, symbol_errors = decode_bits_from_edges(
        signed_counts=signed_counts,
        edge_present=edge_present,
        packet_positions=packet_positions,
        bits=packet_bits,
        polarity=polarity,
    )
    expected_bits_arr = np.asarray([packet_bits[int(position)] for position in packet_positions], dtype=np.uint8)
    score = signed_corr + 0.15 * edge_match_rate + 0.10 * signed_accuracy + 0.10 * symbol_match_rate
    return MatchCandidate(
        score=float(score),
        signed_corr=float(signed_corr),
        edge_match_rate=float(edge_match_rate),
        signed_edge_accuracy=float(signed_accuracy),
        symbol_match_rate=float(symbol_match_rate),
        phase_s=float(phase_s),
        polarity=int(polarity),
        edge_threshold=float(threshold),
        symbols_scored=int(expected_bits_arr.size),
        symbol_errors=int(symbol_errors),
        edges_scored=int(expected_present.size),
        edge_errors=int(edge_errors),
        signed_edges_scored=int(np.sum(signed_mask)),
        signed_edge_errors=int(signed_errors),
        boundary_times_s=boundary_times,
        packet_positions=packet_positions.astype(np.int64),
        expected_signed_edges=expected_edges.astype(np.int8),
        signed_counts=signed_counts.astype(np.float64),
        total_counts=total_counts.astype(np.float64),
        edge_present=edge_present.astype(bool),
        decoded_bits=decoded_bits.astype(np.uint8),
        expected_bits=expected_bits_arr,
    )


def float_range(start: float, stop: float, step: float) -> Iterable[float]:
    count = int(math.floor((stop - start) / step + 0.5))
    for index in range(count + 1):
        yield start + index * step


def find_best_match(
    total_counts: np.ndarray,
    signed_counts: np.ndarray,
    duration_s: float,
    bits: Sequence[int],
    symbol_rate_hz: float,
    bin_us: int,
    edge_window_us: float,
    phase_step_us: float,
) -> MatchCandidate:
    symbol_period_s = 1.0 / float(symbol_rate_hz)
    packet_duration_s = len(bits) * symbol_period_s
    bin_s = bin_us * 1e-6
    edge_window_bins = max(1, int(round(edge_window_us / float(bin_us))))
    total_cum = np.concatenate(([0.0], np.cumsum(total_counts.astype(np.float64))))
    signed_cum = np.concatenate(([0.0], np.cumsum(signed_counts.astype(np.float64))))
    expected_edges_packet = expected_signed_edges(bits)
    phase_step_s = max(bin_s, phase_step_us * 1e-6)

    best: Optional[MatchCandidate] = None
    for phase_s in float_range(0.0, packet_duration_s, phase_step_s):
        candidate = score_phase(
            total_cum=total_cum,
            signed_cum=signed_cum,
            duration_s=duration_s,
            bin_s=bin_s,
            phase_s=phase_s,
            symbol_period_s=symbol_period_s,
            edge_window_bins=edge_window_bins,
            packet_bits=bits,
            expected_edges_packet=expected_edges_packet,
        )
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate
    if best is None:
        raise RuntimeError("No packet phase candidate could be scored.")
    return best


def packet_rows(raw_name: str, candidate: MatchCandidate, packet_len: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if candidate.symbols_scored == 0:
        return rows
    packet_numbers = np.arange(candidate.symbols_scored, dtype=np.int64) // packet_len
    for packet_number in sorted(set(int(value) for value in packet_numbers)):
        mask = packet_numbers == packet_number
        if int(np.sum(mask)) < packet_len:
            continue
        expected_present = candidate.expected_signed_edges[mask] != 0
        row_edge_present = candidate.edge_present[mask]
        row_signed_counts = candidate.signed_counts[mask]
        row_expected_edges = candidate.expected_signed_edges[mask]
        signed_mask = expected_present & row_edge_present
        symbol_errors = int(np.sum(candidate.decoded_bits[mask] != candidate.expected_bits[mask]))
        edge_errors = int(np.sum(row_edge_present != expected_present))
        if np.any(signed_mask):
            sign_errors = int(
                np.sum(
                    np.sign(row_signed_counts[signed_mask] * candidate.polarity)
                    != np.sign(row_expected_edges[signed_mask])
                )
            )
            sign_rate = 1.0 - sign_errors / int(np.sum(signed_mask))
        else:
            sign_errors = 0
            sign_rate = 0.0
        rows.append(
            {
                "raw_file": raw_name,
                "packet_index": packet_number,
                "start_time_s": float(candidate.boundary_times_s[mask][0]),
                "symbols_scored": int(np.sum(mask)),
                "symbol_errors": symbol_errors,
                "symbol_match_rate": 1.0 - symbol_errors / int(np.sum(mask)),
                "edge_errors": edge_errors,
                "edge_match_rate": 1.0 - edge_errors / int(np.sum(mask)),
                "signed_edges_scored": int(np.sum(signed_mask)),
                "signed_edge_errors": sign_errors,
                "signed_edge_accuracy": sign_rate,
            }
        )
    return rows


def match_status(candidate: MatchCandidate) -> str:
    if candidate.signed_corr >= 0.45 and candidate.symbol_match_rate >= 0.90 and candidate.edge_match_rate >= 0.85:
        return "packet_match_candidate"
    if candidate.signed_corr >= 0.25 and candidate.symbol_match_rate >= 0.75:
        return "weak_match_review"
    return "not_recovered"


def save_diagnostic_plot(
    path: str,
    raw_name: str,
    roi_counts_t_s: np.ndarray,
    roi_counts: np.ndarray,
    candidate: MatchCandidate,
    packet_len: int,
    symbol_rate_hz: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(roi_counts_t_s, roi_counts, linewidth=0.9)
    axes[0].set_title(f"{raw_name}: EVK4 packet matcher")
    axes[0].set_xlabel("Time in analysis window (s)")
    axes[0].set_ylabel("ROI events / bin")

    packet_count = min(3, max(1, candidate.symbols_scored // packet_len))
    n_show = packet_count * packet_len
    x = candidate.boundary_times_s[:n_show] * 1e3
    signed = candidate.signed_counts[:n_show] * candidate.polarity
    expected = candidate.expected_signed_edges[:n_show].astype(np.float64)
    scale = max(float(np.max(np.abs(signed))) if signed.size else 1.0, 1.0)
    axes[1].plot(x, signed, marker=".", linewidth=0.9, label="signed edge events")
    axes[1].step(x, expected * scale, where="mid", linewidth=1.0, label="expected signed edge")
    axes[1].axhline(candidate.edge_threshold, color="tab:gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(-candidate.edge_threshold, color="tab:gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Boundary time in analysis window (ms)")
    axes[1].set_ylabel("Signed events")
    axes[1].legend(loc="best")

    rows = packet_rows(raw_name, candidate, packet_len)
    if rows:
        packet_indices = np.asarray([int(row["packet_index"]) for row in rows], dtype=np.int32)
        symbol_match = np.asarray([float(row["symbol_match_rate"]) for row in rows], dtype=np.float64)
        edge_match = np.asarray([float(row["edge_match_rate"]) for row in rows], dtype=np.float64)
        axes[2].plot(packet_indices, symbol_match, marker=".", linewidth=0.8, label="symbol match")
        axes[2].plot(packet_indices, edge_match, marker=".", linewidth=0.8, label="edge/no-edge match")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xlabel("Packet repeat")
        axes[2].set_ylabel("Match fraction")
        axes[2].legend(loc="best")
    else:
        axes[2].axis("off")
    axes[2].text(
        0.01,
        0.05,
        f"status={match_status(candidate)}, corr={candidate.signed_corr:.3f}, "
        f"symrate={symbol_rate_hz:.1f} Hz",
        transform=axes[2].transAxes,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows and not fieldnames:
        return
    names = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def process_raw(
    raw_path: str,
    bits: Sequence[int],
    packet_source: str,
    args: argparse.Namespace,
    repo_root: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    info = scan_raw_info(raw_path)
    activity = detect_activity_window(
        raw_path,
        info,
        bin_width_ms=float(args.activity_bin_ms),
        threshold_sigma=float(args.activity_threshold_sigma),
    )
    roi = detect_roi(raw_path, info, activity, args)

    analysis_start_s = activity.start_s if args.analysis_start_s is None else float(args.analysis_start_s)
    analysis_end_s = activity.end_s if args.analysis_end_s is None else float(args.analysis_end_s)
    analysis_start_s = max(0.0, min(analysis_start_s, info.duration_s))
    analysis_end_s = max(analysis_start_s, min(analysis_end_s, info.duration_s))
    if analysis_end_s <= analysis_start_s:
        raise RuntimeError(f"Analysis window is empty for {raw_path}")

    start_us = info.first_t_us + int(round(analysis_start_s * 1e6))
    end_us = info.first_t_us + int(round(analysis_end_s * 1e6))
    t_us, p = load_roi_events(raw_path, start_us, end_us, roi)
    if t_us.size < 10:
        raise RuntimeError(f"Too few ROI events in analysis window for {raw_path}: {t_us.size}")

    duration_s = (end_us - start_us) * 1e-6
    total_counts, signed_counts = make_histograms(
        t_us,
        p,
        start_us=start_us,
        duration_s=duration_s,
        bin_us=int(args.bin_us),
    )
    candidate = find_best_match(
        total_counts=total_counts,
        signed_counts=signed_counts,
        duration_s=duration_s,
        bits=bits,
        symbol_rate_hz=float(args.symbol_rate_hz),
        bin_us=int(args.bin_us),
        edge_window_us=float(args.edge_window_us),
        phase_step_us=float(args.phase_step_us),
    )

    raw_name = os.path.basename(raw_path)
    raw_stem = Path(raw_path).stem
    plot_path = os.path.join(repo_root, "plots", "3.2", f"{args.out_prefix}_{raw_stem}_diagnostic.png")
    roi_counts_t_s = np.arange(total_counts.size, dtype=np.float64) * int(args.bin_us) * 1e-6
    if not args.no_plot:
        save_diagnostic_plot(
            plot_path,
            raw_name=raw_name,
            roi_counts_t_s=roi_counts_t_s,
            roi_counts=total_counts,
            candidate=candidate,
            packet_len=len(bits),
            symbol_rate_hz=float(args.symbol_rate_hz),
        )

    summary = {
        "raw_file": raw_name,
        "source_path": os.path.abspath(raw_path),
        "packet_source": packet_source,
        "packet_bits": "".join(str(bit) for bit in bits),
        "packet_symbols": len(bits),
        "symbol_rate_hz": float(args.symbol_rate_hz),
        "symbol_period_us": 1e6 / float(args.symbol_rate_hz),
        "packet_duration_ms": len(bits) / float(args.symbol_rate_hz) * 1e3,
        "capture_duration_s": info.duration_s,
        "capture_events": info.events,
        "width": info.width,
        "height": info.height,
        "activity_start_s": activity.start_s,
        "activity_end_s": activity.end_s,
        "analysis_start_s": analysis_start_s,
        "analysis_end_s": analysis_end_s,
        "roi_x0": roi.x0,
        "roi_y0": roi.y0,
        "roi_x1": roi.x1,
        "roi_y1": roi.y1,
        "roi_source": "manual_cli" if args.roi is not None else "auto_activity_blocks",
        "roi_peak_score": roi.peak_score,
        "roi_events": int(t_us.size),
        "polarity_on_fraction": float(np.mean(p > 0)) if p.size else "",
        "bin_us": int(args.bin_us),
        "edge_window_us": float(args.edge_window_us),
        "best_phase_s": candidate.phase_s,
        "best_phase_ms": candidate.phase_s * 1e3,
        "polarity": "normal" if candidate.polarity > 0 else "inverted",
        "match_status": match_status(candidate),
        "signed_edge_correlation": candidate.signed_corr,
        "edge_threshold": candidate.edge_threshold,
        "edges_scored": candidate.edges_scored,
        "edge_errors": candidate.edge_errors,
        "edge_match_rate": candidate.edge_match_rate,
        "signed_edges_scored": candidate.signed_edges_scored,
        "signed_edge_errors": candidate.signed_edge_errors,
        "signed_edge_accuracy": candidate.signed_edge_accuracy,
        "symbols_scored": candidate.symbols_scored,
        "symbol_errors": candidate.symbol_errors,
        "symbol_match_rate": candidate.symbol_match_rate,
        "diagnostic_plot": "" if args.no_plot else plot_path,
    }
    return summary, packet_rows(raw_name, candidate, len(bits))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match EVK4 RAW event timestamps against the repeated OOK packet currently "
            "programmed in pru1_pwm_CFK_continuous_1000Hz/main.c."
        )
    )
    parser.add_argument("--raws", nargs="+", required=True, help="EVK4 .raw capture paths.")
    parser.add_argument(
        "--main_c",
        default=None,
        help="Path to PRU main.c. If --bits is omitted, symbols_to_send[] is parsed from this file.",
    )
    parser.add_argument("--bits", default=None, help="Known packet bits, for example 10101011110000.")
    parser.add_argument("--zero_symbol", type=int, default=DEFAULT_ZERO_SYMBOL)
    parser.add_argument("--one_symbol", type=int, default=DEFAULT_ONE_SYMBOL)
    parser.add_argument("--nonzero_is_one", action="store_true")
    parser.add_argument("--symbol_rate_hz", type=float, default=DEFAULT_SYMBOL_RATE_HZ)
    parser.add_argument("--bin_us", type=int, default=25)
    parser.add_argument("--edge_window_us", type=float, default=100.0)
    parser.add_argument("--phase_step_us", type=float, default=25.0)
    parser.add_argument("--activity_bin_ms", type=float, default=25.0)
    parser.add_argument("--activity_threshold_sigma", type=float, default=4.0)
    parser.add_argument("--block_px", type=int, default=32)
    parser.add_argument("--roi_blocks", type=int, default=4)
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--analysis_start_s", type=float, default=None)
    parser.add_argument("--analysis_end_s", type=float, default=None)
    parser.add_argument("--out_prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--no_plot", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.symbol_rate_hz <= 0:
        raise ValueError("--symbol_rate_hz must be > 0")
    if args.bin_us <= 0:
        raise ValueError("--bin_us must be > 0")
    if args.edge_window_us <= 0:
        raise ValueError("--edge_window_us must be > 0")
    if args.phase_step_us <= 0:
        raise ValueError("--phase_step_us must be > 0")

    repo_root = repo_root_from_this_file(__file__)
    if args.bits:
        bits = parse_bits_argument(args.bits)
        packet_source = "--bits"
    else:
        main_c = args.main_c or os.path.join(repo_root, DEFAULT_MAIN_C)
        if not os.path.isabs(main_c):
            main_c = os.path.abspath(main_c)
        symbols = parse_symbols_from_main_c(main_c)
        bits = symbols_to_bits(
            symbols,
            zero_symbol=int(args.zero_symbol),
            one_symbol=int(args.one_symbol),
            nonzero_is_one=bool(args.nonzero_is_one),
        )
        packet_source = main_c
    if len(bits) < 4:
        raise RuntimeError("Packet must contain at least 4 bits/symbols to match reliably.")

    summary_rows: List[Dict[str, object]] = []
    all_packet_rows: List[Dict[str, object]] = []
    for raw in args.raws:
        summary, rows = process_raw(
            os.path.abspath(raw),
            bits=bits,
            packet_source=packet_source,
            args=args,
            repo_root=repo_root,
        )
        summary_rows.append(summary)
        all_packet_rows.extend(rows)
        print(
            f"{summary['raw_file']}: status={summary['match_status']}, "
            f"corr={float(summary['signed_edge_correlation']):.3f}, "
            f"symbol_match={float(summary['symbol_match_rate']):.3f}, "
            f"edge_match={float(summary['edge_match_rate']):.3f}, "
            f"errors={summary['symbol_errors']}/{summary['symbols_scored']}, "
            f"roi=[{summary['roi_x0']},{summary['roi_y0']},{summary['roi_x1']},{summary['roi_y1']}]"
        )

    data_dir = os.path.join(repo_root, "data", "3.2")
    summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    packet_path = os.path.join(data_dir, f"{args.out_prefix}_per_packet.csv")
    write_csv(summary_path, summary_rows)
    write_csv(packet_path, all_packet_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {packet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
