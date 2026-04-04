import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator

from io_utils import repo_root_from_this_file


# ----------------------------
# Parse values from filenames and inputs
# ----------------------------
def parse_numeric_token(token: str) -> float:
    return float(token.replace("p", "."))


def list_raw_files(input_dir: str) -> List[str]:
    raws = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".raw"):
            raws.append(os.path.join(input_dir, name))
    return sorted(raws)


def load_truth_bits(bits_file: Optional[str], bits_literal: Optional[str]) -> np.ndarray:
    if bits_file and bits_literal:
        raise ValueError("Use either --bits_file or --bits, not both.")
    if not bits_file and not bits_literal:
        raise ValueError("Provide --bits_file or --bits with the repeated message truth bits.")

    if bits_file:
        with open(bits_file, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = bits_literal or ""

    cleaned = "".join(ch for ch in raw if ch in "01")
    if not cleaned:
        raise ValueError("No 0/1 bits were found in the provided truth message.")

    return np.array([1 if ch == "1" else 0 for ch in cleaned], dtype=np.uint8)


# ----------------------------
# Load per-file frequency metadata
# ----------------------------
@dataclass
class ManifestEntry:
    requested_frequency_hz: float
    actual_frequency_hz: float
    symbols_per_bit: int
    message_repeats: int
    guard_bits: int
    total_symbols: int
    duration_s: float
    output_file: str


@dataclass
class CalibrationEntry:
    raw_file: str
    nominal_frequency_hz: float
    active_start_s: float
    active_end_s: float
    active_duration_s: float
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    estimated_bit_frequency_hz: float


def load_frequency_map_csv(path: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Frequency map CSV is missing a header row.")

        freq_col = None
        for candidate in ("frequency_hz", "freq_hz", "bitrate_hz"):
            if candidate in reader.fieldnames:
                freq_col = candidate
                break

        if "raw_file" not in reader.fieldnames or freq_col is None:
            raise ValueError("Frequency map CSV must contain raw_file and frequency_hz/freq_hz/bitrate_hz columns.")

        for row in reader:
            raw_file = row.get("raw_file", "").strip()
            freq_text = row.get(freq_col, "").strip()
            if not raw_file or not freq_text:
                continue
            mapping[os.path.basename(raw_file)] = float(freq_text)

    return mapping


def load_calibration_summaries(calibration_dir: str) -> Dict[str, CalibrationEntry]:
    entries: Dict[str, CalibrationEntry] = {}
    for name in os.listdir(calibration_dir):
        if not name.lower().endswith(".csv") or not name.lower().endswith("_summary.csv"):
            continue
        path = os.path.join(calibration_dir, name)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            required = {
                "raw_file",
                "nominal_frequency_hz",
                "active_start_s",
                "active_end_s",
                "active_duration_s",
                "roi_x0",
                "roi_y0",
                "roi_x1",
                "roi_y1",
                "estimated_bit_frequency_hz",
            }
            if not required.issubset(set(reader.fieldnames)):
                continue
            for row in reader:
                raw_file = os.path.basename(row["raw_file"].strip())
                if not raw_file:
                    continue
                entries[raw_file] = CalibrationEntry(
                    raw_file=raw_file,
                    nominal_frequency_hz=float(row["nominal_frequency_hz"]),
                    active_start_s=float(row["active_start_s"]),
                    active_end_s=float(row["active_end_s"]),
                    active_duration_s=float(row["active_duration_s"]),
                    roi_x0=int(row["roi_x0"]),
                    roi_y0=int(row["roi_y0"]),
                    roi_x1=int(row["roi_x1"]),
                    roi_y1=int(row["roi_y1"]),
                    estimated_bit_frequency_hz=float(row["estimated_bit_frequency_hz"]),
                )
    return entries


def load_transmission_manifest(path: str) -> Dict[float, ManifestEntry]:
    manifest: Dict[float, ManifestEntry] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV is missing a header row.")

        required = {
            "requested_frequency_hz",
            "actual_frequency_hz",
            "symbols_per_bit",
            "message_repeats",
            "guard_bits",
            "total_symbols",
            "duration_s",
            "output_file",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Manifest CSV is missing columns: {', '.join(missing)}")

        for row in reader:
            requested_frequency_hz = float(row["requested_frequency_hz"])
            manifest[round(requested_frequency_hz, 6)] = ManifestEntry(
                requested_frequency_hz=requested_frequency_hz,
                actual_frequency_hz=float(row["actual_frequency_hz"]),
                symbols_per_bit=int(row["symbols_per_bit"]),
                message_repeats=int(row["message_repeats"]),
                guard_bits=int(row["guard_bits"]),
                total_symbols=int(row["total_symbols"]),
                duration_s=float(row["duration_s"]),
                output_file=row["output_file"].strip(),
            )

    return manifest


def lookup_manifest_entry(manifest: Dict[float, ManifestEntry], requested_frequency_hz: float) -> Optional[ManifestEntry]:
    return manifest.get(round(float(requested_frequency_hz), 6))


def extract_frequency_from_name(filename: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, filename)
    if not match:
        return None
    try:
        return parse_numeric_token(match.group(1))
    except Exception:
        return None


# ----------------------------
# Trim and bin captures
# ----------------------------
def load_timestamps_us_filtered(
    raw_path: str,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> tuple[np.ndarray, int, int, int]:
    events = EventsIterator(input_path=raw_path)
    chunks: List[np.ndarray] = []
    capture_start_us: Optional[int] = None
    capture_end_us: Optional[int] = None
    capture_events = 0

    for evs in events:
        if evs.size == 0:
            continue
        t = evs["t"].astype(np.int64)
        capture_events += int(evs.size)
        if capture_start_us is None:
            capture_start_us = int(t[0])
        capture_end_us = int(t[-1])

        if roi is not None:
            x0, y0, x1, y1 = roi
            mask = (
                (evs["x"] >= x0)
                & (evs["x"] < x1)
                & (evs["y"] >= y0)
                & (evs["y"] < y1)
            )
            if not np.any(mask):
                continue
            chunks.append(t[mask])
        else:
            chunks.append(t)

    if capture_start_us is None or capture_end_us is None:
        return np.array([], dtype=np.int64), 0, 0, 0

    if not chunks:
        return np.array([], dtype=np.int64), capture_start_us, capture_end_us, capture_events

    ts_us = np.concatenate(chunks)
    if ts_us.size > 1 and np.any(np.diff(ts_us) < 0):
        ts_us = np.sort(ts_us)
    return ts_us, capture_start_us, capture_end_us, capture_events


def window_and_zero_times(
    ts_rel_s: np.ndarray,
    window_start_s: float,
    window_end_s: float,
    capture_end_s: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    effective_capture_end_s = float(capture_end_s) if capture_end_s is not None else 0.0
    if capture_end_s is None:
        if ts_rel_s.size == 0:
            return np.array([], dtype=np.float64), 0.0
        effective_capture_end_s = float(ts_rel_s.max())

    bounded_start_s = float(min(max(0.0, window_start_s), effective_capture_end_s))
    bounded_end_s = float(min(max(0.0, window_end_s), effective_capture_end_s))
    if bounded_end_s <= bounded_start_s:
        return np.array([], dtype=np.float64), 0.0

    mask = (ts_rel_s >= bounded_start_s) & (ts_rel_s <= bounded_end_s)
    trimmed = ts_rel_s[mask] - bounded_start_s
    duration_s = bounded_end_s - bounded_start_s
    return trimmed, float(duration_s)


def trim_and_zero_times(
    ts_rel_s: np.ndarray,
    trim_start_s: float,
    trim_end_s: float,
) -> tuple[np.ndarray, float]:
    if ts_rel_s.size == 0:
        return np.array([], dtype=np.float64), 0.0

    capture_end_s = float(ts_rel_s.max())
    window_start_s = float(max(0.0, trim_start_s))
    window_end_s = float(capture_end_s - max(0.0, trim_end_s))
    if window_end_s <= window_start_s:
        return np.array([], dtype=np.float64), 0.0

    # Keep only the requested time window, then shift it back to start at 0 seconds.
    mask = (ts_rel_s >= window_start_s) & (ts_rel_s <= window_end_s)
    trimmed = ts_rel_s[mask] - window_start_s
    duration_s = window_end_s - window_start_s
    return trimmed, float(duration_s)


def find_peak_activity_window(
    ts_rel_s: np.ndarray,
    window_duration_s: float,
    bin_width_s: float,
) -> tuple[float, float]:
    if ts_rel_s.size == 0:
        return 0.0, 0.0
    if window_duration_s <= 0.0:
        return 0.0, float(ts_rel_s.max())
    if bin_width_s <= 0.0:
        raise ValueError("bin_width_s must be > 0")

    capture_end_s = float(ts_rel_s.max())
    if capture_end_s <= window_duration_s:
        return 0.0, capture_end_s

    n_bins = max(1, int(np.ceil(capture_end_s / bin_width_s)))
    edges = np.arange(0.0, (n_bins + 1) * bin_width_s, bin_width_s, dtype=np.float64)
    counts, _ = np.histogram(ts_rel_s, bins=edges)
    window_bins = max(1, int(np.ceil(window_duration_s / bin_width_s)))
    if window_bins >= counts.size:
        return 0.0, capture_end_s

    cumulative = np.concatenate(([0], np.cumsum(counts.astype(np.int64))))
    rolling = cumulative[window_bins:] - cumulative[:-window_bins]
    best_idx = int(np.argmax(rolling))
    start_s = float(edges[best_idx])
    end_s = float(min(capture_end_s, start_s + window_bins * bin_width_s))
    return start_s, end_s


def find_manifest_analysis_window(
    ts_rel_s: np.ndarray,
    manifest_entry: ManifestEntry,
    message_len_bits: int,
    search_bin_s: float,
) -> tuple[float, float]:
    actual_frequency_hz = float(manifest_entry.actual_frequency_hz)
    if actual_frequency_hz <= 0.0:
        raise ValueError("Manifest actual_frequency_hz must be > 0")

    payload_duration_s = float((manifest_entry.message_repeats * message_len_bits) / actual_frequency_hz)
    guard_duration_s = float(manifest_entry.guard_bits / actual_frequency_hz)
    payload_start_s, payload_end_s = find_peak_activity_window(
        ts_rel_s=ts_rel_s,
        window_duration_s=payload_duration_s,
        bin_width_s=search_bin_s,
    )

    capture_end_s = float(ts_rel_s.max()) if ts_rel_s.size > 0 else 0.0
    analysis_start_s = max(0.0, payload_start_s - guard_duration_s)
    analysis_end_s = min(capture_end_s, payload_end_s + guard_duration_s)
    return float(analysis_start_s), float(analysis_end_s)


def binned_activity(time_s: np.ndarray, bin_width_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    if duration_s <= 0 or bin_width_s <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Build a fixed set of bins covering the full trimmed capture duration.
    n_bins = max(1, int(np.ceil(duration_s / bin_width_s)))
    edges = np.arange(0.0, (n_bins + 1) * bin_width_s, bin_width_s, dtype=np.float64)
    counts, _ = np.histogram(time_s, bins=edges)
    return edges[:-1], counts.astype(np.float64)


def auto_threshold(sym_sums: np.ndarray) -> float:
    if sym_sums.size == 0:
        return float("nan")
    # Split symbol counts into low/high groups around the median and threshold between them.
    med = float(np.median(sym_sums))
    low = sym_sums[sym_sums <= med]
    high = sym_sums[sym_sums > med]
    if low.size == 0 or high.size == 0:
        return med
    return float(0.5 * (np.mean(low) + np.mean(high)))


def integrate_symbol_counts(
    t_bins: np.ndarray,
    counts: np.ndarray,
    duration_s: float,
    symbol_rate_hz: float,
    start_time_s: float,
) -> np.ndarray:
    if symbol_rate_hz <= 0 or duration_s <= start_time_s:
        return np.array([], dtype=np.float64)

    # Count how many full symbols fit after this candidate start phase.
    symbol_period_s = 1.0 / symbol_rate_hz
    n_symbols = int(np.floor((duration_s - start_time_s) / symbol_period_s))
    if n_symbols <= 0:
        return np.array([], dtype=np.float64)

    cumulative = np.concatenate(([0.0], np.cumsum(counts)))
    symbol_starts = start_time_s + np.arange(n_symbols, dtype=np.float64) * symbol_period_s
    symbol_ends = symbol_starts + symbol_period_s

    left_idx = np.searchsorted(t_bins, symbol_starts, side="left")
    right_idx = np.searchsorted(t_bins, symbol_ends, side="left")
    # Use the cumulative sum so each symbol window can be integrated quickly.
    return cumulative[right_idx] - cumulative[left_idx]


def bits_to_string(bits: np.ndarray) -> str:
    return "".join("1" if int(bit) else "0" for bit in bits.tolist())


# ----------------------------
# Score candidate decodes
# ----------------------------
def compare_candidates(current: Optional["CandidateResult"], challenger: "CandidateResult") -> bool:
    if current is None:
        return True

    current_mar = -1.0 if not np.isfinite(current.mar) else float(current.mar)
    challenger_mar = -1.0 if not np.isfinite(challenger.mar) else float(challenger.mar)
    if challenger_mar != current_mar:
        return challenger_mar > current_mar

    current_ber = float("inf") if not np.isfinite(current.ber) else float(current.ber)
    challenger_ber = float("inf") if not np.isfinite(challenger.ber) else float(challenger.ber)
    if challenger_ber != current_ber:
        return challenger_ber < current_ber

    if challenger.n_messages != current.n_messages:
        return challenger.n_messages > current.n_messages

    if challenger.n_symbols_scored != current.n_symbols_scored:
        return challenger.n_symbols_scored > current.n_symbols_scored

    return challenger.phase_s < current.phase_s


@dataclass
class CandidateResult:
    decode_rate_hz: float
    phase_s: float
    message_offset_bits: int
    init_bit: int
    threshold: float
    n_symbols_total: int
    n_symbols_scored: int
    n_messages: int
    n_correct_messages: int
    mar: float
    n_bit_errors: int
    ber: float
    decoded_bits: np.ndarray


def score_symbol_stream(
    sym_sums: np.ndarray,
    truth_message_bits: np.ndarray,
    phase_s: float,
    decode_rate_hz: float,
) -> Optional[CandidateResult]:
    if sym_sums.size < truth_message_bits.size:
        return None

    threshold = auto_threshold(sym_sums)
    if not np.isfinite(threshold):
        return None

    decoded_bits = (sym_sums >= threshold).astype(np.uint8)
    message_len = int(truth_message_bits.size)
    best: Optional[CandidateResult] = None

    for offset in range(message_len):
        # Drop the partial prefix so the remaining bitstream can be reshaped into full messages.
        drop = (message_len - offset) % message_len
        usable = decoded_bits[drop:]
        n_messages = int(usable.size // message_len)
        if n_messages <= 0:
            continue

        usable = usable[: n_messages * message_len]
        # Compare the repeated decoded message stream against the repeated truth message.
        truth_repeated = np.tile(truth_message_bits, n_messages)
        n_bit_errors = int(np.sum(usable != truth_repeated))
        ber = float(n_bit_errors / usable.size) if usable.size > 0 else float("nan")

        decoded_messages = usable.reshape(n_messages, message_len)
        truth_messages = np.tile(truth_message_bits, (n_messages, 1))
        correct_messages = np.all(decoded_messages == truth_messages, axis=1)
        n_correct_messages = int(np.sum(correct_messages))
        mar = float(n_correct_messages / n_messages) if n_messages > 0 else float("nan")

        candidate = CandidateResult(
            decode_rate_hz=float(decode_rate_hz),
            phase_s=float(phase_s),
            message_offset_bits=int(offset),
            init_bit=int(decoded_bits[0]) if decoded_bits.size else 0,
            threshold=float(threshold),
            n_symbols_total=int(decoded_bits.size),
            n_symbols_scored=int(usable.size),
            n_messages=n_messages,
            n_correct_messages=n_correct_messages,
            mar=mar,
            n_bit_errors=n_bit_errors,
            ber=ber,
            decoded_bits=usable.copy(),
        )

        if compare_candidates(best, candidate):
            best = candidate

    return best


def search_best_decode(
    t_bins: np.ndarray,
    counts: np.ndarray,
    duration_s: float,
    symbol_rate_hz: float,
    truth_message_bits: np.ndarray,
    phase_steps: int,
) -> Optional[CandidateResult]:
    if phase_steps <= 0:
        raise ValueError("phase_steps must be > 0")
    if symbol_rate_hz <= 0:
        raise ValueError("symbol_rate_hz must be > 0")

    symbol_period_s = 1.0 / symbol_rate_hz
    phases = np.linspace(0.0, symbol_period_s, phase_steps, endpoint=False)

    # Try several possible symbol start phases because the capture may not start on a symbol boundary.
    best: Optional[CandidateResult] = None
    for phase_s in phases:
        sym_sums = integrate_symbol_counts(
            t_bins=t_bins,
            counts=counts,
            duration_s=duration_s,
            symbol_rate_hz=symbol_rate_hz,
            start_time_s=float(phase_s),
        )
        candidate = score_symbol_stream(
            sym_sums=sym_sums,
            truth_message_bits=truth_message_bits,
            phase_s=float(phase_s),
            decode_rate_hz=float(symbol_rate_hz),
        )
        if candidate is not None and compare_candidates(best, candidate):
            best = candidate

    return best


def integrate_boundary_counts(
    t_bins: np.ndarray,
    counts: np.ndarray,
    duration_s: float,
    symbol_rate_hz: float,
    start_time_s: float,
    edge_window_fraction: float,
) -> np.ndarray:
    if symbol_rate_hz <= 0 or duration_s <= start_time_s:
        return np.array([], dtype=np.float64)
    if edge_window_fraction <= 0:
        return np.array([], dtype=np.float64)

    symbol_period_s = 1.0 / symbol_rate_hz
    n_boundaries = int(np.floor((duration_s - start_time_s) / symbol_period_s))
    if n_boundaries <= 0:
        return np.array([], dtype=np.float64)

    half_window_s = 0.5 * edge_window_fraction * symbol_period_s
    cumulative = np.concatenate(([0.0], np.cumsum(counts)))
    boundaries = start_time_s + np.arange(n_boundaries, dtype=np.float64) * symbol_period_s
    left_edges = boundaries - half_window_s
    right_edges = boundaries + half_window_s

    left_idx = np.searchsorted(t_bins, left_edges, side="left")
    right_idx = np.searchsorted(t_bins, right_edges, side="left")
    left_idx = np.clip(left_idx, 0, counts.size)
    right_idx = np.clip(right_idx, 0, counts.size)
    return cumulative[right_idx] - cumulative[left_idx]


def score_transition_stream(
    boundary_counts: np.ndarray,
    truth_message_bits: np.ndarray,
    phase_s: float,
    decode_rate_hz: float,
) -> Optional[CandidateResult]:
    if boundary_counts.size + 1 < truth_message_bits.size:
        return None

    threshold = auto_threshold(boundary_counts)
    if not np.isfinite(threshold):
        return None

    edge_bits = (boundary_counts >= threshold).astype(np.uint8)
    best: Optional[CandidateResult] = None
    message_len = int(truth_message_bits.size)

    for init_bit in (0, 1):
        decoded_bits = np.empty(edge_bits.size + 1, dtype=np.uint8)
        decoded_bits[0] = np.uint8(init_bit)
        for i, edge in enumerate(edge_bits, start=1):
            decoded_bits[i] = decoded_bits[i - 1] ^ np.uint8(edge)

        for offset in range(message_len):
            usable = decoded_bits[offset:]
            n_messages = int(usable.size // message_len)
            if n_messages <= 0:
                continue

            usable = usable[: n_messages * message_len]
            truth_repeated = np.tile(truth_message_bits, n_messages)
            n_bit_errors = int(np.sum(usable != truth_repeated))
            ber = float(n_bit_errors / usable.size) if usable.size > 0 else float("nan")

            decoded_messages = usable.reshape(n_messages, message_len)
            truth_messages = np.tile(truth_message_bits, (n_messages, 1))
            correct_messages = np.all(decoded_messages == truth_messages, axis=1)
            n_correct_messages = int(np.sum(correct_messages))
            mar = float(n_correct_messages / n_messages) if n_messages > 0 else float("nan")

            candidate = CandidateResult(
                decode_rate_hz=float(decode_rate_hz),
                phase_s=float(phase_s),
                message_offset_bits=int(offset),
                init_bit=int(init_bit),
                threshold=float(threshold),
                n_symbols_total=int(decoded_bits.size),
                n_symbols_scored=int(usable.size),
                n_messages=n_messages,
                n_correct_messages=n_correct_messages,
                mar=mar,
                n_bit_errors=n_bit_errors,
                ber=ber,
                decoded_bits=usable.copy(),
            )
            if compare_candidates(best, candidate):
                best = candidate

    return best


def search_best_transition_decode(
    t_bins: np.ndarray,
    counts: np.ndarray,
    duration_s: float,
    nominal_rate_hz: float,
    truth_message_bits: np.ndarray,
    phase_steps: int,
    rate_min_scale: float,
    rate_max_scale: float,
    rate_steps: int,
    edge_window_fraction: float,
) -> Optional[CandidateResult]:
    if phase_steps <= 0:
        raise ValueError("phase_steps must be > 0")
    if nominal_rate_hz <= 0:
        raise ValueError("nominal_rate_hz must be > 0")
    if rate_steps <= 0:
        raise ValueError("rate_steps must be > 0")
    if rate_min_scale <= 0 or rate_max_scale <= 0 or rate_max_scale < rate_min_scale:
        raise ValueError("rate scales must be > 0 and max >= min")
    if edge_window_fraction <= 0:
        raise ValueError("edge_window_fraction must be > 0")

    candidate_rates = np.linspace(
        nominal_rate_hz * rate_min_scale,
        nominal_rate_hz * rate_max_scale,
        rate_steps,
        dtype=np.float64,
    )
    best: Optional[CandidateResult] = None
    for decode_rate_hz in candidate_rates:
        symbol_period_s = 1.0 / float(decode_rate_hz)
        phases = np.linspace(0.0, symbol_period_s, phase_steps, endpoint=False)
        for phase_s in phases:
            boundary_counts = integrate_boundary_counts(
                t_bins=t_bins,
                counts=counts,
                duration_s=duration_s,
                symbol_rate_hz=float(decode_rate_hz),
                start_time_s=float(phase_s),
                edge_window_fraction=edge_window_fraction,
            )
            candidate = score_transition_stream(
                boundary_counts=boundary_counts,
                truth_message_bits=truth_message_bits,
                phase_s=float(phase_s),
                decode_rate_hz=float(decode_rate_hz),
            )
            if candidate is not None and compare_candidates(best, candidate):
                best = candidate
    return best


# ----------------------------
# Save summary results
# ----------------------------
@dataclass
class FileResult:
    trial: str
    raw_file: str
    frequency_hz: float
    symbol_rate_hz: float
    decode_rate_hz: float
    capture_events: int
    loaded_events: int
    capture_duration_s: float
    duration_s: float
    total_events: int
    events_per_s: float
    bin_us: float
    analysis_start_s: float
    analysis_end_s: float
    trim_start_s: float
    trim_end_s: float
    expected_transmit_duration_s: float
    active_window_source: str
    roi_source: str
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    decode_mode: str
    phase_s: float
    init_bit: int
    threshold: float
    message_offset_bits: int
    n_symbols_total: int
    n_symbols_scored: int
    n_messages: int
    n_correct_messages: int
    mar: float
    n_bit_errors: int
    ber: float


def aggregate_by_frequency(rows: List[FileResult]) -> List[dict]:
    grouped: Dict[float, List[FileResult]] = {}
    for row in rows:
        grouped.setdefault(float(row.frequency_hz), []).append(row)

    out: List[dict] = []
    for freq in sorted(grouped.keys()):
        trials = grouped[freq]
        # Pool trials at the same frequency so one summary row represents that setting.
        mar_values = np.array([row.mar for row in trials if np.isfinite(row.mar)], dtype=np.float64)
        ber_values = np.array([row.ber for row in trials if np.isfinite(row.ber)], dtype=np.float64)
        total_messages = int(sum(row.n_messages for row in trials))
        total_correct_messages = int(sum(row.n_correct_messages for row in trials))
        total_bits = int(sum(row.n_symbols_scored for row in trials))
        total_bit_errors = int(sum(row.n_bit_errors for row in trials))

        pooled_mar = (
            float(total_correct_messages / total_messages) if total_messages > 0 else float("nan")
        )
        pooled_ber = float(total_bit_errors / total_bits) if total_bits > 0 else float("nan")

        out.append({
            "frequency_hz": freq,
            "n_trials": len(trials),
            "mar_mean": float(np.mean(mar_values)) if mar_values.size > 0 else float("nan"),
            "mar_std": float(np.std(mar_values)) if mar_values.size > 0 else float("nan"),
            "ber_mean": float(np.mean(ber_values)) if ber_values.size > 0 else float("nan"),
            "ber_std": float(np.std(ber_values)) if ber_values.size > 0 else float("nan"),
            "total_messages": total_messages,
            "correct_messages": total_correct_messages,
            "pooled_mar": pooled_mar,
            "total_bits": total_bits,
            "bit_errors": total_bit_errors,
            "pooled_ber": pooled_ber,
        })

    return out


def save_mar_plot(rows: List[dict], out_path: str) -> None:
    if not rows:
        return

    freq = np.array([row["frequency_hz"] for row in rows], dtype=np.float64)
    mar = np.array([row["pooled_mar"] for row in rows], dtype=np.float64)
    mar_std = np.array([row["mar_std"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.errorbar(freq, mar, yerr=mar_std, marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("Beacon frequency (Hz)")
    ax.set_ylabel("Message Accuracy Rate")
    ax.set_title("Section 3.1 replication: MAR vs frequency")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_ber_plot(rows: List[dict], out_path: str) -> None:
    if not rows:
        return

    freq = np.array([row["frequency_hz"] for row in rows], dtype=np.float64)
    ber = np.array([row["pooled_ber"] for row in rows], dtype=np.float64)
    ber_std = np.array([row["ber_std"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.errorbar(freq, ber, yerr=ber_std, marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("Beacon frequency (Hz)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_title("Section 3.1 replication: BER vs frequency")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    root = repo_root_from_this_file(__file__)
    default_input_dir = os.path.abspath(os.path.join(root, "..", "captures", "experiment_replication_3_1"))
    default_manifest_csv = os.path.join(root, "pru1_pwm_CSK_1000Hz", "userspace", "s31_replication_manifest.csv")
    default_calibration_dir = os.path.join(root, "data", "replication_calibration")

    ap = argparse.ArgumentParser(
        description="Analyze the Section 3.1 frequency-sweep replication experiment and compute MAR/BER."
    )
    ap.add_argument(
        "--input_dir",
        default=default_input_dir,
        help="Folder containing replication .raw captures.",
    )
    ap.add_argument(
        "--bits_file",
        default=None,
        help="Text file containing the truth message bits for one repeated message, e.g. 11 bits.",
    )
    ap.add_argument(
        "--bits",
        default=None,
        help="Literal 0/1 truth message bits for one repeated message.",
    )
    ap.add_argument(
        "--freq_map_csv",
        default=None,
        help="Optional CSV mapping raw_file to frequency_hz.",
    )
    ap.add_argument(
        "--manifest_csv",
        default=default_manifest_csv,
        help="Optional symbol manifest CSV used to recover actual transmit frequency/duration per sweep setting.",
    )
    ap.add_argument(
        "--calibration_dir",
        default=default_calibration_dir,
        help="Optional folder of per-file calibration summary CSVs that define active windows and ROI boxes.",
    )
    ap.add_argument(
        "--no_manifest_window",
        action="store_true",
        help="Disable manifest-based active-window detection and decode the full manual-trimmed capture instead.",
    )
    ap.add_argument(
        "--no_calibration_window",
        action="store_true",
        help="Disable use of calibration-summary active windows and ROI boxes even if matching summaries exist.",
    )
    ap.add_argument(
        "--use_calibrated_frequency",
        action="store_true",
        help="When a calibration summary exists, use its estimated bit frequency instead of the nominal/manifest frequency.",
    )
    ap.add_argument(
        "--freq_regex",
        default=r"([0-9]+(?:p[0-9]+|\.[0-9]+)?)Hz",
        help="Regex with one capture group for extracting frequency from filenames.",
    )
    ap.add_argument(
        "--symbol_rate_scale",
        type=float,
        default=1.0,
        help="Symbol rate = extracted frequency_hz * symbol_rate_scale.",
    )
    ap.add_argument(
        "--bin_us",
        type=float,
        default=25.0,
        help="Histogram bin width in microseconds for decoding.",
    )
    ap.add_argument(
        "--decode_mode",
        choices=("activity", "transition"),
        default="activity",
        help="Decode either from per-symbol activity counts or from transition bursts at bit boundaries.",
    )
    ap.add_argument(
        "--phase_steps",
        type=int,
        default=25,
        help="How many start-phase candidates to test within one symbol period.",
    )
    ap.add_argument(
        "--window_search_bin_us",
        type=float,
        default=250.0,
        help="Bin width in microseconds when searching for the active transmit window inside each capture.",
    )
    ap.add_argument(
        "--transition_rate_min_scale",
        type=float,
        default=0.5,
        help="Lower bound of the transition-decoder bit-rate search range, as a multiple of the nominal rate.",
    )
    ap.add_argument(
        "--transition_rate_max_scale",
        type=float,
        default=1.5,
        help="Upper bound of the transition-decoder bit-rate search range, as a multiple of the nominal rate.",
    )
    ap.add_argument(
        "--transition_rate_steps",
        type=int,
        default=81,
        help="How many candidate bit rates to search when using transition decoding.",
    )
    ap.add_argument(
        "--edge_window_fraction",
        type=float,
        default=0.4,
        help="Transition-decoder boundary window width as a fraction of the candidate bit period.",
    )
    ap.add_argument(
        "--trim_start_s",
        type=float,
        default=0.0,
        help="Additional trim applied at the start of the selected analysis window before decoding.",
    )
    ap.add_argument(
        "--trim_end_s",
        type=float,
        default=0.0,
        help="Additional trim applied at the end of the selected analysis window before decoding.",
    )
    ap.add_argument(
        "--out_prefix",
        default="s31_replication",
        help="Prefix for generated CSV and plot files.",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = ap.parse_args()

    if args.symbol_rate_scale <= 0:
        raise ValueError("--symbol_rate_scale must be > 0")
    if args.bin_us <= 0:
        raise ValueError("--bin_us must be > 0")
    if args.phase_steps <= 0:
        raise ValueError("--phase_steps must be > 0")
    if args.window_search_bin_us <= 0:
        raise ValueError("--window_search_bin_us must be > 0")
    if args.transition_rate_min_scale <= 0 or args.transition_rate_max_scale <= 0:
        raise ValueError("--transition_rate_min_scale and --transition_rate_max_scale must be > 0")
    if args.transition_rate_max_scale < args.transition_rate_min_scale:
        raise ValueError("--transition_rate_max_scale must be >= --transition_rate_min_scale")
    if args.transition_rate_steps <= 0:
        raise ValueError("--transition_rate_steps must be > 0")
    if args.edge_window_fraction <= 0:
        raise ValueError("--edge_window_fraction must be > 0")
    if args.trim_start_s < 0 or args.trim_end_s < 0:
        raise ValueError("--trim_start_s and --trim_end_s must be >= 0")

    truth_message_bits = load_truth_bits(args.bits_file, args.bits)
    if truth_message_bits.size < 2:
        raise ValueError("Truth message must contain at least 2 bits.")

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    # Discover all raw captures that belong to this frequency sweep.
    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    # Optional CSV mapping can override whatever frequency is encoded in the filename.
    freq_mapping = load_frequency_map_csv(args.freq_map_csv) if args.freq_map_csv else {}
    manifest = (
        load_transmission_manifest(args.manifest_csv)
        if args.manifest_csv and os.path.exists(args.manifest_csv) and not args.no_manifest_window
        else {}
    )
    calibration = (
        load_calibration_summaries(args.calibration_dir)
        if args.calibration_dir and os.path.isdir(args.calibration_dir) and not args.no_calibration_window
        else {}
    )

    data_dir = os.path.join(root, "data", "replication")
    plot_dir = os.path.join(root, "plots", "replication")
    os.makedirs(data_dir, exist_ok=True)
    if not args.no_plot:
        os.makedirs(plot_dir, exist_ok=True)

    file_summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    freq_summary_path = os.path.join(data_dir, f"{args.out_prefix}_by_frequency_summary.csv")
    per_message_path = os.path.join(data_dir, f"{args.out_prefix}_per_message.csv")
    mar_plot_path = os.path.join(plot_dir, f"{args.out_prefix}_mar_vs_frequency.png")
    ber_plot_path = os.path.join(plot_dir, f"{args.out_prefix}_ber_vs_frequency.png")

    results: List[FileResult] = []
    per_message_rows: List[dict] = []

    for raw_path in raw_files:
        raw_file = os.path.basename(raw_path)
        trial = os.path.splitext(raw_file)[0]

        # Determine the tested beacon frequency for this file.
        frequency_hz = freq_mapping.get(raw_file)
        if frequency_hz is None:
            frequency_hz = extract_frequency_from_name(raw_file, args.freq_regex)
        if frequency_hz is None:
            raise ValueError(
                f"Could not determine frequency for {raw_file}. Update the filename, --freq_regex, or --freq_map_csv."
            )

        manifest_entry = lookup_manifest_entry(manifest, float(frequency_hz))
        calibration_entry = calibration.get(raw_file)
        symbol_rate_hz = (
            float(calibration_entry.estimated_bit_frequency_hz)
            if calibration_entry is not None and args.use_calibrated_frequency and np.isfinite(calibration_entry.estimated_bit_frequency_hz) and calibration_entry.estimated_bit_frequency_hz > 0
            else float(manifest_entry.actual_frequency_hz)
            if manifest_entry is not None
            else float(frequency_hz * args.symbol_rate_scale)
        )

        roi: Optional[tuple[int, int, int, int]] = None
        roi_source = "full_frame"
        if calibration_entry is not None:
            roi = (
                calibration_entry.roi_x0,
                calibration_entry.roi_y0,
                calibration_entry.roi_x1,
                calibration_entry.roi_y1,
            )
            roi_source = "calibration_summary"

        ts_us, capture_start_us, capture_end_us, capture_events = load_timestamps_us_filtered(raw_path, roi=roi)
        loaded_events = int(ts_us.size)
        if loaded_events < 2:
            raise RuntimeError(f"Not enough events in {raw_file} to decode.")

        ts_rel_s = (ts_us - capture_start_us).astype(np.float64) * 1e-6
        capture_duration_s = float((capture_end_us - capture_start_us) * 1e-6)
        if calibration_entry is not None:
            analysis_start_s = float(calibration_entry.active_start_s + args.trim_start_s)
            analysis_end_s = float(calibration_entry.active_end_s - args.trim_end_s)
            expected_transmit_duration_s = float(calibration_entry.active_duration_s)
            active_window_source = "calibration_summary"
        elif manifest_entry is not None:
            auto_window_start_s, auto_window_end_s = find_manifest_analysis_window(
                ts_rel_s=ts_rel_s,
                manifest_entry=manifest_entry,
                message_len_bits=int(truth_message_bits.size),
                search_bin_s=args.window_search_bin_us * 1e-6,
            )
            analysis_start_s = auto_window_start_s + args.trim_start_s
            analysis_end_s = auto_window_end_s - args.trim_end_s
            expected_transmit_duration_s = float(manifest_entry.duration_s)
            active_window_source = "manifest_peak_window"
        else:
            analysis_start_s = float(max(0.0, args.trim_start_s))
            analysis_end_s = float(capture_duration_s - max(0.0, args.trim_end_s))
            expected_transmit_duration_s = float("nan")
            active_window_source = "manual_trim"

        trimmed_times_s, duration_s = window_and_zero_times(
            ts_rel_s=ts_rel_s,
            window_start_s=analysis_start_s,
            window_end_s=analysis_end_s,
            capture_end_s=capture_duration_s,
        )
        if duration_s <= 0 or trimmed_times_s.size < 2:
            raise RuntimeError(
                f"Analysis window for {raw_file} is empty. Adjust trim settings or disable manifest windowing."
            )

        events_per_s = float(trimmed_times_s.size / duration_s) if duration_s > 0 else float("nan")
        # Bin the event stream so symbol windows can be integrated efficiently.
        t_bins, counts = binned_activity(
            time_s=trimmed_times_s,
            bin_width_s=args.bin_us * 1e-6,
            duration_s=duration_s,
        )

        if args.decode_mode == "transition":
            best = search_best_transition_decode(
                t_bins=t_bins,
                counts=counts,
                duration_s=duration_s,
                nominal_rate_hz=symbol_rate_hz,
                truth_message_bits=truth_message_bits,
                phase_steps=args.phase_steps,
                rate_min_scale=args.transition_rate_min_scale,
                rate_max_scale=args.transition_rate_max_scale,
                rate_steps=args.transition_rate_steps,
                edge_window_fraction=args.edge_window_fraction,
            )
        else:
            best = search_best_decode(
                t_bins=t_bins,
                counts=counts,
                duration_s=duration_s,
                symbol_rate_hz=symbol_rate_hz,
                truth_message_bits=truth_message_bits,
                phase_steps=args.phase_steps,
            )
        if best is None:
            raise RuntimeError(
                f"Could not decode any complete messages from {raw_file}. Try smaller --bin_us, more --phase_steps, wider transition-rate search, or trims."
            )

        # Save the best decode summary for this one capture.
        result = FileResult(
            trial=trial,
            raw_file=raw_file,
            frequency_hz=float(frequency_hz),
            symbol_rate_hz=symbol_rate_hz,
            decode_rate_hz=float(best.decode_rate_hz),
            capture_events=int(capture_events),
            loaded_events=int(loaded_events),
            capture_duration_s=capture_duration_s,
            duration_s=float(duration_s),
            total_events=int(trimmed_times_s.size),
            events_per_s=events_per_s,
            bin_us=float(args.bin_us),
            analysis_start_s=float(analysis_start_s),
            analysis_end_s=float(analysis_end_s),
            trim_start_s=float(args.trim_start_s),
            trim_end_s=float(args.trim_end_s),
            expected_transmit_duration_s=expected_transmit_duration_s,
            active_window_source=active_window_source,
            roi_source=roi_source,
            roi_x0=int(roi[0]) if roi is not None else -1,
            roi_y0=int(roi[1]) if roi is not None else -1,
            roi_x1=int(roi[2]) if roi is not None else -1,
            roi_y1=int(roi[3]) if roi is not None else -1,
            decode_mode=args.decode_mode,
            phase_s=float(best.phase_s),
            init_bit=int(best.init_bit),
            threshold=float(best.threshold),
            message_offset_bits=int(best.message_offset_bits),
            n_symbols_total=int(best.n_symbols_total),
            n_symbols_scored=int(best.n_symbols_scored),
            n_messages=int(best.n_messages),
            n_correct_messages=int(best.n_correct_messages),
            mar=float(best.mar),
            n_bit_errors=int(best.n_bit_errors),
            ber=float(best.ber),
        )
        results.append(result)

        if best.n_messages > 0:
            message_len = int(truth_message_bits.size)
            # Also save each repeated message instance for later manual inspection.
            decoded_messages = best.decoded_bits.reshape(best.n_messages, message_len)
            truth_bits_str = bits_to_string(truth_message_bits)
            for idx, decoded_message in enumerate(decoded_messages):
                decoded_str = bits_to_string(decoded_message)
                correct = int(decoded_str == truth_bits_str)
                per_message_rows.append({
                    "trial": trial,
                    "raw_file": raw_file,
                    "frequency_hz": float(frequency_hz),
                    "message_index": idx,
                    "decoded_bits": decoded_str,
                    "truth_bits": truth_bits_str,
                    "correct": correct,
                })

        print(
            f"Done: {raw_file} "
            f"freq={frequency_hz:.1f}Hz "
            f"roi={roi_source} "
            f"mode={args.decode_mode} "
            f"rate={best.decode_rate_hz:.1f}Hz "
            f"window={analysis_start_s:.3f}-{analysis_end_s:.3f}s "
            f"messages={best.n_messages} "
            f"MAR={best.mar:.3f} "
            f"BER={best.ber:.4f}"
        )

    results.sort(key=lambda row: (row.frequency_hz, row.trial))
    # Collapse the trial-level rows into one by-frequency summary table.
    freq_rows = aggregate_by_frequency(results)

    # Save the per-file decode summary.
    with open(file_summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial",
            "raw_file",
            "frequency_hz",
            "symbol_rate_hz",
            "decode_rate_hz",
            "capture_events",
            "loaded_events",
            "capture_duration_s",
            "duration_s",
            "total_events",
            "events_per_s",
            "bin_us",
            "analysis_start_s",
            "analysis_end_s",
            "trim_start_s",
            "trim_end_s",
            "expected_transmit_duration_s",
            "active_window_source",
            "roi_source",
            "roi_x0",
            "roi_y0",
            "roi_x1",
            "roi_y1",
            "decode_mode",
            "phase_s",
            "init_bit",
            "threshold",
            "message_offset_bits",
            "n_symbols_total",
            "n_symbols_scored",
            "n_messages",
            "n_correct_messages",
            "mar",
            "n_bit_errors",
            "ber",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    # Save the pooled per-frequency summary.
    with open(freq_summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frequency_hz",
            "n_trials",
            "mar_mean",
            "mar_std",
            "ber_mean",
            "ber_std",
            "total_messages",
            "correct_messages",
            "pooled_mar",
            "total_bits",
            "bit_errors",
            "pooled_ber",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in freq_rows:
            writer.writerow(row)

    # Save every decoded repeated message for manual spot-checking.
    with open(per_message_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial",
            "raw_file",
            "frequency_hz",
            "message_index",
            "decoded_bits",
            "truth_bits",
            "correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_message_rows:
            writer.writerow(row)

    print("Saved file summary CSV:", file_summary_path)
    print("Saved by-frequency summary CSV:", freq_summary_path)
    print("Saved per-message CSV:", per_message_path)

    if not args.no_plot:
        # Plot pooled MAR and BER trends versus frequency.
        save_mar_plot(freq_rows, mar_plot_path)
        save_ber_plot(freq_rows, ber_plot_path)
        print("Saved plot:", mar_plot_path)
        print("Saved plot:", ber_plot_path)


if __name__ == "__main__":
    main()
