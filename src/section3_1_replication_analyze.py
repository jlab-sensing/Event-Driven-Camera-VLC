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

from io_utils import repo_root_from_this_file
from latency_analyze import load_timestamps_us


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
    phase_s: float
    message_offset_bits: int
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
            phase_s=float(phase_s),
            message_offset_bits=int(offset),
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
    duration_s: float
    total_events: int
    events_per_s: float
    bin_us: float
    trim_start_s: float
    trim_end_s: float
    phase_s: float
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
        "--phase_steps",
        type=int,
        default=25,
        help="How many start-phase candidates to test within one symbol period.",
    )
    ap.add_argument(
        "--trim_start_s",
        type=float,
        default=0.0,
        help="Ignore this many seconds at the start of each capture before decoding.",
    )
    ap.add_argument(
        "--trim_end_s",
        type=float,
        default=0.0,
        help="Ignore this many seconds at the end of each capture before decoding.",
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

        symbol_rate_hz = float(frequency_hz * args.symbol_rate_scale)
        # Load and re-zero the event timestamps for decoding.
        ts_us = load_timestamps_us(raw_path)
        total_events = int(ts_us.size)
        if total_events < 2:
            raise RuntimeError(f"Not enough events in {raw_file} to decode.")

        ts_rel_s = (ts_us - ts_us[0]).astype(np.float64) * 1e-6
        # Optionally trim off unstable capture edges before decoding.
        trimmed_times_s, duration_s = trim_and_zero_times(
            ts_rel_s=ts_rel_s,
            trim_start_s=args.trim_start_s,
            trim_end_s=args.trim_end_s,
        )
        if duration_s <= 0 or trimmed_times_s.size < 2:
            raise RuntimeError(f"Trimmed capture for {raw_file} is empty. Adjust trim settings.")

        events_per_s = float(trimmed_times_s.size / duration_s) if duration_s > 0 else float("nan")
        # Bin the event stream so symbol windows can be integrated efficiently.
        t_bins, counts = binned_activity(
            time_s=trimmed_times_s,
            bin_width_s=args.bin_us * 1e-6,
            duration_s=duration_s,
        )

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
                f"Could not decode any complete messages from {raw_file}. Try smaller --bin_us, more --phase_steps, or trims."
            )

        # Save the best decode summary for this one capture.
        result = FileResult(
            trial=trial,
            raw_file=raw_file,
            frequency_hz=float(frequency_hz),
            symbol_rate_hz=symbol_rate_hz,
            duration_s=float(duration_s),
            total_events=total_events,
            events_per_s=events_per_s,
            bin_us=float(args.bin_us),
            trim_start_s=float(args.trim_start_s),
            trim_end_s=float(args.trim_end_s),
            phase_s=float(best.phase_s),
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
            "duration_s",
            "total_events",
            "events_per_s",
            "bin_us",
            "trim_start_s",
            "trim_end_s",
            "phase_s",
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
