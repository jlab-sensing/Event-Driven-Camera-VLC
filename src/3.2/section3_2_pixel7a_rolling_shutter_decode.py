import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from io_utils import repo_root_from_this_file
from section3_2_pixel7a_video_feasibility import (
    VideoInfo,
    blue_excess,
    detect_roi,
    estimate_active_window,
    find_tool,
    parse_fraction,
    probe_video,
    stream_frames,
)


TARGET_SYMBOL_RATE_HZ = 300.03000300030004
PAYLOAD_BITS = "10110010110"
PREAMBLE_BITS = "10" * 16
GUARD_BITS = 20
MESSAGE_REPEATS = 68
ROI_SIZE = 240


@dataclass
class RoiSamples:
    info: VideoInfo
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    profile_matrix: np.ndarray
    highpass_matrix: np.ndarray
    frame_blue: np.ndarray
    active_start_frame: int
    active_end_frame: int
    selected_rows: np.ndarray
    highpass_selected_rows: np.ndarray
    baseline_profile: np.ndarray
    scale_profile: np.ndarray


@dataclass
class TimingCandidate:
    score: float
    preamble_accuracy: float
    preamble_samples: int
    readout_s: float
    start_s: float
    threshold: float
    polarity: int
    value_mode: str


def expected_bits() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    frame_bits = PREAMBLE_BITS + PAYLOAD_BITS
    full = "0" * GUARD_BITS + frame_bits * MESSAGE_REPEATS + "0" * GUARD_BITS
    bits = np.array([1 if ch == "1" else 0 for ch in full], dtype=np.uint8)
    kind = np.zeros(len(bits), dtype=np.uint8)
    message_index = np.full(len(bits), -1, dtype=np.int16)
    payload_positions: List[Tuple[int, int, int]] = []
    frame_len = len(frame_bits)
    preamble_len = len(PREAMBLE_BITS)
    payload_len = len(PAYLOAD_BITS)
    for msg in range(MESSAGE_REPEATS):
        frame_start = GUARD_BITS + msg * frame_len
        kind[frame_start : frame_start + preamble_len] = 1
        kind[frame_start + preamble_len : frame_start + frame_len] = 2
        message_index[frame_start : frame_start + frame_len] = msg
        for payload_bit in range(payload_len):
            bit_index = frame_start + preamble_len + payload_bit
            payload_positions.append((msg, payload_bit, bit_index))
    return bits, kind, message_index, payload_positions


def extract_roi_samples(
    video_path: str,
    ffmpeg: str,
    ffprobe: str,
    roi_size: int,
    max_detection_frames: int,
) -> RoiSamples:
    info = probe_video(video_path, ffprobe)
    roi = detect_roi(info, ffmpeg, max_detection_frames=max_detection_frames, roi_size=roi_size)
    profiles: List[np.ndarray] = []
    frame_blue: List[float] = []
    for frame in stream_frames(info.path, info.width, info.height, ffmpeg, max_frames=None):
        roi_frame = frame[roi.y0 : roi.y1, roi.x0 : roi.x1, :]
        profile = blue_excess(roi_frame).mean(axis=1)
        profiles.append(profile.astype(np.float32))
        frame_blue.append(float(np.mean(profile)))
    if not profiles:
        raise RuntimeError(f"No frames decoded from {video_path}")

    matrix = np.vstack(profiles)
    highpass_matrix = spatial_highpass(matrix, window=31)
    frame_blue_arr = np.asarray(frame_blue, dtype=np.float32)
    active_start, active_end = estimate_active_window(frame_blue_arr.tolist())
    if active_start is None or active_end is None:
        raise RuntimeError(f"Could not estimate active window for {video_path}")

    inactive_mask = np.ones(matrix.shape[0], dtype=bool)
    margin = 10
    inactive_mask[max(0, active_start - margin) : min(matrix.shape[0], active_end + margin + 1)] = False
    if int(inactive_mask.sum()) < 10:
        baseline = np.percentile(matrix, 10, axis=0)
    else:
        baseline = np.median(matrix[inactive_mask], axis=0)

    active_matrix = matrix[active_start : active_end + 1]
    active_highpass = highpass_matrix[active_start : active_end + 1]
    high_profile = np.percentile(active_matrix, 90, axis=0)
    scale = high_profile - baseline
    highpass_std = np.std(active_highpass, axis=0)
    max_scale = float(np.max(scale))
    if max_scale <= 1e-6:
        raise RuntimeError(f"Detected ROI has no usable active blue signal for {video_path}")

    selected = np.flatnonzero(scale >= max(5.0, 0.25 * max_scale))
    if len(selected) < 12:
        selected = np.flatnonzero(scale >= max(2.5, 0.15 * max_scale))
    if len(selected) < 12:
        raise RuntimeError(f"Too few LED rows selected for {video_path}: {len(selected)}")
    hp_max = float(np.max(highpass_std))
    highpass_selected = np.flatnonzero(highpass_std >= max(1.0, 0.25 * hp_max))
    if len(highpass_selected) < 12:
        highpass_selected = selected

    return RoiSamples(
        info=info,
        roi_x0=roi.x0,
        roi_y0=roi.y0,
        roi_x1=roi.x1,
        roi_y1=roi.y1,
        profile_matrix=matrix,
        highpass_matrix=highpass_matrix.astype(np.float32),
        frame_blue=frame_blue_arr,
        active_start_frame=active_start,
        active_end_frame=active_end,
        selected_rows=selected.astype(np.int32),
        highpass_selected_rows=highpass_selected.astype(np.int32),
        baseline_profile=baseline.astype(np.float32),
        scale_profile=scale.astype(np.float32),
    )


def build_sample_vectors(
    samples: RoiSamples,
    frame_margin: int,
    row_order: str,
    value_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    start_frame = max(0, samples.active_start_frame - frame_margin)
    end_frame = min(samples.profile_matrix.shape[0] - 1, samples.active_end_frame + frame_margin)
    frame_indices = np.arange(start_frame, end_frame + 1, dtype=np.int32)
    selected_rows = samples.highpass_selected_rows if value_mode == "highpass" else samples.selected_rows
    frame_grid, row_grid = np.meshgrid(frame_indices, selected_rows, indexing="ij")
    if value_mode == "normalized":
        values = samples.profile_matrix[frame_grid, row_grid]
        baseline = samples.baseline_profile[row_grid]
        scale = np.maximum(samples.scale_profile[row_grid], 1e-3)
        normalized = ((values - baseline) / scale).astype(np.float32)
        normalized = np.clip(normalized, -0.5, 1.5)
    elif value_mode == "highpass":
        values = samples.highpass_matrix[frame_grid, row_grid].astype(np.float32)
        active_values = samples.highpass_matrix[
            samples.active_start_frame : samples.active_end_frame + 1
        ][:, selected_rows]
        scale = max(float(np.percentile(active_values, 95) - np.percentile(active_values, 5)), 1e-3)
        normalized = values / scale
    else:
        raise ValueError(f"Unknown value mode: {value_mode}")

    frame_times = frame_grid.astype(np.float32) / float(samples.info.avg_fps)
    if row_order == "forward":
        row_rel = row_grid.astype(np.float32)
    elif row_order == "reverse":
        row_rel = (samples.profile_matrix.shape[1] - 1 - row_grid).astype(np.float32)
    else:
        raise ValueError(f"Unknown row order: {row_order}")

    row_fraction = row_rel / float(samples.info.height)
    base_time_and_row = np.column_stack([frame_times.ravel(), row_fraction.ravel()])
    return base_time_and_row, normalized.ravel()


def spatial_highpass(matrix: np.ndarray, window: int) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, matrix.shape[1] - (1 - matrix.shape[1] % 2)))
    pad = window // 2
    padded = np.pad(matrix, ((0, 0), (pad, pad)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smooth = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    return matrix.astype(np.float32) - smooth.astype(np.float32)


def score_candidate(
    base_time_and_row: np.ndarray,
    values: np.ndarray,
    bits: np.ndarray,
    kind: np.ndarray,
    readout_s: float,
    start_s: float,
    value_mode: str,
) -> Optional[TimingCandidate]:
    times = base_time_and_row[:, 0] + base_time_and_row[:, 1] * readout_s
    bit_indices = np.floor((times - start_s) * TARGET_SYMBOL_RATE_HZ).astype(np.int32)
    in_range = (bit_indices >= 0) & (bit_indices < len(bits))
    valid = np.zeros_like(in_range, dtype=bool)
    valid[in_range] = kind[bit_indices[in_range]] == 1
    if int(valid.sum()) < 100:
        return None
    preamble_values = values[valid]
    expected = bits[bit_indices[valid]]
    if not np.any(expected == 0) or not np.any(expected == 1):
        return None

    med0 = float(np.median(preamble_values[expected == 0]))
    med1 = float(np.median(preamble_values[expected == 1]))
    threshold = 0.5 * (med0 + med1)
    normal_pred = (preamble_values > threshold).astype(np.uint8)
    normal_acc = float(np.mean(normal_pred == expected))
    inverted_pred = (preamble_values < threshold).astype(np.uint8)
    inverted_acc = float(np.mean(inverted_pred == expected))
    if inverted_acc > normal_acc:
        accuracy = inverted_acc
        polarity = -1
    else:
        accuracy = normal_acc
        polarity = 1
    separation = abs(med1 - med0) / max(float(np.std(preamble_values)), 1e-6)
    unique_preamble_bits = len(np.unique(bit_indices[valid]))
    coverage_bonus = min(unique_preamble_bits / float(MESSAGE_REPEATS * len(PREAMBLE_BITS)), 1.0)
    score = accuracy + 0.04 * coverage_bonus + 0.01 * min(separation, 5.0)
    return TimingCandidate(
        score=score,
        preamble_accuracy=accuracy,
        preamble_samples=int(valid.sum()),
        readout_s=float(readout_s),
        start_s=float(start_s),
        threshold=float(threshold),
        polarity=polarity,
        value_mode=value_mode,
    )


def find_best_timing(
    samples: RoiSamples,
    row_order: str,
    readout_grid: Sequence[float],
    start_grid: Sequence[float],
    frame_margin: int,
    value_mode: str,
) -> Tuple[TimingCandidate, np.ndarray, np.ndarray]:
    bits, kind, _, _ = expected_bits()
    base_time_and_row, values = build_sample_vectors(
        samples, frame_margin=frame_margin, row_order=row_order, value_mode=value_mode
    )
    best: Optional[TimingCandidate] = None
    for readout_s in readout_grid:
        for start_s in start_grid:
            candidate = score_candidate(base_time_and_row, values, bits, kind, readout_s, start_s, value_mode)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
    if best is None:
        raise RuntimeError(f"No timing candidate could be scored for {samples.info.path}")
    return best, base_time_and_row, values


def refine_timing(
    samples: RoiSamples,
    row_order: str,
    initial: TimingCandidate,
    frame_margin: int,
) -> Tuple[TimingCandidate, np.ndarray, np.ndarray]:
    readout_grid = np.linspace(max(0.001, initial.readout_s - 0.0025), initial.readout_s + 0.0025, 51)
    start_grid = np.linspace(initial.start_s - 0.010, initial.start_s + 0.010, 81)
    return find_best_timing(samples, row_order, readout_grid, start_grid, frame_margin, initial.value_mode)


def decode_with_timing(
    base_time_and_row: np.ndarray,
    values: np.ndarray,
    timing: TimingCandidate,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    bits, kind, message_index, payload_positions = expected_bits()
    times = base_time_and_row[:, 0] + base_time_and_row[:, 1] * timing.readout_s
    bit_indices = np.floor((times - timing.start_s) * TARGET_SYMBOL_RATE_HZ).astype(np.int32)
    valid = (bit_indices >= 0) & (bit_indices < len(bits))

    bit_rows: List[Dict[str, object]] = []
    decoded_by_index: Dict[int, Optional[int]] = {}
    value_by_index: Dict[int, float] = {}
    samples_by_index: Dict[int, int] = {}
    for bit_index in range(len(bits)):
        mask = valid & (bit_indices == bit_index)
        count = int(mask.sum())
        samples_by_index[bit_index] = count
        if count == 0:
            decoded_by_index[bit_index] = None
            value_by_index[bit_index] = float("nan")
            continue
        median_value = float(np.median(values[mask]))
        value_by_index[bit_index] = median_value
        if timing.polarity > 0:
            decoded = int(median_value > timing.threshold)
        else:
            decoded = int(median_value < timing.threshold)
        decoded_by_index[bit_index] = decoded
        bit_rows.append(
            {
                "bit_index": bit_index,
                "kind": bit_kind_name(int(kind[bit_index])),
                "message_index": int(message_index[bit_index]),
                "expected_bit": int(bits[bit_index]),
                "decoded_bit": decoded,
                "median_value": median_value,
                "sample_count": count,
                "correct": int(decoded == int(bits[bit_index])),
            }
        )

    message_rows: List[Dict[str, object]] = []
    payload_errors = 0
    payload_scored = 0
    fully_scored_messages = 0
    correct_messages = 0
    for msg in range(MESSAGE_REPEATS):
        decoded_chars: List[str] = []
        expected_chars: List[str] = []
        message_scored = 0
        message_errors = 0
        for position_msg, payload_bit, bit_index in payload_positions:
            if position_msg != msg:
                continue
            expected = int(bits[bit_index])
            decoded = decoded_by_index.get(bit_index)
            expected_chars.append(str(expected))
            if decoded is None:
                decoded_chars.append("?")
                continue
            decoded_chars.append(str(decoded))
            payload_scored += 1
            message_scored += 1
            if decoded != expected:
                payload_errors += 1
                message_errors += 1
        fully_scored = message_scored == len(PAYLOAD_BITS)
        message_correct = fully_scored and message_errors == 0
        fully_scored_messages += int(fully_scored)
        correct_messages += int(message_correct)
        message_rows.append(
            {
                "message_index": msg,
                "expected_payload": "".join(expected_chars),
                "decoded_payload": "".join(decoded_chars),
                "payload_bits_scored": message_scored,
                "payload_bit_errors": message_errors,
                "fully_scored": int(fully_scored),
                "message_correct": int(message_correct),
            }
        )

    summary = {
        "payload_bits_total": MESSAGE_REPEATS * len(PAYLOAD_BITS),
        "payload_bits_scored": payload_scored,
        "payload_bit_errors": payload_errors,
        "payload_ber": payload_errors / payload_scored if payload_scored else "",
        "messages_total": MESSAGE_REPEATS,
        "messages_fully_scored": fully_scored_messages,
        "messages_correct": correct_messages,
        "message_accuracy": correct_messages / fully_scored_messages if fully_scored_messages else "",
        "all_bits_with_samples": sum(1 for count in samples_by_index.values() if count > 0),
    }
    return bit_rows, message_rows, summary


def decode_status(summary: Dict[str, object], timing: TimingCandidate) -> str:
    payload_total = int(summary["payload_bits_total"])
    payload_scored = int(summary["payload_bits_scored"])
    payload_fraction = payload_scored / payload_total if payload_total else 0.0
    ber = summary["payload_ber"]
    ber_value = float(ber) if ber != "" else 1.0
    if timing.preamble_accuracy < 0.65:
        return "not_recovered_preamble_lock_failed"
    if payload_fraction < 0.5:
        return "not_recovered_low_payload_coverage"
    if ber_value > 0.2:
        return "not_recovered_payload_near_random"
    return "recovered_candidate"


def bit_kind_name(value: int) -> str:
    return {0: "guard_or_pad", 1: "preamble", 2: "payload"}.get(value, "unknown")


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows and not fieldnames:
        return
    names = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def save_decode_plot(
    path: str,
    video_name: str,
    samples: RoiSamples,
    message_rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    times = np.arange(len(samples.frame_blue), dtype=np.float32) / float(samples.info.avg_fps)
    axes[0].plot(times, samples.frame_blue, linewidth=1.0)
    axes[0].axvspan(
        samples.active_start_frame / samples.info.avg_fps,
        samples.active_end_frame / samples.info.avg_fps,
        color="tab:blue",
        alpha=0.12,
        label="active window",
    )
    axes[0].set_title(f"{video_name}: Pixel rolling-shutter decode")
    axes[0].set_ylabel("Mean blue excess")
    axes[0].legend(loc="best")

    bit_errors = np.asarray([int(row["payload_bit_errors"]) for row in message_rows], dtype=np.float32)
    bits_scored = np.asarray([int(row["payload_bits_scored"]) for row in message_rows], dtype=np.float32)
    axes[1].bar(np.arange(len(message_rows)), bit_errors, color="tab:red")
    axes[1].set_ylabel("Payload bit errors")
    axes[1].set_xlabel("Message repeat")
    axes[1].set_ylim(0, max(2, float(np.max(bit_errors)) + 1 if bit_errors.size else 2))

    axes[2].plot(bits_scored, marker="o", linestyle="", markersize=3)
    axes[2].set_ylabel("Payload bits scored")
    axes[2].set_xlabel("Message repeat")
    axes[2].set_ylim(0, len(PAYLOAD_BITS) + 1)
    axes[2].text(
        0.01,
        0.05,
        f"BER={summary['payload_ber']}, scored={summary['payload_bits_scored']}/{summary['payload_bits_total']}",
        transform=axes[2].transAxes,
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode Pixel 7a OOK video with a rolling-shutter row-profile timing search."
    )
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--roi_size", type=int, default=ROI_SIZE)
    parser.add_argument("--max_detection_frames", type=int, default=120)
    parser.add_argument("--frame_margin", type=int, default=45)
    parser.add_argument("--out_prefix", default="s32_pixel7a_rolling_shutter_decode")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = repo_root_from_this_file(__file__)
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    data_dir = os.path.join(repo_root, "data", "3.2")
    plot_dir = os.path.join(repo_root, "plots", "3.2")

    summary_rows: List[Dict[str, object]] = []
    all_bit_rows: List[Dict[str, object]] = []
    all_message_rows: List[Dict[str, object]] = []
    for video in args.videos:
        samples = extract_roi_samples(
            os.path.abspath(video),
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            roi_size=args.roi_size,
            max_detection_frames=args.max_detection_frames,
        )
        coarse_readouts = np.linspace(0.006, 0.033, 56)
        active_start_s = samples.active_start_frame / float(samples.info.avg_fps)
        coarse_starts = np.linspace(active_start_s - 0.25, active_start_s + 0.08, 100)

        search_results = []
        for value_mode in ("normalized", "highpass"):
            for row_order in ("forward", "reverse"):
                initial, _, _ = find_best_timing(
                    samples,
                    row_order=row_order,
                    readout_grid=coarse_readouts,
                    start_grid=coarse_starts,
                    frame_margin=args.frame_margin,
                    value_mode=value_mode,
                )
                refined, base_time_and_row, values = refine_timing(
                    samples,
                    row_order=row_order,
                    initial=initial,
                    frame_margin=args.frame_margin,
                )
                search_results.append((refined, row_order, base_time_and_row, values))

        timing, row_order, base_time_and_row, values = max(search_results, key=lambda item: item[0].score)
        bit_rows, message_rows, decode_summary = decode_with_timing(base_time_and_row, values, timing)
        video_name = os.path.basename(video)

        for row in bit_rows:
            row["video"] = video_name
        for row in message_rows:
            row["video"] = video_name
        all_bit_rows.extend(bit_rows)
        all_message_rows.extend(message_rows)

        summary = {
            "video": video_name,
            "source_path": os.path.abspath(video),
            "codec_name": samples.info.codec_name,
            "width": samples.info.width,
            "height": samples.info.height,
            "avg_fps": samples.info.avg_fps,
            "duration_s": samples.info.duration_s,
            "roi_x0": samples.roi_x0,
            "roi_y0": samples.roi_y0,
            "roi_x1": samples.roi_x1,
            "roi_y1": samples.roi_y1,
            "selected_rows": len(samples.highpass_selected_rows)
            if timing.value_mode == "highpass"
            else len(samples.selected_rows),
            "active_start_s": samples.active_start_frame / float(samples.info.avg_fps),
            "active_end_s": samples.active_end_frame / float(samples.info.avg_fps),
            "value_mode": timing.value_mode,
            "row_order": row_order,
            "estimated_full_frame_readout_s": timing.readout_s,
            "estimated_line_time_us": timing.readout_s / float(samples.info.height) * 1e6,
            "estimated_transmit_start_s": timing.start_s,
            "preamble_sync_accuracy": timing.preamble_accuracy,
            "preamble_sync_samples": timing.preamble_samples,
            "decision_threshold": timing.threshold,
            "polarity": "normal" if timing.polarity > 0 else "inverted",
            **decode_summary,
        }
        summary["decode_status"] = decode_status(summary, timing)
        summary_rows.append(summary)
        save_decode_plot(
            os.path.join(plot_dir, f"{args.out_prefix}_{os.path.splitext(video_name)[0]}_diagnostic.png"),
            video_name,
            samples,
            message_rows,
            summary,
        )
        print(
            f"{video_name}: status={summary['decode_status']}, payload_ber={summary['payload_ber']}, "
            f"payload_scored={summary['payload_bits_scored']}/{summary['payload_bits_total']}, "
            f"messages_correct={summary['messages_correct']}/{summary['messages_fully_scored']}, "
            f"readout={summary['estimated_full_frame_readout_s']:.5f}s, "
            f"mode={timing.value_mode}, row_order={row_order}"
        )

    write_csv(os.path.join(data_dir, f"{args.out_prefix}_summary.csv"), summary_rows)
    write_csv(os.path.join(data_dir, f"{args.out_prefix}_bits.csv"), all_bit_rows)
    write_csv(os.path.join(data_dir, f"{args.out_prefix}_messages.csv"), all_message_rows)
    print(f"Wrote {os.path.join(data_dir, f'{args.out_prefix}_summary.csv')}")
    print(f"Wrote {os.path.join(data_dir, f'{args.out_prefix}_bits.csv')}")
    print(f"Wrote {os.path.join(data_dir, f'{args.out_prefix}_messages.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
