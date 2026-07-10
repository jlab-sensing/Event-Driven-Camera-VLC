import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from io_utils import repo_root_from_this_file


DEFAULT_MAIN_C = os.path.join("pru1_pwm_CFK_continuous_1000Hz", "main.c")
DEFAULT_OUT_PREFIX = "s32_pixel7a_packet_match"
DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_SCALE_WIDTH = 960
DEFAULT_ZERO_SYMBOL = 0
DEFAULT_ONE_SYMBOL = 10


@dataclass
class VideoInfo:
    path: str
    codec_name: str
    width: int
    height: int
    avg_fps: float
    nominal_fps: float
    duration_s: float
    nb_frames: int


@dataclass
class MatchResult:
    corr: float
    symbol_px: float
    phase_symbols: float
    reverse: bool


@dataclass
class FrameMatch:
    frame_name: str
    time_s: float
    result: MatchResult
    symbol_good: int
    symbol_total: int
    recovered_bits: str
    expected_bits: str
    packet_positions: str
    saturation_pct: float
    bright_pct: float
    profile: np.ndarray
    expected_profile: np.ndarray


def parse_fraction(value: str) -> float:
    if not value:
        return float("nan")
    if "/" not in value:
        return float(value)
    num, den = value.split("/", 1)
    den_f = float(den)
    if den_f == 0:
        return float("nan")
    return float(num) / den_f


def find_tool(name: str) -> str:
    tool = shutil.which(name)
    if not tool:
        raise RuntimeError(f"Could not find {name} on PATH.")
    return tool


def probe_video(path: str, ffprobe: str) -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    return VideoInfo(
        path=path,
        codec_name=str(stream.get("codec_name", "")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        avg_fps=parse_fraction(str(stream.get("avg_frame_rate", ""))),
        nominal_fps=parse_fraction(str(stream.get("r_frame_rate", ""))),
        duration_s=float(stream.get("duration") or 0),
        nb_frames=int(stream.get("nb_frames") or 0),
    )


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
    body = match.group("body")
    symbols = [int(value) for value in re.findall(r"\b\d+\b", body)]
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


def extract_sample_frames(
    video_path: str,
    out_dir: str,
    ffmpeg: str,
    sample_fps: float,
    scale_width: int,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%04d.png")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={sample_fps},scale={scale_width}:-1",
        pattern,
    ]
    subprocess.run(command, check=True)
    frames = sorted(str(path) for path in Path(out_dir).glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"ffmpeg extracted no frames from {video_path}")
    return frames


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32).copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) - (1 - len(values) % 2))
    if window <= 1:
        return values.astype(np.float32).copy()
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def zscore(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float32) - float(np.mean(values))
    std = float(np.std(centered))
    if std <= 1e-9:
        return centered
    return centered / std


def build_column_profile(image_path: str, top_fraction: float, detrend_window: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    luma = (54.0 * rgb[:, :, 0] + 183.0 * rgb[:, :, 1] + 19.0 * rgb[:, :, 2]) / 256.0
    sorted_luma = np.sort(luma, axis=0)
    start = int(max(0, min(sorted_luma.shape[0] - 1, round(sorted_luma.shape[0] * (1.0 - top_fraction)))))
    column_signal = np.mean(sorted_luma[start:, :], axis=0)
    trend = moving_average(column_signal, detrend_window)
    residual = column_signal - trend
    residual = moving_average(residual, 5)
    return zscore(residual)


def expected_profile(
    width: int,
    bits: Sequence[int],
    symbol_px: float,
    phase_symbols: float,
    reverse: bool,
) -> np.ndarray:
    packet_len = len(bits)
    values = np.empty(width, dtype=np.float32)
    for x in range(width):
        scan_x = width - 1 - x if reverse else x
        bit_index = int(math.floor(phase_symbols + scan_x / symbol_px)) % packet_len
        values[x] = 1.0 if bits[bit_index] else -1.0
    values = moving_average(values, 7)
    return zscore(values)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return float(np.mean(a * b))


def float_range(start: float, stop: float, step: float) -> Iterable[float]:
    count = int(math.floor((stop - start) / step + 0.5))
    for index in range(count + 1):
        yield start + index * step


def find_best_match(
    profile: np.ndarray,
    bits: Sequence[int],
    min_symbol_px: float,
    max_symbol_px: float,
    symbol_px_step: float,
    phase_step_symbols: float,
) -> Tuple[MatchResult, np.ndarray]:
    best: Optional[MatchResult] = None
    best_expected: Optional[np.ndarray] = None
    for symbol_px in float_range(min_symbol_px, max_symbol_px, symbol_px_step):
        phase = 0.0
        while phase < len(bits):
            for reverse in (False, True):
                expected = expected_profile(len(profile), bits, symbol_px, phase, reverse)
                score = correlation(profile, expected)
                if best is None or score > best.corr:
                    best = MatchResult(
                        corr=score,
                        symbol_px=float(symbol_px),
                        phase_symbols=float(phase),
                        reverse=reverse,
                    )
                    best_expected = expected
            phase += phase_step_symbols
    if best is None or best_expected is None:
        raise RuntimeError("No match candidate could be scored.")
    return best, best_expected


def recover_symbols(
    profile: np.ndarray,
    bits: Sequence[int],
    result: MatchResult,
) -> Tuple[int, int, str, str, str]:
    width = len(profile)
    threshold = float(np.median(profile))
    first_symbol = int(math.floor(result.phase_symbols))
    last_symbol = int(math.floor(result.phase_symbols + width / result.symbol_px))
    recovered: List[int] = []
    expected: List[int] = []
    positions: List[int] = []
    for symbol_index in range(first_symbol + 1, last_symbol):
        x_values: List[float] = []
        for x in range(width):
            scan_x = width - 1 - x if result.reverse else x
            current_symbol = int(math.floor(result.phase_symbols + scan_x / result.symbol_px))
            if current_symbol == symbol_index:
                x_values.append(float(profile[x]))
        if len(x_values) < 5:
            continue
        decoded = 1 if float(np.mean(x_values)) > threshold else 0
        packet_position = symbol_index % len(bits)
        want = bits[packet_position]
        recovered.append(decoded)
        expected.append(want)
        positions.append(packet_position)
    good = sum(1 for got, want in zip(recovered, expected) if got == want)
    return (
        good,
        len(expected),
        "".join(str(bit) for bit in recovered),
        "".join(str(bit) for bit in expected),
        " ".join(f"{position:02d}" for position in positions),
    )


def frame_saturation_stats(image_path: str, luma_threshold: float) -> Tuple[float, float]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    luma = (54.0 * rgb[:, :, 0] + 183.0 * rgb[:, :, 1] + 19.0 * rgb[:, :, 2]) / 256.0
    bright = luma >= luma_threshold
    if int(np.sum(bright)) >= 25:
        ys, xs = np.nonzero(bright)
        margin = 25
        x0 = max(0, int(np.min(xs)) - margin)
        x1 = min(rgb.shape[1], int(np.max(xs)) + margin + 1)
        y0 = max(0, int(np.min(ys)) - margin)
        y1 = min(rgb.shape[0], int(np.max(ys)) + margin + 1)
        rgb = rgb[y0:y1, x0:x1, :]
        luma = luma[y0:y1, x0:x1]
        bright = bright[y0:y1, x0:x1]
    saturated = (np.max(rgb, axis=2) >= 250.0) | (luma >= 248.0)
    return float(np.mean(saturated) * 100.0), float(np.mean(bright) * 100.0)


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows and not fieldnames:
        return
    names = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def save_diagnostic_plot(
    path: str,
    video_name: str,
    frame_matches: Sequence[FrameMatch],
    packet_bits: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best = max(frame_matches, key=lambda item: item.result.corr)
    times = np.asarray([item.time_s for item in frame_matches], dtype=np.float32)
    corr = np.asarray([item.result.corr for item in frame_matches], dtype=np.float32)
    match_rate = np.asarray(
        [
            item.symbol_good / item.symbol_total if item.symbol_total else 0.0
            for item in frame_matches
        ],
        dtype=np.float32,
    )
    sat = np.asarray([item.saturation_pct for item in frame_matches], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(times, corr, marker="o", linewidth=1.0)
    axes[0].set_title(f"{video_name}: Pixel packet matcher")
    axes[0].set_ylabel("Pattern correlation")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylim(0, 1.0)

    axes[1].plot(times, match_rate, marker="o", linewidth=1.0, label="symbol match")
    axes[1].plot(times, sat / 100.0, marker=".", linewidth=1.0, label="saturation fraction")
    axes[1].set_ylabel("Fraction")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="best")

    x = np.arange(len(best.profile))
    axes[2].plot(x, best.profile, linewidth=0.9, label="measured column profile")
    axes[2].plot(x, best.expected_profile, linewidth=0.9, alpha=0.85, label="expected packet")
    axes[2].set_title(
        f"Best frame {best.frame_name}: corr={best.result.corr:.3f}, "
        f"symbol={best.result.symbol_px:.1f}px, "
        f"{'reverse' if best.result.reverse else 'forward'} scan"
    )
    axes[2].set_xlabel("Displayed x column after video rotation/scale")
    axes[2].set_ylabel("Normalized signal")
    axes[2].legend(loc="best")
    axes[2].text(
        0.01,
        0.02,
        f"packet={packet_bits}",
        transform=axes[2].transAxes,
        fontsize=8,
        family="monospace",
        va="bottom",
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize_video(
    video_path: str,
    video_info: VideoInfo,
    bits: Sequence[int],
    frame_matches: Sequence[FrameMatch],
) -> Dict[str, object]:
    total_good = sum(item.symbol_good for item in frame_matches)
    total_symbols = sum(item.symbol_total for item in frame_matches)
    match_rate = total_good / total_symbols if total_symbols else 0.0
    correlations = np.asarray([item.result.corr for item in frame_matches], dtype=np.float32)
    symbol_px = np.asarray([item.result.symbol_px for item in frame_matches], dtype=np.float32)
    saturation = np.asarray([item.saturation_pct for item in frame_matches], dtype=np.float32)
    bright = np.asarray([item.bright_pct for item in frame_matches], dtype=np.float32)
    reverse_count = sum(1 for item in frame_matches if item.result.reverse)
    forward_count = len(frame_matches) - reverse_count
    mean_corr = float(np.mean(correlations)) if len(correlations) else 0.0
    min_corr = float(np.min(correlations)) if len(correlations) else 0.0
    status = (
        "packet_match_candidate"
        if mean_corr >= 0.70 and match_rate >= 0.90
        else "weak_match_review"
        if mean_corr >= 0.45
        else "not_recovered"
    )
    return {
        "video": os.path.basename(video_path),
        "source_path": os.path.abspath(video_path),
        "codec_name": video_info.codec_name,
        "metadata_width": video_info.width,
        "metadata_height": video_info.height,
        "avg_fps": video_info.avg_fps,
        "duration_s": video_info.duration_s,
        "metadata_frames": video_info.nb_frames,
        "sampled_frames": len(frame_matches),
        "packet_bits": "".join(str(bit) for bit in bits),
        "packet_symbols": len(bits),
        "match_status": status,
        "mean_pattern_correlation": mean_corr,
        "min_pattern_correlation": min_corr,
        "mean_symbol_px": float(np.mean(symbol_px)) if len(symbol_px) else 0.0,
        "aggregate_symbols_scored": total_symbols,
        "aggregate_symbol_errors": total_symbols - total_good,
        "aggregate_symbol_match_rate": match_rate,
        "reverse_scan_frames": reverse_count,
        "forward_scan_frames": forward_count,
        "mean_saturation_pct": float(np.mean(saturation)) if len(saturation) else 0.0,
        "mean_bright_pct": float(np.mean(bright)) if len(bright) else 0.0,
    }


def process_video(
    video_path: str,
    bits: Sequence[int],
    ffmpeg: str,
    ffprobe: str,
    args: argparse.Namespace,
    repo_root: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    video_info = probe_video(video_path, ffprobe)
    video_stem = Path(video_path).stem
    if args.keep_sample_frames:
        frame_dir = os.path.join(repo_root, "data", "3.2", f"{args.out_prefix}_sample_frames", video_stem)
        cleanup_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    else:
        cleanup_dir = tempfile.TemporaryDirectory(prefix=f"{video_stem}_packet_frames_")
        frame_dir = cleanup_dir.name

    try:
        frames = extract_sample_frames(
            video_path,
            out_dir=frame_dir,
            ffmpeg=ffmpeg,
            sample_fps=args.sample_fps,
            scale_width=args.scale_width,
        )
        frame_matches: List[FrameMatch] = []
        for index, frame_path in enumerate(frames):
            profile = build_column_profile(
                frame_path,
                top_fraction=args.top_fraction,
                detrend_window=args.detrend_window,
            )
            result, expected = find_best_match(
                profile,
                bits,
                min_symbol_px=args.min_symbol_px,
                max_symbol_px=args.max_symbol_px,
                symbol_px_step=args.symbol_px_step,
                phase_step_symbols=args.phase_step_symbols,
            )
            good, total, recovered, expected_bits, positions = recover_symbols(profile, bits, result)
            saturation_pct, bright_pct = frame_saturation_stats(frame_path, args.bright_luma_threshold)
            frame_matches.append(
                FrameMatch(
                    frame_name=os.path.basename(frame_path),
                    time_s=index / float(args.sample_fps),
                    result=result,
                    symbol_good=good,
                    symbol_total=total,
                    recovered_bits=recovered,
                    expected_bits=expected_bits,
                    packet_positions=positions,
                    saturation_pct=saturation_pct,
                    bright_pct=bright_pct,
                    profile=profile,
                    expected_profile=expected,
                )
            )

        plot_path = os.path.join(
            repo_root,
            "plots",
            "3.2",
            f"{args.out_prefix}_{video_stem}_diagnostic.png",
        )
        save_diagnostic_plot(plot_path, os.path.basename(video_path), frame_matches, "".join(str(bit) for bit in bits))
        summary = summarize_video(video_path, video_info, bits, frame_matches)
        summary["diagnostic_plot"] = plot_path
        summary["sample_frame_dir"] = frame_dir if args.keep_sample_frames else ""

        frame_rows: List[Dict[str, object]] = []
        for item in frame_matches:
            frame_rows.append(
                {
                    "video": os.path.basename(video_path),
                    "frame_name": item.frame_name,
                    "time_s": item.time_s,
                    "pattern_correlation": item.result.corr,
                    "symbol_px": item.result.symbol_px,
                    "phase_symbols": item.result.phase_symbols,
                    "scan_direction": "reverse" if item.result.reverse else "forward",
                    "symbols_scored": item.symbol_total,
                    "symbol_errors": item.symbol_total - item.symbol_good,
                    "symbol_match_rate": item.symbol_good / item.symbol_total if item.symbol_total else 0.0,
                    "saturation_pct": item.saturation_pct,
                    "bright_pct": item.bright_pct,
                    "packet_positions": item.packet_positions,
                    "expected_bits": item.expected_bits,
                    "recovered_bits": item.recovered_bits,
                }
            )
        return summary, frame_rows
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match Pixel 7a rolling-shutter video stripes against a known repeated OOK packet. "
            "This is a packet-pattern sanity check for fixed-settings Pixel videos such as 7-9test5."
        )
    )
    parser.add_argument("--videos", nargs="+", required=True, help="Pixel 7a video paths to analyze.")
    parser.add_argument(
        "--main_c",
        default=None,
        help=(
            "Path to PRU main.c. If --bits is omitted, symbols_to_send[] is parsed from this file. "
            "Defaults to pru1_pwm_CFK_continuous_1000Hz/main.c in the repo."
        ),
    )
    parser.add_argument("--bits", default=None, help="Known packet bits, for example 10101011110000.")
    parser.add_argument("--zero_symbol", type=int, default=DEFAULT_ZERO_SYMBOL)
    parser.add_argument("--one_symbol", type=int, default=DEFAULT_ONE_SYMBOL)
    parser.add_argument(
        "--nonzero_is_one",
        action="store_true",
        help="Map every nonzero symbol in symbols_to_send[] to bit 1.",
    )
    parser.add_argument("--sample_fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--scale_width", type=int, default=DEFAULT_SCALE_WIDTH)
    parser.add_argument("--top_fraction", type=float, default=0.20)
    parser.add_argument("--detrend_window", type=int, default=201)
    parser.add_argument("--bright_luma_threshold", type=float, default=35.0)
    parser.add_argument("--min_symbol_px", type=float, default=22.0)
    parser.add_argument("--max_symbol_px", type=float, default=32.0)
    parser.add_argument("--symbol_px_step", type=float, default=0.5)
    parser.add_argument("--phase_step_symbols", type=float, default=0.5)
    parser.add_argument("--out_prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--keep_sample_frames", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = repo_root_from_this_file(__file__)
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")

    if args.bits:
        bits = parse_bits_argument(args.bits)
        source_description = "--bits"
    else:
        main_c = args.main_c or os.path.join(repo_root, DEFAULT_MAIN_C)
        if not os.path.isabs(main_c):
            main_c = os.path.abspath(main_c)
        symbols = parse_symbols_from_main_c(main_c)
        bits = symbols_to_bits(
            symbols,
            zero_symbol=args.zero_symbol,
            one_symbol=args.one_symbol,
            nonzero_is_one=args.nonzero_is_one,
        )
        source_description = main_c

    if len(bits) < 4:
        raise RuntimeError("Packet must contain at least 4 bits/symbols to match reliably.")

    data_dir = os.path.join(repo_root, "data", "3.2")
    summary_rows: List[Dict[str, object]] = []
    all_frame_rows: List[Dict[str, object]] = []
    for video in args.videos:
        summary, frame_rows = process_video(
            os.path.abspath(video),
            bits,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            args=args,
            repo_root=repo_root,
        )
        summary["packet_source"] = source_description
        summary_rows.append(summary)
        all_frame_rows.extend(frame_rows)
        print(
            f"{summary['video']}: status={summary['match_status']}, "
            f"corr={float(summary['mean_pattern_correlation']):.3f}, "
            f"match={float(summary['aggregate_symbol_match_rate']):.3f}, "
            f"errors={summary['aggregate_symbol_errors']}/{summary['aggregate_symbols_scored']}, "
            f"symbol_px={float(summary['mean_symbol_px']):.2f}"
        )

    summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    frame_path = os.path.join(data_dir, f"{args.out_prefix}_per_frame.csv")
    write_csv(summary_path, summary_rows)
    write_csv(frame_path, all_frame_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {frame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
