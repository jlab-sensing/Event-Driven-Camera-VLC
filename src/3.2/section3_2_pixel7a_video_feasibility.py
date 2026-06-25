import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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


DEFAULT_CAPTURE_SUBDIR = os.path.join("captures", "3.2", "pixel 7a vids")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v")
TARGET_SYMBOL_RATE_HZ = 300.0


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
class Roi:
    x0: int
    y0: int
    x1: int
    y1: int


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


def stream_frames(
    path: str,
    width: int,
    height: int,
    ffmpeg: str,
    max_frames: Optional[int] = None,
) -> Iterable[np.ndarray]:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-noautorotate",
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    frame_bytes = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frames_read = 0
    stopped_early = False
    try:
        while max_frames is None or frames_read < max_frames:
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frames_read += 1
            yield frame
        stopped_early = max_frames is not None and frames_read >= max_frames
    finally:
        if process.stdout:
            process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0 and not stopped_early:
            raise RuntimeError(f"ffmpeg failed for {path} with code {return_code}: {stderr}")


def blue_excess(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32)
    return rgb[:, :, 2] - 0.5 * (rgb[:, :, 0] + rgb[:, :, 1])


def luminance(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def detect_roi(
    info: VideoInfo,
    ffmpeg: str,
    max_detection_frames: int,
    roi_size: int,
) -> Roi:
    detection_map: Optional[np.ndarray] = None
    for frame in stream_frames(info.path, info.width, info.height, ffmpeg, max_detection_frames):
        signal = blue_excess(frame)
        if detection_map is None:
            detection_map = signal
        else:
            detection_map = np.maximum(detection_map, signal)

    if detection_map is None:
        raise RuntimeError(f"No frames decoded from {info.path}")

    threshold = float(np.percentile(detection_map, 99.95))
    mask = detection_map >= threshold
    if int(mask.sum()) < 25:
        threshold = float(np.percentile(detection_map, 99.9))
        mask = detection_map >= threshold

    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) == 0:
        raise RuntimeError(f"Could not detect a bright blue LED ROI in {info.path}")

    weights = np.maximum(detection_map[y_coords, x_coords] - threshold + 1.0, 1.0)
    center_x = int(round(float(np.average(x_coords, weights=weights))))
    center_y = int(round(float(np.average(y_coords, weights=weights))))

    half = roi_size // 2
    x0 = clamp(center_x - half, 0, max(0, info.width - roi_size))
    y0 = clamp(center_y - half, 0, max(0, info.height - roi_size))
    x1 = min(info.width, x0 + roi_size)
    y1 = min(info.height, y0 + roi_size)
    return Roi(x0=x0, y0=y0, x1=x1, y1=y1)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) - (1 - len(values) % 2))
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def profile_metrics(profile: np.ndarray) -> Tuple[float, float]:
    if profile.size < 8:
        return 0.0, 0.0
    window = max(9, int(round(profile.size / 12)))
    baseline = moving_average(profile.astype(np.float32), window)
    highpass = profile.astype(np.float32) - baseline
    raw_span = max(float(np.percentile(profile, 95) - np.percentile(profile, 5)), 1.0)
    hp_span = float(np.percentile(highpass, 95) - np.percentile(highpass, 5))
    row_contrast = hp_span / raw_span

    fft = np.abs(np.fft.rfft(highpass))
    if fft.size <= 4:
        return row_contrast, 0.0
    usable = fft[3:]
    median = float(np.median(usable))
    peak_ratio = float(np.max(usable) / median) if median > 1e-6 else 0.0
    return row_contrast, peak_ratio


def analyze_video(
    info: VideoInfo,
    ffmpeg: str,
    roi: Roi,
    max_analysis_frames: Optional[int],
) -> Tuple[Dict[str, object], List[Dict[str, object]], Optional[np.ndarray], Optional[np.ndarray]]:
    frame_rows: List[Dict[str, object]] = []
    first_roi_frame: Optional[np.ndarray] = None
    first_profile: Optional[np.ndarray] = None

    roi_mean_luma: List[float] = []
    roi_mean_blue_excess: List[float] = []
    row_contrasts: List[float] = []
    fft_ratios: List[float] = []
    saturated_fractions: List[float] = []

    for frame_index, frame in enumerate(stream_frames(info.path, info.width, info.height, ffmpeg, max_analysis_frames)):
        roi_frame = frame[roi.y0 : roi.y1, roi.x0 : roi.x1, :]
        if first_roi_frame is None:
            first_roi_frame = roi_frame.copy()

        roi_luma = luminance(roi_frame)
        roi_blue = blue_excess(roi_frame)
        profile = roi_blue.mean(axis=1)
        if first_profile is None:
            first_profile = profile.copy()

        row_contrast, fft_ratio = profile_metrics(profile)
        mean_luma = float(np.mean(roi_luma))
        mean_blue = float(np.mean(roi_blue))
        sat_fraction = float(np.mean(np.max(roi_frame, axis=2) >= 250.0))

        roi_mean_luma.append(mean_luma)
        roi_mean_blue_excess.append(mean_blue)
        row_contrasts.append(row_contrast)
        fft_ratios.append(fft_ratio)
        saturated_fractions.append(sat_fraction)

        frame_rows.append(
            {
                "video": os.path.basename(info.path),
                "frame_index": frame_index,
                "time_s": frame_index / info.avg_fps if info.avg_fps and math.isfinite(info.avg_fps) else "",
                "roi_mean_luma": mean_luma,
                "roi_mean_blue_excess": mean_blue,
                "row_contrast": row_contrast,
                "row_fft_peak_ratio": fft_ratio,
                "saturated_fraction": sat_fraction,
            }
        )

    if not frame_rows:
        raise RuntimeError(f"No frames analyzed for {info.path}")

    frame_luma_cv = coefficient_of_variation(roi_mean_luma)
    frame_blue_cv = coefficient_of_variation(roi_mean_blue_excess)
    median_row_contrast = safe_median(row_contrasts)
    p95_row_contrast = safe_percentile(row_contrasts, 95)
    median_fft_ratio = safe_median(fft_ratios)
    p95_fft_ratio = safe_percentile(fft_ratios, 95)
    median_saturation = safe_median(saturated_fractions)
    active_start_frame, active_end_frame = estimate_active_window(roi_mean_blue_excess)
    active_rows = frame_rows[active_start_frame : active_end_frame + 1] if active_start_frame is not None else []
    active_row_contrast = [float(row["row_contrast"]) for row in active_rows]
    active_fft_ratios = [float(row["row_fft_peak_ratio"]) for row in active_rows]

    frame_level_possible = info.avg_fps >= TARGET_SYMBOL_RATE_HZ * 2.0
    row_banding_detected = safe_median(active_row_contrast or row_contrasts) >= 0.08 and safe_median(
        active_fft_ratios or fft_ratios
    ) >= 4.0
    feasibility = (
        "frame_level_not_feasible"
        if not frame_level_possible and not row_banding_detected
        else "rolling_shutter_candidate"
        if row_banding_detected
        else "needs_manual_review"
    )

    summary = {
        "video": os.path.basename(info.path),
        "source_path": info.path,
        "codec_name": info.codec_name,
        "width": info.width,
        "height": info.height,
        "avg_fps": info.avg_fps,
        "nominal_fps": info.nominal_fps,
        "duration_s": info.duration_s,
        "metadata_frames": info.nb_frames,
        "analyzed_frames": len(frame_rows),
        "target_symbol_rate_hz": TARGET_SYMBOL_RATE_HZ,
        "active_start_s": (
            active_start_frame / info.avg_fps
            if active_start_frame is not None and info.avg_fps and math.isfinite(info.avg_fps)
            else ""
        ),
        "active_end_s": (
            active_end_frame / info.avg_fps
            if active_end_frame is not None and info.avg_fps and math.isfinite(info.avg_fps)
            else ""
        ),
        "active_frames": len(active_rows),
        "roi_x0": roi.x0,
        "roi_y0": roi.y0,
        "roi_x1": roi.x1,
        "roi_y1": roi.y1,
        "frame_level_possible_by_nyquist": int(frame_level_possible),
        "roi_luma_cv": frame_luma_cv,
        "roi_blue_excess_cv": frame_blue_cv,
        "row_contrast_median": median_row_contrast,
        "row_contrast_p95": p95_row_contrast,
        "row_fft_peak_ratio_median": median_fft_ratio,
        "row_fft_peak_ratio_p95": p95_fft_ratio,
        "active_row_contrast_median": safe_median(active_row_contrast),
        "active_row_fft_peak_ratio_median": safe_median(active_fft_ratios),
        "saturated_fraction_median": median_saturation,
        "row_banding_detected": int(row_banding_detected),
        "feasibility": feasibility,
    }
    return summary, frame_rows, first_roi_frame, first_profile


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(arr))
    if abs(mean) < 1e-6:
        return 0.0
    return float(np.std(arr) / abs(mean))


def safe_median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float32))) if values else 0.0


def safe_percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile)) if values else 0.0


def estimate_active_window(values: Sequence[float]) -> Tuple[Optional[int], Optional[int]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float32)
    edge_count = max(5, int(round(len(arr) * 0.1)))
    baseline = float(np.median(np.concatenate([arr[:edge_count], arr[-edge_count:]])))
    high = float(np.percentile(arr, 95))
    if high <= baseline + 5.0:
        return None, None

    threshold = baseline + 0.35 * (high - baseline)
    active = arr >= threshold
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_len = 0
    start: Optional[int] = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if (not value or index == len(active) - 1) and start is not None:
            end = index if value and index == len(active) - 1 else index - 1
            run_len = end - start + 1
            if run_len > best_len:
                best_start, best_end, best_len = start, end, run_len
            start = None

    if best_start is None or best_len < 5:
        return None, None
    return best_start, best_end


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_roi_png(path: str, roi_frame: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(roi_frame).save(path)


def save_plot(
    path: str,
    video_name: str,
    frame_rows: Sequence[Dict[str, object]],
    first_profile: Optional[np.ndarray],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    times = np.asarray([float(row["time_s"]) for row in frame_rows if row["time_s"] != ""], dtype=np.float32)
    blue = np.asarray([float(row["roi_mean_blue_excess"]) for row in frame_rows], dtype=np.float32)
    row_contrast = np.asarray([float(row["row_contrast"]) for row in frame_rows], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    if len(times) == len(blue):
        axes[0].plot(times, blue, linewidth=1.0)
        axes[0].set_xlabel("Time (s)")
    else:
        axes[0].plot(blue, linewidth=1.0)
        axes[0].set_xlabel("Frame")
    axes[0].set_title(f"{video_name}: ROI blue-excess trace")
    axes[0].set_ylabel("Mean blue excess")

    axes[1].plot(row_contrast, linewidth=1.0)
    axes[1].set_title("Per-frame row-banding contrast")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Contrast")

    if first_profile is not None:
        axes[2].plot(first_profile, linewidth=1.0)
        axes[2].set_title("First-frame ROI row profile")
        axes[2].set_xlabel("Encoded row within ROI")
        axes[2].set_ylabel("Mean blue excess")
    else:
        axes[2].axis("off")

    fig.savefig(path, dpi=200)
    plt.close(fig)


def default_video_dir(repo_root: str) -> str:
    candidates = [
        os.path.join(repo_root, DEFAULT_CAPTURE_SUBDIR),
        os.path.join(os.path.dirname(repo_root), DEFAULT_CAPTURE_SUBDIR),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def collect_videos(video_dir: str, explicit_videos: Sequence[str]) -> List[str]:
    if explicit_videos:
        return [os.path.abspath(path) for path in explicit_videos]
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    return [
        os.path.join(video_dir, name)
        for name in sorted(os.listdir(video_dir))
        if name.lower().endswith(VIDEO_EXTENSIONS)
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether Pixel 7a videos have enough frame or rolling-shutter signal for BER decoding."
    )
    parser.add_argument("--video_dir", default=None, help="Directory containing Pixel 7a videos.")
    parser.add_argument("--videos", nargs="*", default=[], help="Explicit video paths to analyze.")
    parser.add_argument("--roi_size", type=int, default=240, help="Square ROI size around the detected blue LED.")
    parser.add_argument("--max_detection_frames", type=int, default=120)
    parser.add_argument("--max_analysis_frames", type=int, default=0, help="0 means analyze all frames.")
    parser.add_argument("--out_prefix", default="s32_pixel7a_video_feasibility")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = repo_root_from_this_file(__file__)
    video_dir = args.video_dir or default_video_dir(repo_root)
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    videos = collect_videos(video_dir, args.videos)
    if not videos:
        raise RuntimeError(f"No videos found in {video_dir}")

    data_dir = os.path.join(repo_root, "data", "3.2")
    plot_dir = os.path.join(repo_root, "plots", "3.2")
    crop_dir = os.path.join(data_dir, f"{args.out_prefix}_roi_crops")
    summary_rows: List[Dict[str, object]] = []
    all_frame_rows: List[Dict[str, object]] = []
    max_analysis_frames = args.max_analysis_frames if args.max_analysis_frames > 0 else None

    for video_path in videos:
        info = probe_video(video_path, ffprobe)
        roi = detect_roi(info, ffmpeg, args.max_detection_frames, args.roi_size)
        summary, frame_rows, roi_frame, first_profile = analyze_video(
            info, ffmpeg, roi, max_analysis_frames=max_analysis_frames
        )
        summary_rows.append(summary)
        all_frame_rows.extend(frame_rows)

        video_base = os.path.splitext(os.path.basename(video_path))[0]
        if roi_frame is not None:
            save_roi_png(os.path.join(crop_dir, f"{video_base}_roi.png"), roi_frame)
        save_plot(
            os.path.join(plot_dir, f"{args.out_prefix}_{video_base}_diagnostic.png"),
            os.path.basename(video_path),
            frame_rows,
            first_profile,
        )

    summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    frame_path = os.path.join(data_dir, f"{args.out_prefix}_per_frame.csv")
    write_csv(summary_path, summary_rows)
    write_csv(frame_path, all_frame_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {frame_path}")
    print(f"Wrote ROI crops to {crop_dir}")
    for row in summary_rows:
        print(
            f"{row['video']}: feasibility={row['feasibility']}, "
            f"fps={float(row['avg_fps']):.3f}, "
            f"row_contrast_median={float(row['row_contrast_median']):.4f}, "
            f"row_fft_peak_ratio_median={float(row['row_fft_peak_ratio_median']):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
