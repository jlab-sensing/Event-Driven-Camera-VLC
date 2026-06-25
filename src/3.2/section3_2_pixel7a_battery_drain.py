import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from io_utils import repo_root_from_this_file


TARGET_HOURS = [1, 2, 3]
CAMERA_IDLE_SCENARIO = "camera_idle_open_camera_black_not_recording"
SCENARIO_LABELS = {
    "camera_idle_open_camera_black_not_recording": "Open Camera idle",
    "active_recording_black": "Recording black frame",
    "active_recording_moving_phone_screen": "Recording moving screen",
    "idle_screen_on_no_app": "Screen on, no app",
}
SCENARIO_ORDER = [
    "camera_idle_open_camera_black_not_recording",
    "active_recording_black",
    "active_recording_moving_phone_screen",
    "idle_screen_on_no_app",
]


def format_value(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def read_points(path: str) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scenario", "elapsed_min", "battery_percent"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        for row in reader:
            scenario = row["scenario"].strip()
            elapsed_min = float(row["elapsed_min"])
            battery_percent = float(row["battery_percent"])
            if not scenario:
                raise ValueError("Each row must include a scenario.")
            if elapsed_min < 0:
                raise ValueError(f"{scenario}: elapsed_min must be >= 0.")
            if not 0 <= battery_percent <= 100:
                raise ValueError(f"{scenario}: battery_percent must be in [0, 100].")
            item = dict(row)
            item["elapsed_min"] = elapsed_min
            item["battery_percent"] = battery_percent
            grouped[scenario].append(item)

    for scenario, rows in grouped.items():
        rows.sort(key=lambda item: float(item["elapsed_min"]))
        for previous, current in zip(rows, rows[1:]):
            if float(current["elapsed_min"]) <= float(previous["elapsed_min"]):
                raise ValueError(f"{scenario}: elapsed_min values must be strictly increasing.")
            if float(current["battery_percent"]) > float(previous["battery_percent"]):
                raise ValueError(f"{scenario}: battery_percent should not increase over time.")
    return dict(grouped)


def interpolate_percent(points: Sequence[Tuple[float, float]], elapsed_min: float) -> float:
    if elapsed_min < points[0][0] or elapsed_min > points[-1][0]:
        return float("nan")
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if t0 <= elapsed_min <= t1:
            if t1 == t0:
                return p0
            fraction = (elapsed_min - t0) / (t1 - t0)
            return p0 + fraction * (p1 - p0)
    return points[-1][1]


def build_summary(grouped: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    camera_idle_losses = {}
    if CAMERA_IDLE_SCENARIO in grouped:
        camera_points = [
            (float(item["elapsed_min"]), float(item["battery_percent"]))
            for item in grouped[CAMERA_IDLE_SCENARIO]
        ]
        for hour in TARGET_HOURS:
            percent = interpolate_percent(camera_points, hour * 60.0)
            camera_idle_losses[hour] = 100.0 - percent

    rows = []
    for scenario in sorted(grouped, key=lambda item: SCENARIO_ORDER.index(item) if item in SCENARIO_ORDER else 999):
        items = grouped[scenario]
        points = [(float(item["elapsed_min"]), float(item["battery_percent"])) for item in items]
        runtime_min = points[-1][0]
        runtime_hr = runtime_min / 60.0
        row: Dict[str, object] = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
            "runtime_min": runtime_min,
            "runtime_hr": runtime_hr,
            "avg_loss_percent_per_hr": 100.0 / runtime_hr if runtime_hr > 0 else float("nan"),
            "n_checkpoints": len(points),
            "source": "lab_journal/6-22.md",
        }
        for hour in TARGET_HOURS:
            percent_remaining = interpolate_percent(points, hour * 60.0)
            loss = 100.0 - percent_remaining
            row[f"battery_percent_at_{hour}hr"] = percent_remaining
            row[f"loss_{hour}hr"] = loss
            baseline_loss = camera_idle_losses.get(hour, float("nan"))
            row[f"loss_diff_vs_camera_idle_{hour}hr"] = loss - baseline_loss
        rows.append(row)
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, object]], header: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in header})


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario.replace("_", " "))


def plot_battery_percent(grouped: Dict[str, List[Dict[str, object]]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for scenario in sorted(grouped, key=lambda item: SCENARIO_ORDER.index(item) if item in SCENARIO_ORDER else 999):
        rows = grouped[scenario]
        xs = [float(item["elapsed_min"]) / 60.0 for item in rows]
        ys = [float(item["battery_percent"]) for item in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=scenario_label(scenario))
    ax.set_xlabel("Elapsed time (hr)")
    ax.set_ylabel("Battery remaining (%)")
    ax.set_title("Pixel 7a Battery Drain by Operating Mode")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_camera_battery_percent(grouped: Dict[str, List[Dict[str, object]]], out_path: str) -> None:
    camera_scenarios = [
        "camera_idle_open_camera_black_not_recording",
        "active_recording_black",
        "active_recording_moving_phone_screen",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for scenario in camera_scenarios:
        if scenario not in grouped:
            continue
        rows = grouped[scenario]
        xs = [float(item["elapsed_min"]) / 60.0 for item in rows]
        ys = [float(item["battery_percent"]) for item in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=scenario_label(scenario))
    ax.set_xlabel("Elapsed time (hr)")
    ax.set_ylabel("Battery remaining (%)")
    ax.set_title("Pixel 7a Battery Drain: Camera Modes")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_loss_by_duration(summary_rows: Sequence[Dict[str, object]], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x_positions = list(range(len(TARGET_HOURS)))
    width = 0.18
    offsets = [(-1.5 + index) * width for index in range(len(summary_rows))]
    for offset, row in zip(offsets, summary_rows):
        losses = [float(row[f"loss_{hour}hr"]) for hour in TARGET_HOURS]
        xs = [x + offset for x in x_positions]
        ax.bar(xs, losses, width=width, label=str(row["scenario_label"]))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{hour} hr" for hour in TARGET_HOURS])
    ax.set_ylabel("Battery loss (%)")
    ax.set_title("Pixel 7a Battery Loss After Fixed Durations")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_runtime(summary_rows: Sequence[Dict[str, object]], out_path: str) -> None:
    labels = [str(row["scenario_label"]) for row in summary_rows]
    runtimes = [float(row["runtime_hr"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.barh(labels, runtimes)
    ax.set_xlabel("Estimated runtime from 100% to 0% (hr)")
    ax.set_title("Pixel 7a Full-Run Battery Runtime by Mode")
    ax.grid(True, axis="x", alpha=0.25)
    for index, value in enumerate(runtimes):
        ax.text(value + 0.12, index, f"{value:.2f} hr", va="center")
    ax.set_xlim(0, max(runtimes) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def print_summary(summary_rows: Iterable[Dict[str, object]]) -> None:
    for row in summary_rows:
        print(
            f"{row['scenario']}: runtime={float(row['runtime_hr']):.2f} hr, "
            f"avg_loss={float(row['avg_loss_percent_per_hr']):.2f}%/hr, "
            f"loss_1hr={float(row['loss_1hr']):.1f}%, "
            f"loss_2hr={float(row['loss_2hr']):.1f}%, "
            f"loss_3hr={float(row['loss_3hr']):.1f}%"
        )


def main() -> None:
    repo_root = repo_root_from_this_file(__file__)
    data_dir = os.path.join(repo_root, "data", "3.2")
    plot_dir = os.path.join(repo_root, "plots", "3.2")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    default_input = os.path.join(data_dir, "s32_pixel7a_battery_drain_2026_06_22.csv")
    parser = argparse.ArgumentParser(
        description="Analyze Pixel 7a battery-percent drain checkpoints from lab_journal/6-22.md."
    )
    parser.add_argument("--input", default=default_input, help="Battery checkpoint CSV.")
    parser.add_argument(
        "--out_prefix",
        default="s32_pixel7a_battery_drain_2026_06_22",
        help="Output filename prefix for summary CSV and plots.",
    )
    args = parser.parse_args()

    grouped = read_points(args.input)
    summary_rows = build_summary(grouped)

    summary_header = [
        "scenario",
        "scenario_label",
        "runtime_min",
        "runtime_hr",
        "avg_loss_percent_per_hr",
        "n_checkpoints",
        "battery_percent_at_1hr",
        "loss_1hr",
        "loss_diff_vs_camera_idle_1hr",
        "battery_percent_at_2hr",
        "loss_2hr",
        "loss_diff_vs_camera_idle_2hr",
        "battery_percent_at_3hr",
        "loss_3hr",
        "loss_diff_vs_camera_idle_3hr",
        "source",
    ]
    summary_path = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    write_csv(summary_path, summary_rows, summary_header)

    percent_plot = os.path.join(plot_dir, f"{args.out_prefix}_battery_percent_vs_time.png")
    camera_percent_plot = os.path.join(plot_dir, f"{args.out_prefix}_camera_modes_battery_percent_vs_time.png")
    loss_plot = os.path.join(plot_dir, f"{args.out_prefix}_loss_at_1h_2h_3h.png")
    runtime_plot = os.path.join(plot_dir, f"{args.out_prefix}_runtime_by_mode.png")
    plot_battery_percent(grouped, percent_plot)
    plot_camera_battery_percent(grouped, camera_percent_plot)
    plot_loss_by_duration(summary_rows, loss_plot)
    plot_runtime(summary_rows, runtime_plot)

    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved plot: {percent_plot}")
    print(f"Saved plot: {camera_percent_plot}")
    print(f"Saved plot: {loss_plot}")
    print(f"Saved plot: {runtime_plot}")
    print("")
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
