import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from io_utils import repo_root_from_this_file


REQUIRED_COLUMNS = [
    "trial_id",
    "sensor_type",
    "modulation",
    "frequency_hz",
    "lux",
    "lighting_source",
    "visibility_condition",
    "bits_transmitted",
    "bit_errors",
]

OPTIONAL_COLUMNS = [
    "status",
    "actual_frequency_hz",
    "scenario",
    "distance_cm",
    "angle_deg",
    "bits_scored",
    "active_start_s",
    "active_end_s",
    "capture_file",
    "decode_log_file",
    "total_events",
    "raw_duration_s",
    "active_duration_s",
    "event_rate_per_s",
    "max_event_gap_ms",
    "long_event_gap_count",
    "long_event_gap_fraction",
    "notes",
]

NUMERIC_X_FIELDS = ["lux", "frequency_hz", "distance_cm", "angle_deg"]
DEFAULT_SERIES_FIELDS = ["sensor_type", "lighting_source", "visibility_condition"]


@dataclass
class RawActivityMetrics:
    total_events: int
    raw_duration_s: float
    active_duration_s: float
    event_rate_per_s: float
    max_event_gap_ms: float
    long_event_gap_count: int
    long_event_gap_fraction: float


@dataclass
class RobustnessTrial:
    trial_id: str
    sensor_type: str
    modulation: str
    frequency_hz: float
    actual_frequency_hz: float
    lux: float
    lighting_source: str
    visibility_condition: str
    scenario: str
    distance_cm: float
    angle_deg: float
    bits_transmitted: int
    bits_scored: int
    bit_errors: int
    active_start_s: float
    active_end_s: float
    capture_file: str
    decode_log_file: str
    total_events: int
    raw_duration_s: float
    active_duration_s: float
    event_rate_per_s: float
    max_event_gap_ms: float
    long_event_gap_count: int
    long_event_gap_fraction: float
    notes: str
    score_fraction: float
    ber: float
    correct_bits: int
    frequency_error_fraction: float


def require_text(row: Dict[str, str], column: str, row_number: int) -> str:
    value = row.get(column, "").strip()
    if not value:
        raise ValueError(f"Row {row_number}: required column '{column}' is empty.")
    return value


def parse_required_float(row: Dict[str, str], column: str, row_number: int) -> float:
    raw = require_text(row, column, row_number)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: column '{column}' must be numeric, got '{raw}'.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Row {row_number}: column '{column}' must be finite.")
    return value


def parse_optional_float(row: Dict[str, str], column: str) -> float:
    raw = row.get(column, "").strip()
    if not raw:
        return float("nan")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Column '{column}' must be finite when provided.")
    return value


def parse_required_int(row: Dict[str, str], column: str, row_number: int) -> int:
    value = parse_required_float(row, column, row_number)
    if not float(value).is_integer():
        raise ValueError(f"Row {row_number}: column '{column}' must be an integer value.")
    return int(value)


def parse_optional_int(row: Dict[str, str], column: str, default_value: int) -> int:
    raw = row.get(column, "").strip()
    if not raw:
        return default_value
    value = float(raw)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"Column '{column}' must be an integer value when provided.")
    return int(value)


def mean_or_nan(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(statistics.mean(finite))


def std_or_zero(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    return float(statistics.stdev(finite))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def format_value(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def ensure_manifest_header(fieldnames: Sequence[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Manifest is missing required columns: {missing_str}")


def maybe_warn_extra_columns(fieldnames: Sequence[str]) -> None:
    known = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    extras = [field for field in fieldnames if field not in known]
    if extras:
        print(f"Warning: ignoring unknown manifest columns: {', '.join(extras)}")


def resolve_capture_path(repo_root: str, manifest_dir: str, capture_file: str) -> Optional[str]:
    if not capture_file:
        return None
    if os.path.isabs(capture_file) and os.path.exists(capture_file):
        return capture_file

    candidates = [
        os.path.join(manifest_dir, capture_file),
        os.path.join(repo_root, capture_file),
        os.path.join(repo_root, "captures", "3.3", capture_file),
        os.path.join(repo_root, "data", "3.3", capture_file),
        os.path.join(repo_root, "data", "3.1", "replication", capture_file),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def stream_raw_activity_metrics(
    raw_path: str,
    active_start_s: float,
    active_end_s: float,
    dropout_gap_ms: float,
) -> RawActivityMetrics:
    from metavision_core.event_io import EventsIterator

    if dropout_gap_ms <= 0:
        raise ValueError("dropout_gap_ms must be > 0.")

    first_ts_us: Optional[int] = None
    last_ts_us: Optional[int] = None
    active_first_ts_us: Optional[int] = None
    active_last_ts_us: Optional[int] = None
    previous_active_ts_us: Optional[int] = None

    total_events = 0
    active_events = 0
    max_gap_us = 0
    long_gap_count = 0
    long_gap_total_us = 0
    dropout_gap_us = int(round(dropout_gap_ms * 1000.0))
    use_window = math.isfinite(active_start_s) and math.isfinite(active_end_s) and active_end_s > active_start_s

    for evs in EventsIterator(input_path=raw_path):
        if evs.size == 0:
            continue
        ts = evs["t"].astype(np.int64)
        if first_ts_us is None:
            first_ts_us = int(ts[0])
        last_ts_us = int(ts[-1])
        total_events += int(ts.size)

        if use_window:
            rel_s = (ts - first_ts_us) * 1e-6
            mask = (rel_s >= active_start_s) & (rel_s <= active_end_s)
            active_ts = ts[mask]
        else:
            active_ts = ts

        if active_ts.size == 0:
            continue

        active_events += int(active_ts.size)
        if active_first_ts_us is None:
            active_first_ts_us = int(active_ts[0])
        active_last_ts_us = int(active_ts[-1])

        if previous_active_ts_us is not None:
            first_gap = int(active_ts[0]) - previous_active_ts_us
            max_gap_us = max(max_gap_us, first_gap)
            if first_gap > dropout_gap_us:
                long_gap_count += 1
                long_gap_total_us += first_gap

        diffs = np.diff(active_ts)
        if diffs.size:
            max_gap_us = max(max_gap_us, int(np.max(diffs)))
            long_gaps = diffs[diffs > dropout_gap_us]
            if long_gaps.size:
                long_gap_count += int(long_gaps.size)
                long_gap_total_us += int(np.sum(long_gaps))

        previous_active_ts_us = int(active_ts[-1])

    if first_ts_us is None or last_ts_us is None:
        return RawActivityMetrics(
            total_events=0,
            raw_duration_s=0.0,
            active_duration_s=0.0,
            event_rate_per_s=float("nan"),
            max_event_gap_ms=float("nan"),
            long_event_gap_count=0,
            long_event_gap_fraction=float("nan"),
        )

    raw_duration_s = float((last_ts_us - first_ts_us) * 1e-6)
    if use_window:
        active_duration_s = active_end_s - active_start_s
    elif active_first_ts_us is not None and active_last_ts_us is not None:
        active_duration_s = float((active_last_ts_us - active_first_ts_us) * 1e-6)
    else:
        active_duration_s = 0.0

    event_rate_per_s = safe_divide(active_events, active_duration_s)
    long_event_gap_fraction = safe_divide(long_gap_total_us * 1e-6, active_duration_s)

    return RawActivityMetrics(
        total_events=active_events if use_window else total_events,
        raw_duration_s=raw_duration_s,
        active_duration_s=active_duration_s,
        event_rate_per_s=event_rate_per_s,
        max_event_gap_ms=max_gap_us / 1000.0 if active_events > 1 else float("nan"),
        long_event_gap_count=long_gap_count,
        long_event_gap_fraction=long_event_gap_fraction,
    )


def read_manifest_metrics(row: Dict[str, str]) -> RawActivityMetrics:
    total_events = parse_optional_int(row, "total_events", 0)
    raw_duration_s = parse_optional_float(row, "raw_duration_s")
    active_duration_s = parse_optional_float(row, "active_duration_s")
    event_rate_per_s = parse_optional_float(row, "event_rate_per_s")
    max_event_gap_ms = parse_optional_float(row, "max_event_gap_ms")
    long_event_gap_count = parse_optional_int(row, "long_event_gap_count", 0)
    long_event_gap_fraction = parse_optional_float(row, "long_event_gap_fraction")

    return RawActivityMetrics(
        total_events=total_events,
        raw_duration_s=raw_duration_s,
        active_duration_s=active_duration_s,
        event_rate_per_s=event_rate_per_s,
        max_event_gap_ms=max_event_gap_ms,
        long_event_gap_count=long_event_gap_count,
        long_event_gap_fraction=long_event_gap_fraction,
    )


def load_trials(
    manifest_path: str,
    with_raw_metrics: bool,
    dropout_gap_ms: float,
) -> List[RobustnessTrial]:
    repo_root = repo_root_from_this_file(__file__)
    manifest_dir = os.path.abspath(os.path.dirname(manifest_path))
    trials: List[RobustnessTrial] = []

    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV is missing a header row.")
        ensure_manifest_header(reader.fieldnames)
        maybe_warn_extra_columns(reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            status = row.get("status", "").strip().lower()
            if status in {"planned", "todo", "skip", "skipped"}:
                continue

            trial_id = require_text(row, "trial_id", row_number)
            sensor_type = require_text(row, "sensor_type", row_number)
            modulation = require_text(row, "modulation", row_number)
            frequency_hz = parse_required_float(row, "frequency_hz", row_number)
            actual_frequency_hz = parse_optional_float(row, "actual_frequency_hz")
            lux = parse_required_float(row, "lux", row_number)
            lighting_source = require_text(row, "lighting_source", row_number)
            visibility_condition = require_text(row, "visibility_condition", row_number)
            scenario = row.get("scenario", "").strip()
            distance_cm = parse_optional_float(row, "distance_cm")
            angle_deg = parse_optional_float(row, "angle_deg")
            bits_transmitted = parse_required_int(row, "bits_transmitted", row_number)
            bit_errors = parse_required_int(row, "bit_errors", row_number)
            bits_scored = parse_optional_int(row, "bits_scored", bits_transmitted)
            active_start_s = parse_optional_float(row, "active_start_s")
            active_end_s = parse_optional_float(row, "active_end_s")
            capture_file = row.get("capture_file", "").strip()
            decode_log_file = row.get("decode_log_file", "").strip()

            if frequency_hz <= 0:
                raise ValueError(f"Row {row_number}: frequency_hz must be > 0.")
            if bits_transmitted <= 0:
                raise ValueError(f"Row {row_number}: bits_transmitted must be > 0.")
            if bits_scored <= 0:
                raise ValueError(f"Row {row_number}: bits_scored must be > 0.")
            if bits_scored > bits_transmitted:
                raise ValueError(f"Row {row_number}: bits_scored cannot exceed bits_transmitted.")
            if bit_errors < 0:
                raise ValueError(f"Row {row_number}: bit_errors must be >= 0.")
            if bit_errors > bits_scored:
                raise ValueError(f"Row {row_number}: bit_errors cannot exceed bits_scored.")

            metrics = read_manifest_metrics(row)
            if with_raw_metrics and capture_file:
                resolved = resolve_capture_path(repo_root, manifest_dir, capture_file)
                if resolved is None:
                    print(f"Warning: row {row_number} capture_file was not found: {capture_file}")
                else:
                    metrics = stream_raw_activity_metrics(
                        raw_path=resolved,
                        active_start_s=active_start_s,
                        active_end_s=active_end_s,
                        dropout_gap_ms=dropout_gap_ms,
                    )

            score_fraction = safe_divide(bits_scored, bits_transmitted)
            ber = safe_divide(bit_errors, bits_scored)
            correct_bits = bits_scored - bit_errors
            frequency_error_fraction = (
                safe_divide(abs(actual_frequency_hz - frequency_hz), frequency_hz)
                if math.isfinite(actual_frequency_hz)
                else float("nan")
            )

            trials.append(
                RobustnessTrial(
                    trial_id=trial_id,
                    sensor_type=sensor_type,
                    modulation=modulation,
                    frequency_hz=frequency_hz,
                    actual_frequency_hz=actual_frequency_hz,
                    lux=lux,
                    lighting_source=lighting_source,
                    visibility_condition=visibility_condition,
                    scenario=scenario,
                    distance_cm=distance_cm,
                    angle_deg=angle_deg,
                    bits_transmitted=bits_transmitted,
                    bits_scored=bits_scored,
                    bit_errors=bit_errors,
                    active_start_s=active_start_s,
                    active_end_s=active_end_s,
                    capture_file=capture_file,
                    decode_log_file=decode_log_file,
                    total_events=metrics.total_events,
                    raw_duration_s=metrics.raw_duration_s,
                    active_duration_s=metrics.active_duration_s,
                    event_rate_per_s=metrics.event_rate_per_s,
                    max_event_gap_ms=metrics.max_event_gap_ms,
                    long_event_gap_count=metrics.long_event_gap_count,
                    long_event_gap_fraction=metrics.long_event_gap_fraction,
                    notes=row.get("notes", "").strip(),
                    score_fraction=score_fraction,
                    ber=ber,
                    correct_bits=correct_bits,
                    frequency_error_fraction=frequency_error_fraction,
                )
            )

    if not trials:
        raise ValueError(
            "No ready Section 3.3 trial rows were found. Fill in a planned row and set status to blank or 'ready' before running the analyzer."
        )
    return trials


def build_group_fields(x_field: str, series_fields: Sequence[str]) -> List[str]:
    ordered = []
    for field in list(series_fields) + [x_field]:
        if field not in ordered:
            ordered.append(field)
    return ordered


def group_key_for_trial(trial: RobustnessTrial, group_fields: Sequence[str]) -> Tuple[object, ...]:
    return tuple(getattr(trial, field) for field in group_fields)


def aggregate_trials(
    trials: Sequence[RobustnessTrial],
    x_field: str,
    series_fields: Sequence[str],
) -> List[Dict[str, object]]:
    group_fields = build_group_fields(x_field, series_fields)
    grouped: Dict[Tuple[object, ...], List[RobustnessTrial]] = {}
    for trial in trials:
        grouped.setdefault(group_key_for_trial(trial, group_fields), []).append(trial)

    rows: List[Dict[str, object]] = []
    for key, group_trials in grouped.items():
        row: Dict[str, object] = {field: value for field, value in zip(group_fields, key)}

        total_bits_transmitted = sum(trial.bits_transmitted for trial in group_trials)
        total_bits_scored = sum(trial.bits_scored for trial in group_trials)
        total_bit_errors = sum(trial.bit_errors for trial in group_trials)
        total_correct_bits = sum(trial.correct_bits for trial in group_trials)

        row.update(
            {
                "n_trials": len(group_trials),
                "total_bits_transmitted": total_bits_transmitted,
                "total_bits_scored": total_bits_scored,
                "total_bit_errors": total_bit_errors,
                "total_correct_bits": total_correct_bits,
                "pooled_ber": safe_divide(total_bit_errors, total_bits_scored),
                "pooled_score_fraction": safe_divide(total_bits_scored, total_bits_transmitted),
                "mean_ber": mean_or_nan(trial.ber for trial in group_trials),
                "std_ber": std_or_zero(trial.ber for trial in group_trials),
                "mean_score_fraction": mean_or_nan(trial.score_fraction for trial in group_trials),
                "std_score_fraction": std_or_zero(trial.score_fraction for trial in group_trials),
                "mean_actual_frequency_hz": mean_or_nan(trial.actual_frequency_hz for trial in group_trials),
                "mean_frequency_error_fraction": mean_or_nan(
                    trial.frequency_error_fraction for trial in group_trials
                ),
                "mean_event_rate_per_s": mean_or_nan(trial.event_rate_per_s for trial in group_trials),
                "std_event_rate_per_s": std_or_zero(trial.event_rate_per_s for trial in group_trials),
                "mean_max_event_gap_ms": mean_or_nan(trial.max_event_gap_ms for trial in group_trials),
                "mean_long_event_gap_fraction": mean_or_nan(
                    trial.long_event_gap_fraction for trial in group_trials
                ),
            }
        )
        rows.append(row)

    def sort_key(row: Dict[str, object]) -> Tuple[object, ...]:
        prefix = []
        for field in series_fields:
            value = row.get(field)
            if isinstance(value, float):
                prefix.append(float("inf") if not math.isfinite(value) else value)
            else:
                prefix.append("" if value is None else str(value))
        x_value = row.get(x_field)
        if isinstance(x_value, float):
            x_value = float("inf") if not math.isfinite(x_value) else x_value
        return tuple(prefix + [x_value])

    rows.sort(key=sort_key)
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, object]], header: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in header})


def write_trial_csv(path: str, trials: Sequence[RobustnessTrial]) -> None:
    rows = [asdict(trial) for trial in trials]
    header = list(rows[0].keys())
    write_csv(path, rows, header)


def write_summary_csv(path: str, rows: Sequence[Dict[str, object]], x_field: str, series_fields: Sequence[str]) -> None:
    header = build_group_fields(x_field, series_fields) + [
        "n_trials",
        "total_bits_transmitted",
        "total_bits_scored",
        "total_bit_errors",
        "total_correct_bits",
        "pooled_ber",
        "pooled_score_fraction",
        "mean_ber",
        "std_ber",
        "mean_score_fraction",
        "std_score_fraction",
        "mean_actual_frequency_hz",
        "mean_frequency_error_fraction",
        "mean_event_rate_per_s",
        "std_event_rate_per_s",
        "mean_max_event_gap_ms",
        "mean_long_event_gap_fraction",
    ]
    write_csv(path, rows, header)


def format_series_part(field: str, value: object) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if field == "sensor_type":
        return str(value)
    if field == "lighting_source":
        return str(value)
    if field == "visibility_condition":
        return str(value)
    if field == "modulation":
        return str(value)
    if field == "lux":
        return f"{format_value(value)} lux"
    if field == "distance_cm":
        return f"{format_value(value)} cm"
    if field == "angle_deg":
        return f"{format_value(value)} deg"
    return f"{field}={format_value(value)}"


def build_series_label(row: Dict[str, object], series_fields: Sequence[str]) -> str:
    parts = []
    for field in series_fields:
        label_part = format_series_part(field, row[field])
        if label_part:
            parts.append(label_part)
    if not parts:
        return "all trials"
    return " | ".join(parts)


def has_finite_metric(rows: Sequence[Dict[str, object]], x_field: str, y_field: str) -> bool:
    for row in rows:
        try:
            x_value = float(row[x_field])
            y_value = float(row[y_field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            return True
    return False


def plot_metric(
    summary_rows: Sequence[Dict[str, object]],
    x_field: str,
    series_fields: Sequence[str],
    y_field: str,
    yerr_field: Optional[str],
    out_path: str,
    title: str,
    ylabel: str,
) -> bool:
    if not has_finite_metric(summary_rows, x_field, y_field):
        return False

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        label = build_series_label(row, series_fields)
        grouped.setdefault(label, []).append(row)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label, rows in grouped.items():
        usable = []
        for item in rows:
            x_value = float(item[x_field])
            y_value = float(item[y_field])
            if math.isfinite(x_value) and math.isfinite(y_value):
                usable.append(item)
        if not usable:
            continue
        usable.sort(key=lambda item: float(item[x_field]))
        xs = [float(item[x_field]) for item in usable]
        ys = [float(item[y_field]) for item in usable]
        yerr = None
        if yerr_field is not None:
            yerr = [
                float(item[yerr_field]) if math.isfinite(float(item[yerr_field])) else 0.0
                for item in usable
            ]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=4, label=label)

    ax.set_xlabel(x_field.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(grouped) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return True


def validate_group_fields(fields: Sequence[str]) -> None:
    allowed = set(NUMERIC_X_FIELDS + DEFAULT_SERIES_FIELDS + [
        "modulation",
        "scenario",
        "actual_frequency_hz",
    ])
    invalid = [field for field in fields if field not in allowed]
    if invalid:
        raise ValueError(f"Unsupported grouping field(s): {', '.join(invalid)}")


def main() -> None:
    repo_root = repo_root_from_this_file(__file__)
    default_manifest = os.path.join(repo_root, "data", "3.3", "section3_3_robustness_manifest_template.csv")

    ap = argparse.ArgumentParser(
        description="Analyze Section 3.3 robustness trials across lighting and visibility conditions."
    )
    ap.add_argument(
        "--manifest",
        default=default_manifest,
        help="Manifest CSV describing one row per Section 3.3 robustness trial.",
    )
    ap.add_argument(
        "--out_prefix",
        default="s33_robustness",
        help="Prefix for per-trial CSV, summary CSV, and plot filenames.",
    )
    ap.add_argument(
        "--x_field",
        default="lux",
        choices=NUMERIC_X_FIELDS,
        help="Numeric field used on the x-axis for plots and trial aggregation.",
    )
    ap.add_argument(
        "--series_fields",
        nargs="*",
        default=DEFAULT_SERIES_FIELDS,
        help="Fields that define separate plotted series and pooled summary rows.",
    )
    ap.add_argument(
        "--with_raw_metrics",
        action="store_true",
        help="Read capture_file .raw files and compute event-rate and event-gap metrics.",
    )
    ap.add_argument(
        "--dropout_gap_ms",
        type=float,
        default=10.0,
        help="Eventless gap threshold for the long-event-gap proxy metric.",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(args.manifest)
    validate_group_fields([args.x_field] + list(args.series_fields))

    trials = load_trials(
        manifest_path=args.manifest,
        with_raw_metrics=args.with_raw_metrics,
        dropout_gap_ms=args.dropout_gap_ms,
    )
    summary_rows = aggregate_trials(trials, x_field=args.x_field, series_fields=args.series_fields)

    data_dir = os.path.join(repo_root, "data", "3.3")
    plot_dir = os.path.join(repo_root, "plots", "3.3")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    per_trial_csv = os.path.join(data_dir, f"{args.out_prefix}_per_trial.csv")
    summary_csv = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    write_trial_csv(per_trial_csv, trials)
    write_summary_csv(summary_csv, summary_rows, x_field=args.x_field, series_fields=args.series_fields)

    print(f"Saved per-trial CSV: {per_trial_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("")
    for row in summary_rows:
        label = build_series_label(row, args.series_fields)
        print(
            f"{label}, {args.x_field}={format_value(row[args.x_field])}: "
            f"pooled_ber={row['pooled_ber']:.6g}, "
            f"mean_event_rate_per_s={row['mean_event_rate_per_s']:.6g}, "
            f"n_trials={int(row['n_trials'])}"
        )

    if args.no_plot:
        return

    plot_specs = [
        (
            "mean_ber",
            "std_ber",
            f"{args.out_prefix}_ber_vs_{args.x_field}.png",
            "Section 3.3 Bit Error Rate",
            "BER",
        ),
        (
            "mean_event_rate_per_s",
            "std_event_rate_per_s",
            f"{args.out_prefix}_event_rate_vs_{args.x_field}.png",
            "Section 3.3 Event Rate",
            "events/s",
        ),
        (
            "mean_long_event_gap_fraction",
            None,
            f"{args.out_prefix}_long_event_gap_fraction_vs_{args.x_field}.png",
            "Section 3.3 Event-Gap Fraction",
            "fraction of active window in long event gaps",
        ),
    ]

    for y_field, yerr_field, filename, title, ylabel in plot_specs:
        out_path = os.path.join(plot_dir, filename)
        saved = plot_metric(
            summary_rows=summary_rows,
            x_field=args.x_field,
            series_fields=args.series_fields,
            y_field=y_field,
            yerr_field=yerr_field,
            out_path=out_path,
            title=title,
            ylabel=ylabel,
        )
        if saved:
            print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
