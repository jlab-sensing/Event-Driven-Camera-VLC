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
    "bits_transmitted",
    "bit_errors",
    "sensing_window_s",
    "sensing_power_w",
    "sensing_idle_power_w",
    "compute_window_s",
    "compute_power_w",
    "compute_idle_power_w",
]

OPTIONAL_COLUMNS = [
    "status",
    "lux",
    "distance_cm",
    "bits_scored",
    "sensing_power_uncertainty_w",
    "sensing_idle_power_uncertainty_w",
    "compute_power_uncertainty_w",
    "compute_idle_power_uncertainty_w",
    "capture_file",
    "power_log_file",
    "decode_log_file",
    "notes",
]

NUMERIC_X_FIELDS = ["frequency_hz", "lux", "distance_cm"]
DEFAULT_SERIES_FIELDS = ["sensor_type", "lux", "modulation"]
POWER_ESTIMATE_COLUMNS = [
    "sensing_window_s",
    "sensing_power_w",
    "sensing_idle_power_w",
    "compute_window_s",
    "compute_power_w",
    "compute_idle_power_w",
    "sensing_power_uncertainty_w",
    "sensing_idle_power_uncertainty_w",
    "compute_power_uncertainty_w",
    "compute_idle_power_uncertainty_w",
]
MANUAL_POWER_LOG_TOKENS = ("manual", "usb_meter")
DEFAULT_MANUAL_POWER_UNCERTAINTY_W = 0.001


@dataclass
class EnergyTrial:
    trial_id: str
    sensor_type: str
    modulation: str
    frequency_hz: float
    lux: float
    distance_cm: float
    bits_transmitted: int
    bits_scored: int
    bit_errors: int
    sensing_window_s: float
    sensing_power_w: float
    sensing_idle_power_w: float
    compute_window_s: float
    compute_power_w: float
    compute_idle_power_w: float
    sensing_power_uncertainty_w: float
    sensing_idle_power_uncertainty_w: float
    compute_power_uncertainty_w: float
    compute_idle_power_uncertainty_w: float
    capture_file: str
    power_log_file: str
    power_source: str
    decode_log_file: str
    notes: str
    score_fraction: float
    ber: float
    correct_bits: int
    sensing_active_power_w: float
    compute_active_power_w: float
    sensing_active_power_uncertainty_w: float
    compute_active_power_uncertainty_w: float
    sensing_energy_j_gross: float
    sensing_energy_j_active: float
    compute_energy_j_gross: float
    compute_energy_j_active: float
    total_energy_j_gross: float
    total_energy_j_active: float
    sensing_energy_j_gross_uncertainty: float
    sensing_energy_j_active_uncertainty: float
    compute_energy_j_gross_uncertainty: float
    compute_energy_j_active_uncertainty: float
    total_energy_j_gross_uncertainty: float
    total_energy_j_active_uncertainty: float
    sensing_active_j_per_tx_bit: float
    compute_active_j_per_tx_bit: float
    gross_j_per_tx_bit: float
    active_j_per_tx_bit: float
    gross_j_per_scored_bit: float
    active_j_per_scored_bit: float
    gross_j_per_correct_bit: float
    active_j_per_correct_bit: float
    gross_j_per_tx_bit_uncertainty: float
    active_j_per_tx_bit_uncertainty: float
    gross_j_per_scored_bit_uncertainty: float
    active_j_per_scored_bit_uncertainty: float
    gross_j_per_correct_bit_uncertainty: float
    active_j_per_correct_bit_uncertainty: float


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


def rss(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return math.sqrt(sum(value * value for value in finite))


def rms_or_zero(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def format_value(value: object) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def format_series_part(field: str, value: object) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if field == "sensor_type":
        return str(value)
    if field == "modulation":
        return str(value)
    if field == "lux":
        return f"{format_value(value)} lux"
    if field == "distance_cm":
        return f"{format_value(value)} cm"
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


def ensure_manifest_header(fieldnames: Sequence[str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Manifest is missing required columns: {missing_str}")


def parse_float_cell(row: Dict[str, str], column: str) -> Optional[float]:
    raw = row.get(column, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def resolve_power_log_path(repo_root: str, manifest_dir: str, power_log_file: str) -> Optional[str]:
    if not power_log_file:
        return None
    if os.path.isabs(power_log_file) and os.path.exists(power_log_file):
        return power_log_file

    candidates = [
        os.path.join(manifest_dir, power_log_file),
        os.path.join(repo_root, power_log_file),
        os.path.join(repo_root, "data", "3.2", power_log_file),
        os.path.join(repo_root, "captures", "3.2", power_log_file),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def first_finite(row: Dict[str, str], columns: Sequence[str]) -> Optional[float]:
    for column in columns:
        value = parse_float_cell(row, column)
        if value is not None:
            return value
    return None


def mean_finite(values: Iterable[float]) -> Optional[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return float(statistics.mean(finite))


def std_finite(values: Iterable[float]) -> Optional[float]:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return None
    return float(statistics.stdev(finite))


def combine_uncertainties(*values: Optional[float]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value)) and value >= 0]
    if not finite:
        return 0.0
    return math.sqrt(sum(value * value for value in finite))


def mwh_resolution_power_uncertainty(row: Dict[str, str]) -> Optional[float]:
    duration_s = first_finite(row, ["transmission_duration_s", "whole_visible_window_s", "stop_marker_elapsed_s"])
    has_mwh = any(column in row for column in ["delta_mwh", "start_mwh", "end_mwh", "whole_visible_delta_mwh"])
    if duration_s is None or duration_s <= 0 or not has_mwh:
        return None
    return 3.6 / duration_s


def choose_power_log_row(rows: Sequence[Dict[str, str]], trial_id: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("trial_id", "").strip() == trial_id:
            return row
    return rows[0] if rows else None


def estimate_from_power_summary(row: Dict[str, str]) -> Dict[str, float]:
    estimate: Dict[str, float] = {}

    for column in POWER_ESTIMATE_COLUMNS:
        value = parse_float_cell(row, column)
        if value is not None:
            estimate[column] = value

    duration_s = first_finite(row, ["transmission_duration_s", "whole_visible_window_s", "stop_marker_elapsed_s"])
    if duration_s is not None:
        estimate.setdefault("sensing_window_s", duration_s)

    sensing_power = first_finite(
        row,
        [
            "estimated_avg_power_w",
            "mean_active_sample_power_w",
            "mean_sample_power_w",
            "whole_visible_avg_power_w",
            "stop_marker_avg_power_w",
        ],
    )
    if sensing_power is not None:
        estimate.setdefault("sensing_power_w", sensing_power)
        estimate.setdefault("sensing_idle_power_w", 0.0)

    sample_uncertainty = first_finite(
        row,
        [
            "estimated_avg_power_uncertainty_w",
            "std_active_sample_power_w",
            "std_sample_power_w",
        ],
    )
    resolution_uncertainty = mwh_resolution_power_uncertainty(row)
    sensing_uncertainty = combine_uncertainties(sample_uncertainty, resolution_uncertainty)
    if sensing_uncertainty > 0:
        estimate.setdefault("sensing_power_uncertainty_w", sensing_uncertainty)

    estimate.setdefault("compute_window_s", 0.0)
    estimate.setdefault("compute_power_w", 0.0)
    estimate.setdefault("compute_idle_power_w", 0.0)
    return estimate


def estimate_from_power_samples(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    numeric_rows = []
    for row in rows:
        power_w = parse_float_cell(row, "power_w")
        time_s = parse_float_cell(row, "time_s")
        if power_w is None:
            continue
        numeric_rows.append((row, power_w, time_s))

    if not numeric_rows:
        return {}

    def marker_contains(row: Dict[str, str], *needles: str) -> bool:
        marker = row.get("marker", "").strip().lower()
        return any(needle in marker for needle in needles)

    active_values = [
        power_w
        for row, power_w, _ in numeric_rows
        if marker_contains(row, "active", "transmit_end_estimated")
    ]
    idle_values = [
        power_w
        for row, power_w, _ in numeric_rows
        if marker_contains(row, "idle", "baseline")
    ]
    if not active_values:
        active_values = [power_w for _, power_w, _ in numeric_rows]

    estimate: Dict[str, float] = {
        "sensing_power_w": mean_finite(active_values) or 0.0,
        "sensing_idle_power_w": mean_finite(idle_values) if idle_values else 0.0,
        "sensing_power_uncertainty_w": std_finite(active_values) or 0.0,
        "sensing_idle_power_uncertainty_w": std_finite(idle_values) or 0.0,
        "compute_window_s": 0.0,
        "compute_power_w": 0.0,
        "compute_idle_power_w": 0.0,
    }

    start_times = [
        time_s
        for row, _, time_s in numeric_rows
        if time_s is not None and marker_contains(row, "start")
    ]
    end_times = [
        time_s
        for row, _, time_s in numeric_rows
        if time_s is not None and marker_contains(row, "end", "stop")
    ]
    if start_times and end_times:
        start_s = min(start_times)
        end_s = min(time_s for time_s in end_times if time_s > start_s) if any(time_s > start_s for time_s in end_times) else None
        if end_s is not None:
            estimate["sensing_window_s"] = end_s - start_s
    elif all(time_s is not None for _, _, time_s in numeric_rows):
        times = [float(time_s) for _, _, time_s in numeric_rows if time_s is not None]
        estimate["sensing_window_s"] = max(times) - min(times)

    return estimate


def load_power_log_estimate(power_log_path: str, trial_id: str) -> Dict[str, object]:
    with open(power_log_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {"values": {}, "source": power_log_path}
        reader.fieldnames = [(field or "").lstrip("\ufeff") for field in reader.fieldnames]
        rows = [{(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in reader]

    if not rows:
        return {"values": {}, "source": power_log_path}

    fieldnames = set(rows[0].keys())
    if "power_w" in fieldnames:
        values = estimate_from_power_samples(rows)
    else:
        selected = choose_power_log_row(rows, trial_id)
        values = estimate_from_power_summary(selected or {})
    return {"values": values, "source": power_log_path}


def parse_float_with_estimate(
    row: Dict[str, str],
    estimate: Dict[str, float],
    column: str,
    row_number: int,
    default_value: Optional[float] = None,
) -> float:
    raw = row.get(column, "").strip()
    if raw:
        return parse_required_float(row, column, row_number)
    value = estimate.get(column)
    if value is not None and math.isfinite(value):
        return float(value)
    if default_value is not None:
        return float(default_value)
    raise ValueError(
        f"Row {row_number}: column '{column}' is empty and could not be inferred from power_log_file."
    )


def parse_uncertainty(
    row: Dict[str, str],
    estimate: Dict[str, float],
    column: str,
    manual_uncertainty_w: float,
    use_manual_default: bool,
) -> float:
    value = parse_optional_float(row, column)
    if math.isfinite(value):
        return value
    value = estimate.get(column)
    if value is not None and math.isfinite(value):
        return max(0.0, float(value))
    if use_manual_default:
        return max(0.0, manual_uncertainty_w)
    return 0.0


def is_manual_power_source(power_log_file: str) -> bool:
    lower = power_log_file.lower()
    return any(token in lower for token in MANUAL_POWER_LOG_TOKENS)


def load_trials(manifest_path: str, manual_power_uncertainty_w: float) -> List[EnergyTrial]:
    repo_root = repo_root_from_this_file(__file__)
    manifest_dir = os.path.abspath(os.path.dirname(manifest_path))
    trials: List[EnergyTrial] = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV is missing a header row.")
        reader.fieldnames = [(field or "").lstrip("\ufeff") for field in reader.fieldnames]
        ensure_manifest_header(reader.fieldnames)

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
            bits_transmitted = parse_required_int(row, "bits_transmitted", row_number)
            bit_errors = parse_required_int(row, "bit_errors", row_number)
            bits_scored = parse_optional_int(row, "bits_scored", bits_transmitted)
            power_log_file = row.get("power_log_file", "").strip()
            power_estimate: Dict[str, float] = {}
            power_source = "manifest"
            if power_log_file:
                resolved_power_log = resolve_power_log_path(repo_root, manifest_dir, power_log_file)
                if resolved_power_log is not None:
                    loaded_power = load_power_log_estimate(resolved_power_log, trial_id)
                    power_estimate = dict(loaded_power["values"])
                    power_source = f"power_log:{os.path.relpath(str(loaded_power['source']), repo_root)}"
                elif is_manual_power_source(power_log_file):
                    power_source = power_log_file
                elif power_log_file.lower().endswith(".csv"):
                    print(f"Warning: row {row_number} power_log_file was not found: {power_log_file}")

            sensing_window_s = parse_float_with_estimate(row, power_estimate, "sensing_window_s", row_number)
            sensing_power_w = parse_float_with_estimate(row, power_estimate, "sensing_power_w", row_number)
            sensing_idle_power_w = parse_float_with_estimate(
                row,
                power_estimate,
                "sensing_idle_power_w",
                row_number,
                default_value=0.0,
            )
            compute_window_s = parse_float_with_estimate(
                row,
                power_estimate,
                "compute_window_s",
                row_number,
                default_value=0.0,
            )
            compute_power_w = parse_float_with_estimate(
                row,
                power_estimate,
                "compute_power_w",
                row_number,
                default_value=0.0,
            )
            compute_idle_power_w = parse_float_with_estimate(
                row,
                power_estimate,
                "compute_idle_power_w",
                row_number,
                default_value=0.0,
            )
            use_manual_uncertainty = is_manual_power_source(power_log_file)
            sensing_power_uncertainty_w = parse_uncertainty(
                row,
                power_estimate,
                "sensing_power_uncertainty_w",
                manual_power_uncertainty_w,
                use_manual_uncertainty,
            )
            sensing_idle_power_uncertainty_w = parse_uncertainty(
                row,
                power_estimate,
                "sensing_idle_power_uncertainty_w",
                manual_power_uncertainty_w,
                use_manual_uncertainty,
            )
            compute_power_uncertainty_w = parse_uncertainty(
                row,
                power_estimate,
                "compute_power_uncertainty_w",
                manual_power_uncertainty_w,
                use_manual_uncertainty,
            )
            compute_idle_power_uncertainty_w = parse_uncertainty(
                row,
                power_estimate,
                "compute_idle_power_uncertainty_w",
                manual_power_uncertainty_w,
                use_manual_uncertainty,
            )
            if compute_window_s == 0 and compute_power_w == 0 and compute_idle_power_w == 0:
                compute_power_uncertainty_w = 0.0
                compute_idle_power_uncertainty_w = 0.0
            lux = parse_optional_float(row, "lux")
            distance_cm = parse_optional_float(row, "distance_cm")

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
            if sensing_window_s < 0 or compute_window_s < 0:
                raise ValueError(f"Row {row_number}: time windows must be >= 0.")
            if sensing_power_w < 0 or sensing_idle_power_w < 0 or compute_power_w < 0 or compute_idle_power_w < 0:
                raise ValueError(f"Row {row_number}: power values must be >= 0.")

            score_fraction = safe_divide(bits_scored, bits_transmitted)
            ber = safe_divide(bit_errors, bits_scored)
            correct_bits = bits_scored - bit_errors

            sensing_active_power_w = max(0.0, sensing_power_w - sensing_idle_power_w)
            compute_active_power_w = max(0.0, compute_power_w - compute_idle_power_w)
            sensing_active_power_uncertainty_w = combine_uncertainties(
                sensing_power_uncertainty_w,
                sensing_idle_power_uncertainty_w,
            )
            compute_active_power_uncertainty_w = combine_uncertainties(
                compute_power_uncertainty_w,
                compute_idle_power_uncertainty_w,
            )
            sensing_energy_j_gross = sensing_power_w * sensing_window_s
            sensing_energy_j_active = sensing_active_power_w * sensing_window_s
            compute_energy_j_gross = compute_power_w * compute_window_s
            compute_energy_j_active = compute_active_power_w * compute_window_s
            total_energy_j_gross = sensing_energy_j_gross + compute_energy_j_gross
            total_energy_j_active = sensing_energy_j_active + compute_energy_j_active
            sensing_energy_j_gross_uncertainty = sensing_power_uncertainty_w * sensing_window_s
            sensing_energy_j_active_uncertainty = sensing_active_power_uncertainty_w * sensing_window_s
            compute_energy_j_gross_uncertainty = compute_power_uncertainty_w * compute_window_s
            compute_energy_j_active_uncertainty = compute_active_power_uncertainty_w * compute_window_s
            total_energy_j_gross_uncertainty = combine_uncertainties(
                sensing_energy_j_gross_uncertainty,
                compute_energy_j_gross_uncertainty,
            )
            total_energy_j_active_uncertainty = combine_uncertainties(
                sensing_energy_j_active_uncertainty,
                compute_energy_j_active_uncertainty,
            )

            trials.append(
                EnergyTrial(
                    trial_id=trial_id,
                    sensor_type=sensor_type,
                    modulation=modulation,
                    frequency_hz=frequency_hz,
                    lux=lux,
                    distance_cm=distance_cm,
                    bits_transmitted=bits_transmitted,
                    bits_scored=bits_scored,
                    bit_errors=bit_errors,
                    sensing_window_s=sensing_window_s,
                    sensing_power_w=sensing_power_w,
                    sensing_idle_power_w=sensing_idle_power_w,
                    compute_window_s=compute_window_s,
                    compute_power_w=compute_power_w,
                    compute_idle_power_w=compute_idle_power_w,
                    sensing_power_uncertainty_w=sensing_power_uncertainty_w,
                    sensing_idle_power_uncertainty_w=sensing_idle_power_uncertainty_w,
                    compute_power_uncertainty_w=compute_power_uncertainty_w,
                    compute_idle_power_uncertainty_w=compute_idle_power_uncertainty_w,
                    capture_file=row.get("capture_file", "").strip(),
                    power_log_file=power_log_file,
                    power_source=power_source,
                    decode_log_file=row.get("decode_log_file", "").strip(),
                    notes=row.get("notes", "").strip(),
                    score_fraction=score_fraction,
                    ber=ber,
                    correct_bits=correct_bits,
                    sensing_active_power_w=sensing_active_power_w,
                    compute_active_power_w=compute_active_power_w,
                    sensing_active_power_uncertainty_w=sensing_active_power_uncertainty_w,
                    compute_active_power_uncertainty_w=compute_active_power_uncertainty_w,
                    sensing_energy_j_gross=sensing_energy_j_gross,
                    sensing_energy_j_active=sensing_energy_j_active,
                    compute_energy_j_gross=compute_energy_j_gross,
                    compute_energy_j_active=compute_energy_j_active,
                    total_energy_j_gross=total_energy_j_gross,
                    total_energy_j_active=total_energy_j_active,
                    sensing_energy_j_gross_uncertainty=sensing_energy_j_gross_uncertainty,
                    sensing_energy_j_active_uncertainty=sensing_energy_j_active_uncertainty,
                    compute_energy_j_gross_uncertainty=compute_energy_j_gross_uncertainty,
                    compute_energy_j_active_uncertainty=compute_energy_j_active_uncertainty,
                    total_energy_j_gross_uncertainty=total_energy_j_gross_uncertainty,
                    total_energy_j_active_uncertainty=total_energy_j_active_uncertainty,
                    sensing_active_j_per_tx_bit=safe_divide(sensing_energy_j_active, bits_transmitted),
                    compute_active_j_per_tx_bit=safe_divide(compute_energy_j_active, bits_transmitted),
                    gross_j_per_tx_bit=safe_divide(total_energy_j_gross, bits_transmitted),
                    active_j_per_tx_bit=safe_divide(total_energy_j_active, bits_transmitted),
                    gross_j_per_scored_bit=safe_divide(total_energy_j_gross, bits_scored),
                    active_j_per_scored_bit=safe_divide(total_energy_j_active, bits_scored),
                    gross_j_per_correct_bit=safe_divide(total_energy_j_gross, correct_bits),
                    active_j_per_correct_bit=safe_divide(total_energy_j_active, correct_bits),
                    gross_j_per_tx_bit_uncertainty=safe_divide(
                        total_energy_j_gross_uncertainty,
                        bits_transmitted,
                    ),
                    active_j_per_tx_bit_uncertainty=safe_divide(
                        total_energy_j_active_uncertainty,
                        bits_transmitted,
                    ),
                    gross_j_per_scored_bit_uncertainty=safe_divide(
                        total_energy_j_gross_uncertainty,
                        bits_scored,
                    ),
                    active_j_per_scored_bit_uncertainty=safe_divide(
                        total_energy_j_active_uncertainty,
                        bits_scored,
                    ),
                    gross_j_per_correct_bit_uncertainty=safe_divide(
                        total_energy_j_gross_uncertainty,
                        correct_bits,
                    ),
                    active_j_per_correct_bit_uncertainty=safe_divide(
                        total_energy_j_active_uncertainty,
                        correct_bits,
                    ),
                )
            )

    if not trials:
        raise ValueError(
            "No ready Section 3.2 trial rows were found. Fill in a planned row and set status to blank or 'ready' before running the analyzer."
        )
    return trials


def build_group_fields(x_field: str, series_fields: Sequence[str]) -> List[str]:
    ordered = []
    for field in list(series_fields) + [x_field]:
        if field not in ordered:
            ordered.append(field)
    return ordered


def group_key_for_trial(trial: EnergyTrial, group_fields: Sequence[str]) -> Tuple[object, ...]:
    return tuple(getattr(trial, field) for field in group_fields)


def aggregate_trials(
    trials: Sequence[EnergyTrial],
    x_field: str,
    series_fields: Sequence[str],
) -> List[Dict[str, object]]:
    group_fields = build_group_fields(x_field, series_fields)
    grouped: Dict[Tuple[object, ...], List[EnergyTrial]] = {}
    for trial in trials:
        grouped.setdefault(group_key_for_trial(trial, group_fields), []).append(trial)

    rows: List[Dict[str, object]] = []
    for key, group_trials in grouped.items():
        row: Dict[str, object] = {field: value for field, value in zip(group_fields, key)}

        total_bits_transmitted = sum(trial.bits_transmitted for trial in group_trials)
        total_bits_scored = sum(trial.bits_scored for trial in group_trials)
        total_bit_errors = sum(trial.bit_errors for trial in group_trials)
        total_correct_bits = sum(trial.correct_bits for trial in group_trials)
        total_sensing_energy_active = sum(trial.sensing_energy_j_active for trial in group_trials)
        total_compute_energy_active = sum(trial.compute_energy_j_active for trial in group_trials)
        total_energy_active = sum(trial.total_energy_j_active for trial in group_trials)
        total_energy_gross = sum(trial.total_energy_j_gross for trial in group_trials)
        total_energy_active_uncertainty = rss(trial.total_energy_j_active_uncertainty for trial in group_trials)
        total_energy_gross_uncertainty = rss(trial.total_energy_j_gross_uncertainty for trial in group_trials)
        active_j_per_tx_bit_uncertainty_rms = rms_or_zero(
            trial.active_j_per_tx_bit_uncertainty for trial in group_trials
        )
        active_j_per_correct_bit_uncertainty_rms = rms_or_zero(
            trial.active_j_per_correct_bit_uncertainty for trial in group_trials
        )
        gross_j_per_tx_bit_uncertainty_rms = rms_or_zero(
            trial.gross_j_per_tx_bit_uncertainty for trial in group_trials
        )

        row.update(
            {
                "n_trials": len(group_trials),
                "total_bits_transmitted": total_bits_transmitted,
                "total_bits_scored": total_bits_scored,
                "total_bit_errors": total_bit_errors,
                "total_correct_bits": total_correct_bits,
                "pooled_ber": safe_divide(total_bit_errors, total_bits_scored),
                "pooled_score_fraction": safe_divide(total_bits_scored, total_bits_transmitted),
                "pooled_sensing_active_j_per_tx_bit": safe_divide(total_sensing_energy_active, total_bits_transmitted),
                "pooled_compute_active_j_per_tx_bit": safe_divide(total_compute_energy_active, total_bits_transmitted),
                "pooled_active_j_per_tx_bit": safe_divide(total_energy_active, total_bits_transmitted),
                "pooled_gross_j_per_tx_bit": safe_divide(total_energy_gross, total_bits_transmitted),
                "pooled_active_j_per_correct_bit": safe_divide(total_energy_active, total_correct_bits),
                "pooled_active_j_per_tx_bit_uncertainty": safe_divide(
                    total_energy_active_uncertainty,
                    total_bits_transmitted,
                ),
                "pooled_gross_j_per_tx_bit_uncertainty": safe_divide(
                    total_energy_gross_uncertainty,
                    total_bits_transmitted,
                ),
                "pooled_active_j_per_correct_bit_uncertainty": safe_divide(
                    total_energy_active_uncertainty,
                    total_correct_bits,
                ),
                "mean_ber": mean_or_nan(trial.ber for trial in group_trials),
                "std_ber": std_or_zero(trial.ber for trial in group_trials),
                "mean_active_j_per_tx_bit": mean_or_nan(trial.active_j_per_tx_bit for trial in group_trials),
                "std_active_j_per_tx_bit": std_or_zero(trial.active_j_per_tx_bit for trial in group_trials),
                "mean_active_j_per_tx_bit_uncertainty": active_j_per_tx_bit_uncertainty_rms,
                "plot_active_j_per_tx_bit_yerr": combine_uncertainties(
                    std_or_zero(trial.active_j_per_tx_bit for trial in group_trials),
                    active_j_per_tx_bit_uncertainty_rms,
                ),
                "mean_gross_j_per_tx_bit_uncertainty": gross_j_per_tx_bit_uncertainty_rms,
                "mean_active_j_per_correct_bit": mean_or_nan(
                    trial.active_j_per_correct_bit for trial in group_trials
                ),
                "std_active_j_per_correct_bit": std_or_zero(
                    trial.active_j_per_correct_bit for trial in group_trials
                ),
                "mean_active_j_per_correct_bit_uncertainty": active_j_per_correct_bit_uncertainty_rms,
                "mean_sensing_active_j_per_tx_bit": mean_or_nan(
                    trial.sensing_active_j_per_tx_bit for trial in group_trials
                ),
                "mean_compute_active_j_per_tx_bit": mean_or_nan(
                    trial.compute_active_j_per_tx_bit for trial in group_trials
                ),
                "mean_score_fraction": mean_or_nan(trial.score_fraction for trial in group_trials),
                "std_score_fraction": std_or_zero(trial.score_fraction for trial in group_trials),
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


def write_trial_csv(path: str, trials: Sequence[EnergyTrial]) -> None:
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
        "pooled_sensing_active_j_per_tx_bit",
        "pooled_compute_active_j_per_tx_bit",
        "pooled_active_j_per_tx_bit",
        "pooled_gross_j_per_tx_bit",
        "pooled_active_j_per_correct_bit",
        "pooled_active_j_per_tx_bit_uncertainty",
        "pooled_gross_j_per_tx_bit_uncertainty",
        "pooled_active_j_per_correct_bit_uncertainty",
        "mean_ber",
        "std_ber",
        "mean_active_j_per_tx_bit",
        "std_active_j_per_tx_bit",
        "mean_active_j_per_tx_bit_uncertainty",
        "plot_active_j_per_tx_bit_yerr",
        "mean_gross_j_per_tx_bit_uncertainty",
        "mean_active_j_per_correct_bit",
        "std_active_j_per_correct_bit",
        "mean_active_j_per_correct_bit_uncertainty",
        "mean_sensing_active_j_per_tx_bit",
        "mean_compute_active_j_per_tx_bit",
        "mean_score_fraction",
        "std_score_fraction",
    ]
    write_csv(path, rows, header)


def plot_metric(
    summary_rows: Sequence[Dict[str, object]],
    x_field: str,
    series_fields: Sequence[str],
    y_field: str,
    yerr_field: Optional[str],
    out_path: str,
    title: str,
    ylabel: str,
) -> None:
    if not summary_rows:
        return

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


def main() -> None:
    repo_root = repo_root_from_this_file(__file__)
    default_manifest = os.path.join(repo_root, "data", "3.2", "section3_2_energy_manifest_template.csv")

    ap = argparse.ArgumentParser(
        description="Analyze Section 3.2 energy-per-bit trials from a manifest CSV."
    )
    ap.add_argument(
        "--manifest",
        default=default_manifest,
        help="Manifest CSV describing one row per Section 3.2 trial.",
    )
    ap.add_argument(
        "--out_prefix",
        default="s32_energy",
        help="Prefix for per-trial CSV, summary CSV, and plot filenames.",
    )
    ap.add_argument(
        "--x_field",
        default="frequency_hz",
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
        "--no_plot",
        action="store_true",
        help="Disable plot generation.",
    )
    ap.add_argument(
        "--manual_power_uncertainty_w",
        type=float,
        default=DEFAULT_MANUAL_POWER_UNCERTAINTY_W,
        help="Default +/- W uncertainty applied to manual USB-meter manifest readings when no uncertainty column is provided.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(args.manifest)
    if args.manual_power_uncertainty_w < 0:
        raise ValueError("--manual_power_uncertainty_w must be >= 0.")

    trials = load_trials(args.manifest, manual_power_uncertainty_w=args.manual_power_uncertainty_w)
    summary_rows = aggregate_trials(trials, x_field=args.x_field, series_fields=args.series_fields)

    data_dir = os.path.join(repo_root, "data", "3.2")
    plot_dir = os.path.join(repo_root, "plots", "3.2")
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
            f"pooled_active_j_per_tx_bit={row['pooled_active_j_per_tx_bit']:.6g}, "
            f"pooled_ber={row['pooled_ber']:.6g}, "
            f"n_trials={int(row['n_trials'])}"
        )

    if args.no_plot:
        return

    energy_plot = os.path.join(plot_dir, f"{args.out_prefix}_active_j_per_tx_bit_vs_{args.x_field}.png")
    ber_plot = os.path.join(plot_dir, f"{args.out_prefix}_ber_vs_{args.x_field}.png")

    plot_metric(
        summary_rows=summary_rows,
        x_field=args.x_field,
        series_fields=args.series_fields,
        y_field="mean_active_j_per_tx_bit",
        yerr_field="plot_active_j_per_tx_bit_yerr",
        out_path=energy_plot,
        title="Section 3.2 Active Energy per Transmitted Bit",
        ylabel="Active energy per transmitted bit (J/bit)",
    )
    plot_metric(
        summary_rows=summary_rows,
        x_field=args.x_field,
        series_fields=args.series_fields,
        y_field="mean_ber",
        yerr_field="std_ber",
        out_path=ber_plot,
        title="Section 3.2 Bit Error Rate",
        ylabel="BER",
    )

    print(f"Saved plot: {energy_plot}")
    print(f"Saved plot: {ber_plot}")


if __name__ == "__main__":
    main()
