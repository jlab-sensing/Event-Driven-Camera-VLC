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


SYMBOL_TO_BITS = {
    "1": "00",
    "2": "01",
    "3": "10",
    "4": "11",
}

REQUIRED_COLUMNS = [
    "trial_id",
    "sensor_type",
    "modulation",
    "csk_scheme",
    "symbol_rate_hz",
    "lux",
]

OPTIONAL_COLUMNS = [
    "status",
    "actual_symbol_rate_hz",
    "bit_rate_hz",
    "bits_per_symbol",
    "symbol_alphabet",
    "lighting_source",
    "visibility_condition",
    "distance_cm",
    "angle_deg",
    "truth_symbols",
    "decoded_symbols",
    "symbols_transmitted",
    "symbols_scored",
    "symbol_errors",
    "bits_transmitted",
    "bits_scored",
    "bit_errors",
    "capture_file",
    "symbol_file",
    "decode_log_file",
    "energy_j",
    "notes",
]

NUMERIC_X_FIELDS = ["symbol_rate_hz", "bit_rate_hz", "lux", "distance_cm", "angle_deg"]
DEFAULT_SERIES_FIELDS = ["sensor_type", "csk_scheme", "lighting_source"]


@dataclass
class CskTrial:
    trial_id: str
    sensor_type: str
    modulation: str
    csk_scheme: str
    symbol_rate_hz: float
    actual_symbol_rate_hz: float
    bit_rate_hz: float
    bits_per_symbol: float
    symbol_alphabet: str
    lux: float
    lighting_source: str
    visibility_condition: str
    distance_cm: float
    angle_deg: float
    truth_symbols: str
    decoded_symbols: str
    symbols_transmitted: int
    symbols_scored: int
    symbol_errors: int
    bits_transmitted: int
    bits_scored: int
    bit_errors: int
    capture_file: str
    symbol_file: str
    decode_log_file: str
    energy_j: float
    notes: str
    symbol_error_rate: float
    symbol_accuracy: float
    score_fraction: float
    bit_error_rate: float
    bit_accuracy: float
    energy_j_per_symbol: float
    energy_j_per_bit: float
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


def parse_optional_int(row: Dict[str, str], column: str, default_value: Optional[int] = None) -> Optional[int]:
    raw = row.get(column, "").strip()
    if not raw:
        return default_value
    value = float(raw)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"Column '{column}' must be an integer value when provided.")
    return int(value)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


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
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")


def maybe_warn_extra_columns(fieldnames: Sequence[str]) -> None:
    known = set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    extras = [field for field in fieldnames if field not in known]
    if extras:
        print(f"Warning: ignoring unknown manifest columns: {', '.join(extras)}")


def clean_symbols(raw: str, alphabet: str) -> str:
    allowed = set(alphabet)
    return "".join(ch for ch in raw.strip() if ch in allowed)


def infer_bits_per_symbol(alphabet: str) -> float:
    if set(alphabet).issubset(set(SYMBOL_TO_BITS)):
        return 2.0
    if len(alphabet) > 1:
        candidate = math.log2(len(alphabet))
        if abs(candidate - round(candidate)) < 1e-9:
            return float(round(candidate))
    return float("nan")


def symbols_to_bits(symbols: str) -> Optional[str]:
    bits = []
    for symbol in symbols:
        if symbol not in SYMBOL_TO_BITS:
            return None
        bits.append(SYMBOL_TO_BITS[symbol])
    return "".join(bits)


def compare_symbol_sequences(truth_symbols: str, decoded_symbols: str) -> Tuple[int, int]:
    scored = min(len(truth_symbols), len(decoded_symbols))
    errors = sum(1 for truth, decoded in zip(truth_symbols[:scored], decoded_symbols[:scored]) if truth != decoded)
    return scored, errors


def compare_bit_sequences(truth_symbols: str, decoded_symbols: str) -> Tuple[Optional[int], Optional[int]]:
    scored_symbols = min(len(truth_symbols), len(decoded_symbols))
    truth_bits = symbols_to_bits(truth_symbols[:scored_symbols])
    decoded_bits = symbols_to_bits(decoded_symbols[:scored_symbols])
    if truth_bits is None or decoded_bits is None:
        return None, None
    errors = sum(1 for truth, decoded in zip(truth_bits, decoded_bits) if truth != decoded)
    return len(truth_bits), errors


def normalize_counts(
    row: Dict[str, str],
    row_number: int,
    alphabet: str,
    bits_per_symbol: float,
) -> Dict[str, object]:
    truth_symbols = clean_symbols(row.get("truth_symbols", ""), alphabet)
    decoded_symbols = clean_symbols(row.get("decoded_symbols", ""), alphabet)
    have_sequences = bool(truth_symbols and decoded_symbols)

    symbols_transmitted = parse_optional_int(row, "symbols_transmitted")
    symbols_scored = parse_optional_int(row, "symbols_scored")
    symbol_errors = parse_optional_int(row, "symbol_errors")

    if have_sequences:
        inferred_scored, inferred_errors = compare_symbol_sequences(truth_symbols, decoded_symbols)
        symbols_scored = symbols_scored if symbols_scored is not None else inferred_scored
        symbol_errors = symbol_errors if symbol_errors is not None else inferred_errors
        symbols_transmitted = symbols_transmitted if symbols_transmitted is not None else len(truth_symbols)

    if symbols_transmitted is None or symbols_scored is None or symbol_errors is None:
        raise ValueError(
            f"Row {row_number}: provide symbols_transmitted/symbols_scored/symbol_errors "
            "or provide truth_symbols and decoded_symbols."
        )

    if symbols_transmitted <= 0:
        raise ValueError(f"Row {row_number}: symbols_transmitted must be > 0.")
    if symbols_scored <= 0:
        raise ValueError(f"Row {row_number}: symbols_scored must be > 0.")
    if symbols_scored > symbols_transmitted:
        raise ValueError(f"Row {row_number}: symbols_scored cannot exceed symbols_transmitted.")
    if symbol_errors < 0:
        raise ValueError(f"Row {row_number}: symbol_errors must be >= 0.")
    if symbol_errors > symbols_scored:
        raise ValueError(f"Row {row_number}: symbol_errors cannot exceed symbols_scored.")

    bits_transmitted = parse_optional_int(row, "bits_transmitted")
    bits_scored = parse_optional_int(row, "bits_scored")
    bit_errors = parse_optional_int(row, "bit_errors")

    if have_sequences:
        inferred_bits_scored, inferred_bit_errors = compare_bit_sequences(truth_symbols, decoded_symbols)
        if inferred_bits_scored is not None and inferred_bit_errors is not None:
            bits_scored = bits_scored if bits_scored is not None else inferred_bits_scored
            bit_errors = bit_errors if bit_errors is not None else inferred_bit_errors

    if bits_transmitted is None and math.isfinite(bits_per_symbol):
        bits_transmitted = int(round(symbols_transmitted * bits_per_symbol))

    if bits_scored is None and math.isfinite(bits_per_symbol):
        bits_scored = int(round(symbols_scored * bits_per_symbol))

    if bit_errors is None:
        bit_errors = 0 if bits_scored == 0 else None

    if bits_transmitted is None:
        bits_transmitted = 0
    if bits_scored is None:
        bits_scored = 0
    if bit_errors is None:
        bit_errors = 0

    if bits_scored < 0 or bit_errors < 0 or bits_transmitted < 0:
        raise ValueError(f"Row {row_number}: bit count fields must be >= 0 when provided.")
    if bits_scored > bits_transmitted and bits_transmitted > 0:
        raise ValueError(f"Row {row_number}: bits_scored cannot exceed bits_transmitted.")
    if bit_errors > bits_scored:
        raise ValueError(f"Row {row_number}: bit_errors cannot exceed bits_scored.")

    return {
        "truth_symbols": truth_symbols,
        "decoded_symbols": decoded_symbols,
        "symbols_transmitted": int(symbols_transmitted),
        "symbols_scored": int(symbols_scored),
        "symbol_errors": int(symbol_errors),
        "bits_transmitted": int(bits_transmitted),
        "bits_scored": int(bits_scored),
        "bit_errors": int(bit_errors),
    }


def load_trials(manifest_path: str) -> List[CskTrial]:
    trials: List[CskTrial] = []
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
            csk_scheme = require_text(row, "csk_scheme", row_number)
            symbol_rate_hz = parse_required_float(row, "symbol_rate_hz", row_number)
            actual_symbol_rate_hz = parse_optional_float(row, "actual_symbol_rate_hz")
            lux = parse_required_float(row, "lux", row_number)
            distance_cm = parse_optional_float(row, "distance_cm")
            angle_deg = parse_optional_float(row, "angle_deg")
            energy_j = parse_optional_float(row, "energy_j")
            alphabet = row.get("symbol_alphabet", "").strip() or "1234"
            bits_per_symbol = parse_optional_float(row, "bits_per_symbol")
            if not math.isfinite(bits_per_symbol):
                bits_per_symbol = infer_bits_per_symbol(alphabet)

            if symbol_rate_hz <= 0:
                raise ValueError(f"Row {row_number}: symbol_rate_hz must be > 0.")
            if bits_per_symbol < 0 and math.isfinite(bits_per_symbol):
                raise ValueError(f"Row {row_number}: bits_per_symbol must be >= 0.")

            counts = normalize_counts(row, row_number, alphabet, bits_per_symbol)
            bit_rate_hz = parse_optional_float(row, "bit_rate_hz")
            if not math.isfinite(bit_rate_hz) and math.isfinite(bits_per_symbol):
                bit_rate_hz = symbol_rate_hz * bits_per_symbol

            symbol_error_rate = safe_divide(counts["symbol_errors"], counts["symbols_scored"])
            symbol_accuracy = 1.0 - symbol_error_rate if math.isfinite(symbol_error_rate) else float("nan")
            score_fraction = safe_divide(counts["symbols_scored"], counts["symbols_transmitted"])
            bit_error_rate = safe_divide(counts["bit_errors"], counts["bits_scored"])
            bit_accuracy = 1.0 - bit_error_rate if math.isfinite(bit_error_rate) else float("nan")
            energy_j_per_symbol = safe_divide(energy_j, counts["symbols_transmitted"]) if math.isfinite(energy_j) else float("nan")
            energy_j_per_bit = safe_divide(energy_j, counts["bits_transmitted"]) if math.isfinite(energy_j) else float("nan")
            frequency_error_fraction = (
                safe_divide(abs(actual_symbol_rate_hz - symbol_rate_hz), symbol_rate_hz)
                if math.isfinite(actual_symbol_rate_hz)
                else float("nan")
            )

            trials.append(
                CskTrial(
                    trial_id=trial_id,
                    sensor_type=sensor_type,
                    modulation=modulation,
                    csk_scheme=csk_scheme,
                    symbol_rate_hz=symbol_rate_hz,
                    actual_symbol_rate_hz=actual_symbol_rate_hz,
                    bit_rate_hz=bit_rate_hz,
                    bits_per_symbol=bits_per_symbol,
                    symbol_alphabet=alphabet,
                    lux=lux,
                    lighting_source=row.get("lighting_source", "").strip(),
                    visibility_condition=row.get("visibility_condition", "").strip(),
                    distance_cm=distance_cm,
                    angle_deg=angle_deg,
                    truth_symbols=str(counts["truth_symbols"]),
                    decoded_symbols=str(counts["decoded_symbols"]),
                    symbols_transmitted=int(counts["symbols_transmitted"]),
                    symbols_scored=int(counts["symbols_scored"]),
                    symbol_errors=int(counts["symbol_errors"]),
                    bits_transmitted=int(counts["bits_transmitted"]),
                    bits_scored=int(counts["bits_scored"]),
                    bit_errors=int(counts["bit_errors"]),
                    capture_file=row.get("capture_file", "").strip(),
                    symbol_file=row.get("symbol_file", "").strip(),
                    decode_log_file=row.get("decode_log_file", "").strip(),
                    energy_j=energy_j,
                    notes=row.get("notes", "").strip(),
                    symbol_error_rate=symbol_error_rate,
                    symbol_accuracy=symbol_accuracy,
                    score_fraction=score_fraction,
                    bit_error_rate=bit_error_rate,
                    bit_accuracy=bit_accuracy,
                    energy_j_per_symbol=energy_j_per_symbol,
                    energy_j_per_bit=energy_j_per_bit,
                    frequency_error_fraction=frequency_error_fraction,
                )
            )

    if not trials:
        raise ValueError(
            "No ready Section 3.4 CSK rows were found. Fill in a planned row and set status to blank or 'ready' before running the analyzer."
        )
    return trials


def build_group_fields(x_field: str, series_fields: Sequence[str]) -> List[str]:
    ordered = []
    for field in list(series_fields) + [x_field]:
        if field not in ordered:
            ordered.append(field)
    return ordered


def aggregate_trials(trials: Sequence[CskTrial], x_field: str, series_fields: Sequence[str]) -> List[Dict[str, object]]:
    group_fields = build_group_fields(x_field, series_fields)
    grouped: Dict[Tuple[object, ...], List[CskTrial]] = {}
    for trial in trials:
        grouped.setdefault(tuple(getattr(trial, field) for field in group_fields), []).append(trial)

    rows: List[Dict[str, object]] = []
    for key, group_trials in grouped.items():
        row: Dict[str, object] = {field: value for field, value in zip(group_fields, key)}
        total_symbols_transmitted = sum(trial.symbols_transmitted for trial in group_trials)
        total_symbols_scored = sum(trial.symbols_scored for trial in group_trials)
        total_symbol_errors = sum(trial.symbol_errors for trial in group_trials)
        total_bits_transmitted = sum(trial.bits_transmitted for trial in group_trials)
        total_bits_scored = sum(trial.bits_scored for trial in group_trials)
        total_bit_errors = sum(trial.bit_errors for trial in group_trials)
        total_energy_j = sum(trial.energy_j for trial in group_trials if math.isfinite(trial.energy_j))

        pooled_ser = safe_divide(total_symbol_errors, total_symbols_scored)
        pooled_ber = safe_divide(total_bit_errors, total_bits_scored)
        row.update(
            {
                "n_trials": len(group_trials),
                "total_symbols_transmitted": total_symbols_transmitted,
                "total_symbols_scored": total_symbols_scored,
                "total_symbol_errors": total_symbol_errors,
                "pooled_symbol_error_rate": pooled_ser,
                "pooled_symbol_accuracy": 1.0 - pooled_ser if math.isfinite(pooled_ser) else float("nan"),
                "total_bits_transmitted": total_bits_transmitted,
                "total_bits_scored": total_bits_scored,
                "total_bit_errors": total_bit_errors,
                "pooled_bit_error_rate": pooled_ber,
                "pooled_bit_accuracy": 1.0 - pooled_ber if math.isfinite(pooled_ber) else float("nan"),
                "mean_symbol_error_rate": mean_or_nan(trial.symbol_error_rate for trial in group_trials),
                "std_symbol_error_rate": std_or_zero(trial.symbol_error_rate for trial in group_trials),
                "mean_bit_error_rate": mean_or_nan(trial.bit_error_rate for trial in group_trials),
                "std_bit_error_rate": std_or_zero(trial.bit_error_rate for trial in group_trials),
                "mean_score_fraction": mean_or_nan(trial.score_fraction for trial in group_trials),
                "mean_actual_symbol_rate_hz": mean_or_nan(trial.actual_symbol_rate_hz for trial in group_trials),
                "mean_frequency_error_fraction": mean_or_nan(trial.frequency_error_fraction for trial in group_trials),
                "total_energy_j": total_energy_j if total_energy_j > 0 else float("nan"),
                "mean_energy_j_per_symbol": mean_or_nan(trial.energy_j_per_symbol for trial in group_trials),
                "mean_energy_j_per_bit": mean_or_nan(trial.energy_j_per_bit for trial in group_trials),
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


def write_trial_csv(path: str, trials: Sequence[CskTrial]) -> None:
    rows = [asdict(trial) for trial in trials]
    write_csv(path, rows, list(rows[0].keys()))


def write_summary_csv(path: str, rows: Sequence[Dict[str, object]], x_field: str, series_fields: Sequence[str]) -> None:
    header = build_group_fields(x_field, series_fields) + [
        "n_trials",
        "total_symbols_transmitted",
        "total_symbols_scored",
        "total_symbol_errors",
        "pooled_symbol_error_rate",
        "pooled_symbol_accuracy",
        "total_bits_transmitted",
        "total_bits_scored",
        "total_bit_errors",
        "pooled_bit_error_rate",
        "pooled_bit_accuracy",
        "mean_symbol_error_rate",
        "std_symbol_error_rate",
        "mean_bit_error_rate",
        "std_bit_error_rate",
        "mean_score_fraction",
        "mean_actual_symbol_rate_hz",
        "mean_frequency_error_fraction",
        "total_energy_j",
        "mean_energy_j_per_symbol",
        "mean_energy_j_per_bit",
    ]
    write_csv(path, rows, header)


def write_confusion_csv(path: str, trials: Sequence[CskTrial]) -> bool:
    counts: Dict[Tuple[str, str], int] = {}
    alphabet = set()
    for trial in trials:
        if not trial.truth_symbols or not trial.decoded_symbols:
            continue
        scored = min(len(trial.truth_symbols), len(trial.decoded_symbols))
        alphabet.update(trial.symbol_alphabet)
        for truth, decoded in zip(trial.truth_symbols[:scored], trial.decoded_symbols[:scored]):
            counts[(truth, decoded)] = counts.get((truth, decoded), 0) + 1

    if not counts:
        return False

    labels = sorted(alphabet)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth_symbol"] + [f"decoded_{label}" for label in labels])
        for truth in labels:
            writer.writerow([truth] + [counts.get((truth, decoded), 0) for decoded in labels])
    return True


def format_series_part(field: str, value: object) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if field in {"sensor_type", "csk_scheme", "lighting_source", "visibility_condition", "modulation"}:
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
    return " | ".join(parts) if parts else "all trials"


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
        grouped.setdefault(build_series_label(row, series_fields), []).append(row)

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
            yerr = [float(item[yerr_field]) if math.isfinite(float(item[yerr_field])) else 0.0 for item in usable]
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
        "visibility_condition",
        "symbol_alphabet",
    ])
    invalid = [field for field in fields if field not in allowed]
    if invalid:
        raise ValueError(f"Unsupported grouping field(s): {', '.join(invalid)}")


def main() -> None:
    repo_root = repo_root_from_this_file(__file__)
    default_manifest = os.path.join(repo_root, "data", "3.4", "section3_4_csk_manifest_template.csv")

    ap = argparse.ArgumentParser(description="Analyze Section 3.4 CSK trial metrics from a manifest CSV.")
    ap.add_argument("--manifest", default=default_manifest, help="Manifest CSV describing one row per CSK trial.")
    ap.add_argument("--out_prefix", default="s34_csk", help="Prefix for output CSVs and plots.")
    ap.add_argument(
        "--x_field",
        default="symbol_rate_hz",
        choices=NUMERIC_X_FIELDS,
        help="Numeric field used on the x-axis for plots and grouping.",
    )
    ap.add_argument(
        "--series_fields",
        nargs="*",
        default=DEFAULT_SERIES_FIELDS,
        help="Fields that define separate plotted series and pooled summary rows.",
    )
    ap.add_argument("--no_plot", action="store_true", help="Disable plot generation.")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(args.manifest)
    validate_group_fields([args.x_field] + list(args.series_fields))

    trials = load_trials(args.manifest)
    summary_rows = aggregate_trials(trials, x_field=args.x_field, series_fields=args.series_fields)

    data_dir = os.path.join(repo_root, "data", "3.4")
    plot_dir = os.path.join(repo_root, "plots", "3.4")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    per_trial_csv = os.path.join(data_dir, f"{args.out_prefix}_per_trial.csv")
    summary_csv = os.path.join(data_dir, f"{args.out_prefix}_summary.csv")
    confusion_csv = os.path.join(data_dir, f"{args.out_prefix}_confusion_matrix.csv")
    write_trial_csv(per_trial_csv, trials)
    write_summary_csv(summary_csv, summary_rows, x_field=args.x_field, series_fields=args.series_fields)
    confusion_saved = write_confusion_csv(confusion_csv, trials)

    print(f"Saved per-trial CSV: {per_trial_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    if confusion_saved:
        print(f"Saved confusion matrix CSV: {confusion_csv}")
    print("")

    for row in summary_rows:
        label = build_series_label(row, args.series_fields)
        print(
            f"{label}, {args.x_field}={format_value(row[args.x_field])}: "
            f"pooled_SER={row['pooled_symbol_error_rate']:.6g}, "
            f"pooled_BER={row['pooled_bit_error_rate']:.6g}, "
            f"n_trials={int(row['n_trials'])}"
        )

    if args.no_plot:
        return

    plot_specs = [
        (
            "mean_symbol_error_rate",
            "std_symbol_error_rate",
            f"{args.out_prefix}_ser_vs_{args.x_field}.png",
            "Section 3.4 CSK Symbol Error Rate",
            "SER",
        ),
        (
            "mean_bit_error_rate",
            "std_bit_error_rate",
            f"{args.out_prefix}_ber_vs_{args.x_field}.png",
            "Section 3.4 CSK Bit Error Rate",
            "BER",
        ),
        (
            "mean_energy_j_per_bit",
            None,
            f"{args.out_prefix}_energy_j_per_bit_vs_{args.x_field}.png",
            "Section 3.4 CSK Energy per Bit",
            "energy per bit (J/bit)",
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
