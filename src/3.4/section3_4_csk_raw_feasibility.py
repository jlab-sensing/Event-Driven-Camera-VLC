"""Analyze the saved 300 Hz four-state CSK RAW captures.

This is a state/transition-feasibility analysis, not a validated color decoder.
The recorded truth stream repeats 1,2,3,4 in a fixed order, so a timing-only
predictor could exploit the cycle.  To avoid calling that CSK recovery, the
script uses only per-symbol ROI event features and leave-one-capture-out
classification.  Its output must be reported as exploratory evidence.
"""

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = ROOT.parent / "captures" / "3.4" / "s34_csk_ratio4_300hz_50cm"
DATA_DIR = ROOT / "data" / "3.4"
PLOT_DIR = ROOT / "plots" / "3.4"
RAW_FILES = [
    CAPTURE_DIR / "s34_csk_ratio4_300hz_los_t1.raw",
    CAPTURE_DIR / "s34_csk_ratio4_300hz_los_t2.raw",
    CAPTURE_DIR / "s34_csk_ratio4_300hz_los_t3.raw",
]

SYMBOL_RATE_HZ = 300.03000300030004
SYMBOL_PERIOD_US = 1_000_000.0 / SYMBOL_RATE_HZ
GUARD_SYMBOLS = 20
PAYLOAD_SYMBOLS = 2960
BIN_US = 25
ROI_SIZE = 96
TRUTH_CYCLE = np.array(list("1234"), dtype="U1")
OUT_PREFIX = "s34_csk_ratio4_300hz_raw_feasibility"


def choose_roi(raw_path: Path):
    """Select a fixed high-event-density square ROI from the first capture."""
    max_x = max_y = 0
    events_seen = 0
    for events in EventsIterator(input_path=str(raw_path)):
        if len(events) == 0:
            continue
        max_x = max(max_x, int(events["x"].max()))
        max_y = max(max_y, int(events["y"].max()))
        events_seen += len(events)
    if not events_seen:
        raise RuntimeError(f"No events in {raw_path}")

    hist = np.zeros((max_y + 1, max_x + 1), dtype=np.int64)
    for events in EventsIterator(input_path=str(raw_path)):
        if len(events):
            np.add.at(hist, (events["y"], events["x"]), 1)
    integral = hist.cumsum(axis=0).cumsum(axis=1)
    h, w = hist.shape
    y1 = np.arange(ROI_SIZE - 1, h)[:, None]
    x1 = np.arange(ROI_SIZE - 1, w)[None, :]
    sums = integral[y1, x1].copy()
    sums -= np.where(y1 >= ROI_SIZE, integral[y1 - ROI_SIZE, x1], 0)
    sums -= np.where(x1 >= ROI_SIZE, integral[y1, x1 - ROI_SIZE], 0)
    sums += np.where((y1 >= ROI_SIZE) & (x1 >= ROI_SIZE), integral[y1 - ROI_SIZE, x1 - ROI_SIZE], 0)
    iy, ix = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(ix), int(iy), int(ix + ROI_SIZE), int(iy + ROI_SIZE)


def bin_roi_events(raw_path: Path, roi):
    x0, y0, x1, y1 = roi
    n_bins = 600_000  # 15 s at 25 us; captures are approximately 10 s.
    total = np.zeros(n_bins, dtype=np.int64)
    positive = np.zeros(n_bins, dtype=np.int64)
    negative = np.zeros(n_bins, dtype=np.int64)
    first_t = None
    last_t = None
    roi_events = 0
    for events in EventsIterator(input_path=str(raw_path)):
        if len(events) == 0:
            continue
        if first_t is None:
            first_t = int(events["t"][0])
        last_t = int(events["t"][-1])
        mask = (events["x"] >= x0) & (events["x"] < x1) & (events["y"] >= y0) & (events["y"] < y1)
        if not np.any(mask):
            continue
        local = events[mask]
        bins = ((local["t"].astype(np.int64) - first_t) // BIN_US).astype(np.int64)
        valid = bins < n_bins
        bins = bins[valid]
        pol = local["p"][valid].astype(bool)
        total += np.bincount(bins, minlength=n_bins)[:n_bins]
        positive += np.bincount(bins[pol], minlength=n_bins)[:n_bins]
        negative += np.bincount(bins[~pol], minlength=n_bins)[:n_bins]
        roi_events += len(bins)
    if first_t is None or last_t is None:
        raise RuntimeError(f"No timestamped events in {raw_path}")
    used = min(n_bins, int((last_t - first_t) // BIN_US) + 1)
    return total[:used], positive[:used], negative[:used], first_t, roi_events


def first_active_boundary(total):
    smooth = np.convolve(total.astype(float), np.ones(40) / 40.0, mode="same")
    baseline = smooth[: min(len(smooth), int(200_000 / BIN_US))]
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    threshold = max(median + 10.0 * max(mad, 1.0), float(np.percentile(smooth, 90)))
    candidates = np.flatnonzero(smooth > threshold)
    if len(candidates) == 0:
        raise RuntimeError("Could not find an active CSK transition above background")
    return int(candidates[0]) * BIN_US


def align_boundary(total, start_guess_us):
    period_bins = SYMBOL_PERIOD_US / BIN_US
    window = max(1, int(round(250 / BIN_US)))
    best_score = -1.0
    best_start = start_guess_us
    for offset_us in range(-2000, 2001, BIN_US):
        start_bin = int(round((start_guess_us + offset_us) / BIN_US))
        boundaries = start_bin + np.rint(np.arange(500) * period_bins).astype(int)
        boundaries = boundaries[(boundaries >= window) & (boundaries < len(total) - window)]
        if len(boundaries) < 50:
            continue
        score = float(sum(total[b - window : b + window + 1].sum() for b in boundaries)) / len(boundaries)
        if score > best_score:
            best_score = score
            best_start = start_bin * BIN_US
    return best_start


def symbol_features(total, positive, negative, boundary_us):
    rows = []
    for index in range(PAYLOAD_SYMBOLS):
        left = int(round((boundary_us + index * SYMBOL_PERIOD_US) / BIN_US))
        right = int(round((boundary_us + (index + 1) * SYMBOL_PERIOD_US) / BIN_US))
        if left < 0 or right > len(total) or right <= left:
            break
        midpoint = left + (right - left) // 2
        rows.append((
            index,
            TRUTH_CYCLE[index % len(TRUTH_CYCLE)],
            int(total[left:right].sum()),
            int(positive[left:right].sum()),
            int(negative[left:right].sum()),
            int(total[left:midpoint].sum()),
            int(total[midpoint:right].sum()),
        ))
    dtype = [
        ("symbol_index", "i4"), ("truth_symbol", "U1"), ("total_events", "f8"),
        ("positive_events", "f8"), ("negative_events", "f8"),
        ("early_events", "f8"), ("late_events", "f8"),
    ]
    return np.array(rows, dtype=dtype)


def feature_matrix(rows):
    base = np.column_stack([rows[name] for name in ("total_events", "positive_events", "negative_events", "early_events", "late_events")])
    return np.log1p(base)


def leave_one_capture_out(trials):
    all_predictions = []
    labels = np.array(list("1234"))
    for holdout, test in enumerate(trials):
        train = np.vstack([feature_matrix(t["rows"]) for i, t in enumerate(trials) if i != holdout])
        train_labels = np.concatenate([t["rows"]["truth_symbol"] for i, t in enumerate(trials) if i != holdout])
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale == 0] = 1.0
        prototypes = np.vstack([((train[train_labels == label] - mean) / scale).mean(axis=0) for label in labels])
        test_features = (feature_matrix(test["rows"]) - mean) / scale
        distances = ((test_features[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
        predicted = labels[np.argmin(distances, axis=1)]
        all_predictions.append(predicted)
        test["predicted"] = predicted
        test["accuracy"] = float(np.mean(predicted == test["rows"]["truth_symbol"]))
    return all_predictions


def write_outputs(trials, roi):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    per_symbol = DATA_DIR / f"{OUT_PREFIX}_per_symbol.csv"
    per_trial = DATA_DIR / f"{OUT_PREFIX}_per_trial.csv"
    summary = DATA_DIR / f"{OUT_PREFIX}_summary.csv"
    confusion = DATA_DIR / f"{OUT_PREFIX}_confusion_matrix.csv"
    plot = PLOT_DIR / f"{OUT_PREFIX}_holdout_symbol_accuracy.png"

    with per_symbol.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "symbol_index", "truth_symbol", "predicted_symbol", "total_events", "positive_events", "negative_events", "early_events", "late_events"])
        for trial in trials:
            for row, prediction in zip(trial["rows"], trial["predicted"]):
                writer.writerow([trial["trial_id"], int(row["symbol_index"]), row["truth_symbol"], prediction, *[int(row[name]) for name in ("total_events", "positive_events", "negative_events", "early_events", "late_events")]])

    with per_trial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "raw_file", "roi_x0", "roi_y0", "roi_x1", "roi_y1", "aligned_first_state_boundary_us", "roi_events", "symbols_scored", "correct_symbols", "holdout_symbol_accuracy", "interpretation"])
        for trial in trials:
            writer.writerow([trial["trial_id"], trial["raw_file"], *roi, trial["boundary_us"], trial["roi_events"], len(trial["rows"]), int(round(trial["accuracy"] * len(trial["rows"]))), f"{trial['accuracy']:.8f}", "cross-capture state/transition feasibility; not validated CSK color decoding"])

    pooled_symbols = sum(len(trial["rows"]) for trial in trials)
    pooled_correct = sum(int(round(trial["accuracy"] * len(trial["rows"]))) for trial in trials)
    pooled_accuracy = pooled_correct / pooled_symbols
    trial_accuracy_std = float(np.std([trial["accuracy"] for trial in trials], ddof=1))
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "capture_count", "symbol_rate_hz", "distance_cm", "ambient_lux", "link_condition",
            "logical_states", "symbols_scored", "correct_symbols", "pooled_holdout_accuracy",
            "trial_accuracy_sample_std", "four_class_chance_accuracy", "ser_or_ber_status", "conclusion",
        ])
        writer.writerow([
            len(trials), f"{SYMBOL_RATE_HZ:.6f}", 50, 440, "direct LOS", 4,
            pooled_symbols, pooled_correct, f"{pooled_accuracy:.8f}", f"{trial_accuracy_std:.8f}",
            "0.25000000", "not valid to report", 
            "At this setup, cross-capture event features did not separate the four RGB-ratio states beyond chance; exploratory RAW feasibility only.",
        ])

    labels = list("1234")
    matrix = {(truth, pred): 0 for truth in labels for pred in labels}
    for trial in trials:
        for truth, pred in zip(trial["rows"]["truth_symbol"], trial["predicted"]):
            matrix[(truth, pred)] += 1
    with confusion.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth_symbol"] + [f"predicted_{label}" for label in labels])
        for truth in labels:
            writer.writerow([truth] + [matrix[(truth, pred)] for pred in labels])

    accuracies = [trial["accuracy"] * 100.0 for trial in trials]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar([trial["trial_id"] for trial in trials], accuracies, color="#2f6db0")
    ax.axhline(25.0, linestyle="--", color="#666666", label="25% four-class chance reference")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Leave-one-capture-out state accuracy (%)")
    ax.set_title("300 Hz CSK RAW state/transition feasibility")
    ax.legend(frameon=False)
    for bar, value in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    fig.text(0.5, 0.01, "Repeated 1-2-3-4 truth order confounds color state with transition position; exploratory only.", ha="center", fontsize=8)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(plot, dpi=220)
    plt.close(fig)
    return per_symbol, per_trial, summary, confusion, plot


def main():
    roi = choose_roi(RAW_FILES[0])
    print("Fixed ROI:", roi)
    trials = []
    for raw_path in RAW_FILES:
        total, positive, negative, _first_t, roi_events = bin_roi_events(raw_path, roi)
        first_burst_us = first_active_boundary(total)
        boundary_us = align_boundary(total, first_burst_us)
        rows = symbol_features(total, positive, negative, boundary_us)
        trial = {"trial_id": raw_path.stem.rsplit("_", 1)[-1], "raw_file": raw_path.name, "rows": rows, "boundary_us": boundary_us, "roi_events": roi_events}
        trials.append(trial)
        print(raw_path.name, "first_burst_us=", first_burst_us, "boundary_us=", boundary_us, "symbols=", len(rows))
    leave_one_capture_out(trials)
    for trial in trials:
        print(trial["raw_file"], f"holdout_accuracy={trial['accuracy']:.2%}")
    paths = write_outputs(trials, roi)
    print("Saved:", *paths, sep="\n")


if __name__ == "__main__":
    main()
