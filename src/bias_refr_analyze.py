import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator

# Reuse repo output conventions
from io_utils import repo_root_from_this_file


DEFAULT_INPUT_DIR = r"C:\Users\rabis\OneDrive\Documents\School\LAB aka 195\captures\testing_bias_refr"
DEFAULT_MAX_IEI_EVENTS = 5_000_000


# ----------------------------
# Loading events
# ----------------------------
def load_event_fields(raw_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t_us: timestamps in microseconds (int64)
      p: polarity (uint8) if present; else zeros
    """
    it = EventsIterator(input_path=raw_path)
    ts_chunks = []
    p_chunks = []
    for evs in it:
        if evs.size == 0:
            continue
        # Keep timestamps and polarity aligned chunk by chunk.
        ts_chunks.append(evs["t"].astype(np.int64))
        if "p" in evs.dtype.names:
            p_chunks.append(evs["p"].astype(np.uint8))
        else:
            p_chunks.append(np.zeros(evs.size, dtype=np.uint8))

    if not ts_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

    t_us = np.concatenate(ts_chunks)
    p = np.concatenate(p_chunks)
    return t_us, p


# ----------------------------
# Signal + peak finding
# ----------------------------
def binned_activity(time_s: np.ndarray, bin_width_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram event counts vs time. Returns (t_left_edges, counts)."""
    if time_s.size == 0:
        return np.array([]), np.array([])
    t0, t1 = float(time_s.min()), float(time_s.max())
    bins = np.arange(t0, t1 + bin_width_s, bin_width_s)
    counts, edges = np.histogram(time_s, bins=bins)
    return edges[:-1], counts


def find_peaks_simple(t: np.ndarray, y: np.ndarray, min_height: float, min_distance_s: float) -> np.ndarray:
    """
    Very simple peak finder:
    peak at i if y[i] > y[i-1] and y[i] >= y[i+1] and y[i] >= min_height.
    Enforces a minimum spacing between peaks.
    Returns peak times (seconds).
    """
    if t.size < 3:
        return np.array([])

    candidates = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height:
            candidates.append(i)

    if not candidates:
        return np.array([])

    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if (t[idx] - t[peaks[-1]]) >= min_distance_s:
            peaks.append(idx)

    return t[np.array(peaks, dtype=int)]


def robust_peak_threshold(counts: np.ndarray, k: float) -> float:
    """
    min_height = median + k * robust_sigma, where robust_sigma ~ 1.4826*MAD
    """
    med = float(np.median(counts))
    mad = float(np.median(np.abs(counts - med))) + 1e-9
    robust_sigma = 1.4826 * mad
    return med + k * robust_sigma


# ----------------------------
# Clustering metrics
# ----------------------------
def fano_factor(counts: np.ndarray) -> float:
    """Fano = Var/Mean for binned counts (bigger => more bursty/clustery)."""
    m = float(np.mean(counts))
    if m <= 0:
        return float("nan")
    return float(np.var(counts) / m)


def burstiness_index(counts: np.ndarray) -> float:
    """
    Burstiness (std-mean)/(std+mean) in [-1, 1]
    - near -1 => very regular
    - near 0  => Poisson-ish
    - near +1 => very bursty
    """
    m = float(np.mean(counts))
    s = float(np.std(counts))
    if (s + m) == 0:
        return float("nan")
    return float((s - m) / (s + m))


def iei_cv(t_us: np.ndarray, max_events: int = DEFAULT_MAX_IEI_EVENTS) -> float:
    """
    Coefficient of variation of inter-event intervals.
    Uses stride-based downsampling for very large captures to avoid OOM.
    """
    if t_us.size < 3:
        return float("nan")

    if max_events <= 0:
        return float("nan")

    if t_us.size > max_events:
        step = int(np.ceil(t_us.size / max_events))
        ts = t_us[::step]
    else:
        ts = t_us

    if ts.size < 3:
        return float("nan")

    dt = np.diff(ts)
    mu = float(np.mean(dt))
    if mu <= 0:
        return float("nan")
    return float(np.std(dt) / mu)


# ----------------------------
# Step 3 metrics struct
# ----------------------------
@dataclass
class RefrMetrics:
    raw_file: str
    bias_refr: float
    duration_s: float
    total_events: int
    events_per_s: float
    on_frac: float

    # clustering / burstiness
    fano: float
    burstiness: float
    iei_cv: float

    # edge detection / missed edges
    peaks_detected: int
    expected_edges: float
    missed_edge_frac: float
    peak_interval_mean_ms: float
    peak_interval_std_ms: float


# ----------------------------
# Parsing bias_refr from filename
# ----------------------------
def extract_refr_from_name(filename: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, filename)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def list_raw_files(input_dir: str) -> List[str]:
    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".raw")
    )


# ----------------------------
# One-file analysis
# ----------------------------
def analyze_one_file(
    raw_path: str,
    bias_refr: float,
    bin_ms: float,
    peak_k: float,
    min_peak_dist_ms: float,
    led_freq_hz: float,
    edges_per_cycle: int,
    max_iei_events: int,
) -> RefrMetrics:
    # Load both timestamps and polarity so the analysis can report ON-event fraction too.
    t_us, p = load_event_fields(raw_path)
    total_events = int(t_us.size)

    if total_events < 2:
        return RefrMetrics(
            raw_file=os.path.basename(raw_path),
            bias_refr=bias_refr,
            duration_s=0.0,
            total_events=total_events,
            events_per_s=float("nan"),
            on_frac=float("nan"),
            fano=float("nan"),
            burstiness=float("nan"),
            iei_cv=float("nan"),
            peaks_detected=0,
            expected_edges=float("nan"),
            missed_edge_frac=float("nan"),
            peak_interval_mean_ms=float("nan"),
            peak_interval_std_ms=float("nan"),
        )

    time_s = t_us.astype(np.float64) * 1e-6
    duration_s = float(time_s.max() - time_s.min())
    events_per_s = float(total_events / duration_s) if duration_s > 0 else float("nan")

    # This tells you whether the capture is dominated by ON or OFF events.
    on_frac = float(np.mean(p > 0)) if p.size == total_events else float("nan")

    # clustering metrics on binned counts
    t_bins, counts = binned_activity(time_s, bin_ms / 1000.0)
    ff = fano_factor(counts)
    bi = burstiness_index(counts)
    cv = iei_cv(t_us, max_events=max_iei_events)

    # peak detection -> edges detected
    min_height = robust_peak_threshold(counts, peak_k)
    peak_times_s = find_peaks_simple(
        t_bins, counts,
        min_height=min_height,
        min_distance_s=min_peak_dist_ms / 1000.0
    )

    peaks_detected = int(peak_times_s.size)

    # expected edges: for a square wave, typically 2 edges per cycle (ON and OFF)
    edge_rate = led_freq_hz * edges_per_cycle
    expected_edges = float(edge_rate * duration_s) if duration_s > 0 else float("nan")

    if np.isfinite(expected_edges) and expected_edges > 0:
        missed_edge_frac = float(max(0.0, 1.0 - (peaks_detected / expected_edges)))
    else:
        missed_edge_frac = float("nan")

    if peak_times_s.size >= 3:
        intervals = np.diff(peak_times_s)  # seconds
        peak_interval_mean_ms = float(np.mean(intervals) * 1e3)
        peak_interval_std_ms = float(np.std(intervals) * 1e3)
    else:
        peak_interval_mean_ms = float("nan")
        peak_interval_std_ms = float("nan")

    return RefrMetrics(
        raw_file=os.path.basename(raw_path),
        bias_refr=bias_refr,
        duration_s=duration_s,
        total_events=total_events,
        events_per_s=events_per_s,
        on_frac=on_frac,
        fano=ff,
        burstiness=bi,
        iei_cv=cv,
        peaks_detected=peaks_detected,
        expected_edges=expected_edges,
        missed_edge_frac=missed_edge_frac,
        peak_interval_mean_ms=peak_interval_mean_ms,
        peak_interval_std_ms=peak_interval_std_ms,
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 3: Sweep bias_refr and measure clustering + missed edges.")
    ap.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing bias_refr sweep .raw files (default: {DEFAULT_INPUT_DIR})"
    )
    ap.add_argument(
        "--refr_regex",
        default=r"(?:refr|biasrefr)[_-]([+-]?[0-9]+(?:\.[0-9]+)?)",
        help="Regex (with one capture group) to parse bias_refr from filename (e.g., refr_-20.raw)"
    )

    ap.add_argument("--led_freq_hz", type=float, required=True, help="LED square-wave frequency (e.g., 1000)")
    ap.add_argument("--edges_per_cycle", type=int, default=2, help="2 for square wave (ON+OFF edges)")
    ap.add_argument("--bin_ms", type=float, default=1.0, help="Bin width for activity histogram (ms)")
    ap.add_argument("--peak_k", type=float, default=6.0, help="Peak threshold median+k*robust_sigma")
    ap.add_argument("--min_peak_dist_ms", type=float, default=0.3, help="Minimum spacing between detected peaks (ms)")
    ap.add_argument(
        "--max_iei_events",
        type=int,
        default=DEFAULT_MAX_IEI_EVENTS,
        help=f"Max events used for IEI CV (default {DEFAULT_MAX_IEI_EVENTS}; larger uses more RAM)"
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV filename (saved into repo data/)")
    ap.add_argument("--plot_prefix", default=None, help="Optional prefix for plot filenames (saved into repo plots/)")

    ap.add_argument("--no_plot", action="store_true", help="Disable plots")
    args = ap.parse_args()

    raw_files = list_raw_files(args.input_dir)
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {args.input_dir}")

    rows: List[RefrMetrics] = []
    for rp in raw_files:
        base = os.path.basename(rp)
        # Each file should encode the tested refractory bias in its name.
        refr = extract_refr_from_name(base, args.refr_regex)
        if refr is None:
            print(f"Skipping (can't parse bias_refr): {base}")
            continue

        m = analyze_one_file(
            raw_path=rp,
            bias_refr=float(refr),
            bin_ms=args.bin_ms,
            peak_k=args.peak_k,
            min_peak_dist_ms=args.min_peak_dist_ms,
            led_freq_hz=args.led_freq_hz,
            edges_per_cycle=args.edges_per_cycle,
            max_iei_events=args.max_iei_events,
        )
        rows.append(m)
        print(f"Done {base}: refr={m.bias_refr} events/s={m.events_per_s:.2f} missed={m.missed_edge_frac:.3f}")

    if not rows:
        raise RuntimeError("No files analyzed. Check your filenames or --refr_regex.")

    rows.sort(key=lambda r: r.bias_refr)

    # Save summary CSV into repo data/
    root = repo_root_from_this_file(__file__)
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)

    out_name = args.out_csv
    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"
    out_path = os.path.join(out_dir, out_name)

    header = [
        "raw_file", "bias_refr",
        "duration_s", "total_events", "events_per_s", "on_frac",
        "fano", "burstiness", "iei_cv",
        "peaks_detected", "expected_edges", "missed_edge_frac",
        "peak_interval_mean_ms", "peak_interval_std_ms"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.raw_file, r.bias_refr,
                r.duration_s, r.total_events, r.events_per_s, r.on_frac,
                r.fano, r.burstiness, r.iei_cv,
                r.peaks_detected, r.expected_edges, r.missed_edge_frac,
                r.peak_interval_mean_ms, r.peak_interval_std_ms
            ])

    print("Saved summary CSV:", out_path)

    if args.no_plot:
        return

    plot_dir = os.path.join(root, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_prefix = args.plot_prefix.strip() if args.plot_prefix else os.path.splitext(out_name)[0]

    # Pull each summary metric into a plain array for plotting against bias_refr.
    refr = np.array([r.bias_refr for r in rows], dtype=float)
    missed = np.array([r.missed_edge_frac for r in rows], dtype=float)
    fano = np.array([r.fano for r in rows], dtype=float)
    burst = np.array([r.burstiness for r in rows], dtype=float)
    evrate = np.array([r.events_per_s for r in rows], dtype=float)
    pstd = np.array([r.peak_interval_std_ms for r in rows], dtype=float)

    fig1 = plt.figure()
    plt.plot(refr, missed, marker="o")
    plt.xlabel("bias_refr")
    plt.ylabel("missed edge fraction")
    plt.title("Missed edges vs bias_refr")
    plt.grid(True)
    plot1_path = os.path.join(plot_dir, f"{plot_prefix}_missed_edges_vs_bias_refr.png")
    fig1.savefig(plot1_path, dpi=300)
    print("Saved plot:", plot1_path)

    fig2 = plt.figure()
    plt.plot(refr, fano, marker="o")
    plt.xlabel("bias_refr")
    plt.ylabel("Fano factor (binned counts)")
    plt.title("Event clustering (burstiness) vs bias_refr")
    plt.grid(True)
    plot2_path = os.path.join(plot_dir, f"{plot_prefix}_fano_vs_bias_refr.png")
    fig2.savefig(plot2_path, dpi=300)
    print("Saved plot:", plot2_path)

    fig3 = plt.figure()
    plt.plot(refr, burst, marker="o")
    plt.xlabel("bias_refr")
    plt.ylabel("Burstiness index")
    plt.title("Burstiness index vs bias_refr")
    plt.grid(True)
    plot3_path = os.path.join(plot_dir, f"{plot_prefix}_burstiness_vs_bias_refr.png")
    fig3.savefig(plot3_path, dpi=300)
    print("Saved plot:", plot3_path)

    fig4 = plt.figure()
    plt.plot(refr, evrate, marker="o")
    plt.xlabel("bias_refr")
    plt.ylabel("events/s")
    plt.title("Event rate vs bias_refr")
    plt.grid(True)
    plot4_path = os.path.join(plot_dir, f"{plot_prefix}_event_rate_vs_bias_refr.png")
    fig4.savefig(plot4_path, dpi=300)
    print("Saved plot:", plot4_path)

    fig5 = plt.figure()
    plt.plot(refr, pstd, marker="o")
    plt.xlabel("bias_refr")
    plt.ylabel("peak interval std (ms)")
    plt.title("Peak timing spread vs bias_refr")
    plt.grid(True)
    plot5_path = os.path.join(plot_dir, f"{plot_prefix}_peak_timing_spread_vs_bias_refr.png")
    fig5.savefig(plot5_path, dpi=300)
    print("Saved plot:", plot5_path)

    plt.show()


if __name__ == "__main__":
    main()
