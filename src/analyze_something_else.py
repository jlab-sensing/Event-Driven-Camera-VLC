import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from metavision_core.event_io import EventsIterator
from cli_utils import add_common_args
from io_utils import save_metrics_csv, save_plot


# ----------------------------
# Custom analysis functions
# ----------------------------
def load_timestamps_us(raw_path: str) -> np.ndarray:
    """Load all event timestamps (microseconds) from a Metavision .raw file."""
    events = EventsIterator(input_path=raw_path)
    ts = []
    for evs in events:
        ts.append(evs["t"])
    return np.concatenate(ts).astype(np.int64) if ts else np.array([], dtype=np.int64)


def plot_activity(t: np.ndarray, y: np.ndarray, title: str):
    """Create the plot (does not save by itself)."""
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Your signal")
    plt.title(title)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)  # adds --raw --save_csv --out --no_plot
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        raise FileNotFoundError(args.raw)

    ts_us = load_timestamps_us(args.raw)

    # ---- Example NEW metrics ----
    total_events = int(ts_us.size)
    duration_s = float((ts_us.max() - ts_us.min()) * 1e-6) if ts_us.size else 0.0
    events_per_s = (total_events / duration_s) if duration_s > 0 else float("nan")

    # ---- CSV (same place, custom name) ----
    if args.save_csv:
        header = ["raw_file", "total_events", "duration_s", "events_per_s"]
        row = [os.path.basename(args.raw), total_events, duration_s, events_per_s]
        csv_path = save_metrics_csv(__file__, args.out, header, row)
        print("Saved CSV:", csv_path)

    # ---- Plot + auto-save to plots/ ----
    if not args.no_plot:
        if ts_us.size:
            t = (ts_us - ts_us.min()) * 1e-6

            # Replace this with your real signal later
            y = np.ones_like(t)

            plot_activity(t, y, f"Custom plot ({os.path.basename(args.raw)})")

            # Save using the same base name as --out (but .png)
            plot_path = save_plot(__file__, args.out)
            print("Saved plot:", plot_path)

            # Show the plot window (optional)
            plt.show()
        else:
            print("No events found in file; skipping plot.")


if __name__ == "__main__":
    main()