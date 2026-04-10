
# How results are saved


import os
import csv
from typing import List, Any, Dict
import matplotlib.pyplot as plt


# ----------------------------
# Repo path helpers
# ----------------------------
def repo_root_from_this_file(this_file: str) -> str:
    # Walk upward until we hit the repo's src/ folder, then return its parent.
    # This supports scripts stored directly in src/ and nested paths like src/3.1/.
    cursor = os.path.abspath(os.path.dirname(this_file))
    while True:
        if os.path.basename(cursor) == "src":
            return os.path.abspath(os.path.join(cursor, ".."))
        parent = os.path.abspath(os.path.join(cursor, ".."))
        if parent == cursor:
            raise ValueError(f"Could not locate repo root from path: {this_file}")
        cursor = parent


def data_dir(this_file: str) -> str:
    # Make sure the repo-level data folder exists before returning it.
    root = repo_root_from_this_file(this_file)
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ----------------------------
# Save CSV summaries
# ----------------------------
def save_metrics_csv(this_file: str, out_name: str, header: list, row: list) -> str:
    root = repo_root_from_this_file(this_file)
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Add the extension automatically so callers can pass a short base name.
    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"

    out_path = os.path.join(data_dir, out_name)

    # Write one header row and one metrics row.
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row)) + "\n")

    print("Saved CSV:", out_path)
    return out_path


# ----------------------------
# Save plots
# ----------------------------
def save_plot(this_file: str, out_name: str, dpi: int = 300) -> str:
    root = repo_root_from_this_file(this_file)
    plot_dir = os.path.join(root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Reuse the output name but always save figures as PNGs.
    base = os.path.splitext(out_name)[0]
    plot_path = os.path.join(plot_dir, base + ".png")

    plt.savefig(plot_path, dpi=dpi)

    print("Saved plot:", plot_path)
    return plot_path
