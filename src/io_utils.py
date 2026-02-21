
# How results are saved 


import os
import csv
from typing import List, Any, Dict
import matplotlib.pyplot as plt


def repo_root_from_this_file(this_file: str) -> str:
    # this_file is __file__ from a script inside src/
    return os.path.abspath(os.path.join(os.path.dirname(this_file), ".."))

def data_dir(this_file: str) -> str:
    root = repo_root_from_this_file(this_file)
    out_dir = os.path.join(root, "data")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_metrics_csv(this_file: str, out_name: str, header: list, row: list) -> str:
    root = repo_root_from_this_file(this_file)
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    if not out_name.lower().endswith(".csv"):
        out_name += ".csv"

    out_path = os.path.join(data_dir, out_name)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(map(str, row)) + "\n")

    print("Saved CSV:", out_path)  # ← moved here
    return out_path

def save_plot(this_file: str, out_name: str, dpi: int = 300) -> str:
    root = repo_root_from_this_file(this_file)
    plot_dir = os.path.join(root, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    base = os.path.splitext(out_name)[0]
    plot_path = os.path.join(plot_dir, base + ".png")

    plt.savefig(plot_path, dpi=dpi)

    print("Saved plot:", plot_path)  # ← moved here
    return plot_path