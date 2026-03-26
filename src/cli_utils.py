

# defines how the user talks to the script from the terminal


import argparse


# ----------------------------
# Shared CLI arguments
# ----------------------------
def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Input capture to analyze.
    ap.add_argument("--raw", required=True, help="Path to .raw file")
    # Optional CSV export for summary numbers.
    ap.add_argument("--save_csv", action="store_true", help="Save CSV summary into repo data folder")
    # Output filename stem used by the helper save functions.
    ap.add_argument("--out", type=str, required=True, help="Custom CSV filename (saved inside data folder)")
    # Skip interactive plotting when running batch analyses.
    ap.add_argument("--no_plot", action="store_true", help="Do not show plots")
    return ap
