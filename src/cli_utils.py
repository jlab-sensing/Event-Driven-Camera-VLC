

# defines how the user talks to the script from the terminal


import argparse

def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--raw", required=True, help="Path to .raw file")
    ap.add_argument("--save_csv", action="store_true", help="Save CSV summary into repo data folder")
    ap.add_argument("--out", type=str, required=True, help="Custom CSV filename (saved inside data folder)")
    ap.add_argument("--no_plot", action="store_true", help="Do not show plots")
    return ap