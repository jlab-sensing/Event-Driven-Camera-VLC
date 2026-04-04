import argparse
import csv
import os
from dataclasses import dataclass
from typing import List

from io_utils import repo_root_from_this_file


@dataclass
class CalibrationPlan:
    requested_frequency_hz: float
    actual_frequency_hz: float
    symbols_per_bit: int
    active_bits: int
    guard_bits: int
    total_symbols: int
    duration_s: float
    output_file: str


def build_alternating_stream(
    symbols_per_bit: int,
    active_bits: int,
    guard_bits: int,
    on_symbol: str,
    off_symbol: str,
) -> str:
    if symbols_per_bit <= 0:
        raise ValueError("symbols_per_bit must be > 0")
    if active_bits <= 0:
        raise ValueError("active_bits must be > 0")
    if guard_bits < 0:
        raise ValueError("guard_bits must be >= 0")

    alt_pair = (on_symbol * symbols_per_bit) + (off_symbol * symbols_per_bit)
    pair_repeats = active_bits // 2
    active = alt_pair * pair_repeats
    if active_bits % 2:
        active += on_symbol * symbols_per_bit

    guard = off_symbol * (guard_bits * symbols_per_bit)
    return guard + active + guard


def main() -> None:
    root = repo_root_from_this_file(__file__)
    default_out_dir = os.path.join(root, "pru1_pwm_CSK_1000Hz", "userspace")

    ap = argparse.ArgumentParser(
        description="Generate alternating-pattern calibration symbol files for BBB timing validation."
    )
    ap.add_argument(
        "--frequencies_hz",
        nargs="+",
        type=float,
        default=[500.0],
        help="Requested calibration frequencies to generate.",
    )
    ap.add_argument(
        "--symbol_us",
        type=float,
        default=1.0,
        help="Duration of one PRU symbol in microseconds. Current firmware uses 1.0 us.",
    )
    ap.add_argument(
        "--active_bits",
        type=int,
        default=1500,
        help="How many alternating bits to transmit in the active calibration segment.",
    )
    ap.add_argument(
        "--guard_bits",
        type=int,
        default=100,
        help="How many OFF bit periods to prepend and append as guard space.",
    )
    ap.add_argument(
        "--on_symbol",
        default="4",
        help="PRU symbol character used for logical 1. Current OOK default is '4'.",
    )
    ap.add_argument(
        "--off_symbol",
        default="0",
        help="PRU symbol character used for logical 0. Current OOK default is '0'.",
    )
    ap.add_argument(
        "--out_dir",
        default=default_out_dir,
        help="Folder where generated calibration symbol files will be written.",
    )
    ap.add_argument(
        "--out_prefix",
        default="s31_cal_alt",
        help="Prefix for generated calibration symbol filenames.",
    )
    args = ap.parse_args()

    if args.symbol_us <= 0:
        raise ValueError("--symbol_us must be > 0")
    if args.active_bits <= 0:
        raise ValueError("--active_bits must be > 0")
    if args.guard_bits < 0:
        raise ValueError("--guard_bits must be >= 0")
    if len(args.on_symbol) != 1 or len(args.off_symbol) != 1:
        raise ValueError("--on_symbol and --off_symbol must each be a single character.")

    os.makedirs(args.out_dir, exist_ok=True)
    plans: List[CalibrationPlan] = []

    for requested_frequency_hz in args.frequencies_hz:
        if requested_frequency_hz <= 0:
            raise ValueError("All frequencies must be > 0")

        symbols_per_bit = max(1, int(round((1_000_000.0 / args.symbol_us) / requested_frequency_hz)))
        actual_frequency_hz = (1_000_000.0 / args.symbol_us) / symbols_per_bit
        symbol_stream = build_alternating_stream(
            symbols_per_bit=symbols_per_bit,
            active_bits=args.active_bits,
            guard_bits=args.guard_bits,
            on_symbol=args.on_symbol,
            off_symbol=args.off_symbol,
        )

        file_name = f"{args.out_prefix}_{int(round(requested_frequency_hz))}Hz_symbols.txt"
        out_path = os.path.join(args.out_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(symbol_stream)

        total_symbols = len(symbol_stream)
        duration_s = (total_symbols * args.symbol_us) * 1e-6
        plans.append(
            CalibrationPlan(
                requested_frequency_hz=float(requested_frequency_hz),
                actual_frequency_hz=float(actual_frequency_hz),
                symbols_per_bit=int(symbols_per_bit),
                active_bits=int(args.active_bits),
                guard_bits=int(args.guard_bits),
                total_symbols=int(total_symbols),
                duration_s=float(duration_s),
                output_file=file_name,
            )
        )
        print(
            f"Saved {file_name} "
            f"(requested={requested_frequency_hz:.1f}Hz, actual={actual_frequency_hz:.3f}Hz, "
            f"symbols_per_bit={symbols_per_bit}, duration={duration_s:.3f}s)"
        )

    manifest_path = os.path.join(args.out_dir, f"{args.out_prefix}_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "requested_frequency_hz",
            "actual_frequency_hz",
            "symbols_per_bit",
            "active_bits",
            "guard_bits",
            "total_symbols",
            "duration_s",
            "output_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for plan in plans:
            writer.writerow(plan.__dict__)

    print("Saved manifest CSV:", manifest_path)


if __name__ == "__main__":
    main()
