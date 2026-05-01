import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from io_utils import repo_root_from_this_file


DEFAULT_BITS = "10110010110"
DEFAULT_FREQUENCIES_HZ = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]


def load_truth_bits(bits_file: Optional[str], bits_literal: Optional[str]) -> str:
    if bits_file and bits_literal:
        raise ValueError("Use either --bits_file or --bits, not both.")

    if bits_file:
        with open(bits_file, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = bits_literal or DEFAULT_BITS

    bits = "".join(ch for ch in raw if ch in "01")
    if not bits:
        raise ValueError("No 0/1 bits were found in the truth-message input.")
    if len(bits) != 11:
        raise ValueError(f"Section 3.1 replication expects an 11-bit truth message, got {len(bits)} bits.")
    return bits


@dataclass
class SymbolPlan:
    requested_frequency_hz: float
    actual_frequency_hz: float
    symbols_per_bit: int
    message_repeats: int
    guard_bits: int
    target_duration_s: Optional[float]
    duration_pad_symbols: int
    total_symbols: int
    duration_s: float
    output_file: str


def duration_tag(seconds: float) -> str:
    return f"{seconds:g}s".replace(".", "p")


def build_symbol_stream(
    truth_bits: str,
    on_symbol: str,
    off_symbol: str,
    symbols_per_bit: int,
    message_repeats: int,
    guard_bits: int,
) -> str:
    if symbols_per_bit <= 0:
        raise ValueError("symbols_per_bit must be > 0")
    if message_repeats <= 0:
        raise ValueError("message_repeats must be > 0")
    if guard_bits < 0:
        raise ValueError("guard_bits must be >= 0")

    one_message = "".join(
        (on_symbol if bit == "1" else off_symbol) * symbols_per_bit
        for bit in truth_bits
    )
    guard = off_symbol * (guard_bits * symbols_per_bit)
    return guard + (one_message * message_repeats) + guard


def build_duration_symbol_stream(
    truth_bits: str,
    on_symbol: str,
    off_symbol: str,
    symbols_per_bit: int,
    target_total_symbols: int,
    guard_bits: int,
) -> Tuple[str, int, int]:
    if symbols_per_bit <= 0:
        raise ValueError("symbols_per_bit must be > 0")
    if target_total_symbols <= 0:
        raise ValueError("target_total_symbols must be > 0")
    if guard_bits < 0:
        raise ValueError("guard_bits must be >= 0")

    one_message = "".join(
        (on_symbol if bit == "1" else off_symbol) * symbols_per_bit
        for bit in truth_bits
    )
    guard = off_symbol * (guard_bits * symbols_per_bit)
    available_symbols = target_total_symbols - (2 * len(guard))
    if available_symbols < len(one_message):
        raise ValueError(
            "Target duration is too short for one complete message plus guard space."
        )

    message_repeats = available_symbols // len(one_message)
    duration_pad_symbols = available_symbols - (message_repeats * len(one_message))
    symbol_stream = guard + (one_message * message_repeats) + guard
    symbol_stream += off_symbol * duration_pad_symbols
    return symbol_stream, int(message_repeats), int(duration_pad_symbols)


def main() -> None:
    root = repo_root_from_this_file(__file__)
    default_output_dir = os.path.join(root, "pru1_pwm_CSK_1000Hz", "userspace")
    default_bits_file = os.path.join(default_output_dir, "replication_bits_11b.txt")

    ap = argparse.ArgumentParser(
        description="Generate Section 3.1 replication PRU symbol files from one repeated 11-bit message."
    )
    ap.add_argument(
        "--bits_file",
        default=default_bits_file,
        help="Path to the 11-bit truth message file.",
    )
    ap.add_argument(
        "--bits",
        default=None,
        help="Literal 11-bit truth message, e.g. 10110010110.",
    )
    ap.add_argument(
        "--frequencies_hz",
        nargs="+",
        type=float,
        default=DEFAULT_FREQUENCIES_HZ,
        help="Requested OOK bit frequencies to generate.",
    )
    ap.add_argument(
        "--symbol_us",
        type=float,
        default=1.0,
        help="Duration of one PRU symbol in microseconds. Current firmware uses 1.0 us.",
    )
    ap.add_argument(
        "--message_repeats",
        type=int,
        default=128,
        help="How many complete 11-bit messages to repeat in each file.",
    )
    ap.add_argument(
        "--target_duration_s",
        type=float,
        default=None,
        help=(
            "If set, override --message_repeats so each file lasts this many seconds. "
            "Only complete messages are repeated; any leftover time is padded OFF."
        ),
    )
    ap.add_argument(
        "--guard_bits",
        type=int,
        default=20,
        help="How many OFF bit-periods to prepend and append as guard space.",
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
        default=default_output_dir,
        help="Folder where generated PRU symbol files will be written.",
    )
    ap.add_argument(
        "--out_prefix",
        default="s31_replication",
        help="Prefix for generated symbol filenames.",
    )
    ap.add_argument(
        "--manifest_name",
        default=None,
        help="Manifest CSV filename. Defaults to '<out_prefix>_manifest.csv'.",
    )
    args = ap.parse_args()

    if args.symbol_us <= 0:
        raise ValueError("--symbol_us must be > 0")
    if args.message_repeats <= 0:
        raise ValueError("--message_repeats must be > 0")
    if args.target_duration_s is not None and args.target_duration_s <= 0:
        raise ValueError("--target_duration_s must be > 0")
    if args.guard_bits < 0:
        raise ValueError("--guard_bits must be >= 0")
    if len(args.on_symbol) != 1 or len(args.off_symbol) != 1:
        raise ValueError("--on_symbol and --off_symbol must each be a single character.")

    os.makedirs(args.out_dir, exist_ok=True)
    truth_bits = load_truth_bits(args.bits_file, args.bits)

    if not os.path.exists(args.bits_file):
        with open(args.bits_file, "w", encoding="utf-8") as f:
            f.write(truth_bits + "\n")
        print("Saved truth bits file:", args.bits_file)

    plans: List[SymbolPlan] = []
    for requested_freq_hz in args.frequencies_hz:
        if requested_freq_hz <= 0:
            raise ValueError("All frequencies must be > 0")

        symbols_per_bit = max(1, int(round((1_000_000.0 / args.symbol_us) / requested_freq_hz)))
        actual_freq_hz = (1_000_000.0 / args.symbol_us) / symbols_per_bit
        message_repeats = args.message_repeats
        duration_pad_symbols = 0
        if args.target_duration_s is None:
            symbol_stream = build_symbol_stream(
                truth_bits=truth_bits,
                on_symbol=args.on_symbol,
                off_symbol=args.off_symbol,
                symbols_per_bit=symbols_per_bit,
                message_repeats=message_repeats,
                guard_bits=args.guard_bits,
            )
        else:
            target_total_symbols = int(round((args.target_duration_s * 1_000_000.0) / args.symbol_us))
            symbol_stream, message_repeats, duration_pad_symbols = build_duration_symbol_stream(
                truth_bits=truth_bits,
                on_symbol=args.on_symbol,
                off_symbol=args.off_symbol,
                symbols_per_bit=symbols_per_bit,
                target_total_symbols=target_total_symbols,
                guard_bits=args.guard_bits,
            )

        file_suffix = "symbols"
        if args.target_duration_s is not None:
            file_suffix = f"{duration_tag(args.target_duration_s)}_symbols"
        file_name = f"{args.out_prefix}_{int(round(requested_freq_hz))}Hz_{file_suffix}.txt"
        out_path = os.path.join(args.out_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(symbol_stream)

        total_symbols = len(symbol_stream)
        duration_s = (total_symbols * args.symbol_us) * 1e-6
        plans.append(
            SymbolPlan(
                requested_frequency_hz=float(requested_freq_hz),
                actual_frequency_hz=float(actual_freq_hz),
                symbols_per_bit=int(symbols_per_bit),
                message_repeats=int(message_repeats),
                guard_bits=int(args.guard_bits),
                target_duration_s=args.target_duration_s,
                duration_pad_symbols=int(duration_pad_symbols),
                total_symbols=int(total_symbols),
                duration_s=float(duration_s),
                output_file=file_name,
            )
        )
        print(
            f"Saved {file_name} "
            f"(requested={requested_freq_hz:.1f}Hz, actual={actual_freq_hz:.3f}Hz, "
            f"symbols_per_bit={symbols_per_bit}, repeats={message_repeats}, "
            f"duration={duration_s:.3f}s)"
        )

    manifest_name = args.manifest_name or f"{args.out_prefix}_manifest.csv"
    manifest_path = os.path.join(args.out_dir, manifest_name)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "truth_bits",
            "requested_frequency_hz",
            "actual_frequency_hz",
            "symbols_per_bit",
            "message_repeats",
            "guard_bits",
            "target_duration_s",
            "duration_pad_symbols",
            "total_symbols",
            "duration_s",
            "output_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for plan in plans:
            writer.writerow({
                "truth_bits": truth_bits,
                "requested_frequency_hz": plan.requested_frequency_hz,
                "actual_frequency_hz": plan.actual_frequency_hz,
                "symbols_per_bit": plan.symbols_per_bit,
                "message_repeats": plan.message_repeats,
                "guard_bits": plan.guard_bits,
                "target_duration_s": plan.target_duration_s,
                "duration_pad_symbols": plan.duration_pad_symbols,
                "total_symbols": plan.total_symbols,
                "duration_s": plan.duration_s,
                "output_file": plan.output_file,
            })

    print("Saved manifest CSV:", manifest_path)


if __name__ == "__main__":
    main()
