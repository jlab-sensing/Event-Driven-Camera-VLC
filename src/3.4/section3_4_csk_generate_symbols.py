import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from io_utils import repo_root_from_this_file


DEFAULT_SYMBOL_MESSAGE = "1234123412341234"
BIT_PAIR_TO_SYMBOL = {
    "00": "1",
    "01": "2",
    "10": "3",
    "11": "4",
}


@dataclass
class CskSymbolPlan:
    csk_scheme: str
    truth_symbols: str
    requested_symbol_rate_hz: float
    actual_symbol_rate_hz: float
    bits_per_symbol: float
    symbols_per_csk_symbol: int
    message_repeats: int
    guard_symbols: int
    target_duration_s: Optional[float]
    duration_pad_pru_symbols: int
    csk_symbols_transmitted: int
    total_pru_symbols: int
    duration_s: float
    output_file: str


def load_bits(bits_file: Optional[str], bits_literal: Optional[str]) -> Optional[str]:
    if bits_file and bits_literal:
        raise ValueError("Use either --bits_file or --bits, not both.")
    if not bits_file and not bits_literal:
        return None

    if bits_file:
        with open(bits_file, "r", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raw = bits_literal or ""

    bits = "".join(ch for ch in raw if ch in "01")
    if not bits:
        raise ValueError("No 0/1 bits were found in the provided bit input.")
    if len(bits) % 2 != 0:
        raise ValueError("CSK 4-state symbol generation expects an even number of bits.")
    return bits


def bits_to_symbols(bits: str) -> str:
    return "".join(BIT_PAIR_TO_SYMBOL[bits[i : i + 2]] for i in range(0, len(bits), 2))


def clean_symbol_message(raw: str, alphabet: str) -> str:
    symbols = "".join(ch for ch in raw.strip() if ch in set(alphabet))
    if not symbols:
        raise ValueError("No valid CSK symbols were found in the provided symbol message.")
    return symbols


def duration_tag(seconds: float) -> str:
    return f"{seconds:g}s".replace(".", "p")


def build_symbol_stream(
    truth_symbols: str,
    symbols_per_csk_symbol: int,
    message_repeats: int,
    guard_symbols: int,
    off_symbol: str,
) -> str:
    if symbols_per_csk_symbol <= 0:
        raise ValueError("symbols_per_csk_symbol must be > 0.")
    if message_repeats <= 0:
        raise ValueError("message_repeats must be > 0.")
    if guard_symbols < 0:
        raise ValueError("guard_symbols must be >= 0.")

    one_message = "".join(symbol * symbols_per_csk_symbol for symbol in truth_symbols)
    guard = off_symbol * (guard_symbols * symbols_per_csk_symbol)
    return guard + (one_message * message_repeats) + guard


def build_duration_symbol_stream(
    truth_symbols: str,
    symbols_per_csk_symbol: int,
    target_total_pru_symbols: int,
    guard_symbols: int,
    off_symbol: str,
) -> tuple[str, int, int]:
    if target_total_pru_symbols <= 0:
        raise ValueError("target_total_pru_symbols must be > 0.")

    one_message = "".join(symbol * symbols_per_csk_symbol for symbol in truth_symbols)
    guard = off_symbol * (guard_symbols * symbols_per_csk_symbol)
    available = target_total_pru_symbols - (2 * len(guard))
    if available < len(one_message):
        raise ValueError("Target duration is too short for one complete CSK message plus guard space.")

    message_repeats = available // len(one_message)
    duration_pad_pru_symbols = available - (message_repeats * len(one_message))
    stream = guard + (one_message * message_repeats) + guard + (off_symbol * duration_pad_pru_symbols)
    return stream, int(message_repeats), int(duration_pad_pru_symbols)


def main() -> None:
    root = repo_root_from_this_file(__file__)
    default_out_dir = os.path.join(root, "pru1_pwm_CSK_1000Hz", "userspace")

    ap = argparse.ArgumentParser(
        description="Generate Section 3.4 4-state CSK PRU symbol files using the current '1'..'4' RGB state mapping."
    )
    ap.add_argument("--symbol_message", default=DEFAULT_SYMBOL_MESSAGE, help="Literal CSK symbols, e.g. 12341234.")
    ap.add_argument("--symbol_alphabet", default="1234", help="Allowed CSK symbols. Current PRU mapping uses 1234.")
    ap.add_argument("--bits", default=None, help="Optional literal bits. Pairs map 00->1, 01->2, 10->3, 11->4.")
    ap.add_argument("--bits_file", default=None, help="Optional bit file. Mutually exclusive with --bits.")
    ap.add_argument("--symbol_rates_hz", nargs="+", type=float, default=[1000.0], help="Requested CSK symbol rates.")
    ap.add_argument("--symbol_us", type=float, default=1.0, help="Duration of one PRU symbol in microseconds.")
    ap.add_argument("--message_repeats", type=int, default=128, help="Complete CSK messages to repeat.")
    ap.add_argument(
        "--target_duration_s",
        type=float,
        default=None,
        help="If set, repeat complete messages until this duration is filled, then pad OFF.",
    )
    ap.add_argument("--guard_symbols", type=int, default=20, help="OFF CSK-symbol periods before and after the message.")
    ap.add_argument("--off_symbol", default="0", help="PRU symbol character used for off/guard.")
    ap.add_argument("--bits_per_symbol", type=float, default=2.0, help="Logical payload bits per CSK symbol.")
    ap.add_argument("--csk_scheme", default="ratio_4state", help="Scheme label written into the manifest.")
    ap.add_argument("--out_dir", default=default_out_dir, help="Folder where generated symbol files are written.")
    ap.add_argument("--out_prefix", default="s34_csk_ratio4", help="Prefix for generated symbol filenames.")
    ap.add_argument("--manifest_name", default=None, help="Manifest CSV filename. Defaults to '<out_prefix>_manifest.csv'.")
    args = ap.parse_args()

    if args.symbol_us <= 0:
        raise ValueError("--symbol_us must be > 0.")
    if args.message_repeats <= 0:
        raise ValueError("--message_repeats must be > 0.")
    if args.target_duration_s is not None and args.target_duration_s <= 0:
        raise ValueError("--target_duration_s must be > 0.")
    if args.guard_symbols < 0:
        raise ValueError("--guard_symbols must be >= 0.")
    if len(args.off_symbol) != 1:
        raise ValueError("--off_symbol must be a single character.")

    bits = load_bits(args.bits_file, args.bits)
    truth_symbols = bits_to_symbols(bits) if bits is not None else clean_symbol_message(args.symbol_message, args.symbol_alphabet)

    os.makedirs(args.out_dir, exist_ok=True)
    plans: List[CskSymbolPlan] = []

    for requested_symbol_rate_hz in args.symbol_rates_hz:
        if requested_symbol_rate_hz <= 0:
            raise ValueError("All symbol rates must be > 0.")

        symbols_per_csk_symbol = max(
            1,
            int(round((1_000_000.0 / args.symbol_us) / requested_symbol_rate_hz)),
        )
        actual_symbol_rate_hz = (1_000_000.0 / args.symbol_us) / symbols_per_csk_symbol
        message_repeats = args.message_repeats
        duration_pad_pru_symbols = 0

        if args.target_duration_s is None:
            stream = build_symbol_stream(
                truth_symbols=truth_symbols,
                symbols_per_csk_symbol=symbols_per_csk_symbol,
                message_repeats=message_repeats,
                guard_symbols=args.guard_symbols,
                off_symbol=args.off_symbol,
            )
        else:
            target_total_pru_symbols = int(round((args.target_duration_s * 1_000_000.0) / args.symbol_us))
            stream, message_repeats, duration_pad_pru_symbols = build_duration_symbol_stream(
                truth_symbols=truth_symbols,
                symbols_per_csk_symbol=symbols_per_csk_symbol,
                target_total_pru_symbols=target_total_pru_symbols,
                guard_symbols=args.guard_symbols,
                off_symbol=args.off_symbol,
            )

        suffix = "symbols"
        if args.target_duration_s is not None:
            suffix = f"{duration_tag(args.target_duration_s)}_symbols"
        file_name = f"{args.out_prefix}_{int(round(requested_symbol_rate_hz))}Hz_{suffix}.txt"
        out_path = os.path.join(args.out_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(stream)

        total_pru_symbols = len(stream)
        duration_s = (total_pru_symbols * args.symbol_us) * 1e-6
        csk_symbols_transmitted = len(truth_symbols) * message_repeats
        plans.append(
            CskSymbolPlan(
                csk_scheme=args.csk_scheme,
                truth_symbols=truth_symbols,
                requested_symbol_rate_hz=float(requested_symbol_rate_hz),
                actual_symbol_rate_hz=float(actual_symbol_rate_hz),
                bits_per_symbol=float(args.bits_per_symbol),
                symbols_per_csk_symbol=int(symbols_per_csk_symbol),
                message_repeats=int(message_repeats),
                guard_symbols=int(args.guard_symbols),
                target_duration_s=args.target_duration_s,
                duration_pad_pru_symbols=int(duration_pad_pru_symbols),
                csk_symbols_transmitted=int(csk_symbols_transmitted),
                total_pru_symbols=int(total_pru_symbols),
                duration_s=float(duration_s),
                output_file=file_name,
            )
        )
        print(
            f"Saved {file_name} "
            f"(requested={requested_symbol_rate_hz:.1f}Hz, actual={actual_symbol_rate_hz:.3f}Hz, "
            f"symbols_per_csk_symbol={symbols_per_csk_symbol}, repeats={message_repeats}, "
            f"duration={duration_s:.3f}s)"
        )

    manifest_name = args.manifest_name or f"{args.out_prefix}_manifest.csv"
    manifest_path = os.path.join(args.out_dir, manifest_name)
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(CskSymbolPlan.__dataclass_fields__.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for plan in plans:
            writer.writerow(plan.__dict__)

    print("Saved manifest CSV:", manifest_path)


if __name__ == "__main__":
    main()
