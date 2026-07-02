"""
Adds IPA (G2P) transcriptions to an existing CSV file.

Input schema (--input CSV):
    Must contain columns x_1 and x_2.
    Existing t_1 / t_2 columns are overwritten.

Usage:
    python scripts/g2p.py --input _results/D.csv --in-place y
    python scripts/g2p.py --input _data/P.csv --in-place n
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import COL_T1, COL_T2, COL_X1, COL_X2
from src.phonemes import transcribe_dataframe

_REQUIRED_COLS: frozenset = frozenset({COL_X1, COL_X2})


def g2p(input_path: Path, in_place: bool) -> None:
    df = pd.read_csv(input_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    print(f"[g2p] Loaded {len(df):,} rows from {input_path}")

    df = transcribe_dataframe(df, verbose=True)

    if in_place:
        out_path = input_path
    else:
        out_path = input_path.parent / f"{input_path.stem}-g2p.csv"

    df.to_csv(out_path, index=False)

    print(f"\n[g2p] Done.")
    print(f"  Rows transcribed : {len(df):,}")
    print(f"  Output           : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add IPA transcriptions (t_1, t_2) to a drug-pair CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV with x_1, x_2 columns")
    parser.add_argument(
        "--in-place",
        required=True,
        choices=["y", "n"],
        dest="in_place",
        help="Overwrite input file (y) or write to <input>-g2p.csv (n)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    g2p(args.input, in_place=(args.in_place == "y"))


if __name__ == "__main__":
    main()
