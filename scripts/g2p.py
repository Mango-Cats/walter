"""
Adds IPA (G2P) transcriptions to an existing CSV file, one column pair per
language: English (t_eng_1/t_eng_2) and/or Filipino (t_fil_1/t_fil_2).

Input schema (--input CSV):
    Must contain columns x_1 and x_2.
    Existing transcription columns for the selected language(s) are overwritten.

Usage:
    python scripts/g2p.py --input _results/D.csv --in-place y
    python scripts/g2p.py --input _data/P.csv --in-place n --lang eng
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import COL_X1, COL_X2
from src.eng_g2p import transcribe_dataframe as transcribe_eng
from src.fil_g2p import transcribe_dataframe as transcribe_fil

_REQUIRED_COLS: frozenset = frozenset({COL_X1, COL_X2})
_TRANSCRIBERS = {"eng": transcribe_eng, "fil": transcribe_fil}


def g2p(input_path: Path, in_place: bool, langs: list[str]) -> None:
    df = pd.read_csv(input_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    print(f"[g2p] Loaded {len(df):,} rows from {input_path}")

    for lang in langs:
        print(f"\n[g2p] Transcribing: {lang}")
        df = _TRANSCRIBERS[lang](df, verbose=True)

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
        description="Add per-language IPA transcriptions to a drug-pair CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV with x_1, x_2 columns")
    parser.add_argument(
        "--in-place",
        required=True,
        choices=["y", "n"],
        dest="in_place",
        help="Overwrite input file (y) or write to <input>-g2p.csv (n)",
    )
    parser.add_argument(
        "--lang",
        default="both",
        choices=["eng", "fil", "both"],
        help="Which transcription(s) to add (default: both)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    langs = ["eng", "fil"] if args.lang == "both" else [args.lang]
    g2p(args.input, in_place=(args.in_place == "y"), langs=langs)


if __name__ == "__main__":
    main()
