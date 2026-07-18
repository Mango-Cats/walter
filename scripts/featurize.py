#!/usr/bin/env python3
"""
Runs the whole post-pair-generation pipeline over one CSV of drug-name pairs:

    1. G2P        — IPA transcriptions, one column pair per language
                    (src/eng_g2p.py, src/fil_g2p.py)
    2. phoc       — phonetic-similarity columns, one per bin/pho_conf/*.toml,
                    fanned out per language for transcription-dependent configs
                    (src/phoc_runner.py)
    3. engineering — the META_FEATURES (src/feature_engineering.py)

This is exactly the tail of walter.py, minus the pair generation, so you can
featurize an arbitrary pair CSV without regenerating P/U.

Every step writes its own CSV next to the output, named <input>_<suffix>.csv:
    <input>_t.csv     transcriptions
    <input>_pho.csv   + phonetic-similarity columns
    <input>_engi.csv  + META_FEATURES (the default --output)

Input schema (--input CSV):
    Must contain columns x_1 and x_2. Every other column (label, ...) is
    preserved verbatim. Existing transcription columns are reused as-is unless
    --retranscribe is passed.

Usage:
    python scripts/featurize.py --input _data/P_ph.csv
    python scripts/featurize.py --input pairs.csv --output feats.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import COL_X1, COL_X2, TRANSCRIPTION_LANGS
from src.eng_g2p import transcribe_dataframe as transcribe_eng
from src.feature_engineering import run_engineering
from src.fil_g2p import transcribe_dataframe as transcribe_fil
from src.phoc_runner import run_phoc_multilingual

_TRANSCRIBERS = {"eng": transcribe_eng, "fil": transcribe_fil}


def _transcribe(df: pd.DataFrame, retranscribe: bool) -> pd.DataFrame:
    """Add each language's transcription columns, skipping any already present."""
    for lang, cols in TRANSCRIPTION_LANGS.items():
        if not retranscribe and all(c in df.columns for c in cols):
            print(f"[featurize] {lang}: columns {list(cols)} already present, reusing")
            continue
        print(f"\n[featurize] Transcribing: {lang}")
        df = _TRANSCRIBERS[lang](df, verbose=True)
    return df


def featurize(
    input_path: Path,
    output_path: Path,
    retranscribe: bool,
) -> None:
    df = pd.read_csv(input_path)
    missing = [c for c in (COL_X1, COL_X2) if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    print(f"[featurize] Loaded {len(df):,} rows from {input_path}")

    stage_dir = output_path.parent
    stage_dir.mkdir(parents=True, exist_ok=True)
    t_csv = stage_dir / f"{input_path.stem}_t.csv"
    pho_csv = stage_dir / f"{input_path.stem}_pho.csv"

    df = _transcribe(df, retranscribe)
    df.to_csv(t_csv, index=False)
    print(f"[featurize] Transcriptions: {t_csv}")

    print("\n[featurize] Adding phonetic features (phoc)...")
    feats = run_phoc_multilingual(t_csv, pho_csv)
    print(f"[featurize] Phonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"[featurize] Phonetic features: {pho_csv}")

    print("\n[featurize] Engineering meta-features...")
    meta = run_engineering(pho_csv, output_path)
    print(f"[featurize] Meta-features ({len(meta)}): {', '.join(meta)}")

    final = pd.read_csv(output_path)
    print(f"\n[featurize] Done.")
    print(f"  Rows    : {len(final):,}")
    print(f"  Columns : {len(final.columns)}")
    print(f"  Output  : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe, phoc, and feature-engineer a drug-pair CSV."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input CSV with x_1, x_2 columns"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: <input>_engi.csv)",
    )
    parser.add_argument(
        "--retranscribe",
        action="store_true",
        help="Re-run G2P even if transcription columns already exist",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    output = args.output or args.input.parent / f"{args.input.stem}_engi.csv"
    if output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")

    featurize(args.input, output, args.retranscribe)


if __name__ == "__main__":
    main()
