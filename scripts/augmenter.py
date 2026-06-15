"""
Augments an existing dataset CSV with additional random unlabeled pairs.

Input schema (--input CSV):
    x_1, t_1, x_2, t_2, label

Registry (--registry CSV):
    One-column CSV of drug names (header ignored, first column used).

Usage:
    python scripts/augmenter.py --input _results/D.csv --registry _data/R_ph.csv \
        --in-place n --csize 5000
"""

import argparse
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    COL_LABEL,
    COL_T1,
    COL_T2,
    COL_X1,
    COL_X2,
    REGISTRY_COL,
    SEED,
    SHUFFLE_SEED,
    UNLABELED_LABEL,
)
from src.dataset import _canonical_key, clean_and_deduplicate
from src.phonemes import transcribe_dataframe
from src.preprocessing import clean_registry

_REQUIRED_COLS: frozenset = frozenset({COL_X1, COL_T1, COL_X2, COL_T2, COL_LABEL})


def _load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")
    df = clean_and_deduplicate(df)
    print(f"[augmenter] Input loaded: {len(df):,} pairs after cleaning")
    return df


def _load_registry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[0], header=0)
    df.columns = [REGISTRY_COL]
    df = clean_registry(df)
    print(f"[augmenter] Registry loaded: {len(df):,} names after cleaning")
    return df


def _build_existing_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    keys = set(df.apply(lambda r: _canonical_key(r[COL_X1], r[COL_X2]), axis=1))
    print(f"[augmenter] Existing canonical pairs: {len(keys):,}")
    return keys


def _sample_new_pairs(
    names: list[str],
    existing_keys: set[tuple[str, str]],
    csize: int,
    seed: int,
) -> list[dict]:
    n = len(names)
    if n < 2:
        warnings.warn(
            "[augmenter] Registry has fewer than 2 names — cannot generate pairs."
        )
        return []

    max_possible = n * (n - 1) // 2
    estimated_available = max(0, max_possible - len(existing_keys))
    target = min(csize, estimated_available)

    if target < csize:
        warnings.warn(
            f"[augmenter] Requested {csize:,} pairs but only ~{estimated_available:,} "
            "estimated valid pairs exist. Will return all available."
        )

    max_attempts = max(target * 20, 100_000)
    rng = random.Random(seed)
    new_pairs: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    attempts = 0

    while len(new_pairs) < target and attempts < max_attempts:
        a, b = rng.sample(names, 2)
        key = _canonical_key(a, b)
        attempts += 1
        if key in existing_keys or key in seen_keys:
            continue
        seen_keys.add(key)
        new_pairs.append({COL_X1: a, COL_X2: b})

    if len(new_pairs) < csize and len(new_pairs) < estimated_available:
        warnings.warn(
            f"[augmenter] Generated {len(new_pairs):,} pairs (requested {csize:,}) "
            f"after {max_attempts:,} attempts. Consider a larger registry."
        )

    print(f"[augmenter] New pairs sampled: {len(new_pairs):,}")
    return new_pairs


def augment(
    input_path: Path,
    registry_path: Path,
    target_total: int,
    in_place: bool,
) -> None:
    input_df = _load_input(input_path)

    csize = target_total - len(input_df)
    if csize <= 0:
        print(
            f"[augmenter] Dataset already has {len(input_df):,} rows "
            f"(target: {target_total:,}). Nothing to add."
        )
        return

    registry_df = _load_registry(registry_path)
    existing_keys = _build_existing_keys(input_df)

    names = registry_df[REGISTRY_COL].tolist()
    raw_pairs = _sample_new_pairs(names, existing_keys, csize, SEED)

    if not raw_pairs:
        print("[augmenter] No new pairs generated — output unchanged.")
        return

    new_df = pd.DataFrame(raw_pairs)
    print("\n[augmenter] Adding IPA transcriptions to new pairs...")
    new_df = transcribe_dataframe(new_df, verbose=True)
    new_df[COL_LABEL] = UNLABELED_LABEL

    final_cols = [COL_X1, COL_T1, COL_X2, COL_T2, COL_LABEL]
    new_df = new_df.reindex(columns=final_cols, fill_value="")

    combined = pd.concat([input_df, new_df], ignore_index=True)
    combined = combined[final_cols]
    combined = combined.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    if in_place:
        out_path = input_path
    else:
        out_path = input_path.parent / f"{input_path.stem}-aug.csv"

    combined.to_csv(out_path, index=False)

    print(f"\n[augmenter] Done.")
    print(f"  Existing pairs : {len(input_df):,}")
    print(f"  New pairs added: {len(raw_pairs):,}")
    print(f"  Total rows     : {len(combined):,}")
    print(f"  Output         : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment a dataset CSV with random unlabeled drug-name pairs."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Existing dataset CSV"
    )
    parser.add_argument(
        "--registry", required=True, type=Path, help="One-column drug name registry CSV"
    )
    parser.add_argument(
        "--in-place",
        required=True,
        choices=["y", "n"],
        dest="in_place",
        help="Overwrite input file (y) or write to <input>-aug.csv (n)",
    )
    parser.add_argument(
        "--csize",
        required=True,
        type=int,
        help="Target total number of rows after augmenting",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    if not args.registry.exists():
        parser.error(f"Registry file not found: {args.registry}")
    if args.csize <= 0:
        parser.error("--csize must be a positive integer")

    augment(args.input, args.registry, args.csize, in_place=(args.in_place == "y"))


if __name__ == "__main__":
    main()
