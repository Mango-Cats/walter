"""
Augments an existing dataset CSV with additional random unlabeled pairs.

Every new pair attaches one never-before-seen registry name to a name
that already appears in the input dataset. This can only grow an
existing connected component, never merge two of them together, so
the name graph stays split-safe (see src/clustering.py and src/noise.py
for why that matters). It also means a new pair is never both-unseen,
which would form a useless all-negative component.

Input schema (--input CSV):
    x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label

Registry (--registry CSV):
    One-column CSV of drug names (header ignored, first column used).

Writes the augmented pairs (same schema as --input).

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
    COL_T_ENG_1,
    COL_T_ENG_2,
    COL_T_FIL_1,
    COL_T_FIL_2,
    COL_X1,
    COL_X2,
    REGISTRY_COL,
    SEED,
    SHUFFLE_SEED,
    UNLABELED_LABEL,
)
from src.dataset import clean_and_deduplicate
from src.transcribe import transcribe_all
from src.preprocessing import clean_registry

# x_1's transcriptions, then x_2's, in language order.
_T1_COLS: list[str] = [COL_T_ENG_1, COL_T_FIL_1]
_T2_COLS: list[str] = [COL_T_ENG_2, COL_T_FIL_2]
_FINAL_COLS: list[str] = [COL_X1, *_T1_COLS, COL_X2, *_T2_COLS, COL_LABEL]

_REQUIRED_COLS: frozenset = frozenset(_FINAL_COLS)


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


def _sample_new_pairs(
    anchors: list[str],
    fresh_names: list[str],
    csize: int,
    seed: int,
) -> list[dict]:
    """
    Each new pair = (random existing anchor, random never-seen registry name).
    Every fresh name is consumed at most once, so it can attach to exactly
    one existing cluster and never bridge two of them.
    """
    if not anchors or not fresh_names:
        warnings.warn(
            "[augmenter] No existing names or no fresh registry names available "
            "— cannot generate cluster-safe pairs."
        )
        return []

    rng = random.Random(seed)
    pool = fresh_names[:]
    rng.shuffle(pool)

    target = min(csize, len(pool))
    if target < csize:
        warnings.warn(
            f"[augmenter] Requested {csize:,} pairs but only {len(pool):,} fresh "
            "registry names are available to attach without bridging clusters. "
            "Will return all available."
        )

    new_pairs = [{COL_X1: rng.choice(anchors), COL_X2: b} for b in pool[:target]]

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

    existing_names = set(input_df[COL_X1]) | set(input_df[COL_X2])
    anchors = sorted(existing_names)
    fresh_names = [n for n in registry_df[REGISTRY_COL].tolist() if n not in existing_names]
    print(
        f"[augmenter] Existing names: {len(anchors):,}  "
        f"Fresh (unused) registry names: {len(fresh_names):,}"
    )

    raw_pairs = _sample_new_pairs(anchors, fresh_names, csize, SEED)

    if not raw_pairs:
        print("[augmenter] No new pairs generated — output unchanged.")
        return

    new_df = pd.DataFrame(raw_pairs)
    new_df = transcribe_all(new_df, tag="augmenter", verbose=True)
    new_df[COL_LABEL] = UNLABELED_LABEL

    new_df = new_df.reindex(columns=_FINAL_COLS, fill_value="")

    combined = pd.concat([input_df, new_df], ignore_index=True)
    combined = combined[_FINAL_COLS]
    combined = combined.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    if in_place:
        out_path = input_path
    else:
        out_path = input_path.parent / f"{input_path.stem}-aug.csv"

    combined.to_csv(out_path, index=False)

    print("\n[augmenter] Done.")
    print(f"  Existing pairs  : {len(input_df):,}")
    print(f"  New pairs added : {len(raw_pairs):,}")
    print(f"  Total rows      : {len(combined):,}")
    print(f"  Output (pairs)  : {out_path}")


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
