"""
Noise pairs generation.
"""
from typing import Any
from itertools import combinations
import pandas as pd
from .preprocessing import TARGET_COL
from .proposer.inference import LABEL_COL

TRUE_EXPECTED_COLS: list[str] = [TARGET_COL, LABEL_COL]


def validate_columns(true_df: pd.DataFrame) -> None:
    """
    Validates the columns of the true LASA dataset.
    """
    true_df_cols = list(true_df.columns)
    if true_df_cols != TRUE_EXPECTED_COLS:
        raise ValueError(
            f"Invalid `true_df` columns. "
            f"Found: {true_df_cols}. Expected: {TRUE_EXPECTED_COLS}."
        )


def get_positive_set(true_df: pd.DataFrame) -> set[Any]:
    """
    Returns all unique drug names appearing in the positive pairs.
    """
    return set(true_df[TARGET_COL]) | set(true_df[LABEL_COL])


def get_positive_pairs(true_df: pd.DataFrame) -> set[tuple]:
    """
    Returns the set of known positive LASA pairs as frozensets
    so that (A, B) and (B, A) are treated as the same pair.
    """
    return {
        frozenset([row[TARGET_COL], row[LABEL_COL]])
        for _, row in true_df.iterrows()
    }


def make_noise(true_df: pd.DataFrame):
    """
    Generates unlabeled pairs from the `true_df` itself.

    All combinatorial pairs of `true_df` are generated, then known
    positive LASA pairs are removed. The remaining pairs serve as
    unlabeled examples — they may or may not be true LASA pairs,
    but none are confirmed positives. This mirrors Kondrak & Dorr's
    dataset construction exactly.
    """
    validate_columns(true_df=true_df)

    ismp_vocab: set[Any] = get_positive_set(true_df=true_df)
    positive_pairs: set[frozenset] = get_positive_pairs(true_df=true_df)

    unlabeled_rows = []
    for a, b in combinations(sorted(ismp_vocab), 2):
        if frozenset([a, b]) not in positive_pairs:
            unlabeled_rows.append({TARGET_COL: a, LABEL_COL: b})

    noise_df = pd.DataFrame(unlabeled_rows, columns=[TARGET_COL, LABEL_COL])

    print(f"<walter> ISMP vocabulary size: {len(ismp_vocab)}")
    print(f"<walter> Known positive pairs: {len(positive_pairs)}")
    print(f"<walter> Unlabeled pairs generated: {len(noise_df):,}")

    return noise_df