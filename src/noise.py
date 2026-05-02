"""
Noise pairs generation.
"""

from typing import Any
from itertools import permutations
import pandas as pd
from .preprocessing import TARGET_COL
from .proposer.inference import LABEL_COL

FDA_EXPECTED_COLS: list[str] = [TARGET_COL]
TRUE_EXPECTED_COLS: list[str] = [TARGET_COL, LABEL_COL]


def validate_columns(fda_df: pd.DataFrame, true_df: pd.DataFrame) -> None:
    """
    This validates the columns of the Philippine drug dataset and the
    true LASA dataset. This returns nothing but may raise an Exception.
    This is an internal function of `make_noise()`.
    """

    fda_df_cols = list(fda_df.columns)
    if fda_df_cols != FDA_EXPECTED_COLS:
        raise ValueError(
            f"Invalid `fda_df` columns. "
            f"Found: {fda_df_cols}. Expected: {FDA_EXPECTED_COLS}."
        )

    true_df_cols = list(true_df.columns)
    if true_df_cols != TRUE_EXPECTED_COLS:
        raise ValueError(
            f"Invalid `true_df` columns. "
            f"Found: {true_df_cols}. Expected: {TRUE_EXPECTED_COLS}."
        )


def get_lasa_set(true_df: pd.DataFrame) -> set[Any]:
    """
    This returns the set of all brand names included in the true
    LASA dataset. This is an internal function of `make_noise()`.
    """

    lasa_set = set(true_df[TARGET_COL])
    return lasa_set.union(set(true_df[LABEL_COL]))


def make_noise(fda_df: pd.DataFrame, true_df: pd.DataFrame, n: int):
    """
    This orchestrates the creation of noise drug pairs. This returns
    a 2-permutations from a subset of brand name drugs in the
    Philippine drug dataset but are not in the true LASA dataset.

    This function should only be called after cleaning the datasets.
    """

    validate_columns(fda_df=fda_df, true_df=true_df)

    lasa_set: set[Any] = get_lasa_set(true_df=true_df)

    set_diff: pd.DataFrame = fda_df[~fda_df[TARGET_COL].isin(values=lasa_set)]
    noise_set: set[Any] = set(set_diff[TARGET_COL].sample(n=n))

    return pd.DataFrame(
        data=permutations(iterable=noise_set, r=2),
        columns=[TARGET_COL, LABEL_COL],
    )
