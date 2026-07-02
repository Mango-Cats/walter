"""
Feature-engineering step: appends META_FEATURES onto an already-assembled
(and typically phoc'd) pair CSV.

Every feature is a pure function of the two drug names ``(x_1, x_2)`` — cheap,
O(1) per pair, deterministic. They complement the phonetic-similarity columns
phoc adds with orthographic / edit-distance signal (lengths, prefixes,
Levenshtein, Jaro-Winkler, fuzzy ratios, Soundex/Metaphone agreement).

FEATURE_REGISTRY is the single source of truth for which columns get added: an
ordered ``{column_name: fn(x_1, x_2) -> value}`` map. Add or remove an entry
here and the pipeline picks it up — column names are taken straight from the
keys. Existing columns are never overwritten (see ``engineer``), so re-running
over a file that already has some META_FEATURES only fills in the missing ones.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jellyfish
import pandas as pd
from rapidfuzz import fuzz

from config import COL_X1, COL_X2


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _common_suffix_len(a: str, b: str) -> int:
    return _common_prefix_len(a[::-1], b[::-1])


def _len_ratio(a: str, b: str) -> float:
    longer = max(len(a), len(b))
    return min(len(a), len(b)) / longer if longer else 1.0


# Ordered map of META_FEATURE column name -> pair function. Insertion order is
# the column order in the output CSV.
FEATURE_REGISTRY: dict[str, Callable[[str, str], float]] = {
    "len_1": lambda a, b: len(a),
    "len_2": lambda a, b: len(b),
    "len_diff": lambda a, b: abs(len(a) - len(b)),
    "len_ratio": _len_ratio,
    "prefix_match": _common_prefix_len,
    "suffix_match": _common_suffix_len,
    "same_first_char": lambda a, b: int(bool(a) and bool(b) and a[0] == b[0]),
    "same_last_char": lambda a, b: int(bool(a) and bool(b) and a[-1] == b[-1]),
    "levenshtein": lambda a, b: jellyfish.levenshtein_distance(a, b),
    "damerau_levenshtein": lambda a, b: jellyfish.damerau_levenshtein_distance(a, b),
    "hamming": lambda a, b: jellyfish.hamming_distance(a, b),
    "jaro": lambda a, b: jellyfish.jaro_similarity(a, b),
    "jaro_winkler": lambda a, b: jellyfish.jaro_winkler_similarity(a, b),
    "ratio": lambda a, b: fuzz.ratio(a, b),
    "partial_ratio": lambda a, b: fuzz.partial_ratio(a, b),
    "token_sort_ratio": lambda a, b: fuzz.token_sort_ratio(a, b),
    "wratio": lambda a, b: fuzz.WRatio(a, b),
    "soundex_match": lambda a, b: int(
        bool(a) and bool(b) and jellyfish.soundex(a) == jellyfish.soundex(b)
    ),
    "metaphone_match": lambda a, b: int(
        bool(a) and bool(b) and jellyfish.metaphone(a) == jellyfish.metaphone(b)
    ),
}


def engineer(
    df: pd.DataFrame,
    x1_col: str = COL_X1,
    x2_col: str = COL_X2,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Add every FEATURE_REGISTRY column that isn't already present to ``df``,
    computed from ``x1_col`` / ``x2_col``. Mutates and returns ``df`` along
    with the lists of columns added and skipped (already present).
    """
    added: list[str] = []
    skipped: list[str] = []
    for col, fn in FEATURE_REGISTRY.items():
        if col in df.columns:
            skipped.append(col)
            continue
        df[col] = df.apply(
            lambda row, _fn=fn: _fn(str(row[x1_col]), str(row[x2_col])), axis=1
        )
        added.append(col)
    return df, added, skipped


def run_engineering(
    input_csv: Path,
    output_csv: Path,
    x1_col: str = COL_X1,
    x2_col: str = COL_X2,
) -> list[str]:
    """
    Read ``input_csv``, append META_FEATURES, and write the augmented CSV to
    ``output_csv``. Every input column is preserved verbatim. Returns the list
    of feature columns added.

    Raises ValueError if the required pair columns are missing.
    """
    df = pd.read_csv(input_csv)

    missing = [c for c in (x1_col, x2_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    df, added, _skipped = engineer(df, x1_col, x2_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return added
