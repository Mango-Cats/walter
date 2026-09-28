"""Feature engineering for drug name pairs.

This module computes structural, prosodic, and phonetic similarity features
between two drug names (x_1 and x_2) to help machine learning models decide
how similar or confusable they are:
    - Structural: Word length difference, common prefixes, common suffixes.
    - Prosodic: Differences in syllable counts, vowels, and consonants.
    - Filipino Phonetics: Shared initial consonants, ending sounds, and vowel shapes
      after adapting English words to Filipino pronunciation.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from config import COL_X1, COL_X2
from src.adapters.tbb import adapt as _adapt
from src.adapters.tbb import nativize as _nativize

_VOWELS: frozenset[str] = frozenset("aeiou")


def len_diff(x1: str, x2: str) -> int:
    """Calculate the absolute difference in character length between two drug names.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Absolute difference in length as an integer.

    """
    return abs(len(x1) - len(x2))


def common_prefix_len(x1: str, x2: str) -> int:
    """Count the number of starting characters shared by both drug names.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Length of the common prefix as an integer.

    """
    n = 0
    for c1, c2 in zip(x1, x2):
        if c1 != c2:
            break
        n += 1
    return n


def common_suffix_len(x1: str, x2: str) -> int:
    """Count the number of trailing characters shared by both drug names.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Length of the common suffix as an integer.

    """
    return common_prefix_len(x1[::-1], x2[::-1])


def _vowel_seq(word: str) -> str:
    """Extract the vowel sequence of a word under Filipino 3-vowel reduction (e->i, o->u).

    Args:
        word: Input word string.

    Returns:
        Reduced vowel sequence string.

    """
    nat = _nativize(word)
    return "".join(
        "i" if v == "e" else "u" if v == "o" else v for v in nat if v in _VOWELS
    )


def fil_onset_match(x1: str, x2: str) -> int:
    """Check whether both drug names start with the same sound in Filipino pronunciation.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        1 if initial sounds match, 0 otherwise.

    """
    a, b = _nativize(x1), _nativize(x2)
    return int(bool(a) and bool(b) and a[0] == b[0])


def fil_coda_match(x1: str, x2: str) -> int:
    """Check whether both drug names end with the same sound in Filipino pronunciation.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        1 if final sounds match, 0 otherwise.

    """
    a, b = _nativize(x1), _nativize(x2)
    return int(bool(a) and bool(b) and a[-1] == b[-1])


def fil_vowel_skeleton_match(x1: str, x2: str) -> int:
    """Check whether both drug names have identical vowel patterns in Filipino pronunciation.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        1 if vowel patterns match, 0 otherwise.

    """
    return int(_vowel_seq(x1) == _vowel_seq(x2))


def fil_penult_vowel_match(x1: str, x2: str) -> int:
    """Check whether both drug names share the same second-to-last (penultimate) vowel.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        1 if penultimate vowels match, 0 otherwise.

    """
    v1, v2 = _vowel_seq(x1), _vowel_seq(x2)
    p1 = v1[-2] if len(v1) >= 2 else (v1[-1:] or "")
    p2 = v2[-2] if len(v2) >= 2 else (v2[-1:] or "")
    return int(p1 != "" and p1 == p2)


def fil_phonetic_equal(x1: str, x2: str) -> int:
    """Check whether two drug names become identical after Filipino pronunciation adaptation.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        1 if nativized names are identical, 0 otherwise.

    """
    return int(_nativize(x1) == _nativize(x2))


def n_marked(x1: str, x2: str) -> int:
    """Count how many names in the pair received penult-length vowel stress marking.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Total count (0, 1, or 2) of stress-marked names.

    """
    a, b = _adapt(x1).stressed, _adapt(x2).stressed
    return sum(c.isupper() for c in a) + sum(c.isupper() for c in b)


def english_prominence_match(x1: str, x2: str) -> int:
    """Check whether both words have the same English stress placement (penultimate vs other).

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        1 if stress placement matches, 0 otherwise.

    """
    a = _adapt(x1).english_stress_on_penult
    b = _adapt(x2).english_stress_on_penult
    if a is None or b is None:
        return 0
    return int(a == b)


def _count_syllables(s: str) -> int:
    """Estimate the syllable count of a string from consecutive vowel runs.

    Args:
        s: Input text string.

    Returns:
        Estimated syllable count.

    """
    count = 0
    prev_vowel = False
    for c in s.lower():
        is_vowel = c in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return count


def _count_vowels(s: str) -> int:
    """Count the total number of vowel letters in a string.

    Args:
        s: Input text string.

    Returns:
        Total vowel count.

    """
    return sum(c in _VOWELS for c in s.lower())


def _count_consonants(s: str) -> int:
    """Count the total number of consonant letters in a string.

    Args:
        s: Input text string.

    Returns:
        Total consonant count.

    """
    return sum(c.isalpha() and c.lower() not in _VOWELS for c in s)


def syllable_diff(x1: str, x2: str) -> int:
    """Calculate the absolute difference in estimated syllable counts.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Absolute syllable difference.

    """
    return abs(_count_syllables(x1) - _count_syllables(x2))


def _fil_syllable_count(word: str) -> int:
    """Count syllables in a word using Filipino syllabification rules.

    Args:
        word: Input word string.

    Returns:
        Number of syllables.

    """
    syllabified = _adapt(word).syllabified
    return len(syllabified.split("-")) if syllabified else 0


def syllable_count_diff(x1: str, x2: str) -> int:
    """Calculate the absolute difference in Filipino syllabified syllable counts.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Absolute difference in Filipino syllable counts.

    """
    return abs(_fil_syllable_count(x1) - _fil_syllable_count(x2))


def vowel_count_diff(x1: str, x2: str) -> int:
    """Calculate the absolute difference in vowel counts between two names.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Absolute difference in vowel counts.

    """
    return abs(_count_vowels(x1) - _count_vowels(x2))


def consonant_count_diff(x1: str, x2: str) -> int:
    """Calculate the absolute difference in consonant counts between two names.

    Args:
        x1: First drug name.
        x2: Second drug name.

    Returns:
        Absolute difference in consonant counts.

    """
    return abs(_count_consonants(x1) - _count_consonants(x2))


FEATURE_REGISTRY: dict[str, Callable[[str, str], float | int]] = {
    "len_diff": len_diff,
    "common_prefix_len": common_prefix_len,
    "common_suffix_len": common_suffix_len,
    "consonant_count_diff": consonant_count_diff,
    "syllable_diff": syllable_diff,
    "syllable_count_diff": syllable_count_diff,
    "vowel_count_diff": vowel_count_diff,
    "fil_vowel_skeleton_match": fil_vowel_skeleton_match,
    "fil_penult_vowel_match": fil_penult_vowel_match,
    "fil_onset_match": fil_onset_match,
    "fil_coda_match": fil_coda_match,
    "fil_phonetic_equal": fil_phonetic_equal,
    "n_marked": n_marked,
    "english_prominence_match": english_prominence_match,
}


def engineer(
    df: pd.DataFrame,
    x1_col: str = COL_X1,
    x2_col: str = COL_X2,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add all engineered feature columns to a DataFrame.

    Args:
        df: Input DataFrame containing pair columns.
        x1_col: Column name for first drug name.
        x2_col: Column name for second drug name.

    Returns:
        A tuple of (updated_df, added_column_names, skipped_column_names).

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
    """Compute engineered features for pairs in a CSV and save the result.

    Args:
        input_csv: Path to input CSV containing drug pairs.
        output_csv: Path to write the feature-augmented CSV.
        x1_col: Column name for first drug name.
        x2_col: Column name for second drug name.

    Returns:
        List of feature column names that were added.

    Raises:
        ValueError: If required drug name columns are missing from the input CSV.

    """
    df = pd.read_csv(input_csv)

    missing = [c for c in (x1_col, x2_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    df, added, _skipped = engineer(df, x1_col, x2_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return added
