"""
Feature-engineering step: appends META_FEATURES onto an already-assembled
(and typically phoc'd) pair CSV.

Every feature is a pure function of the two drug names ``(x_1, x_2)`` - cheap,
O(1) per pair, deterministic. They complement the phonetic-similarity columns
phoc adds with orthographic / edit-distance signal (lengths, prefixes,
Levenshtein, Jaro-Winkler, fuzzy ratios, Soundex/Metaphone agreement).

The features here fall into three groups:

* **structural** - length and shared-affix comparisons of the raw strings.
* **prosodic** - syllable / vowel / consonant-count differences, i.e. how the
  two names differ in spoken "weight" and length.
* **phonetic (Filipino nativization)** - indicators describing *structural*
  properties of the pair the way a Filipino (Tagalog) speaker would hear them,
  so a downstream gate can decide which string-similarity score to trust. They
  are indicators, not similarity scores themselves.

FEATURE_REGISTRY is the single source of truth for which columns get added: an
ordered ``{column_name: fn(x_1, x_2) -> value}`` map. Add or remove an entry
here and the pipeline picks it up - column names are taken straight from the
keys. Existing columns are never overwritten (see ``engineer``), so re-running
over a file that already has some META_FEATURES only fills in the missing ones.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from config import COL_X1, COL_X2
from src.adapters.tbb import nativize as _nativize

_VOWELS: frozenset[str] = frozenset("aeiou")


# Structural features (lengths / shared affixes)
def len_diff(x1: str, x2: str) -> int:
    """Absolute difference in raw string length."""
    return abs(len(x1) - len(x2))


def common_prefix_len(x1: str, x2: str) -> int:
    """Number of leading characters the two names share."""
    n = 0
    for c1, c2 in zip(x1, x2):
        if c1 != c2:
            break
        n += 1
    return n


def common_suffix_len(x1: str, x2: str) -> int:
    """Number of trailing characters the two names share."""
    return common_prefix_len(x1[::-1], x2[::-1])


# Phonetic features - Filipino (Tagalog) nativization
def _vowel_seq(word: str) -> str:
    """Nativized vowel skeleton with the native 3-vowel collapse (e→i, o→u)."""
    nat = _nativize(word)
    return "".join(
        "i" if v == "e" else "u" if v == "o" else v for v in nat if v in _VOWELS
    )


def fil_onset_match(x1: str, x2: str) -> int:
    """1 if the nativized initial phonemes agree (shared onset)."""
    a, b = _nativize(x1), _nativize(x2)
    return int(bool(a) and bool(b) and a[0] == b[0])


def fil_coda_match(x1: str, x2: str) -> int:
    """1 if the nativized final phonemes agree (shared coda)."""
    a, b = _nativize(x1), _nativize(x2)
    return int(bool(a) and bool(b) and a[-1] == b[-1])


def fil_vowel_skeleton_match(x1: str, x2: str) -> int:
    """1 if the collapsed vowel sequences are identical (prosodic shape)."""
    return int(_vowel_seq(x1) == _vowel_seq(x2))


def fil_penult_vowel_match(x1: str, x2: str) -> int:
    """1 if the penultimate (default-stress) vowels agree.

    Falls back to the final vowel for monosyllables; 0 if either has no vowel.
    """
    v1, v2 = _vowel_seq(x1), _vowel_seq(x2)
    p1 = v1[-2] if len(v1) >= 2 else (v1[-1:] or "")
    p2 = v2[-2] if len(v2) >= 2 else (v2[-1:] or "")
    return int(p1 != "" and p1 == p2)


def fil_phonetic_equal(x1: str, x2: str) -> int:
    """1 if the two names are homographs after Filipino nativization."""
    return int(_nativize(x1) == _nativize(x2))


# Prosodic features
def _count_syllables(s: str) -> int:
    """Count syllable nuclei as maximal runs of vowel letters."""
    count = 0
    prev_vowel = False
    for c in s.lower():
        is_vowel = c in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return count


def _count_vowels(s: str) -> int:
    return sum(c in _VOWELS for c in s.lower())


def _count_consonants(s: str) -> int:
    return sum(c.isalpha() and c.lower() not in _VOWELS for c in s)


def syllable_diff(x1: str, x2: str) -> int:
    """Absolute difference in syllable counts (prosodic length mismatch)."""
    return abs(_count_syllables(x1) - _count_syllables(x2))


def vowel_count_diff(x1: str, x2: str) -> int:
    """Absolute difference in vowel-nucleus counts (prosodic weight)."""
    return abs(_count_vowels(x1) - _count_vowels(x2))


def consonant_count_diff(x1: str, x2: str) -> int:
    """Absolute difference in consonant counts (segmental complexity)."""
    return abs(_count_consonants(x1) - _count_consonants(x2))


FEATURE_REGISTRY: dict[str, Callable[[str, str], float | int]] = {
    # structural
    "len_diff": len_diff,
    "common_prefix_len": common_prefix_len,
    "common_suffix_len": common_suffix_len,
    "consonant_count_diff": consonant_count_diff,
    # prosodic
    "syllable_diff": syllable_diff,
    "vowel_count_diff": vowel_count_diff,
    "fil_vowel_skeleton_match": fil_vowel_skeleton_match,
    "fil_penult_vowel_match": fil_penult_vowel_match,
    # phonetic (Filipino nativization)
    "fil_onset_match": fil_onset_match,
    "fil_coda_match": fil_coda_match,
    "fil_phonetic_equal": fil_phonetic_equal,
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
