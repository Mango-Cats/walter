"""Finds the core word of a drug name.

Registry names often carry salts, dosage forms, strengths, and product qualifiers,
such as "sevelamer hydrochloride" or "tored 10". These words inflate the distance
between names that a person would easily confuse. The core word is scored in
addition to the full name, never in place of it, because the extra words can be
what separates two products.
"""

import re

from config import CORE_LEADING_EXCLUDED, CORE_LEADING_WORDS, CORE_NAME_STOPWORDS

_HAS_DIGIT = re.compile(r"\d")


def core_name(name: str) -> str:
    """Return the part of a drug name that identifies the product.

    Skips stopwords, words containing digits, and single letters. A name opening
    with a leading word keeps the next usable word too, including a salt, so
    "neo penotran" and "sodium chloride" stay whole.

    Args:
        name: Cleaned drug name string.

    Returns:
        The core word or words, or the first word if every word would be skipped.

    """
    words = str(name).split()
    if words and words[0] in CORE_LEADING_WORDS:
        for word in words[1:]:
            if (
                word not in CORE_LEADING_EXCLUDED
                and not _HAS_DIGIT.search(word)
                and len(word) > 1
            ):
                return f"{words[0]} {word}"
        return words[0]
    for word in words:
        if (
            word not in CORE_NAME_STOPWORDS
            and not _HAS_DIGIT.search(word)
            and len(word) > 1
        ):
            return word
    return words[0] if words else ""


def stripped_name(name: str) -> str:
    """Remove stopwords and words containing digits from a drug name.

    Args:
        name: Cleaned drug name string.

    Returns:
        The remaining words, or the whole name if none remain.

    """
    words = str(name).split()
    kept = [
        w for w in words if w not in CORE_NAME_STOPWORDS and not _HAS_DIGIT.search(w)
    ]
    return " ".join(kept or words)


def is_variant_pair(a: str, b: str) -> bool:
    """Check whether two names differ only in strength, salt, form, or qualifier.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        True if the names match once those words are removed.

    """
    return stripped_name(a) == stripped_name(b)
