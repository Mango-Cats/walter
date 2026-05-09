"""
Constructs the unlabeled set U for PU learning via two-tier
similarity-filtered sampling.

Tier 1 (~65%): Anchor-based hard negatives.
    For each known LASA drug x, find registry names outside the
    P-vocabulary that are similar under ANY of: WRatio, Soundex, Metaphone.
    These are the hard near-boundary cases.

Tier 2 (~35%): Broader coverage.
    Pre-sample TIER_2_SAMPLE_SIZE names from outside the P-vocabulary,
    then score all pairs within that sample. Avoids O(n²) over the full
    registry. Coverage is approximate (seeded random sample).

A pair qualifies for U if it exceeds SIMILARITY_THRESHOLD on ANY measure,
and is not already a known positive pair.
"""

import random
import unicodedata
from itertools import combinations

import jellyfish
import pandas as pd
from rapidfuzz import fuzz

from config import (
    COL_X1,
    COL_X2,
    COL_LABEL,
    REGISTRY_COL,
    UNLABELED_LABEL,
    UNLABELED_TO_POSITIVE_RATIO,
    SIMILARITY_THRESHOLD,
    TIER_1_PROPORTION,
    TIER_2_PROPORTION,
    TIER_2_SAMPLE_SIZE,
    SEED,
)


def normalize(name: str) -> str:
    """Lowercase, strip, remove diacritics. Keeps digits."""
    name = name.strip().lower()
    nfd = unicodedata.normalize("NFD", name)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _soundex_match(a: str, b: str) -> bool:
    try:
        return jellyfish.soundex(a) == jellyfish.soundex(b)
    except Exception:
        return False


def _metaphone_match(a: str, b: str) -> bool:
    try:
        return jellyfish.metaphone(a) == jellyfish.metaphone(b)
    except Exception:
        return False


def is_similar_enough(
    a: str,
    b: str,
    threshold: int = SIMILARITY_THRESHOLD,
) -> tuple[bool, int]:
    """
    Returns (qualifies, wratio_score).
    Qualifies if ANY of WRatio >= threshold, Soundex match, Metaphone match.
    Using ANY avoids Levenshtein bias for phonetically similar but
    orthographically distant pairs (e.g. Xanax / Zantac).
    """
    score = fuzz.WRatio(a, b)
    if score >= threshold:
        return True, score
    if _soundex_match(a, b):
        return True, score
    if _metaphone_match(a, b):
        return True, score
    return False, score


def get_positive_vocabulary(pairs_df: pd.DataFrame) -> set[str]:
    """All unique normalized drug names that appear in any confirmed pair."""
    return {normalize(v) for v in pairs_df[COL_X1].tolist() + pairs_df[COL_X2].tolist()}


def get_positive_pairs(pairs_df: pd.DataFrame) -> set[frozenset]:
    """
    Known positive pairs as frozensets so (A,B) == (B,A).
    Normalized to avoid case mismatches.
    """
    return {
        frozenset([normalize(row[COL_X1]), normalize(row[COL_X2])])
        for _, row in pairs_df.iterrows()
    }


def _build_tier_1(
    p_vocabulary: set[str],
    outside: list[str],
    positive_pairs: set[frozenset],
    threshold: int,
) -> list[dict]:
    """
    Tier 1: for each anchor in P-vocabulary, score every outside-vocab name.
    Complexity: O(|p_vocabulary| × |outside|).
    """
    rows = []
    for anchor in sorted(p_vocabulary):
        for candidate in outside:
            pair = frozenset([anchor, candidate])
            if pair in positive_pairs:
                continue
            qualifies, score = is_similar_enough(anchor, candidate, threshold)
            if qualifies:
                rows.append(
                    {
                        COL_X1: anchor,
                        COL_X2: candidate,
                        "similarity": score,
                        "tier": 1,
                        COL_LABEL: UNLABELED_LABEL,
                    }
                )
    return rows


def _build_tier_2(
    p_vocabulary: set[str],
    outside: list[str],
    positive_pairs: set[frozenset],
    threshold: int,
    sample_size: int,
    seed: int | None,
) -> list[dict]:
    """
    Tier 2: pre-sample `sample_size` outside-vocab names, then score all pairs
    within that sample. C(10000,2) ≈ 50M — still fast with rapidfuzz.
    """
    rng = random.Random(seed)
    sample = rng.sample(outside, min(sample_size, len(outside)))

    rows = []
    for a, b in combinations(sample, 2):
        pair = frozenset([a, b])
        if pair in positive_pairs:
            continue
        qualifies, score = is_similar_enough(a, b, threshold)
        if qualifies:
            rows.append(
                {
                    COL_X1: a,
                    COL_X2: b,
                    "similarity": score,
                    "tier": 2,
                    COL_LABEL: UNLABELED_LABEL,
                }
            )
    return rows


def make_noise(
    pairs_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    ratio: int = UNLABELED_TO_POSITIVE_RATIO,
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_1_proportion: float = TIER_1_PROPORTION,
    tier_2_proportion: float = TIER_2_PROPORTION,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    seed: int | None = SEED,
) -> pd.DataFrame:
    """
    Construct and return the unlabeled set U.

    Args:
        pairs_df:            Confirmed LASA pairs DataFrame [COL_X1, COL_X2].
        registry_df:         Cleaned drug registry DataFrame [REGISTRY_COL].
        ratio:               Target |U| / |P| ratio.
        similarity_threshold:Min score for ANY measure to qualify a pair.
        tier_1_proportion:   Fraction of U from Tier 1.
        tier_2_proportion:   Fraction of U from Tier 2 (must sum to 1 with tier_1).
        tier_2_sample_size:  Outside-vocab names pre-sampled for Tier 2.
        seed:                Random seed.

    Returns:
        DataFrame with columns: COL_X1, COL_X2, similarity, tier, COL_LABEL.
        All rows have COL_LABEL = UNLABELED_LABEL (0).
    """
    if abs(tier_1_proportion + tier_2_proportion - 1.0) > 1e-6:
        raise ValueError(
            f"tier_1_proportion + tier_2_proportion must equal 1.0 "
            f"(got {tier_1_proportion} + {tier_2_proportion})"
        )

    if seed is not None:
        random.seed(seed)

    p_vocab = get_positive_vocabulary(pairs_df)
    positive_pairs = get_positive_pairs(pairs_df)

    all_names_norm = [normalize(n) for n in registry_df[REGISTRY_COL].dropna().tolist()]
    outside = [n for n in all_names_norm if n not in p_vocab]

    num_positives = len(positive_pairs)
    if num_positives == 0:
        raise ValueError("No positive pairs found — cannot construct U.")

    target_total = num_positives * ratio
    tier_1_target = int(target_total * tier_1_proportion)
    tier_2_target = int(target_total * tier_2_proportion)

    print(f"\n[noise] P-vocabulary size   : {len(p_vocab):,}")
    print(f"[noise] Known positive pairs: {num_positives:,}")
    print(f"[noise] Registry size       : {len(all_names_norm):,}")
    print(f"[noise] Outside vocab       : {len(outside):,}")
    print(f"[noise] Target |U|          : {target_total:,}  (ratio 1:{ratio})")
    print(f"[noise] Similarity threshold: {similarity_threshold} (ANY measure)")
    print(f"[noise] Tier 2 sample size  : {tier_2_sample_size:,}")

    print("\n[noise] Building Tier 1 (anchor-based hard negatives)...")
    t1_candidates = _build_tier_1(
        p_vocab, outside, positive_pairs, similarity_threshold
    )
    print(f"[noise] Tier 1 candidates: {len(t1_candidates):,}")

    print(
        f"[noise] Building Tier 2 (broader coverage, sample={tier_2_sample_size:,})..."
    )
    t2_candidates = _build_tier_2(
        p_vocab,
        outside,
        positive_pairs,
        similarity_threshold,
        tier_2_sample_size,
        seed,
    )
    print(f"[noise] Tier 2 candidates: {len(t2_candidates):,}")

    rng = random.Random(seed)

    t1_sampled = rng.sample(t1_candidates, min(tier_1_target, len(t1_candidates)))
    shortfall = tier_1_target - len(t1_sampled)
    if shortfall > 0:
        print(
            f"[noise] WARNING: Tier 1 short by {shortfall:,}; reallocating to Tier 2."
        )
        tier_2_target += shortfall

    t2_sampled = rng.sample(t2_candidates, min(tier_2_target, len(t2_candidates)))
    shortfall2 = tier_2_target - len(t2_sampled)
    if shortfall2 > 0:
        print(
            f"[noise] WARNING: Tier 2 short by {shortfall2:,}. "
            "Total U will be smaller than target. "
            "Consider lowering SIMILARITY_THRESHOLD or raising TIER_2_SAMPLE_SIZE."
        )

    all_unlabeled = t1_sampled + t2_sampled
    if not all_unlabeled:
        raise ValueError(
            "No unlabeled pairs generated. SIMILARITY_THRESHOLD may be too strict."
        )

    u_df = pd.DataFrame(all_unlabeled)

    print(f"\n[noise] Final Tier 1: {len(t1_sampled):,}")
    print(f"[noise] Final Tier 2: {len(t2_sampled):,}")
    print(
        f"[noise] Total U     : {len(u_df):,}  (actual ratio 1:{len(u_df) / num_positives:.1f})"
    )

    return u_df
