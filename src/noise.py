"""
PU Learning Dataset Construction for LASA Drug Name Detection
=============================================================
Constructs the unlabeled set U using two-tier similarity-filtered sampling.

Tier 1 (~65%): Anchor-based hard negatives from P-vocabulary.
               For each known drug x, find candidates i outside P-vocabulary
               that are similar under ANY of: orthographic, phonetic measures.
               These are the critical hard cases near the decision boundary.

Tier 2 (~35%): Broader coverage from outside P-vocabulary.
               Pre-sample a tractable subset of outside-vocabulary names,
               then find similar pairs within that subset.
               Avoids O(n^2) over 56,000 names (1.5B pairs — infeasible).

Similarity gate uses multiple measures (Option B) to avoid biasing U
toward pairs detectable only by Levenshtein. A pair qualifies if it
exceeds the threshold on ANY of: WRatio, Soundex match, Metaphone match.
"""

from __future__ import annotations

import random
import unicodedata
from itertools import combinations

import jellyfish  # pip install jellyfish  (Soundex + Metaphone)
import pandas as pd
from rapidfuzz import fuzz  # pip install rapidfuzz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILE_DIR = "data/"
PRIMARY_FNAME = "drug_products.csv"
CLEANED_FNAME = FILE_DIR + "cleaned_" + PRIMARY_FNAME.removesuffix(".csv")

TARGET_COL = "Brand Name"
LABEL_COL = "Confusible"
TRUE_EXPECTED_COLS: list[str] = [TARGET_COL, LABEL_COL]

SIMILARITY_THRESHOLD: int = 20

DEFAULT_UNLABELED_TO_POSITIVE_RATIO: int = 30

TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 0.35

TIER_2_SAMPLE_SIZE: int = 2_000

# Labels
UNLABELED_LABEL: int = 0
POSITIVE_LABEL: int = 1


def validate_columns(true_df: pd.DataFrame) -> None:
    """Validates that true_df has exactly the expected columns."""
    cols = list(true_df.columns)
    if cols != TRUE_EXPECTED_COLS:
        raise ValueError(
            f"Invalid true_df columns. Found: {cols}. Expected: {TRUE_EXPECTED_COLS}."
        )


def normalize_name(name: str) -> str:
    """
    Lowercase, strip whitespace, remove diacritics.
    Does NOT strip digits (e.g. B12 is meaningful).
    """
    name = name.strip().lower()
    # Decompose unicode and drop combining characters (diacritics)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


def get_positive_set(true_df: pd.DataFrame) -> set[str]:
    """All unique drug names appearing in either column of true_df."""
    return set(true_df[TARGET_COL]) | set(true_df[LABEL_COL])


def get_positive_pairs(true_df: pd.DataFrame) -> set[frozenset]:
    """
    Known positive LASA pairs as frozensets so (A, B) == (B, A).
    Uses normalized names to avoid missing pairs due to case differences.
    """
    return {
        frozenset([normalize_name(row[TARGET_COL]), normalize_name(row[LABEL_COL])])
        for _, row in true_df.iterrows()
    }


def _soundex_match(a: str, b: str) -> bool:
    """True if both names share the same Soundex code."""
    try:
        return jellyfish.soundex(a) == jellyfish.soundex(b)
    except Exception:
        return False


def _metaphone_match(a: str, b: str) -> bool:
    """True if both names share the same Metaphone code."""
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

    A pair qualifies if ANY of the following hold:
      1. fuzz.WRatio(a, b) >= threshold   (orthographic / Levenshtein-based)
      2. Soundex codes match              (phonetic, coarse)
      3. Metaphone codes match            (phonetic, finer-grained)

    Using ANY rather than ALL avoids the Levenshtein bias you flagged:
    phonetically similar but orthographically distant pairs (Xanax/Zantac)
    will clear via Soundex/Metaphone even if WRatio is low.
    """
    score = fuzz.WRatio(a, b)
    if score >= threshold:
        return True, score
    if _soundex_match(a, b):
        return True, score
    if _metaphone_match(a, b):
        return True, score
    return False, score


def build_tier_1_negatives(
    p_vocabulary: set[str],
    full_registry: list[str],
    positive_pairs: set[frozenset],
    similarity_threshold: int = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Tier 1: Anchor-based hard negatives.

    For each anchor x in P-vocabulary, exhaustively score all candidates i
    outside P-vocabulary. Include (x, i) if:
      - frozenset({x, i}) not in positive_pairs
      - is_similar_enough(x, i) is True

    Complexity: O(|p_vocabulary| * |outside_vocab|)
               = O(537 * ~55,000) = ~29M WRatio calls.
    rapidfuzz makes each call ~1–2 µs, so ~30–60s total. Acceptable.
    """
    outside_vocabulary = [
        d
        for d in full_registry
        if normalize_name(d) not in {normalize_name(v) for v in p_vocabulary}
    ]

    rows = []
    for anchor in sorted(p_vocabulary):
        anchor_norm = normalize_name(anchor)
        for candidate in outside_vocabulary:
            candidate_norm = normalize_name(candidate)
            pair = frozenset([anchor_norm, candidate_norm])
            if pair in positive_pairs:
                continue
            qualifies, score = is_similar_enough(
                anchor_norm, candidate_norm, similarity_threshold
            )
            if qualifies:
                rows.append(
                    {
                        TARGET_COL: anchor,
                        LABEL_COL: candidate,
                        "similarity": score,
                        "tier": 1,
                        LABEL_COL + "_label": UNLABELED_LABEL,
                    }
                )
    return rows


def build_tier_2_negatives(
    p_vocabulary: set[str],
    full_registry: list[str],
    positive_pairs: set[frozenset],
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    seed: int | None = None,
) -> list[dict]:
    """
    Tier 2: Broader coverage outside P-vocabulary.

    Pre-samples `tier_2_sample_size` names from outside P-vocabulary,
    then does exhaustive pairwise scoring within that sample.

    C(2000, 2) = ~2M pairs — fast.
    C(55000, 2) = ~1.5B pairs — infeasible, hence the pre-sample.

    The pre-sample is random (with seed for reproducibility), which means
    Tier 2 coverage is approximate. Document this as a limitation.
    """
    rng = random.Random(seed)

    outside_vocabulary = [
        d
        for d in full_registry
        if normalize_name(d) not in {normalize_name(v) for v in p_vocabulary}
    ]

    # Pre-sample to make O(n^2) tractable
    if len(outside_vocabulary) > tier_2_sample_size:
        outside_sample = rng.sample(outside_vocabulary, tier_2_sample_size)
    else:
        outside_sample = outside_vocabulary

    rows = []
    for drug_a, drug_b in combinations(outside_sample, 2):
        a_norm = normalize_name(drug_a)
        b_norm = normalize_name(drug_b)
        pair = frozenset([a_norm, b_norm])
        if pair in positive_pairs:
            continue
        qualifies, score = is_similar_enough(a_norm, b_norm, similarity_threshold)
        if qualifies:
            rows.append(
                {
                    TARGET_COL: drug_a,
                    LABEL_COL: drug_b,
                    "similarity": score,
                    "tier": 2,
                    LABEL_COL + "_label": UNLABELED_LABEL,
                }
            )
    return rows


def make_unlabeled_set(
    true_df: pd.DataFrame,
    full_registry: pd.DataFrame,
    ratio: int = DEFAULT_UNLABELED_TO_POSITIVE_RATIO,
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_1_proportion: float = TIER_1_PROPORTION,
    tier_2_proportion: float = TIER_2_PROPORTION,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Constructs the unlabeled set U for PU learning.

    U is composed of two tiers of similarity-filtered drug name pairs,
    neither of which contains any known positive pair from true_df.
    Total size is capped at ratio * |P|.

    Args:
        true_df:             DataFrame of confirmed LASA pairs [TARGET_COL, LABEL_COL].
        full_registry:       DataFrame of all FDA-registered drug names [TARGET_COL].
        ratio:               Target |U| / |P| ratio.
        similarity_threshold:Minimum score (0-100) for ANY similarity measure.
        tier_1_proportion:   Fraction of U from Tier 1 (anchor-based hard negatives).
        tier_2_proportion:   Fraction of U from Tier 2 (broader coverage).
        tier_2_sample_size:  How many outside-vocab names to pre-sample for Tier 2.
        seed:                Random seed for reproducibility.

    Returns:
        DataFrame with columns: TARGET_COL, LABEL_COL, similarity, tier, <LABEL_COL>_label.
        All rows have <LABEL_COL>_label = 0 (unlabeled).
    """
    if abs(tier_1_proportion + tier_2_proportion - 1.0) > 1e-6:
        raise ValueError(
            f"<walter>\ttier_1_proportion + tier_2_proportion must equal 1.0. "
            f"\tGot {tier_1_proportion} + {tier_2_proportion} = {tier_1_proportion + tier_2_proportion}."
        )

    validate_columns(true_df)

    if seed is not None:
        random.seed(seed)

    p_vocabulary = get_positive_set(true_df)
    positive_pairs = get_positive_pairs(true_df)
    full_registry_list = full_registry[TARGET_COL].dropna().tolist()

    num_positives = len(positive_pairs)
    if num_positives == 0:
        raise ValueError("No positive pairs found in true_df. Cannot construct U.")

    target_total = num_positives * ratio
    tier_1_target = int(target_total * tier_1_proportion)
    tier_2_target = int(target_total * tier_2_proportion)

    print(f"<walter> P-vocabulary size:      {len(p_vocabulary):,}")
    print(f"<walter> Known positive pairs:   {num_positives:,}")
    print(f"<walter> FDA registry size:      {len(full_registry_list):,}")
    print(
        f"<walter> Target |U|:             {target_total:,}  (ratio 1:{ratio})"
    )
    print(
        f"<walter> Similarity threshold:   {similarity_threshold} (ANY measure)"
    )
    print(f"<walter> Tier 1 target:          {tier_1_target:,}")
    print(f"<walter> Tier 2 target:          {tier_2_target:,}")

    print("\n<walter> Building anchor-based hard negatives...")
    tier_1_candidates = build_tier_1_negatives(
        p_vocabulary=p_vocabulary,
        full_registry=full_registry_list,
        positive_pairs=positive_pairs,
        similarity_threshold=similarity_threshold,
    )
    print(f"<walter> Candidates generated: {len(tier_1_candidates):,}")

    print(
        f"\n<walter> Building broader coverage (pre-sample size: {tier_2_sample_size:,})..."
    )
    tier_2_candidates = build_tier_2_negatives(
        p_vocabulary=p_vocabulary,
        full_registry=full_registry_list,
        positive_pairs=positive_pairs,
        similarity_threshold=similarity_threshold,
        tier_2_sample_size=tier_2_sample_size,
        seed=seed,
    )
    print(f"<walter> Candidates generated: {len(tier_2_candidates):,}")

    tier_1_sampled = (
        random.sample(tier_1_candidates, min(tier_1_target, len(tier_1_candidates)))
        if tier_1_candidates
        else []
    )
    tier_1_shortfall = tier_1_target - len(tier_1_sampled)
    if tier_1_shortfall > 0:
        print(
            f"\n<walter>\t[WARNING] Tier 1 short by {tier_1_shortfall:,} pairs "
            f"\t(threshold may be too strict). Reallocating to Tier 2."
        )
        tier_2_target += tier_1_shortfall

    tier_2_sampled = (
        random.sample(tier_2_candidates, min(tier_2_target, len(tier_2_candidates)))
        if tier_2_candidates
        else []
    )
    tier_2_shortfall = tier_2_target - len(tier_2_sampled)
    if tier_2_shortfall > 0:
        print(
            f"<walter>\t[WARNING] Tier 2 short by {tier_2_shortfall:,} pairs. "
            f"\tTotal U will be smaller than target. Consider lowering similarity_threshold "
            f"\tor increasing tier_2_sample_size."
        )

    all_unlabeled = tier_1_sampled + tier_2_sampled
    if not all_unlabeled:
        raise ValueError(
            "No unlabeled pairs generated. "
            "similarity_threshold is likely too strict — try lowering it."
        )

    u_df = pd.DataFrame(all_unlabeled)

    print(f"\n<walter> Final Tier 1 pairs:    {len(tier_1_sampled):,}")
    print(f"<walter> Final Tier 2 pairs:    {len(tier_2_sampled):,}")
    print(f"<walter> Total U:               {len(u_df):,}")
    print(
        f"<walter> Actual ratio (U/P):    1:{len(u_df) / num_positives:.1f}"
    )

    return u_df


def make_noise(
    true_df: pd.DataFrame,
    full_registry: pd.DataFrame,
    ratio: int = DEFAULT_UNLABELED_TO_POSITIVE_RATIO,
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Creates the unlabeled set U for PU learning.

    Wrapper around make_unlabeled_set for convenience.

    Args:
        true_df:        Confirmed LASA pairs [TARGET_COL, LABEL_COL].
        full_registry:  FDA registry [TARGET_COL].
        ratio:          |U| / |P| target ratio.
        similarity_threshold: Minimum similarity for inclusion in U.
        tier_2_sample_size:   Pre-sample size for Tier 2.
        seed:           Random seed.

    Returns:
        DataFrame with columns:
          TARGET_COL, LABEL_COL, similarity, tier, <LABEL_COL>_label
        where all rows have <LABEL_COL>_label = 0 (unlabeled).
    """
    return make_unlabeled_set(
        true_df=true_df,
        full_registry=full_registry,
        ratio=ratio,
        similarity_threshold=similarity_threshold,
        tier_2_sample_size=tier_2_sample_size,
        seed=seed,
    )
