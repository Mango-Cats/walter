"""Constructs the unlabeled set of non-confusable drug pairs (U).

Why this is needed:
To train machine learning models to detect confusable drugs, we need both confusable
pairs (positives) and realistic non-confusable pairs (negatives).

This module samples negatives in two tiers:
    - Tier 1 (~65%): Anchor-based hard negatives that sound or look similar to confirmed drugs.
    - Tier 2 (~35%): Broader random samples within cluster pools to provide general coverage.

Each outside drug is claimed exclusively by one cluster, preventing different clusters
from merging together.
"""

import random
import unicodedata
from itertools import combinations

import jellyfish
import pandas as pd
from rapidfuzz import fuzz

from config import (
    CANDIDATE_MIN_POOL,
    CANDIDATE_OVERSAMPLE_FACTOR,
    COL_LABEL,
    COL_X1,
    COL_X2,
    POSITIVE_PREVALENCE,
    REGISTRY_COL,
    SEED,
    SIMILARITY_THRESHOLD,
    TIER_1_PROPORTION,
    TIER_2_MAX_POOL_PER_CLUSTER,
    TIER_2_PROPORTION,
    TIER_2_SAMPLE_SIZE,
    UNLABELED_LABEL,
)
from src.adapters.tbb import nativize as _nativize
from src.pipeline.clustering import build_components


def normalize(name: str) -> str:
    """Normalize a drug name by lowercasing, stripping whitespace, and removing diacritics.

    Args:
        name: The input drug name string.

    Returns:
        Cleaned and normalized drug name string.

    """
    name = name.strip().lower()
    nfd = unicodedata.normalize("NFD", name)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def is_qualifier_pair(a: str, b: str) -> bool:
    """Check if one drug name is a prefix of another with an added qualifier.

    For example, 'Zantac' vs 'Zantac 360'. Confirmed pairs treat these as confusable,
    so they are excluded from the negative set.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        True if one name is a word-boundary prefix of the other, False otherwise.

    """
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return False
    short, long_ = (na, nb) if len(na) < len(nb) else (nb, na)
    return long_.startswith(short) and long_[len(short) : len(short) + 1] == " "


def _metaphone_match(a: str, b: str) -> bool:
    """Check if two drug names produce the same Metaphone phonetic code.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        True if their Metaphone codes match, False otherwise.

    """
    try:
        return jellyfish.metaphone(a) == jellyfish.metaphone(b)
    except Exception:
        return False


def _nativized_soundex_match(a: str, b: str) -> bool:
    """Check if two drug names produce the same Soundex code under Filipino pronunciation.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        True if their nativized Soundex codes match, False otherwise.

    """
    try:
        return jellyfish.soundex(_nativize(a)) == jellyfish.soundex(_nativize(b))
    except Exception:
        return False


def is_similar_enough(
    a: str,
    b: str,
    threshold: int = SIMILARITY_THRESHOLD,
) -> tuple[bool, int]:
    """Check if two drug names meet any similarity criteria (spelling, Soundex, or Metaphone).

    Args:
        a: First drug name.
        b: Second drug name.
        threshold: Minimum RapidFuzz WRatio score required to qualify.

    Returns:
        A tuple of (qualifies, wratio_score).

    """
    score = fuzz.WRatio(a, b)
    if score >= threshold:
        return True, score
    if _nativized_soundex_match(a, b):
        return True, score
    if _metaphone_match(a, b):
        return True, score
    return False, score


def get_positive_vocabulary(pairs_df: pd.DataFrame) -> set[str]:
    """Return all unique normalized drug names appearing in confirmed confusable pairs.

    Args:
        pairs_df: DataFrame of confirmed confusable pairs.

    Returns:
        Set of unique normalized drug names.

    """
    return {normalize(v) for v in pairs_df[COL_X1].tolist() + pairs_df[COL_X2].tolist()}


def get_positive_pairs(pairs_df: pd.DataFrame) -> set[frozenset]:
    """Return confirmed confusable pairs as frozensets to treat (A, B) and (B, A) identically.

    Args:
        pairs_df: DataFrame of confirmed confusable pairs.

    Returns:
        Set of frozensets, each containing a normalized pair of drug names.

    """
    return {
        frozenset([normalize(row[COL_X1]), normalize(row[COL_X2])])
        for _, row in pairs_df.iterrows()
    }


def build_clusters(pairs_df: pd.DataFrame) -> dict[str, set[str]]:
    """Group confirmed confusable pairs into connected confusion clusters.

    Args:
        pairs_df: DataFrame of confirmed confusable pairs.

    Returns:
        Dictionary mapping each cluster root name to the set of member drug names.

    """
    edges = [
        (normalize(row[COL_X1]), normalize(row[COL_X2]))
        for _, row in pairs_df.iterrows()
    ]
    return build_components(edges)


def _positives_per_cluster(
    clusters: dict[str, set[str]],
    positive_pairs: set[frozenset],
) -> dict[str, int]:
    """Count how many confirmed positive pairs belong to each cluster.

    Args:
        clusters: Dictionary mapping cluster IDs to member drug names.
        positive_pairs: Set of confirmed positive pair frozensets.

    Returns:
        Dictionary mapping cluster IDs to positive pair counts.

    """
    node_to_cluster = {
        name: cid for cid, members in clusters.items() for name in members
    }
    counts = {cid: 0 for cid in clusters}
    for pair in positive_pairs:
        anchor = next(iter(pair))
        cid = node_to_cluster.get(anchor)
        if cid is not None:
            counts[cid] += 1
    return counts


def _claimable(candidate: str, cluster_id: str, owner: dict[str, str]) -> bool:
    """Check whether a candidate name can be claimed by a cluster.

    Args:
        candidate: The drug name under consideration.
        cluster_id: The ID of the claiming cluster.
        owner: Dictionary tracking which cluster currently owns each candidate name.

    Returns:
        True if candidate is unclaimed or already claimed by this cluster, False otherwise.

    """
    current = owner.get(candidate)
    return current is None or current == cluster_id


def _cluster_tier_targets(
    cluster_order: list[str],
    pos_counts: dict[str, int],
    num_positives: int,
    target_total: float,
    tier_1_proportion: float,
    tier_2_proportion: float,
) -> dict[str, tuple[int, int]]:
    """Compute target counts of Tier 1 and Tier 2 negative samples for each cluster.

    Args:
        cluster_order: List of cluster IDs.
        pos_counts: Number of positive pairs per cluster.
        num_positives: Total number of positive pairs.
        target_total: Total target count of negative samples.
        tier_1_proportion: Fraction of negatives from Tier 1.
        tier_2_proportion: Fraction of negatives from Tier 2.

    Returns:
        Dictionary mapping cluster IDs to (tier_1_target, tier_2_target) tuples.

    """
    targets: dict[str, tuple[int, int]] = {}
    for cid in cluster_order:
        cluster_total = target_total * pos_counts.get(cid, 0) / num_positives
        t1 = round(cluster_total * tier_1_proportion)
        t2 = round(cluster_total * tier_2_proportion)
        targets[cid] = (t1, t2)
    return targets


def _accumulation_caps(
    targets: dict[str, tuple[int, int]],
    which: int,
    factor: int,
    floor: int,
) -> dict[str, int]:
    """Calculate the maximum candidates a cluster may accumulate before downsampling.

    Args:
        targets: Dictionary mapping cluster IDs to tier target counts.
        which: Index selecting the tier (0 for Tier 1, 1 for Tier 2).
        factor: Oversample multiplier factor.
        floor: Minimum candidate floor.

    Returns:
        Dictionary mapping cluster IDs to candidate accumulation caps.

    """
    return {cid: max(floor, factor * tgt[which]) for cid, tgt in targets.items()}


def _build_tier_1(
    clusters: dict[str, set[str]],
    cluster_order: list[str],
    outside: list[str],
    positive_pairs: set[frozenset],
    threshold: int,
    owner: dict[str, str],
    caps: dict[str, int],
) -> dict[str, list[dict]]:
    """Sample anchor-based hard negatives (Tier 1) for each cluster.

    Args:
        clusters: Dictionary of clusters and their member drug names.
        cluster_order: Order in which to process clusters.
        outside: List of registry drug names outside the positive vocabulary.
        positive_pairs: Set of confirmed positive pair frozensets.
        threshold: Minimum RapidFuzz WRatio similarity score.
        owner: Dictionary tracking candidate ownership by cluster.
        caps: Per-cluster candidate accumulation caps.

    Returns:
        Dictionary mapping cluster IDs to lists of candidate pair dictionaries.

    """
    rows_by_cluster: dict[str, list[dict]] = {cid: [] for cid in clusters}
    for cluster_id in cluster_order:
        cap = caps.get(cluster_id, 0)
        rows = rows_by_cluster[cluster_id]
        if cap <= 0:
            continue
        for anchor in sorted(clusters[cluster_id]):
            if len(rows) >= cap:
                break
            for candidate in outside:
                if not _claimable(candidate, cluster_id, owner):
                    continue
                pair = frozenset([anchor, candidate])
                if pair in positive_pairs:
                    continue
                if is_qualifier_pair(anchor, candidate):
                    continue
                qualifies, score = is_similar_enough(anchor, candidate, threshold)
                if qualifies:
                    owner[candidate] = cluster_id
                    rows.append(
                        {
                            COL_X1: anchor,
                            COL_X2: candidate,
                            "similarity": score,
                            "tier": 1,
                            COL_LABEL: UNLABELED_LABEL,
                        }
                    )
                    if len(rows) >= cap:
                        break
    return rows_by_cluster


def _build_tier_2(
    clusters: dict[str, set[str]],
    cluster_order: list[str],
    tier_1_pool: dict[str, set[str]],
    outside: list[str],
    positive_pairs: set[frozenset],
    threshold: int,
    extra_per_cluster: int,
    owner: dict[str, str],
    rng: random.Random,
    caps: dict[str, int],
    max_pool_per_cluster: int = TIER_2_MAX_POOL_PER_CLUSTER,
) -> dict[str, list[dict]]:
    """Sample broader pairwise negatives (Tier 2) for each cluster.

    Args:
        clusters: Dictionary of clusters and their member drug names.
        cluster_order: Order in which to process clusters.
        tier_1_pool: Dictionary of Tier 1 candidate names claimed per cluster.
        outside: List of registry drug names outside the positive vocabulary.
        positive_pairs: Set of confirmed positive pair frozensets.
        threshold: Minimum RapidFuzz WRatio similarity score.
        extra_per_cluster: Number of extra random outside names each cluster claims.
        owner: Dictionary tracking candidate ownership by cluster.
        rng: Random number generator instance.
        caps: Per-cluster candidate accumulation caps.
        max_pool_per_cluster: Maximum combined pool size per cluster to avoid OOM.

    Returns:
        Dictionary mapping cluster IDs to lists of candidate pair dictionaries.

    """
    rows_by_cluster: dict[str, list[dict]] = {cid: [] for cid in clusters}
    free = [n for n in outside if n not in owner]
    rng.shuffle(free)

    capped_clusters = 0
    idx = 0
    for cluster_id in cluster_order:
        extra = free[idx : idx + extra_per_cluster]
        idx += extra_per_cluster
        for name in extra:
            owner[name] = cluster_id

        cap = caps.get(cluster_id, 0)
        rows = rows_by_cluster[cluster_id]
        if cap <= 0:
            continue

        pool = sorted(tier_1_pool.get(cluster_id, set()) | set(extra))
        if len(pool) > max_pool_per_cluster:
            pool = sorted(rng.sample(pool, max_pool_per_cluster))
            capped_clusters += 1
        for a, b in combinations(pool, 2):
            if len(rows) >= cap:
                break
            pair = frozenset([a, b])
            if pair in positive_pairs:
                continue
            if is_qualifier_pair(a, b):
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
    if capped_clusters:
        print(
            f"[noise] Capped {capped_clusters:,} oversized cluster pool(s) to "
            f"{max_pool_per_cluster:,} names before pairwise scoring (hub anchors)"
        )
    return rows_by_cluster


def _fallback_negative(
    cluster_id: str,
    members: set[str],
    outside: list[str],
    positive_pairs: set[frozenset],
    owner: dict[str, str],
) -> dict | None:
    """Find a best-effort negative pair for a cluster with zero qualifying matches.

    Args:
        cluster_id: ID of the cluster needing a negative.
        members: Set of anchor drug names in the cluster.
        outside: List of outside drug names.
        positive_pairs: Set of confirmed positive pair frozensets.
        owner: Dictionary tracking candidate ownership.

    Returns:
        A candidate pair dictionary, or None if no candidate could be claimed.

    """
    best = None
    best_score = -1
    for anchor in sorted(members):
        for candidate in outside:
            if not _claimable(candidate, cluster_id, owner):
                continue
            if frozenset([anchor, candidate]) in positive_pairs:
                continue
            if is_qualifier_pair(anchor, candidate):
                continue
            score = fuzz.WRatio(anchor, candidate)
            if score > best_score:
                best_score = score
                best = (anchor, candidate)
    if best is None:
        return None
    anchor, candidate = best
    owner[candidate] = cluster_id
    return {
        COL_X1: anchor,
        COL_X2: candidate,
        "similarity": best_score,
        "tier": 1,
        COL_LABEL: UNLABELED_LABEL,
    }


def make_noise(
    pairs_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    positive_prevalence: float = POSITIVE_PREVALENCE,
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_1_proportion: float = TIER_1_PROPORTION,
    tier_2_proportion: float = TIER_2_PROPORTION,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    tier_2_max_pool_per_cluster: int = TIER_2_MAX_POOL_PER_CLUSTER,
    candidate_oversample_factor: int = CANDIDATE_OVERSAMPLE_FACTOR,
    candidate_min_pool: int = CANDIDATE_MIN_POOL,
    seed: int | None = SEED,
) -> pd.DataFrame:
    """Construct the unlabeled negative set U using two-tier similarity sampling.

    Args:
        pairs_df: Confirmed confusable drug pairs DataFrame.
        registry_df: Cleaned drug registry DataFrame.
        positive_prevalence: Target share of positive pairs in the final dataset.
        similarity_threshold: Minimum RapidFuzz WRatio score for candidates.
        tier_1_proportion: Fraction of negatives drawn from Tier 1 (anchor-based).
        tier_2_proportion: Fraction of negatives drawn from Tier 2 (broader coverage).
        tier_2_sample_size: Total outside names sampled for Tier 2 across all clusters.
        tier_2_max_pool_per_cluster: Maximum candidate pool size per cluster.
        candidate_oversample_factor: Multiple of target to accumulate before sampling.
        candidate_min_pool: Minimum candidate accumulation floor per cluster.
        seed: Random seed for reproducible sampling.

    Returns:
        A DataFrame of sampled non-confusable pairs with columns x_1, x_2, similarity, tier, and label.

    Raises:
        ValueError: If parameters are invalid or no pairs could be sampled.

    """
    if not 0.0 < positive_prevalence < 1.0:
        raise ValueError(
            f"positive_prevalence must be in (0, 1) (got {positive_prevalence})"
        )
    if abs(tier_1_proportion + tier_2_proportion - 1.0) > 1e-6:
        raise ValueError(
            f"tier_1_proportion + tier_2_proportion must equal 1.0 "
            f"(got {tier_1_proportion} + {tier_2_proportion})"
        )

    rng = random.Random(seed)

    clusters = build_clusters(pairs_df)
    positive_pairs = get_positive_pairs(pairs_df)
    p_vocab = {n for members in clusters.values() for n in members}

    num_positives = len(positive_pairs)
    if num_positives == 0:
        raise ValueError("No positive pairs found - cannot construct U.")

    all_names_norm = [normalize(n) for n in registry_df[REGISTRY_COL].dropna().tolist()]
    outside = [n for n in all_names_norm if n not in p_vocab]
    rng.shuffle(outside)

    cluster_order = list(clusters.keys())
    rng.shuffle(cluster_order)

    pos_counts = _positives_per_cluster(clusters, positive_pairs)

    target_total = round(
        num_positives * (1 - positive_prevalence) / positive_prevalence
    )
    extra_per_cluster = max(1, tier_2_sample_size // max(1, len(clusters)))

    tier_targets = _cluster_tier_targets(
        cluster_order,
        pos_counts,
        num_positives,
        target_total,
        tier_1_proportion,
        tier_2_proportion,
    )
    t1_caps = _accumulation_caps(
        tier_targets, 0, candidate_oversample_factor, candidate_min_pool
    )
    t2_caps = _accumulation_caps(
        tier_targets, 1, candidate_oversample_factor, candidate_min_pool
    )

    print(f"\n[noise] P-vocabulary size   : {len(p_vocab):,}")
    print(f"[noise] Known positive pairs: {num_positives:,}")
    print(f"[noise] Clusters (P groups) : {len(clusters):,}")
    print(f"[noise] Registry size       : {len(all_names_norm):,}")
    print(f"[noise] Outside vocab       : {len(outside):,}")
    print(
        f"[noise] Target |U|          : {target_total:,}  "
        f"(positive prevalence {positive_prevalence:.6f})"
    )
    print(f"[noise] Similarity threshold: {similarity_threshold} (ANY measure)")
    print(f"[noise] Tier 2 extra/cluster: {extra_per_cluster:,}")
    print(f"[noise] Tier 2 max pool/cluster: {tier_2_max_pool_per_cluster:,}")
    print(
        f"[noise] Candidate cap         : {candidate_oversample_factor}x tier target "
        f"(min {candidate_min_pool}); Tier 1 cap total {sum(t1_caps.values()):,}"
    )

    owner: dict[str, str] = {}

    print("\n[noise] Building Tier 1 (anchor-based hard negatives, per cluster)...")
    t1_by_cluster = _build_tier_1(
        clusters,
        cluster_order,
        outside,
        positive_pairs,
        similarity_threshold,
        owner,
        t1_caps,
    )
    t1_pool = {
        cid: {row[COL_X2] for row in rows} for cid, rows in t1_by_cluster.items()
    }
    print(f"[noise] Tier 1 candidates: {sum(len(v) for v in t1_by_cluster.values()):,}")

    print("[noise] Building Tier 2 (broader coverage, per cluster)...")
    t2_by_cluster = _build_tier_2(
        clusters,
        cluster_order,
        t1_pool,
        outside,
        positive_pairs,
        similarity_threshold,
        extra_per_cluster,
        owner,
        rng,
        t2_caps,
        tier_2_max_pool_per_cluster,
    )
    print(f"[noise] Tier 2 candidates: {sum(len(v) for v in t2_by_cluster.values()):,}")

    all_rows: list[dict] = []
    empty_clusters = 0
    for cluster_id in cluster_order:
        tier_1_target, tier_2_target = tier_targets[cluster_id]

        t1_candidates = t1_by_cluster.get(cluster_id, [])
        t2_candidates = t2_by_cluster.get(cluster_id, [])

        t1_sampled = rng.sample(t1_candidates, min(tier_1_target, len(t1_candidates)))
        shortfall = tier_1_target - len(t1_sampled)
        if shortfall > 0:
            tier_2_target += shortfall
        t2_sampled = rng.sample(t2_candidates, min(tier_2_target, len(t2_candidates)))

        cluster_rows = t1_sampled + t2_sampled
        if not cluster_rows:
            fallback = _fallback_negative(
                cluster_id, clusters[cluster_id], outside, positive_pairs, owner
            )
            if fallback is not None:
                cluster_rows = [fallback]
            else:
                empty_clusters += 1
        all_rows.extend(cluster_rows)

    if empty_clusters:
        print(
            f"[noise] WARNING: {empty_clusters:,} cluster(s) got no negatives at all "
            "(outside vocab exhausted) - they will be all-positive and likely "
            "dropped by the downstream split."
        )

    if not all_rows:
        raise ValueError(
            "No unlabeled pairs generated. SIMILARITY_THRESHOLD may be too strict."
        )

    u_df = pd.DataFrame(all_rows)

    print(
        f"\n[noise] Total U     : {len(u_df):,}  "
        f"(actual positive prevalence {num_positives / (num_positives + len(u_df)):.6f})"
    )

    return u_df
