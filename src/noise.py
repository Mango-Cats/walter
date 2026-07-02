"""
Constructs the unlabeled set U for PU learning via two-tier
similarity-filtered sampling, constrained per-cluster so the emitted
name graph stays a union of disjoint components instead of one giant
blob.

Downstream, the LASA classifier splits train/test by connected
component of the name graph (x_1/x_2 pairs = edges). That only works
if the graph actually decomposes into many components. P's confirmed
pairs already do this naturally — each connected component of P is a
LASA confusion group. The risk is entirely in how U is built: if
negatives are drawn from one shared global pool, the same
outside-vocabulary name can end up paired with anchors from two
different P clusters, bridging them into one component.

To avoid that, every outside-vocabulary name is claimed *exclusively*
by at most one cluster (tracked in `owner`). Once claimed, no other
cluster may use it, so no name can ever bridge two clusters.

Tier 1 (~65%): Anchor-based hard negatives, per cluster.
    For each anchor in a cluster, scan the still-unclaimed outside pool
    for names similar under ANY of: WRatio, Soundex, Metaphone. First
    cluster to match a given outside name claims it.

Tier 2 (~35%): Broader coverage, per cluster.
    Each cluster claims a further small random sample of unclaimed
    outside names (TIER_2_SAMPLE_SIZE distributed across all clusters),
    then all pairs within that cluster's combined claimed pool
    (Tier 1 + this sample) are scored pairwise.

A pair qualifies for U if it exceeds SIMILARITY_THRESHOLD on ANY measure,
and is not already a known positive pair. A cluster that would otherwise
end up with zero negatives (all-positive, useless for train/test) gets a
best-effort fallback negative instead.
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
    CLASS_RATIO,
    SIMILARITY_THRESHOLD,
    TIER_1_PROPORTION,
    TIER_2_PROPORTION,
    TIER_2_SAMPLE_SIZE,
    SEED,
)
from src.clustering import build_components


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


def build_clusters(pairs_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Connected components of the confirmed P pairs. Each component is a
    LASA confusion group and becomes one train/test-split-safe cluster.
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
    """Count of confirmed positive edges whose endpoints fall in each cluster."""
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
    """True if `candidate` is unclaimed, or already claimed by this cluster."""
    current = owner.get(candidate)
    return current is None or current == cluster_id


def _build_tier_1(
    clusters: dict[str, set[str]],
    cluster_order: list[str],
    outside: list[str],
    positive_pairs: set[frozenset],
    threshold: int,
    owner: dict[str, str],
) -> dict[str, list[dict]]:
    """
    Per-cluster anchor-based hard negatives. Scans outside names still
    unclaimed by another cluster; on a qualifying match, claims that
    name exclusively for this cluster so it can never bridge to another.
    """
    rows_by_cluster: dict[str, list[dict]] = {cid: [] for cid in clusters}
    for cluster_id in cluster_order:
        for anchor in sorted(clusters[cluster_id]):
            for candidate in outside:
                if not _claimable(candidate, cluster_id, owner):
                    continue
                pair = frozenset([anchor, candidate])
                if pair in positive_pairs:
                    continue
                qualifies, score = is_similar_enough(anchor, candidate, threshold)
                if qualifies:
                    owner[candidate] = cluster_id
                    rows_by_cluster[cluster_id].append(
                        {
                            COL_X1: anchor,
                            COL_X2: candidate,
                            "similarity": score,
                            "tier": 1,
                            COL_LABEL: UNLABELED_LABEL,
                        }
                    )
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
) -> dict[str, list[dict]]:
    """
    Per-cluster broader coverage. Each cluster claims a further small
    random sample of still-unclaimed outside names (not anchored to a
    specific match), then all pairs within its combined claimed pool
    (Tier 1 matches + this sample) are scored pairwise.
    """
    rows_by_cluster: dict[str, list[dict]] = {cid: [] for cid in clusters}
    free = [n for n in outside if n not in owner]
    rng.shuffle(free)

    idx = 0
    for cluster_id in cluster_order:
        extra = free[idx : idx + extra_per_cluster]
        idx += extra_per_cluster
        for name in extra:
            owner[name] = cluster_id

        pool = sorted(tier_1_pool.get(cluster_id, set()) | set(extra))
        for a, b in combinations(pool, 2):
            pair = frozenset([a, b])
            if pair in positive_pairs:
                continue
            qualifies, score = is_similar_enough(a, b, threshold)
            if qualifies:
                rows_by_cluster[cluster_id].append(
                    {
                        COL_X1: a,
                        COL_X2: b,
                        "similarity": score,
                        "tier": 2,
                        COL_LABEL: UNLABELED_LABEL,
                    }
                )
    return rows_by_cluster


def _fallback_negative(
    cluster_id: str,
    members: set[str],
    outside: list[str],
    positive_pairs: set[frozenset],
    owner: dict[str, str],
) -> dict | None:
    """
    Best-effort single negative for a cluster that matched nothing: pick
    the anchor/unclaimed-candidate pair with the highest raw WRatio,
    ignoring the similarity threshold. Keeps the cluster from being
    all-positive (which downstream would drop as useless anyway).
    """
    best = None
    best_score = -1
    for anchor in sorted(members):
        for candidate in outside:
            if not _claimable(candidate, cluster_id, owner):
                continue
            if frozenset([anchor, candidate]) in positive_pairs:
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
    ratio: int = CLASS_RATIO,
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_1_proportion: float = TIER_1_PROPORTION,
    tier_2_proportion: float = TIER_2_PROPORTION,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    seed: int | None = SEED,
) -> pd.DataFrame:
    """
    Construct and return the unlabeled set U, one cluster at a time so
    the resulting name graph is a union of disjoint components.

    Args:
        pairs_df:            Confirmed LASA pairs DataFrame [COL_X1, COL_X2].
        registry_df:         Cleaned drug registry DataFrame [REGISTRY_COL].
        ratio:               Target |U| / |P| ratio.
        similarity_threshold:Min score for ANY measure to qualify a pair.
        tier_1_proportion:   Fraction of U from Tier 1.
        tier_2_proportion:   Fraction of U from Tier 2 (must sum to 1 with tier_1).
        tier_2_sample_size:  Total outside-vocab names sampled for Tier 2,
                              distributed evenly across clusters.
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

    rng = random.Random(seed)

    clusters = build_clusters(pairs_df)
    positive_pairs = get_positive_pairs(pairs_df)
    p_vocab = {n for members in clusters.values() for n in members}

    num_positives = len(positive_pairs)
    if num_positives == 0:
        raise ValueError("No positive pairs found — cannot construct U.")

    all_names_norm = [normalize(n) for n in registry_df[REGISTRY_COL].dropna().tolist()]
    outside = [n for n in all_names_norm if n not in p_vocab]

    cluster_order = list(clusters.keys())
    rng.shuffle(cluster_order)  # claim order shouldn't systematically favor any cluster

    pos_counts = _positives_per_cluster(clusters, positive_pairs)

    target_total = num_positives * ratio
    extra_per_cluster = max(1, tier_2_sample_size // max(1, len(clusters)))

    print(f"\n[noise] P-vocabulary size   : {len(p_vocab):,}")
    print(f"[noise] Known positive pairs: {num_positives:,}")
    print(f"[noise] Clusters (P groups) : {len(clusters):,}")
    print(f"[noise] Registry size       : {len(all_names_norm):,}")
    print(f"[noise] Outside vocab       : {len(outside):,}")
    print(f"[noise] Target |U|          : {target_total:,}  (ratio 1:{ratio})")
    print(f"[noise] Similarity threshold: {similarity_threshold} (ANY measure)")
    print(f"[noise] Tier 2 extra/cluster: {extra_per_cluster:,}")

    owner: dict[str, str] = {}

    print("\n[noise] Building Tier 1 (anchor-based hard negatives, per cluster)...")
    t1_by_cluster = _build_tier_1(
        clusters, cluster_order, outside, positive_pairs, similarity_threshold, owner
    )
    t1_pool = {
        cid: {row[COL_X2] for row in rows} for cid, rows in t1_by_cluster.items()
    }
    print(f"[noise] Tier 1 candidates: {sum(len(v) for v in t1_by_cluster.values()):,}")

    print(f"[noise] Building Tier 2 (broader coverage, per cluster)...")
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
    )
    print(f"[noise] Tier 2 candidates: {sum(len(v) for v in t2_by_cluster.values()):,}")

    all_rows: list[dict] = []
    empty_clusters = 0
    for cluster_id in cluster_order:
        cluster_total = target_total * pos_counts.get(cluster_id, 0) / num_positives
        tier_1_target = round(cluster_total * tier_1_proportion)
        tier_2_target = round(cluster_total * tier_2_proportion)

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
            "(outside vocab exhausted) — they will be all-positive and likely "
            "dropped by the downstream split."
        )

    if not all_rows:
        raise ValueError(
            "No unlabeled pairs generated. SIMILARITY_THRESHOLD may be too strict."
        )

    u_df = pd.DataFrame(all_rows)

    print(f"\n[noise] Total U     : {len(u_df):,}  (actual ratio 1:{len(u_df) / num_positives:.1f})")

    return u_df
