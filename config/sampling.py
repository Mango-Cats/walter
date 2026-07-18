"""
How the unlabeled set U is built (src/noise.py): class balance, the Tier 1 /
Tier 2 split, and the caps that keep the pairwise scoring from blowing up
memory on "hub" clusters.

Most of the caps below exist because of a specific OOM — read the comments
before raising one.
"""

# Target share of D that is confirmed positives, i.e. |P| / (|P| + |U|).
# 1/451 reproduces the previous 1:450 U-to-P ratio.
POSITIVE_PREVALENCE: float = 1 / 451

# Fraction of U drawn from each tier
TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 1 - TIER_1_PROPORTION

# Total outside-vocabulary names sampled for Tier 2, split evenly across
# clusters (each cluster only ever scores pairs within its own share —
# see src/noise.py for why sampling must stay per-cluster)
TIER_2_SAMPLE_SIZE: int = 10_000

# Hard cap on a cluster's combined Tier 1 + Tier 2-extra pool before it's
# fed into Tier 2's pairwise combinations. Some anchors are generic
# enough (short, digit-heavy names) that Soundex/Metaphone collide with a
# large slice of the outside vocabulary — is_similar_enough() qualifies
# on ANY of WRatio/Soundex/Metaphone, so that collision isn't filtered by
# SIMILARITY_THRESHOLD. Without this cap, one such "hub" cluster turns
# into combinations(n, 2) with n in the thousands, which is quadratic in
# both CPU and the number of qualifying rows held in memory before
# down-sampling — that's what blows up RAM. Oversized pools are
# subsampled down to this size (seeded, so still deterministic).
TIER_2_MAX_POOL_PER_CLUSTER: int = 300

# Oversample factor: cap each cluster's accumulated Tier-1/Tier-2 candidates at
# factor × that cluster's tier target before down-sampling. This bounds memory
# to O(|U|) instead of O(all qualifying pairs). Tier 1 was previously uncapped,
# so on a large registry "hub" anchors (short/digit-heavy names that Soundex/
# Metaphone-collide with a huge slice of the vocabulary) made the candidate list
# grow into the millions of rows and OOM-killed the process. We only ever need
# an oversample of each cluster's target to still draw a representative random
# sample, so accumulation stops once a cluster hits its cap. Higher = more
# sampling diversity but more memory.
CANDIDATE_OVERSAMPLE_FACTOR: int = 4

# Floor on a cluster's candidate cap, so tiny clusters (target of 0-1) still
# accumulate a small spread to sample from rather than the first match only.
CANDIDATE_MIN_POOL: int = 50

# Minimum similarity for a pair to qualify for U (ANY measure)
SIMILARITY_THRESHOLD: int = 65

# Random seed — set to None for non-deterministic runs
SEED: int = 42

# Random state for final shuffle
SHUFFLE_SEED: int = 67
