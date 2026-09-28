"""Settings for sampling non-confusable drug pairs (U).

This module defines proportions, similarity thresholds, random seeds, and pool
limits used to generate negative samples without running out of memory.
"""

POSITIVE_PREVALENCE: float = 1 / 451

TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 1 - TIER_1_PROPORTION

TIER_2_SAMPLE_SIZE: int = 10_000
TIER_2_MAX_POOL_PER_CLUSTER: int = 300

CANDIDATE_OVERSAMPLE_FACTOR: int = 4
CANDIDATE_MIN_POOL: int = 50

SIMILARITY_THRESHOLD: int = 65

SEED: int = 42
SHUFFLE_SEED: int = 67
