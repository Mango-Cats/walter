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

# With an exact noise target, how many times that target to request before
# removing labelled pairs and sampling down.
NOISE_OVERSAMPLE: int = 2

# With an exact noise target, the fraction drawn as unfiltered random registry pairs,
# and whether to leave out pairs differing only in strength, salt, form, or qualifier.
NOISE_RANDOM_SHARE: float = 0.1
NOISE_SKIP_VARIANTS: bool = True

SEED: int = 42
SHUFFLE_SEED: int = 67
