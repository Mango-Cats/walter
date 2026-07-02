"""
Single source of truth for the walter pipeline.
All paths, column names, and tunable parameters live here.
No other file should hardcode these values.
"""

from enum import Enum, auto
from pathlib import Path


class DataSource(Enum):
    PH = auto()  # Philippine FDA registry  →  _data/drug_set_ph.csv
    US = auto()  # US FDA registry          →  _data/drug_set_us.csv


# Active data source — change this to switch registries
DATA_SOURCE: DataSource = DataSource.PH


DATA_DIR = Path("_data")
RESULTS_DIR = Path("_results")

# Models are downloaded here by scripts/model_setup.py
MODELS_DIR = Path("models")


# Raw drug name registries — one-column CSVs, header ignored, first column used
R: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph.csv",
    DataSource.US: DATA_DIR / "R_us.csv",
}

# Cleaned registry cache — written by src/preprocessing.py every time it
# cleans a raw registry, so a slow clean never has to be repeated.
R_CLEAN: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph_clean.csv",
    DataSource.US: DATA_DIR / "R_us_clean.csv",
}

# If True, preprocessing loads R_CLEAN[source] directly instead of
# re-cleaning R[source]. Toggle on once you have a cached clean registry
# you trust; toggle off (or delete the cache file) to force a re-clean.
USE_PRECLEANED_REGISTRY: bool = False

# Confirmed LASA pairs input — used when FROM_FILE = True
# Must have columns x_1, x_2 (see P_INPUT_COLS below)
P: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "P_ph.csv",
    DataSource.US: DATA_DIR / "P_us.csv",
}

# Final output — classification: pair + label, schema unchanged
D_OUT_CSV: Path = RESULTS_DIR / "D.csv"  # full assembled dataset

# Ranking output — identical rows to D_OUT_CSV plus COL_GROUP, so the
# downstream repo can group rows by connected component for a ranking task
D_RANK_OUT_CSV: Path = RESULTS_DIR / "D_rank.csv"

# Phonetic-feature step — bin/phoc (Rust CLI, see bin/phoc summary). Reads a
# scored pair CSV, preserves every input column, and appends one similarity
# feature column per .toml in PHOC_CONFIG_DIR (column name = file stem).
PHOC_BIN: Path = Path("bin/phoc")
PHOC_CONFIG_DIR: Path = Path("bin/pho_conf")

# phoc outputs — D / D_rank with the phonetic-similarity feature columns added
D_PHO_OUT_CSV: Path = RESULTS_DIR / "D_pho.csv"
D_RANK_PHO_OUT_CSV: Path = RESULTS_DIR / "D_rank_pho.csv"

# Feature-engineering step — src/feature_engineering.py appends META_FEATURES
# (orthographic / edit-distance features) onto the phoc outputs, so _engi is
# "engineered and pho'd": every phonetic column plus the META_FEATURES.
D_ENGI_OUT_CSV: Path = RESULTS_DIR / "D_engi.csv"
D_RANK_ENGI_OUT_CSV: Path = RESULTS_DIR / "D_rank_engi.csv"

# Single-column name in the raw / cleaned registry
REGISTRY_COL: str = "drug_name"

# Pair dataset columns
COL_X1: str = "x_1"  # drug name A
COL_X2: str = "x_2"  # drug name B
COL_T1: str = "t_1"  # IPA transcription of x_1
COL_T2: str = "t_2"  # IPA transcription of x_2
COL_LABEL: str = "label"  # 1 = known positive (LASA), 0 = unlabeled
COL_GROUP: str = "group"  # connected-component id (ranking output only)

# P.csv (input) must have exactly these two columns
P_INPUT_COLS: list[str] = [COL_X1, COL_X2]


# Target ratio of unlabeled pairs to confirmed positives
CLASS_RATIO: int = 450

# Fraction of U drawn from each tier
TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 0.35

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

# Labels
POSITIVE_LABEL: int = 1
UNLABELED_LABEL: int = 0

# Random seed — set to None for non-deterministic runs
SEED: int = 42


# If True, read confirmed LASA pairs from P_CSV.
# If False, generate them via the local LLM proposer.
FROM_FILE: bool = True

# LLM proposer settings (only used when FROM_FILE = False)
LLM_ITERATIONS: int = 400
LLM_N_PROPOSALS: int = 5
LLM_OUTPUT_JSON: Path = RESULTS_DIR / "lasa_run.json"


# Batch size for transcription — tune down if eSpeak is slow on your machine
IPA_BATCH_SIZE: int = 256


# Random state for final shuffle
SHUFFLE_SEED: int = 67
