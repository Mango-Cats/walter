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
CLASS_RATIO: int = 30

# Fraction of U drawn from each tier
TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 0.35

# Total outside-vocabulary names sampled for Tier 2, split evenly across
# clusters (each cluster only ever scores pairs within its own share —
# see src/noise.py for why sampling must stay per-cluster)
TIER_2_SAMPLE_SIZE: int = 10_000

# Minimum similarity for a pair to qualify for U (ANY measure)
SIMILARITY_THRESHOLD: int = 20

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
