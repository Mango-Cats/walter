"""
Single source of truth for the walter pipeline.
All paths, column names, and tunable parameters live here.
No other file should hardcode these values.
"""

from enum import Enum, auto
from pathlib import Path




class DataSource(Enum):
    PH = auto()   # Philippine FDA registry  →  _data/drug_set_ph.csv
    US = auto()   # US FDA registry          →  _data/drug_set_us.csv


# Active data source — change this to switch registries
DATA_SOURCE: DataSource = DataSource.PH




DATA_DIR    = Path("_data")
RESULTS_DIR = Path("_results")

# Models are downloaded here by scripts/model_setup.py
MODELS_DIR  = Path("models")





# Raw drug name registries — one-column CSVs, header ignored, first column used
R: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph.csv",
    DataSource.US: DATA_DIR / "R_us.csv",
}

# Confirmed LASA pairs input — used when FROM_FILE = True
# Must have columns x_1, x_2 (see P_INPUT_COLS below)
P_INPUT_CSV: Path = DATA_DIR / "P.csv"





# R_clean is an in-memory intermediate only — not saved to disk

# Final output
D_OUT_CSV: Path = RESULTS_DIR / "D.csv"         # full assembled dataset





# Single-column name in the raw / cleaned registry
REGISTRY_COL: str = "drug_name"

# Pair dataset columns
COL_X1:    str = "x_1"      # drug name A
COL_X2:    str = "x_2"      # drug name B
COL_T1:    str = "t_1"      # IPA transcription of x_1
COL_T2:    str = "t_2"      # IPA transcription of x_2
COL_LABEL: str = "label"    # 1 = known positive (LASA), 0 = unlabeled

# P.csv (input) must have exactly these two columns
P_INPUT_COLS: list[str] = [COL_X1, COL_X2]





# Target ratio of unlabeled pairs to confirmed positives
UNLABELED_TO_POSITIVE_RATIO: int = 30

# Fraction of U drawn from each tier
TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 0.35

# Number of outside-vocabulary names pre-sampled for Tier 2
# C(10000, 2) ≈ 50M candidate pairs — lower if runtime is a concern
TIER_2_SAMPLE_SIZE: int = 10_000

# Minimum similarity for a pair to qualify for U (ANY measure)
SIMILARITY_THRESHOLD: int = 20

# Labels
POSITIVE_LABEL:  int = 1
UNLABELED_LABEL: int = 0

# Random seed — set to None for non-deterministic runs
SEED: int = 42





# If True, read confirmed LASA pairs from P_CSV.
# If False, generate them via the local LLM proposer.
FROM_FILE: bool = False

# LLM proposer settings (only used when FROM_FILE = False)
LLM_ITERATIONS:   int = 400
LLM_N_PROPOSALS:  int = 5
LLM_OUTPUT_JSON:  Path = RESULTS_DIR / "lasa_run.json"

# Same file as LLM_OUTPUT_JSON — used by Section 7 of the notebook to mine
# unselected candidates (candidates - x_2) as additional label=0 pairs.
LASA_RUN_JSON: Path = LLM_OUTPUT_JSON

# Standalone CSV of (x_1, unselected candidate) pairs — see Section 7
LASA_RUN_U_CSV: Path = RESULTS_DIR / "lasa_run_U.csv"

# If True, use the DeepSeek API instead of a local model for generation.
# Requires openai package: uv add openai  (or pip install openai)
USE_API_MODEL: bool = True

# DeepSeek API settings (only used when USE_API_MODEL = True)
DEEPSEEK_MODEL:   str = "deepseek-v4-pro"
DEEPSEEK_API_KEY: str = "------------------------------------------------------"





# Batch size for transcription — tune down if eSpeak is slow on your machine
IPA_BATCH_SIZE: int = 256





# Random state for final shuffle
SHUFFLE_SEED: int = 67