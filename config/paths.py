"""
Where things live: the active registry, the directories, and the dataset
files each pipeline stage reads and writes.

Stage-specific binaries and their outputs live with their stage
(see phonetics.py, proposer.py). This module owns the shared roots.
"""

from enum import Enum, auto
from pathlib import Path


class DataSource(Enum):
    PH = auto()  # Philippine FDA registry  ->  _data/drug_set_ph.csv
    US = auto()  # US FDA registry          ->  _data/drug_set_us.csv


# Active data source -- change this to switch registries
DATA_SOURCE: DataSource = DataSource.PH

DATA_DIR = Path("_data")
RESULTS_DIR = Path("results")

# Models are downloaded here by scripts/model_setup.py
MODELS_DIR = Path("models")


# Raw drug name registries -- one-column CSVs, header ignored, first column used
R: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph_raw.csv",
    DataSource.US: DATA_DIR / "R_us_raw.csv",
}

# Cleaned registry cache -- written by src/preprocessing.py every time it
# cleans a raw registry, so a slow clean never has to be repeated.
R_CLEAN: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph.csv",
    DataSource.US: DATA_DIR / "R_us.csv",
}

# If True, preprocessing loads R_CLEAN[source] directly instead of
# re-cleaning R[source]. Toggle on once you have a cached clean registry
# you trust; toggle off (or delete the cache file) to force a re-clean.
USE_PRECLEANED_REGISTRY: bool = True

# Confirmed LASA pairs input -- used when FROM_FILE = True
# Must have columns x_1, x_2 (see schema.P_INPUT_COLS)
P: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "P_ph.csv",
    DataSource.US: DATA_DIR / "P_us.csv",
}

# Sampled unlabeled pairs, written so `walter noise` and `walter assemble`
# can run as separate invocations. The full run keeps U in memory and writes
# this as a checkpoint.
U_CSV: Path = RESULTS_DIR / "U.csv"

# Final output -- classification: pair + label, schema unchanged
D_CSV: Path = RESULTS_DIR / "D.csv"  # full assembled dataset

# phoc output -- D with the phonetic-similarity feature columns added
D_PHO_CSV: Path = RESULTS_DIR / "D_pho.csv"

# Feature-engineering step -- src/feature_engineering.py appends META_FEATURES
# (orthographic / edit-distance features) onto the phoc output, so _engi is
# "engineered and pho'd": every phonetic column plus the META_FEATURES.
D_ENGI_CSV: Path = RESULTS_DIR / "D_engi.csv"
