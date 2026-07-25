"""
Where things live: the active registry, the directories, and the dataset
files each pipeline stage reads and writes.

Stage-specific binaries and their outputs live with their stage
(see phonetics.py, proposer.py). This module owns the shared roots.
"""

from enum import Enum, auto
from pathlib import Path


class DataSource(Enum):
    PH = auto()
    US = auto()


# Active data source -- change this to switch registries
DATA_SOURCE: DataSource = DataSource.PH

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Models are downloaded here by scripts/model_setup.py
MODELS_DIR = Path("models")


# Raw drug name registries -- one-column CSVs, header ignored, first column used
R: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph.csv",
    DataSource.US: DATA_DIR / "R_us.csv",
}

# Cleaned registry cache -- written by src/pipeline/preprocessing.py every time it
# cleans a raw registry, so a slow clean never has to be repeated. Kept under a
# distinct _clean suffix: mapping this back onto the raw filename would make
# USE_PRECLEANED_REGISTRY load uncleaned names and skip cleaning silently.
R_CLEAN: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph_clean.csv",
    DataSource.US: DATA_DIR / "R_us_clean.csv",
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

# Predefined rejected (non-LASA) pairs -- read only under SOFT_LABELS, and
# optional even then: absent means the LLM's rejections are the only source of
# NEGATIVE_LABEL rows. Same columns as P (see schema.P_INPUT_COLS).
N: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "N_ph.csv",
    DataSource.US: DATA_DIR / "N_us.csv",
}

# --------------------------------------------------------------------------
# Canonical artifact filenames
#
# Stages take directories, not files. An artifact keeps the same filename
# wherever it is written, so the name a stage writes into its output directory
# is the name the next stage looks for in its input directory. Pointing two
# stages at one directory is all it takes to chain them; see src/artifacts.py.
#
# Changing a name here changes it on both sides of every stage at once, which
# is the point -- a stage must never write one name and read another.
# --------------------------------------------------------------------------

U_FILENAME: str = "U.csv"
D_FILENAME: str = "D.csv"
D_PHO_FILENAME: str = "D_pho.csv"
D_ENGI_FILENAME: str = "D_engi.csv"

# Default locations, i.e. those filenames under RESULTS_DIR. Stages accept any
# directory; these are what the CLI falls back to.

# Sampled unlabeled pairs, written so `walter noise` and `walter assemble`
# can run as separate invocations. The full run keeps U in memory and writes
# this as a checkpoint.
U_CSV: Path = RESULTS_DIR / U_FILENAME

# Final output -- classification: pair + label, schema unchanged
D_CSV: Path = RESULTS_DIR / D_FILENAME  # full assembled dataset

# phoc output -- D with the phonetic-similarity feature columns added
D_PHO_CSV: Path = RESULTS_DIR / D_PHO_FILENAME

# Feature-engineering step -- src/pipeline/features.py appends META_FEATURES
# (orthographic / edit-distance features) onto the phoc output, so _engi is
# "engineered and pho'd": every phonetic column plus the META_FEATURES.
D_ENGI_CSV: Path = RESULTS_DIR / D_ENGI_FILENAME
