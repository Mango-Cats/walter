"""
File paths and directories used across the Walter pipeline.

This module defines the locations of raw drug registries, cleaned cache files,
confirmed confusable pairs, downloaded AI models, and final output datasets.
"""

from enum import Enum, auto
from pathlib import Path


class DataSource(Enum):
    """Supported drug registries: Philippines (PH) or United States (US)."""

    PH = auto()
    US = auto()


DATA_SOURCE: DataSource = DataSource.PH

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

R: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph.csv",
    DataSource.US: DATA_DIR / "R_us.csv",
}

R_CLEAN: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph_clean.csv",
    DataSource.US: DATA_DIR / "R_us_clean.csv",
}

USE_PRECLEANED_REGISTRY: bool = True

P: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "P_ph.csv",
    DataSource.US: DATA_DIR / "P_us.csv",
}

N: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "N_ph.csv",
    DataSource.US: DATA_DIR / "N_us.csv",
}

ALIASES: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "aliases_ph.csv",
    DataSource.US: DATA_DIR / "aliases_us.csv",
}

U_FILENAME: str = "U.csv"
D_FILENAME: str = "D.csv"
D_PHO_FILENAME: str = "D_pho.csv"

U_CSV: Path = RESULTS_DIR / U_FILENAME
D_CSV: Path = RESULTS_DIR / D_FILENAME
D_PHO_CSV: Path = RESULTS_DIR / D_PHO_FILENAME

