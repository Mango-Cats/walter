"""Loads and cleans a drug name registry for the selected DataSource.

This module cleans raw drug names from national registries (such as the Philippines
FDA or US FDA), removing accents, punctuation, duplicates, and invalid entries.
"""

import re
import unicodedata
from pathlib import Path
from random import randint

import pandas as pd

from config import (
    ALIASES,
    REGISTRY_COL,
    USE_PRECLEANED_REGISTRY,
    DataSource,
    R,
    R_CLEAN,
)


def load_registry(source: DataSource) -> pd.DataFrame:
    """Load raw drug names from the CSV file for the given data source.

    Args:
        source: The drug registry data source (PH or US).

    Returns:
        A DataFrame containing raw drug names in a single column.

    Raises:
        FileNotFoundError: If the raw data file does not exist.

    """
    path: Path = R[source]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\nExpected one-column CSV of drug names."
        )
    df = pd.read_csv(path, usecols=[0], header=None, names=[REGISTRY_COL])

    if str(df.iloc[0, 0]).strip().lower() == REGISTRY_COL:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _remove_diacritics(text: str) -> str:
    """Strip accent marks and diacritics from a string.

    Args:
        text: Input text string.

    Returns:
        Cleaned string without accents.

    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def clean_name(name: str) -> str:
    """Normalize a single drug name.

    Lowercases the name, removes accents, replaces punctuation with spaces,
    strips non-alphanumeric characters, and collapses whitespace.

    Args:
        name: Raw drug name string.

    Returns:
        Cleaned drug name string.

    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = _remove_diacritics(name)
    name = re.sub(r"[-/']", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def clean_registry(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all drug names in a registry DataFrame and remove duplicates and empties.

    Args:
        df: DataFrame containing raw drug names.

    Returns:
        Cleaned DataFrame with unique, non-empty drug names.

    """
    df = df.copy()
    df[REGISTRY_COL] = df[REGISTRY_COL].apply(clean_name)

    bad = {"", "none", "nan", "n/a"}
    df = df[~df[REGISTRY_COL].isin(bad)]
    df = df.drop_duplicates(subset=[REGISTRY_COL]).dropna()
    df = df.reset_index(drop=True)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Print warning notices for unusual or suspicious raw registry data.

    Args:
        df: Raw registry DataFrame.

    """
    if len(df) < 500:
        print(f"[preprocessing] WARNING: only {len(df)} rows - unusually small.")
    nulls = df[REGISTRY_COL].isna().sum()
    if nulls:
        print(f"[preprocessing] WARNING: {nulls} null values in raw data.")
    dupes = df.duplicated(subset=[REGISTRY_COL]).sum()
    if dupes:
        print(f"[preprocessing] WARNING: {dupes} duplicate entries in raw data.")


def cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    """Print a summary of row counts before and after cleaning.

    Args:
        raw: Original raw DataFrame.
        clean: Cleaned DataFrame.

    """
    dropped = len(raw) - len(clean)
    print(
        f"[preprocessing] Rows: {len(raw):,} raw → {len(clean):,} clean  (dropped {dropped:,})"
    )


def save_clean_registry(df: pd.DataFrame, source: DataSource) -> None:
    """Save a cleaned registry DataFrame to disk for future reuse.

    Args:
        df: Cleaned registry DataFrame.
        source: Data source enum key.

    """
    path: Path = R_CLEAN[source]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[preprocessing] Saved cleaned registry → {path}")


def load_clean_registry(source: DataSource) -> pd.DataFrame:
    """Load a previously saved cleaned registry from disk.

    Args:
        source: Data source enum key.

    Returns:
        Cleaned registry DataFrame.

    Raises:
        FileNotFoundError: If the pre-cleaned file does not exist.

    """
    path: Path = R_CLEAN[source]
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-cleaned registry not found: {path}\n"
            "Set USE_PRECLEANED_REGISTRY = False in config.py to generate it first."
        )
    df = pd.read_csv(path, usecols=[0], header=0)
    df.columns = [REGISTRY_COL]
    return df.dropna().reset_index(drop=True)


def apply_aliases(df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
    """Drop registry names confirmed as misspellings of another registry name.

    Reads the alias file for the source (columns `typo`, `canonical`). A missing
    file leaves the registry unchanged.

    Args:
        df: Cleaned registry DataFrame.
        source: Data source enum key.

    Returns:
        Registry DataFrame without the misspelled names.

    """
    path: Path = ALIASES[source]
    if not path.exists():
        return df
    aliases = pd.read_csv(path, dtype=str)
    typos = set(aliases["typo"].map(clean_name)) - set(aliases["canonical"].map(clean_name))
    out = df[~df[REGISTRY_COL].isin(typos)].reset_index(drop=True)
    print(f"[preprocessing] Removed {len(df) - len(out):,} misspelled names listed in {path}")
    return out


def run(source: DataSource) -> pd.DataFrame:
    """Run the complete registry preprocessing pipeline for the given data source.

    Args:
        source: Data source enum key (PH or US).

    Returns:
        A cleaned single-column DataFrame of unique drug names.

    """
    if USE_PRECLEANED_REGISTRY:
        print(f"[preprocessing] Source: {source.name} (pre-cleaned cache)")
        clean = load_clean_registry(source)
        print(
            f"[preprocessing] Loaded {len(clean):,} pre-cleaned rows from {R_CLEAN[source]}"
        )
        return apply_aliases(clean, source)

    print(f"[preprocessing] Source: {source.name}")
    raw = load_registry(source)
    validate_raw(raw)
    clean = clean_registry(raw)
    cleaning_report(raw, clean)
    save_clean_registry(clean, source)
    return apply_aliases(clean, source)


def get_rand_entries(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """Return a random slice of consecutive rows from a DataFrame for inspection.

    Args:
        df: Source DataFrame.
        count: Number of rows to return.

    Returns:
        A slice of the DataFrame containing up to count rows.

    """
    n = randint(0, max(0, len(df) - count))
    return df.iloc[n : n + count]
