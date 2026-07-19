"""
Loads and cleans a drug name registry for the selected DataSource.

Input:  the raw registry at R[source] (data/R_ph.csv or data/R_us.csv).
        One-column CSV of raw drug names
        (header optional; first column used).

Output: Single-column DataFrame with column REGISTRY_COL ("drug_name"),
        lowercase, symbols stripped, duplicates removed.

Both sources go through the same cleaning pipeline.
"""

import re
import unicodedata
from pathlib import Path
from random import randint

import pandas as pd

from config import (
    DataSource,
    R,
    R_CLEAN,
    REGISTRY_COL,
    USE_PRECLEANED_REGISTRY,
)


def load_registry(source: DataSource) -> pd.DataFrame:
    """
    Load raw drug names from the CSV for the given source.
    Always reads the first column regardless of its header,
    then renames it to REGISTRY_COL.
    """
    path: Path = R[source]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\nExpected one-column CSV of drug names."
        )
    df = pd.read_csv(path, usecols=[0], header=None, names=[REGISTRY_COL])
    # The registries ship headerless, so header=0 would consume a real drug
    # name. Tolerate a header anyway, in case a hand-exported file carries one.
    if str(df.iloc[0, 0]).strip().lower() == REGISTRY_COL:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _remove_diacritics(text: str) -> str:
    """Strip combining diacritical marks (accents, umlauts, etc.)."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def clean_name(name: str) -> str:
    """
    Normalize a single drug name:
      1. Lowercase
      2. Strip leading/trailing whitespace
      3. Remove diacritics
      4. Replace hyphens, slashes, apostrophes with a space
      5. Remove any remaining non-alphanumeric, non-space characters
      6. Collapse runs of whitespace to a single space
    Digits are kept (e.g. B12, D3).
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
    """
    Apply clean_name to every row, then drop empties and duplicates.
    Returns a reset-index DataFrame with column REGISTRY_COL.
    """
    df = df.copy()
    df[REGISTRY_COL] = df[REGISTRY_COL].apply(clean_name)

    bad = {"", "none", "nan", "n/a"}
    df = df[~df[REGISTRY_COL].isin(bad)]
    df = df.drop_duplicates(subset=[REGISTRY_COL]).dropna()
    df = df.reset_index(drop=True)
    return df


def validate_raw(df: pd.DataFrame) -> None:
    """Print warnings for suspicious raw data. Never raises."""
    if len(df) < 500:
        print(f"[preprocessing] WARNING: only {len(df)} rows — unusually small.")
    nulls = df[REGISTRY_COL].isna().sum()
    if nulls:
        print(f"[preprocessing] WARNING: {nulls} null values in raw data.")
    dupes = df.duplicated(subset=[REGISTRY_COL]).sum()
    if dupes:
        print(f"[preprocessing] WARNING: {dupes} duplicate entries in raw data.")


def cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    dropped = len(raw) - len(clean)
    print(
        f"[preprocessing] Rows: {len(raw):,} raw → {len(clean):,} clean  (dropped {dropped:,})"
    )


def save_clean_registry(df: pd.DataFrame, source: DataSource) -> None:
    """Write the cleaned registry to R_CLEAN[source] for reuse by later runs."""
    path: Path = R_CLEAN[source]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[preprocessing] Saved cleaned registry → {path}")


def load_clean_registry(source: DataSource) -> pd.DataFrame:
    """
    Load a previously-saved cleaned registry, skipping clean_registry()
    entirely. Used when USE_PRECLEANED_REGISTRY = True.
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


def run(source: DataSource) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the given DataSource.

    If USE_PRECLEANED_REGISTRY is True, loads the cached clean registry
    from R_CLEAN[source] directly, skipping cleaning entirely. Otherwise:

    1. Load raw CSV (R_ph.csv or R_us.csv)
    2. Validate
    3. Clean
    4. Save the cleaned registry to R_CLEAN[source] for next time

    Returns cleaned single-column DataFrame [REGISTRY_COL].
    """
    if USE_PRECLEANED_REGISTRY:
        print(f"[preprocessing] Source: {source.name} (pre-cleaned cache)")
        clean = load_clean_registry(source)
        print(f"[preprocessing] Loaded {len(clean):,} pre-cleaned rows from {R_CLEAN[source]}")
        return clean

    print(f"[preprocessing] Source: {source.name}")
    raw = load_registry(source)
    validate_raw(raw)
    clean = clean_registry(raw)
    cleaning_report(raw, clean)
    save_clean_registry(clean, source)
    return clean


def get_rand_entries(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """Random slice of `count` rows — useful for spot-checking."""
    n = randint(0, max(0, len(df) - count))
    return df.iloc[n : n + count]
