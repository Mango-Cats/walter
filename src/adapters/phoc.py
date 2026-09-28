"""Phonetic similarity scoring using the bundled `phoc` tool.

What this file does:
1. Takes a CSV of drug pairs and their spoken pronunciations (IPA).
2. Runs the fast `phoc` tool (located in `bin/phoc`) to compare each pair.
3. Calculates similarity scores using multiple phonetic algorithms:
   - Compares English spoken pronunciations.
   - Compares Filipino spoken pronunciations.
   - Compares raw drug name spellings (edit distance, prefix/suffix match).
4. Appends these similarity scores as new columns and saves the enriched dataset (D_pho.csv).
"""

import csv
import errno
import os
import stat
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

import pandas as pd

from config import (
    COL_T1,
    COL_T2,
    PHOC_BIN,
    PHOC_CONFIG_DIR,
    PHONETIC_ALGORITHMS,
    TRANSCRIPTION_LANGS,
)

_SEPARATE_SUFFIXES = ("_substitutions", "_insertions", "_deletions")


def _header(csv_path: Path) -> list[str]:
    """Read the header row of column names from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of column name strings.

    """
    with csv_path.open(newline="") as f:
        return next(csv.reader(f), [])


def run_phoc(
    input_csv: Path,
    output_csv: Path,
    config_dir: Path = PHOC_CONFIG_DIR,
) -> list[str]:
    """Run the phoc command-line tool to calculate similarity scores between pairs.

    Args:
        input_csv: Path to input CSV file.
        output_csv: Path to output CSV file with added feature columns.
        config_dir: Directory containing algorithm configuration TOML files.

    Returns:
        List of new column names added by phoc.

    Raises:
        FileNotFoundError: If the phoc binary or configuration folder is missing.
        PermissionError: If the phoc binary cannot be made executable.
        RuntimeError: If phoc execution returns a non-zero exit code.

    """
    if not PHOC_BIN.exists():
        raise FileNotFoundError(
            f"phoc binary not found at {PHOC_BIN}. "
            "It ships in the repo under bin/phoc - check it out or rebuild it."
        )

    if not os.access(PHOC_BIN, os.X_OK):
        try:
            mode = PHOC_BIN.stat().st_mode
            PHOC_BIN.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError as e:
            raise PermissionError(
                f"phoc binary at {PHOC_BIN} is not executable and could not be "
                f"made executable ({e}). Run: chmod +x {PHOC_BIN}"
            ) from e
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"phoc config dir not found at {config_dir}. "
            "It must contain one .toml per feature column."
        )

    input_cols = _header(input_csv)

    cmd = [
        str(PHOC_BIN),
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        "--config-dir",
        str(config_dir),
    ]

    for attempt in range(5):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            break
        except OSError as e:
            if e.errno == errno.ETXTBSY and attempt < 4:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    if result.returncode != 0:
        raise RuntimeError(
            f"phoc failed (exit {result.returncode}) on {input_csv}:\n"
            f"{result.stderr.strip()}"
        )

    output_cols = _header(output_csv)
    return [c for c in output_cols if c not in input_cols]


def _stem_columns(stem: str, separate: bool) -> list[str]:
    """Determine the output column names for a configuration stem.

    Args:
        stem: Base name of the configuration file.
        separate: Whether the algorithm emits separate insertion/deletion/substitution scores.

    Returns:
        List of column name strings.

    """
    if separate:
        return [f"{stem}{suffix}" for suffix in _SEPARATE_SUFFIXES]
    return [stem]


def _classify_configs(
    config_dir: Path,
) -> tuple[list[str], list[str], dict[str, bool]]:
    """Classify configuration files into phonetic (IPA-based) and orthographic (spelling-based).

    Args:
        config_dir: Directory containing configuration TOML files.

    Returns:
        A tuple of (phonetic_stems, orthographic_stems, separate_mapping).

    Raises:
        FileNotFoundError: If no configuration files are found in the directory.

    """
    phonetic: list[str] = []
    orthographic: list[str] = []
    separate: dict[str, bool] = {}
    for path in sorted(config_dir.glob("*.toml")):
        with path.open("rb") as f:
            conf = tomllib.load(f)
        algorithm = str(conf.get("algorithm", "")).lower()
        target = phonetic if algorithm in PHONETIC_ALGORITHMS else orthographic
        target.append(path.stem)
        separate[path.stem] = bool(conf.get("separate", False))

    if not phonetic and not orthographic:
        raise FileNotFoundError(f"No .toml configs found in {config_dir}")
    return phonetic, orthographic, separate


def _feature_names(
    phonetic: list[str],
    orthographic: list[str],
    langs: dict[str, tuple[str, str]],
    separate: dict[str, bool],
) -> list[str]:
    """Generate the full ordered list of feature column names produced by phoc.

    Args:
        phonetic: List of phonetic configuration stems.
        orthographic: List of orthographic configuration stems.
        langs: Mapping of language codes to transcription column names.
        separate: Mapping of stems to their separation boolean flag.

    Returns:
        List of all feature column names in output order.

    """
    orth_cols = [
        c for stem in orthographic for c in _stem_columns(stem, separate[stem])
    ]
    phon_cols = [
        f"{col}_{lang}"
        for stem in phonetic
        for col in _stem_columns(stem, separate[stem])
        for lang in langs
    ]
    return orth_cols + phon_cols


def run_phoc_multilingual(
    input_csv: Path,
    output_csv: Path,
    config_dir: Path = PHOC_CONFIG_DIR,
    langs: dict[str, tuple[str, str]] = TRANSCRIPTION_LANGS,
) -> list[str]:
    """Run phoc across all supported languages and merge the resulting feature columns.

    Args:
        input_csv: Path to input CSV containing drug pairs and IPA pronunciations.
        output_csv: Path to write the feature-augmented CSV file.
        config_dir: Directory containing algorithm configuration TOML files.
        langs: Dictionary mapping language codes to pairs of transcription columns.

    Returns:
        List of feature column names appended to the dataset.

    Raises:
        ValueError: If any required transcription columns are missing from the input CSV.
        RuntimeError: If a supposedly language-independent algorithm produces varying scores.

    """
    base = pd.read_csv(input_csv)

    lang_cols = [c for pair in langs.values() for c in pair]
    missing = [c for c in lang_cols if c not in base.columns]
    if missing:
        raise ValueError(
            f"{input_csv} is missing transcription columns {missing}. "
            f"Expected one pair per language in TRANSCRIPTION_LANGS: {lang_cols}"
        )

    phonetic, orthographic, separate = _classify_configs(config_dir)

    stale = [
        c
        for c in _feature_names(phonetic, orthographic, langs, separate)
        if c in base.columns
    ]
    if stale:
        base = base.drop(columns=stale)

    features: dict[str, pd.Series] = {}
    orth_reference: dict[str, pd.Series] = {}

    with tempfile.TemporaryDirectory(prefix="phoc_") as tmpdir:
        tmp = Path(tmpdir)
        for lang, (col_1, col_2) in langs.items():
            staged = base.drop(columns=lang_cols)
            staged[COL_T1] = base[col_1].fillna("")
            staged[COL_T2] = base[col_2].fillna("")

            staged_csv = tmp / f"in_{lang}.csv"
            scored_csv = tmp / f"out_{lang}.csv"
            staged.to_csv(staged_csv, index=False)

            run_phoc(staged_csv, scored_csv, config_dir)
            scored = pd.read_csv(scored_csv)

            for stem in phonetic:
                for col in _stem_columns(stem, separate[stem]):
                    features[f"{col}_{lang}"] = scored[col]

            for stem in orthographic:
                for col in _stem_columns(stem, separate[stem]):
                    if col not in orth_reference:
                        orth_reference[col] = scored[col]
                    elif not orth_reference[col].equals(scored[col]):
                        raise RuntimeError(
                            f"phoc config '{stem}' was treated as transcription-"
                            f"independent but its values changed for lang "
                            f"'{lang}'. Add its `algorithm` to "
                            "config.PHONETIC_ALGORITHMS."
                        )

    ordered = _feature_names(phonetic, orthographic, langs, separate)

    merged = base.copy()
    for col, values in orth_reference.items():
        merged[col] = values
    for name in ordered:
        if name in features:
            merged[name] = features[name]
        elif name in orth_reference:
            pass

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return ordered
