"""
Phonetic-feature step: runs the bundled `bin/phoc` Rust CLI over an already
assembled pair CSV.

phoc reads a CSV with x_1/x_2, t_1/t_2, label, and any other
columns, preserves every input column verbatim, and appends one similarity
feature column per .toml config in PHOC_CONFIG_DIR, the algorithm is
chosen by that file's `algorithm` key.

Only the algorithms that actually read the transcription need duplicating.
Those are listed in config.PHONETIC_ALGORITHMS, and their columns come
back suffixed with the language code (e.g., _fil).

This has no side effect on the input file.
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


def _header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as f:
        return next(csv.reader(f), [])


def run_phoc(
    input_csv: Path,
    output_csv: Path,
    config_dir: Path = PHOC_CONFIG_DIR,
) -> list[str]:
    """
    Run phoc on ``input_csv`` and write the feature-augmented CSV to
    ``output_csv``. Returns the list of feature columns phoc appended
    (output header minus input header).

    Raises FileNotFoundError if the phoc binary or config dir is missing,
    and RuntimeError (with phoc's stderr) if phoc exits non-zero.
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


_SEPARATE_SUFFIXES = ("_substitutions", "_insertions", "_deletions")


def _stem_columns(stem: str, separate: bool) -> list[str]:
    """
    The output column(s) phoc emits for one config stem: a single column
    named after the stem, or - when the config sets `separate = true` - three
    columns (substitutions/insertions/deletions), per `phoc --help`.
    """
    if separate:
        return [f"{stem}{suffix}" for suffix in _SEPARATE_SUFFIXES]
    return [stem]


def _classify_configs(
    config_dir: Path,
) -> tuple[list[str], list[str], dict[str, bool]]:
    """
    Split the config stems into (phonetic, orthographic) by reading each
    .toml's `algorithm` key. Phonetic configs read t_1/t_2 and so must be
    computed once per language; orthographic ones read only x_1/x_2.

    Also returns a stem -> `separate` map, since a `separate = true` config
    emits three output columns instead of one.
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
    """
    Every feature column a run emits, in output order: the orthographic ones
    (computed once), then each phonetic one with its languages grouped together
    (aline_ph_mc_eng, aline_ph_mc_fil, ...). Stems with `separate = true`
    expand to their three sub-columns.
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
    """
    Run phoc once per language in ``langs`` and merge the results.

    ``input_csv`` must carry every language's transcription columns. For each
    language, phoc sees a temp CSV where that language's columns are presented
    as t_1/t_2 (and the other languages' are dropped, so they can't leak into
    the output). Transcription-dependent features come back as
    ``<config_stem>_<lang>``; transcription-independent ones are taken once.

    Writes the merged frame - every original column, then the features - to
    ``output_csv``. Returns the list of appended feature column names.
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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return ordered
