"""
Phonetic-feature step: runs the bundled `bin/phoc` Rust CLI over an already
assembled pair CSV.

phoc reads a CSV with x_1/x_2 (plus optional t_1/t_2, label, and any other
columns), preserves every input column verbatim, and appends one similarity
feature column per .toml config in PHOC_CONFIG_DIR — the column name is the
config file's stem, the algorithm is chosen by that file's `algorithm` key.

Multilingual transcriptions
---------------------------
Our datasets carry one transcription per language (t_eng_1/t_eng_2,
t_fil_1/t_fil_2, ...), but the phoc binary hardcodes t_1/t_2 as the columns it
reads. run_phoc_multilingual() bridges that: it runs phoc once per language
against a temp CSV in which that language's transcription columns have been
renamed to t_1/t_2.

Only the algorithms that actually read the transcription need duplicating.
Those are listed in config.PHONETIC_ALGORITHMS (currently just `aline`), and
their columns come back suffixed — aline_ph_mc_eng, aline_ph_mc_fil. The
remaining configs (bisim, editex, levenshtein) score x_1/x_2 alone, so they are
identical in every run and are emitted once, unsuffixed.

Because walter.py writes D.csv before calling into here, a phoc failure never
destroys that base dataset: it just aborts the run with a clear error and
leaves the base CSV on disk.
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
            "It ships in the repo under bin/phoc — check it out or rebuild it."
        )
    # bin/phoc is gitignored (194 MB) and gets copied between machines, which
    # can strip the execute bit. Restore it rather than fail with Errno 13.
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
    # ETXTBSY (Errno 26) means the binary is still open for writing elsewhere
    # (e.g. bin/phoc is 194 MB and may still be syncing/copying when the run
    # starts). It clears on its own, so retry a few times with a short backoff.
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


def _classify_configs(config_dir: Path) -> tuple[list[str], list[str]]:
    """
    Split the config stems into (phonetic, orthographic) by reading each
    .toml's `algorithm` key. Phonetic configs read t_1/t_2 and so must be
    computed once per language; orthographic ones read only x_1/x_2.
    """
    phonetic: list[str] = []
    orthographic: list[str] = []
    for path in sorted(config_dir.glob("*.toml")):
        with path.open("rb") as f:
            algorithm = str(tomllib.load(f).get("algorithm", "")).lower()
        target = phonetic if algorithm in PHONETIC_ALGORITHMS else orthographic
        target.append(path.stem)

    if not phonetic and not orthographic:
        raise FileNotFoundError(f"No .toml configs found in {config_dir}")
    return phonetic, orthographic


def _feature_names(
    phonetic: list[str],
    orthographic: list[str],
    langs: dict[str, tuple[str, str]],
) -> list[str]:
    """
    Every feature column a run emits, in output order: the orthographic ones
    (computed once), then each phonetic one with its languages grouped together
    (aline_ph_mc_eng, aline_ph_mc_fil, ...).
    """
    return list(orthographic) + [
        f"{stem}_{lang}" for stem in phonetic for lang in langs
    ]


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

    Writes the merged frame — every original column, then the features — to
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

    phonetic, orthographic = _classify_configs(config_dir)

    # Re-running over an already-scored CSV: phoc appends its column regardless
    # of whether one of the same name came in, so the output would carry two
    # `bisim` columns and pandas would resolve reads to the stale first one.
    # Drop any prior feature columns up front, so they are always recomputed.
    stale = [c for c in _feature_names(phonetic, orthographic, langs) if c in base.columns]
    if stale:
        base = base.drop(columns=stale)

    features: dict[str, pd.Series] = {}
    orth_reference: dict[str, pd.Series] = {}

    with tempfile.TemporaryDirectory(prefix="phoc_") as tmpdir:
        tmp = Path(tmpdir)
        for lang, (col_1, col_2) in langs.items():
            # Present this language as t_1/t_2 and hide every other language's
            # columns, so phoc's verbatim passthrough can't duplicate them.
            staged = base.drop(columns=lang_cols)
            staged[COL_T1] = base[col_1].fillna("")
            staged[COL_T2] = base[col_2].fillna("")

            staged_csv = tmp / f"in_{lang}.csv"
            scored_csv = tmp / f"out_{lang}.csv"
            staged.to_csv(staged_csv, index=False)

            run_phoc(staged_csv, scored_csv, config_dir)
            scored = pd.read_csv(scored_csv)

            for stem in phonetic:
                features[f"{stem}_{lang}"] = scored[stem]

            # Orthographic features ignore t_1/t_2, so every language must
            # produce the same numbers. If not, a config we classified as
            # orthographic actually reads the transcription and would be
            # silently collapsed to one language's value.
            for stem in orthographic:
                if stem not in orth_reference:
                    orth_reference[stem] = scored[stem]
                elif not orth_reference[stem].equals(scored[stem]):
                    raise RuntimeError(
                        f"phoc config '{stem}' was treated as transcription-"
                        f"independent but its values changed for lang '{lang}'. "
                        f"Add its `algorithm` to config.PHONETIC_ALGORITHMS."
                    )

    ordered = _feature_names(phonetic, orthographic, langs)

    merged = base.copy()
    for stem in orthographic:
        merged[stem] = orth_reference[stem]
    for name in ordered:
        if name in features:
            merged[name] = features[name]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return ordered
