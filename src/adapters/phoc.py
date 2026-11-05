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
import shutil
import stat
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COL_LABEL,
    COL_T1,
    COL_T2,
    COL_X1,
    COL_X2,
    CORE_COLUMN_PREFIX,
    CORE_ORTHOGRAPHIC_CONFIGS,
    CORE_PHONETIC_CONFIG,
    PHOC_BIN,
    PHOC_CONFIG_DIR,
    PHONETIC_ALGORITHMS,
    PHONETIC_CONFIG_LANGS,
    POSITIVE_LABEL,
    TRANSCRIPTION_LANGS,
)
from src.adapters.g2p.fil import stressed_transcriptions, transcribe_stressed
from src.pipeline.core_name import core_name

_STRESSED_LANG = "fil"
_SEPARATE_SUFFIXES = ("_substitutions", "_indels", "_indel_diff")


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
        "csv",
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


def _config_langs(stem: str, langs: dict[str, tuple[str, str]]) -> list[str]:
    """List the transcription languages a phonetic configuration is scored over.

    Args:
        stem: Base name of the phonetic configuration file.
        langs: Mapping of language codes to transcription column names.

    Returns:
        Language codes from `langs` that apply to this configuration.

    """
    allowed = PHONETIC_CONFIG_LANGS.get(stem, tuple(langs))
    return [lang for lang in langs if lang in allowed]


def _phonetic_column(col: str, lang: str) -> str:
    """Name the feature column for a phonetic configuration and transcription.

    Args:
        col: phoc output column, named `aline_<variant>` after its configuration file.
        lang: Language code of the transcription it was scored over.

    Returns:
        Column name of the form `aline_var:<variant>_g2p:<lang>`.

    """
    algorithm, _, variant = col.partition("_")
    return f"{algorithm}_var:{variant}_g2p:{lang}"


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
        _phonetic_column(col, lang)
        for stem in phonetic
        for col in _stem_columns(stem, separate[stem])
        for lang in _config_langs(stem, langs)
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
            if lang == _STRESSED_LANG:
                for name_col, t_col in ((COL_X1, COL_T1), (COL_X2, COL_T2)):
                    staged[t_col] = stressed_transcriptions(
                        base[name_col].fillna("").tolist(), staged[t_col].tolist()
                    )

            staged_csv = tmp / f"in_{lang}.csv"
            scored_csv = tmp / f"out_{lang}.csv"
            staged.to_csv(staged_csv, index=False)

            lang_config_dir = tmp / f"conf_{lang}"
            lang_config_dir.mkdir()
            for path in config_dir.glob("*.toml"):
                if path.stem in orthographic or lang in _config_langs(path.stem, langs):
                    shutil.copy(path, lang_config_dir / path.name)

            run_phoc(staged_csv, scored_csv, lang_config_dir)
            scored = pd.read_csv(scored_csv)

            for stem in phonetic:
                if lang not in _config_langs(stem, langs):
                    continue
                for col in _stem_columns(stem, separate[stem]):
                    features[_phonetic_column(col, lang)] = scored[col]

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


def score_lexicon(
    input_csv: Path,
    output_path: Path,
    config_dir: Path = PHOC_CONFIG_DIR,
    langs: dict[str, tuple[str, str]] = TRANSCRIPTION_LANGS,
) -> list[str]:
    """Score every LASA name against every name in a held-out pair CSV.

    Each name in a LASA pair becomes a query, ranked against all other names in
    the CSV. Feature columns match `run_phoc_multilingual` and `run_core_features`.
    Engineered features are not computed, so a model trained on them cannot be
    scored over this pool.

    Args:
        input_csv: CSV with x_1, x_2, label, and the transcription columns.
        output_path: Path to write the pool as Parquet.
        config_dir: Directory containing algorithm configuration TOML files.
        langs: Dictionary mapping language codes to pairs of transcription columns.

    Returns:
        List of feature column names in the pool.

    Raises:
        ValueError: If a required column is missing from the input CSV.

    """
    base = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    lang_cols = [c for pair in langs.values() for c in pair]
    missing = [c for c in (COL_X1, COL_X2, COL_LABEL, *lang_cols) if c not in base]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    transcripts: dict[str, dict[str, str]] = {}
    for lang, (col_1, col_2) in langs.items():
        transcripts[lang] = dict(zip(base[COL_X1], base[col_1]))
        transcripts[lang].update(zip(base[COL_X2], base[col_2]))

    names = sorted(set(base[COL_X1]) | set(base[COL_X2]))
    positives = base[base[COL_LABEL] == str(POSITIVE_LABEL)]
    queries = sorted(set(positives[COL_X1]) | set(positives[COL_X2]))

    stressed = dict(zip(names, transcribe_stressed(names)))
    x_1 = pd.Series([q for q in queries for n in names if n != q], dtype=object)
    x_2 = pd.Series([n for q in queries for n in names if n != q], dtype=object)

    phonetic, orthographic, _separate = _classify_configs(config_dir)
    features: dict[str, np.ndarray] = {}

    def score(stems: list[str], t_map: dict[str, str] | None, tmp: Path) -> pd.DataFrame:
        conf = tmp / "conf"
        shutil.rmtree(conf, ignore_errors=True)
        conf.mkdir()
        for stem in stems:
            shutil.copy(config_dir / f"{stem}.toml", conf / f"{stem}.toml")
        staged = pd.DataFrame({COL_X1: x_1, COL_X2: x_2})
        if t_map is not None:
            staged[COL_T1] = x_1.map(t_map)
            staged[COL_T2] = x_2.map(t_map)
        staged.to_csv(tmp / "in.csv", index=False)
        run_phoc(tmp / "in.csv", tmp / "out.csv", conf)
        scored = pd.read_csv(tmp / "out.csv")
        return scored.drop(columns=list(staged.columns))

    with tempfile.TemporaryDirectory(prefix="phoc_lexicon_") as tmpdir:
        tmp = Path(tmpdir)
        for col, values in score(orthographic, None, tmp).items():
            features[col] = values.to_numpy(np.float32)
        for lang in langs:
            stems = [s for s in phonetic if lang in _config_langs(s, langs)]
            t_map = stressed if lang == _STRESSED_LANG else transcripts[lang]
            for col, values in score(stems, t_map, tmp).items():
                features[_phonetic_column(col, lang)] = values.to_numpy(np.float32)

        core = {name: core_name(name) for name in names}
        core_stressed = dict(
            zip(sorted(set(core.values())), transcribe_stressed(sorted(set(core.values()))))
        )
        x_1, x_2 = x_1.map(core), x_2.map(core)
        core_stems = [*CORE_ORTHOGRAPHIC_CONFIGS, CORE_PHONETIC_CONFIG]
        scored = score(core_stems, core_stressed, tmp)
        for column, stem in zip(core_feature_names(), core_stems):
            features[column] = scored[stem].to_numpy(np.float32)

    pool = pd.DataFrame(
        {
            COL_X1: [q for q in queries for n in names if n != q],
            COL_X2: [n for q in queries for n in names if n != q],
            **features,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(output_path, index=False)
    return list(features)


def core_feature_names() -> list[str]:
    """List the feature columns added by run_core_features.

    Returns:
        Column names in output order.

    """
    stems = list(CORE_ORTHOGRAPHIC_CONFIGS)
    phonetic = _phonetic_column(CORE_PHONETIC_CONFIG, _STRESSED_LANG)
    return [f"{CORE_COLUMN_PREFIX}{col}" for col in [*stems, phonetic]]


def run_core_features(
    input_csv: Path,
    output_csv: Path,
    config_dir: Path = PHOC_CONFIG_DIR,
) -> list[str]:
    """Score each pair's core words and append the scores to the dataset.

    Existing columns are written back unchanged, so this can extend a finished
    dataset without recomputing or reformatting its features.

    Args:
        input_csv: Path to input CSV containing x_1 and x_2.
        output_csv: Path to write the CSV with added core feature columns.
        config_dir: Directory containing algorithm configuration TOML files.

    Returns:
        List of feature column names appended to the dataset.

    Raises:
        ValueError: If x_1 or x_2 is missing from the input CSV.

    """
    base = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    missing = [c for c in (COL_X1, COL_X2) if c not in base.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    names = core_feature_names()
    base = base.drop(columns=[c for c in names if c in base.columns])

    core_1 = base[COL_X1].map(core_name)
    core_2 = base[COL_X2].map(core_name)
    unique = sorted(set(core_1) | set(core_2))
    stressed = dict(zip(unique, transcribe_stressed(unique)))

    staged = pd.DataFrame(
        {
            COL_X1: core_1,
            COL_X2: core_2,
            COL_T1: core_1.map(stressed),
            COL_T2: core_2.map(stressed),
        }
    )

    with tempfile.TemporaryDirectory(prefix="phoc_core_") as tmpdir:
        tmp = Path(tmpdir)
        conf = tmp / "conf"
        conf.mkdir()
        stems = [*CORE_ORTHOGRAPHIC_CONFIGS, CORE_PHONETIC_CONFIG]
        for stem in stems:
            shutil.copy(config_dir / f"{stem}.toml", conf / f"{stem}.toml")
        staged_csv = tmp / "in.csv"
        scored_csv = tmp / "out.csv"
        staged.to_csv(staged_csv, index=False)
        run_phoc(staged_csv, scored_csv, conf)
        scored = pd.read_csv(scored_csv, dtype=str, keep_default_na=False)

    for name, stem in zip(names, stems):
        base[name] = scored[stem].to_numpy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_csv, index=False)
    return names
