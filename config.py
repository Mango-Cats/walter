"""
Single source of truth for the walter pipeline.
All paths, column names, and tunable parameters live here.
No other file should hardcode these values.
"""

from enum import Enum, auto
from pathlib import Path

class DataSource(Enum):
    PH = auto()  # Philippine FDA registry  →  _data/drug_set_ph.csv
    US = auto()  # US FDA registry          →  _data/drug_set_us.csv


# Active data source — change this to switch registries
DATA_SOURCE: DataSource = DataSource.PH

DATA_DIR = Path("_data")
RESULTS_DIR = Path("_results")

# Models are downloaded here by scripts/model_setup.py
MODELS_DIR = Path("models")


# Raw drug name registries — one-column CSVs, header ignored, first column used
R: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph_raw.csv",
    DataSource.US: DATA_DIR / "R_us_raw.csv",
}

# Cleaned registry cache — written by src/preprocessing.py every time it
# cleans a raw registry, so a slow clean never has to be repeated.
R_CLEAN: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "R_ph.csv",
    DataSource.US: DATA_DIR / "R_us.csv",
}

# If True, preprocessing loads R_CLEAN[source] directly instead of
# re-cleaning R[source]. Toggle on once you have a cached clean registry
# you trust; toggle off (or delete the cache file) to force a re-clean.
USE_PRECLEANED_REGISTRY: bool = True

# Confirmed LASA pairs input — used when FROM_FILE = True
# Must have columns x_1, x_2 (see P_INPUT_COLS below)
P: dict[DataSource, Path] = {
    DataSource.PH: DATA_DIR / "P_ph.csv",
    DataSource.US: DATA_DIR / "P_us.csv",
}

# Final output — classification: pair + label, schema unchanged
D_OUT_CSV: Path = RESULTS_DIR / "D.csv"  # full assembled dataset

# Phonetic-feature step — bin/phoc (Rust CLI, see bin/phoc summary). Reads a
# scored pair CSV, preserves every input column, and appends one similarity
# feature column per .toml in PHOC_CONFIG_DIR (column name = file stem).
PHOC_BIN: Path = Path("bin/phoc")
PHOC_CONFIG_DIR: Path = Path("bin/pho_conf")

# phoc algorithms that score the IPA transcription (t_1/t_2) rather than the
# raw name (x_1/x_2). A config whose `algorithm` key is in this set produces
# one feature column PER LANGUAGE (aline_ph_mc_eng, aline_ph_mc_fil, ...);
# every other config reads only x_1/x_2 and so is computed once, unsuffixed.
# Verified against bin/phoc: aline raises MissingTranscription when t_1/t_2 are
# absent, while bisim/editex/levenshtein are unchanged when t_1/t_2 vary.
PHONETIC_ALGORITHMS: frozenset[str] = frozenset({"aline"})

# Filipino nativization — bin/tbb-cli (TagaBaybay Rust worker, see its module
# docstring). A long-lived JSONL stream worker that adapts loanwords into
# Filipino orthography; src/feature_engineering.py drives it (via
# src/tbb_client.py) to build its phonetic (nativization) features.
TBB_BIN: Path = Path("bin/tbb-cli")

# phoc output — D with the phonetic-similarity feature columns added
D_PHO_OUT_CSV: Path = RESULTS_DIR / "D_pho.csv"

# Feature-engineering step — src/feature_engineering.py appends META_FEATURES
# (orthographic / edit-distance features) onto the phoc output, so _engi is
# "engineered and pho'd": every phonetic column plus the META_FEATURES.
D_ENGI_OUT_CSV: Path = RESULTS_DIR / "D_engi.csv"

# Single-column name in the raw / cleaned registry
REGISTRY_COL: str = "drug_name"

# Pair dataset columns
COL_X1: str = "x_1"  # drug name A
COL_X2: str = "x_2"  # drug name B

# phoc's wire format ONLY. The phoc binary hardcodes t_1/t_2 as the names of
# the transcription columns it reads, so these never appear in a dataset we
# write — src/phoc_runner.py materializes them into a temp CSV, one run per
# language. The real, persisted transcription columns are the per-language
# ones below.
COL_T1: str = "t_1"
COL_T2: str = "t_2"

# Persisted per-language IPA transcriptions of x_1 / x_2.
COL_T_ENG_1: str = "t_eng_1"  # English (en-us), src/eng_g2p.py
COL_T_ENG_2: str = "t_eng_2"
COL_T_FIL_1: str = "t_fil_1"  # Filipino (Tagalog), src/fil_g2p.py
COL_T_FIL_2: str = "t_fil_2"

# The languages every transcription-dependent feature is computed once per.
# Adding a language here fans out the aline columns automatically (phoc_runner
# emits <config_stem>_<lang> for each) — no other file needs to change.
TRANSCRIPTION_LANGS: dict[str, tuple[str, str]] = {
    "eng": (COL_T_ENG_1, COL_T_ENG_2),
    "fil": (COL_T_FIL_1, COL_T_FIL_2),
}

COL_LABEL: str = "label"  # 1 = known positive (LASA), 0 = unlabeled

# P.csv (input) must have exactly these two columns
P_INPUT_COLS: list[str] = [COL_X1, COL_X2]


# Target ratio of unlabeled pairs to confirmed positives
CLASS_RATIO: int = 450

# Fraction of U drawn from each tier
TIER_1_PROPORTION: float = 0.65
TIER_2_PROPORTION: float = 0.35

# Total outside-vocabulary names sampled for Tier 2, split evenly across
# clusters (each cluster only ever scores pairs within its own share —
# see src/noise.py for why sampling must stay per-cluster)
TIER_2_SAMPLE_SIZE: int = 10_000

# Hard cap on a cluster's combined Tier 1 + Tier 2-extra pool before it's
# fed into Tier 2's pairwise combinations. Some anchors are generic
# enough (short, digit-heavy names) that Soundex/Metaphone collide with a
# large slice of the outside vocabulary — is_similar_enough() qualifies
# on ANY of WRatio/Soundex/Metaphone, so that collision isn't filtered by
# SIMILARITY_THRESHOLD. Without this cap, one such "hub" cluster turns
# into combinations(n, 2) with n in the thousands, which is quadratic in
# both CPU and the number of qualifying rows held in memory before
# down-sampling — that's what blows up RAM. Oversized pools are
# subsampled down to this size (seeded, so still deterministic).
TIER_2_MAX_POOL_PER_CLUSTER: int = 300

# Oversample factor: cap each cluster's accumulated Tier-1/Tier-2 candidates at
# factor × that cluster's tier target before down-sampling. This bounds memory
# to O(|U|) instead of O(all qualifying pairs). Tier 1 was previously uncapped,
# so on a large registry "hub" anchors (short/digit-heavy names that Soundex/
# Metaphone-collide with a huge slice of the vocabulary) made the candidate list
# grow into the millions of rows and OOM-killed the process. We only ever need
# an oversample of each cluster's target to still draw a representative random
# sample, so accumulation stops once a cluster hits its cap. Higher = more
# sampling diversity but more memory.
CANDIDATE_OVERSAMPLE_FACTOR: int = 4

# Floor on a cluster's candidate cap, so tiny clusters (target of 0-1) still
# accumulate a small spread to sample from rather than the first match only.
CANDIDATE_MIN_POOL: int = 50

# Minimum similarity for a pair to qualify for U (ANY measure)
SIMILARITY_THRESHOLD: int = 65

# Labels
POSITIVE_LABEL: int = 1
UNLABELED_LABEL: int = 0

# Random seed — set to None for non-deterministic runs
SEED: int = 42


# If True, read confirmed LASA pairs from P_CSV.
# If False, generate them via the local LLM proposer.
FROM_FILE: bool = False

# LLM proposer settings (only used when FROM_FILE = False)
LLM_ITERATIONS: int = 400
LLM_N_PROPOSALS: int = 5
LLM_OUTPUT_JSON: Path = RESULTS_DIR / "lasa_run.json"

# Same file as LLM_OUTPUT_JSON — used by Section 7 of the notebook to mine
# unselected candidates (candidates - x_2) as additional label=0 pairs.
LASA_RUN_JSON: Path = LLM_OUTPUT_JSON

# Standalone CSV of (x_1, unselected candidate) pairs — see Section 7
LASA_RUN_U_CSV: Path = RESULTS_DIR / "lasa_run_U.csv"

# If True, use the DeepSeek API instead of a local model for generation.
# Requires openai package: uv add openai  (or pip install openai)
USE_API_MODEL: bool = True

# DeepSeek API settings (only used when USE_API_MODEL = True)
DEEPSEEK_MODEL: str = "deepseek-v4-pro"
DEEPSEEK_API_KEY: str = ""


# Batch size for transcription — tune down if eSpeak is slow on your machine
IPA_BATCH_SIZE: int = 256

# Filipino G2P — a Phonetisaurus WFST trained on Tagalog Wiktionary
# (clean_tgl_wik.csv).
#
# The decoder comes from the `phonetisaurus` PyPI wheel (a project dependency),
# which bundles both the phonetisaurus-* binaries and the OpenFst shared
# libraries they link against. The binaries carry no RPATH, so src/fil_g2p.py
# locates the bundled lib dir and injects it as LD_LIBRARY_PATH — nothing needs
# to be on PATH, and no system OpenFst is required. Set FIL_G2P_BIN to override
# with an external phonetisaurus-g2pfst (e.g. a system build).
#
# The wheel ships x86_64 Linux binaries only.
FIL_G2P_BIN: str | None = None
FIL_G2P_MODEL: Path = Path("bin/cwik_model.fst")

# Names per phonetisaurus invocation. Phonetisaurus takes a --wordlist file, so
# one process handles a whole batch; this exists only to bound the temp file.
FIL_G2P_BATCH_SIZE: int = 2048


# Random state for final shuffle
SHUFFLE_SEED: int = 67
