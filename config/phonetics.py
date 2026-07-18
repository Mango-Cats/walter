"""
The phonetic toolchain: the external binaries (phoc, tbb-cli, phonetisaurus),
the grapheme-to-phoneme settings, and which phoc algorithms need a
transcription rather than the raw name.
"""

from pathlib import Path

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
