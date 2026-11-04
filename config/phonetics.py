"""
Settings for phonetic tools, pronunciation models, and speech similarity binaries.

This module configures paths to external helper tools:
    - phoc: Fast phonetic similarity scoring tool.
    - tbb-cli: Filipino loanword adaptation tool.
    - Phonetisaurus: Filipino grapheme-to-phoneme pronunciation model.
"""

from pathlib import Path

PHOC_BIN: Path = Path("bin/phoc")
PHOC_CONFIG_DIR: Path = Path("bin/pho_conf")

PHONETIC_ALGORITHMS: frozenset[str] = frozenset({"aline"})

# Transcriptions each phonetic config is scored over, keyed by config file stem.
# A phonetic config not listed here is scored over every TRANSCRIPTION_LANGS entry.
PHONETIC_CONFIG_LANGS: dict[str, tuple[str, ...]] = {"aline_cangu": ("fil",)}

TBB_BIN: Path = Path("bin/tbb-cli")

IPA_BATCH_SIZE: int = 256

FIL_G2P_BIN: str | None = None
FIL_G2P_MODEL: Path = Path("bin/cwik_model.fst")
FIL_G2P_BATCH_SIZE: int = 2048

