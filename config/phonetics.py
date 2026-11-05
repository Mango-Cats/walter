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

# Words skipped when finding a drug name's core word (see src/pipeline/core_name.py).
CORE_NAME_STOPWORDS: frozenset[str] = frozenset(
    {
        # Salts and hydrates
        "acetate", "bromide", "carbonate", "chloride", "citrate", "dihydrate",
        "dihydrochloride", "dipropionate", "fumarate", "gluconate", "hcl",
        "hydrobromide", "hydrochloride", "hydroxide", "maleate", "monohydrate",
        "mononitrate", "phosphate", "potassium", "propionate", "sodium",
        "succinate", "sulfate", "sulphate", "trihydrate", "valerate", "besilate",
        # Dosage forms and routes
        "adsorbed", "cap", "capsule", "capsules", "cream", "drops", "er", "eye",
        "formulation", "gel", "iv", "mg", "mr", "odt", "ointment", "reformulated",
        "solution", "sr", "suspension", "syrup", "tab", "tablet", "tablets",
        "topical", "xr",
        # Product qualifiers and connecting words
        "advance", "and", "ds", "for", "forte", "kids", "od", "plus", "with",
    }
)  # fmt: skip

# First words that identify little alone, so the core keeps the word after them:
# "neo penotran" stays whole, and "sodium valproate" keeps its active ingredient.
CORE_LEADING_WORDS: frozenset[str] = frozenset(
    {
        # Brand-style prefixes
        "bio", "co", "dr", "fin", "go", "md", "mega", "neo", "neuro", "new", "one",
        "rapid", "st", "tri",
        # Ingredient and class words shared by many products
        "calcium", "human", "insulin", "iron", "magnesium", "multivitamins",
        "potassium", "purified", "sodium", "tetanus", "vitamin", "zinc",
    }
)  # fmt: skip

# Words that cannot complete a leading word: forms, routes, and qualifiers.
CORE_LEADING_EXCLUDED: frozenset[str] = frozenset(
    {
        "adsorbed", "advance", "and", "cap", "capsule", "capsules", "cream", "drops",
        "ds", "er", "eye", "for", "formulation", "forte", "gel", "iv", "kids", "mg",
        "mr", "od", "odt", "ointment", "plus", "reformulated", "solution", "sr",
        "suspension", "syrup", "tab", "tablet", "tablets", "topical", "with", "xr",
    }
)  # fmt: skip

# Config file stems scored over each pair's core words (see run_core_features).
CORE_ORTHOGRAPHIC_CONFIGS: tuple[str, ...] = (
    "bisim",
    "editex",
    "lcsubstring",
    "lcsuf",
    "levenshtein",
    "levenshtein_consonant",
    "prefix",
    "visual_weighted",
)
CORE_PHONETIC_CONFIG: str = "aline_cangu"
CORE_COLUMN_PREFIX: str = "core_"

TBB_BIN: Path = Path("bin/tbb-cli")

IPA_BATCH_SIZE: int = 256

FIL_G2P_BIN: str | None = None
FIL_G2P_MODEL: Path = Path("bin/cwik_model.fst")
FIL_G2P_BATCH_SIZE: int = 2048
