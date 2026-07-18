"""
Single source of truth for the walter pipeline.
All paths, column names, and tunable parameters live here.
No other file should hardcode these values.

Split by pipeline stage:

    paths      the active registry, directories, dataset files
    schema     column names and label values
    sampling   how U is built (class balance, tiers, memory caps)
    proposer   where P comes from (file / local LLM / DeepSeek API)
    phonetics  phoc, tbb-cli, and the G2P toolchain

Every name is re-exported here, so `from config import COL_X1` keeps working.
Import from the submodule (`from config.sampling import SEED`) when you want
to be explicit about which stage a knob belongs to.
"""

from .paths import (
    D_CSV,
    D_ENGI_CSV,
    D_PHO_CSV,
    DATA_DIR,
    DATA_SOURCE,
    DataSource,
    MODELS_DIR,
    P,
    R,
    R_CLEAN,
    RESULTS_DIR,
    U_CSV,
    USE_PRECLEANED_REGISTRY,
)
from .phonetics import (
    FIL_G2P_BATCH_SIZE,
    FIL_G2P_BIN,
    FIL_G2P_MODEL,
    IPA_BATCH_SIZE,
    PHOC_BIN,
    PHOC_CONFIG_DIR,
    PHONETIC_ALGORITHMS,
    TBB_BIN,
)
from .proposer import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    FROM_FILE,
    LASA_RUN_JSON,
    LASA_RUN_U_CSV,
    LLM_ITERATIONS,
    LLM_N_PROPOSALS,
    LLM_OUTPUT_JSON,
    USE_API_MODEL,
)
from .sampling import (
    CANDIDATE_MIN_POOL,
    CANDIDATE_OVERSAMPLE_FACTOR,
    POSITIVE_PREVALENCE,
    SEED,
    SHUFFLE_SEED,
    SIMILARITY_THRESHOLD,
    TIER_1_PROPORTION,
    TIER_2_MAX_POOL_PER_CLUSTER,
    TIER_2_PROPORTION,
    TIER_2_SAMPLE_SIZE,
)
from .schema import (
    COL_LABEL,
    COL_T_ENG_1,
    COL_T_ENG_2,
    COL_T_FIL_1,
    COL_T_FIL_2,
    COL_T1,
    COL_T2,
    COL_X1,
    COL_X2,
    P_INPUT_COLS,
    POSITIVE_LABEL,
    REGISTRY_COL,
    TRANSCRIPTION_LANGS,
    UNLABELED_LABEL,
)

__all__ = [
    # paths
    "DataSource",
    "DATA_SOURCE",
    "DATA_DIR",
    "RESULTS_DIR",
    "MODELS_DIR",
    "R",
    "R_CLEAN",
    "USE_PRECLEANED_REGISTRY",
    "P",
    "U_CSV",
    "D_CSV",
    "D_PHO_CSV",
    "D_ENGI_CSV",
    # schema
    "REGISTRY_COL",
    "COL_X1",
    "COL_X2",
    "COL_T1",
    "COL_T2",
    "COL_T_ENG_1",
    "COL_T_ENG_2",
    "COL_T_FIL_1",
    "COL_T_FIL_2",
    "TRANSCRIPTION_LANGS",
    "COL_LABEL",
    "P_INPUT_COLS",
    "POSITIVE_LABEL",
    "UNLABELED_LABEL",
    # sampling
    "POSITIVE_PREVALENCE",
    "TIER_1_PROPORTION",
    "TIER_2_PROPORTION",
    "TIER_2_SAMPLE_SIZE",
    "TIER_2_MAX_POOL_PER_CLUSTER",
    "CANDIDATE_OVERSAMPLE_FACTOR",
    "CANDIDATE_MIN_POOL",
    "SIMILARITY_THRESHOLD",
    "SEED",
    "SHUFFLE_SEED",
    # proposer
    "FROM_FILE",
    "LLM_ITERATIONS",
    "LLM_N_PROPOSALS",
    "LLM_OUTPUT_JSON",
    "LASA_RUN_JSON",
    "LASA_RUN_U_CSV",
    "USE_API_MODEL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_API_KEY",
    # phonetics
    "PHOC_BIN",
    "PHOC_CONFIG_DIR",
    "PHONETIC_ALGORITHMS",
    "TBB_BIN",
    "IPA_BATCH_SIZE",
    "FIL_G2P_BIN",
    "FIL_G2P_MODEL",
    "FIL_G2P_BATCH_SIZE",
]
