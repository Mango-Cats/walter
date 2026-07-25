"""
Where the confirmed LASA pairs (P) come from -- a checked-in CSV, a local LLM,
or the DeepSeek API -- and whether the pairs those same sources *reject* are
kept as a class of their own. See src/proposer/.
"""

from pathlib import Path

from .paths import RESULTS_DIR

# If True, read confirmed LASA pairs from P[DATA_SOURCE].
# If False, generate them via the local LLM proposer.
FROM_FILE: bool = False

# Soft labels. If True, D carries three label values instead of two:
#
#     POSITIVE_LABEL  ( 1)  proposed by the LLM, or predefined in P[DATA_SOURCE]
#     NEGATIVE_LABEL  (-1)  shown to the LLM and not proposed, or predefined
#                           in N[DATA_SOURCE]
#     UNLABELED_LABEL ( 0)  combinatorially induced -- the sampled pairs of U
#
# If False, no rejections are read at all and D is the two-value dataset it has
# always been. Either way U stays 0: a rejection is a judgement, an unlabeled
# pair is the absence of one, and collapsing them would throw that away.
# `walter assemble --soft-labels / --no-soft-labels` overrides this per run.
SOFT_LABELS: bool = False

# LLM proposer settings (only used when FROM_FILE = False).
# There is no iteration count: the proposer augments a predefined pair file
# and runs exactly once per pair in it, so the seed file sets the workload.
LLM_N_PROPOSALS: int = 5

# Canonical filename for the proposer's output, kept alongside the other
# artifact names so `walter propose --output <dir>` and `walter noise
# --input <dir>` agree on it. See config/paths.py.
LLM_OUTPUT_FILENAME: str = "lasa_run.json"
LLM_OUTPUT_JSON: Path = RESULTS_DIR / LLM_OUTPUT_FILENAME

# Same file as LLM_OUTPUT_JSON -- used by Section 7 of the notebook to mine
# unselected candidates (candidates - x_2) as additional label=0 pairs.
LASA_RUN_JSON: Path = LLM_OUTPUT_JSON

# Standalone CSV of (x_1, unselected candidate) pairs -- see Section 7
LASA_RUN_U_CSV: Path = RESULTS_DIR / "lasa_run_U.csv"

# If True, use the DeepSeek API instead of a local model for generation.
# Requires openai package: uv add openai  (or pip install openai)
USE_API_MODEL: bool = True

# DeepSeek API settings (only used when USE_API_MODEL = True)
DEEPSEEK_MODEL: str = "deepseek-v4-pro"

# Prefer the DEEPSEEK_API_KEY environment variable. src/adapters/llm/api.py
# falls back to it when this is empty. Leave blank so a key is never committed.
DEEPSEEK_API_KEY: str = ""
