"""
Where the confirmed LASA pairs (P) come from: a checked-in CSV, a local LLM,
or the DeepSeek API. See src/proposer/.
"""

from pathlib import Path

from .paths import RESULTS_DIR

# If True, read confirmed LASA pairs from P[DATA_SOURCE].
# If False, generate them via the local LLM proposer.
FROM_FILE: bool = False

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
