"""Settings for finding confusable drug pairs using AI models.

This module specifies where confirmed pairs come from (a CSV file or an AI proposer),
whether rejected pairs are saved with negative labels (-1), and settings for local
models or remote API providers such as DeepSeek.
"""

from pathlib import Path

from .paths import RESULTS_DIR

FROM_FILE: bool = False
SOFT_LABELS: bool = False

LLM_N_PROPOSALS: int = 5

LLM_OUTPUT_FILENAME: str = "lasa_run.json"
LLM_OUTPUT_JSON: Path = RESULTS_DIR / LLM_OUTPUT_FILENAME
LASA_RUN_JSON: Path = LLM_OUTPUT_JSON
LASA_RUN_U_CSV: Path = RESULTS_DIR / "lasa_run_U.csv"

USE_API_MODEL: bool = True
DEEPSEEK_MODEL: str = "deepseek-v4-pro"
DEEPSEEK_API_KEY: str = ""
