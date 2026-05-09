"""
LLM-based LASA pair proposer.
Used when FROM_FILE = False in config.py.
"""

import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from config import (
    REGISTRY_COL,
    COL_X1,
    COL_X2,
    LLM_ITERATIONS,
    LLM_N_PROPOSALS,
    LLM_OUTPUT_JSON,
    RESULTS_DIR,
)
from src.proposer.llm import LocalModel, response
from src.proposer.prompt import construct_user_prompt


def run_inference(
    registry_df: pd.DataFrame,
    model_choice: LocalModel,
    iterations: int = LLM_ITERATIONS,
    n_proposals: int = LLM_N_PROPOSALS,
    output_path: Path = LLM_OUTPUT_JSON,
) -> Path:
    """
    Randomly sample drugs from registry_df, find similar candidates via
    fuzzy matching, then ask the LLM which are true confusibles.

    Writes results to a JSON file and returns the path.

    Args:
        registry_df:   Cleaned drug registry [REGISTRY_COL].
        model_choice:  Which LocalModel to use.
        iterations:    Number of drugs to sample.
        n_proposals:   Number of confusibles to request per drug.
        output_path:   Where to write the JSON output.

    Returns:
        Path to the written JSON file.
    """
    all_drugs = registry_df[REGISTRY_COL].tolist()
    results = []

    for i in range(iterations):
        sample_drug = registry_df.sample(n=1)[REGISTRY_COL].iloc[0]
        top_matches = process.extract(
            sample_drug, all_drugs, scorer=fuzz.WRatio, limit=11
        )
        candidates = [m[0] for m in top_matches if m[0] != sample_drug][:10]

        user_prompt = construct_user_prompt(
            sample_drug, "\n".join(candidates), n_proposals
        )
        proposed = response(
            user_prompt,
            model=model_choice,
            candidates=candidates,
            new_toks_len=64,
        )

        results.append(
            {
                "run": i + 1,
                COL_X1: sample_drug,
                "candidates": candidates,
                COL_X2: proposed,
            }
        )

        proposed_str = ", ".join(proposed) if proposed else "(none proposed)"
        print(f"[inference] Iteration {i + 1}: {sample_drug!r} → {proposed_str}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[inference] Results saved → {output_path.resolve()}")
    return output_path


def load_inference(json_path: Path | str) -> pd.DataFrame:
    """
    Parse the JSON written by run_inference() into a pairs DataFrame
    with columns [COL_X1, COL_X2]. Entries with no proposals are skipped.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = []
    for entry in data:
        drug = entry[COL_X1]
        proposed = entry.get(COL_X2, [])
        if not proposed:
            continue
        for confusible in proposed:
            rows.append({COL_X1: drug, COL_X2: confusible})
    return pd.DataFrame(rows, columns=[COL_X1, COL_X2])
