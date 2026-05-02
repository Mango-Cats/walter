from .local import LocalModel, response
from .prompt import construct_user_prompt
from rapidfuzz import process, fuzz
from src.preprocessing import TARGET_COL
import pandas as pd
import json
from pathlib import Path

LABEL_COL = "Confusible"


def run_inference(
    output_path: str,
    D_clean: pd.DataFrame,
    model_choice: LocalModel = LocalModel.DEEPSEEK_1_5B,
    iterations: int = 1,
    n_proposals: int = 5,
) -> Path:
    """
    Runs LASA inference and writes results to a JSON file.
    Returns the path to the output file.
    """
    all_drugs = D_clean[TARGET_COL].to_list()
    results = []

    for i in range(iterations):
        sample_drug = D_clean.sample(n=1)[TARGET_COL].iloc[0]
        top_matches = process.extract(
            sample_drug, all_drugs, scorer=fuzz.WRatio, limit=11
        )
        candidate_list = [match[0] for match in top_matches if match[0] != sample_drug][
            :10
        ]

        user_prompt = construct_user_prompt(
            sample_drug, "\n".join(candidate_list), n_proposals
        )
        raw_output = response(
            user_prompt,
            model=model_choice,
            candidates=candidate_list,
            new_toks_len=64,
        )

        results.append(
            {
                "iteration": i + 1,
                TARGET_COL: sample_drug,
                "candidates": candidate_list,
                LABEL_COL: raw_output,
            }
        )

        proposed_str = ", ".join(raw_output) if raw_output else "(No drugs proposed)"
        print(f"<walter> Iteration {i + 1}:")
        print(f"\tSelected Drug Name: {sample_drug}")
        print(f"\tProposed Confusibles: {proposed_str}\n")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"<walter> Results saved to {out.resolve()}")
    return out


def load_inference(json_path: str | Path) -> pd.DataFrame:
    """
    Converts the inference JSON output of `run_inference()` into a
    DataFrame of LASA pairs. Drugs with no proposals are skipped.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    rows = []
    for entry in data:
        drug = entry[TARGET_COL]
        proposed = entry[LABEL_COL]
        if not proposed:
            continue
        for confusible in proposed:
            rows.append({TARGET_COL: drug, LABEL_COL: confusible})

    return pd.DataFrame(rows, columns=[TARGET_COL, LABEL_COL])
