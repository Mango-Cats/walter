"""
LLM-based LASA pair proposer.

The proposer augments rather than invents: it is seeded with a CSV of
predefined LASA pairs (columns x_1, x_2), and for each seed pair it asks the
LLM which *other* registry drugs are also confusible with x_1. The seed pair
always survives into the output -- it is confirmed input, not a proposal, and
the LLM is never given the chance to drop it.

This is the one stage that takes a file rather than a directory, because the
seed pairs are supplied by the user and are not produced by another stage.
"""

import json
from pathlib import Path

import jellyfish
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from config import (
    REGISTRY_COL,
    COL_X1,
    COL_X2,
    LLM_N_PROPOSALS,
    LLM_OUTPUT_JSON,
    P_INPUT_COLS,
    USE_API_MODEL,
)
from src.adapters.llm.api import api_response
from src.adapters.llm.local import LocalModel, response
from src.proposer.prompt import SYSTEM_PROMPT, construct_user_prompt

# Fuzzy candidates pulled per seed drug before the LLM sees them. One extra is
# requested so the seed drug itself can be dropped without going short.
_CANDIDATE_LIMIT = 20


def _soundex_code(name: str) -> str:
    """Safely compute 4-char Soundex code for name."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    try:
        return jellyfish.soundex(name)
    except Exception:
        return ""


def _soundex_similarity(code_a: str, code_b: str) -> float:
    """Levenshtein similarity on 4-char Soundex codes in [0.0, 1.0]."""
    if not code_a or not code_b:
        return 0.0
    dist = Levenshtein.distance(code_a, code_b)
    return max(0.0, 1.0 - (dist / 4.0))


def extract_candidates(
    anchor: str,
    known: str,
    all_drugs: list[str],
    drug_soundex_map: dict[str, str] | None = None,
    limit: int = _CANDIDATE_LIMIT,
) -> list[str]:
    """
    Extract candidate confusibles for an anchor drug satisfying:
      ((fuzzy_score > 60) or (soundex_similarity >= 0.75)) and (edit_distance > 2)

    Excludes exact matches to anchor and known. Candidates are ranked by:
      composite_score = 0.5 * (fuzzy_score / 100.0) + 0.5 * soundex_similarity

    Returns at most `limit` unique candidate names.
    """
    if drug_soundex_map is None:
        drug_soundex_map = {d: _soundex_code(d) for d in all_drugs}

    s_anchor = _soundex_code(anchor)
    scored: list[tuple[float, float, str]] = []

    for candidate in all_drugs:
        if candidate == anchor or candidate == known:
            continue
        if Levenshtein.distance(anchor, candidate) <= 2:
            continue

        s_candidate = drug_soundex_map.get(candidate, "")
        s_sim = _soundex_similarity(s_anchor, s_candidate)
        f_score = fuzz.WRatio(anchor, candidate)

        if f_score > 60 or s_sim >= 0.75:
            composite = 0.5 * (f_score / 100.0) + 0.5 * s_sim
            scored.append((composite, f_score, candidate))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen: set[str] = set()
    result: list[str] = []
    for _, _, candidate in scored:
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
            if len(result) >= limit:
                break
    return result


def load_seed_pairs(seed_csv: Path | str) -> pd.DataFrame:
    """
    Read the predefined LASA pairs the proposer augments.

    Raises rather than silently proposing from nothing, since an empty or
    mis-columned seed file would otherwise produce an empty P that only
    surfaces as a confusing failure two stages later.
    """
    pairs = pd.read_csv(seed_csv)
    missing = [c for c in P_INPUT_COLS if c not in pairs.columns]
    if missing:
        raise ValueError(
            f"{seed_csv} is missing column(s) {missing}. Predefined LASA pairs "
            f"need {list(P_INPUT_COLS)}."
        )
    pairs = pairs[list(P_INPUT_COLS)].dropna().drop_duplicates()
    if pairs.empty:
        raise ValueError(f"{seed_csv} has no usable pairs to augment.")
    return pairs.reset_index(drop=True)


def run_inference(
    registry_df: pd.DataFrame,
    model_choice: LocalModel,
    seed_pairs: pd.DataFrame,
    n_proposals: int = LLM_N_PROPOSALS,
    output_path: Path = LLM_OUTPUT_JSON,
) -> Path:
    """
    Augment predefined LASA pairs with additional confusibles.

    For each seed pair, x_1 is the anchor: registry drugs similar to it are
    gathered by similarity constraints (fuzzy match > 0.6 or Soundex
    similarity >= 0.75, edit distance > 2) and the LLM picks the true
    confusibles among them. The seed's own x_2 is excluded from the candidate
    list (it is already confirmed) and carried into the output directly.

    Writes results to a JSON file and returns the path.

    Args:
        registry_df:   Cleaned drug registry [REGISTRY_COL].
        model_choice:  Which LocalModel to use.
        seed_pairs:    Predefined LASA pairs [COL_X1, COL_X2] to augment.
        n_proposals:   Number of extra confusibles to request per seed pair.
        output_path:   Where to write the JSON output.

    Returns:
        Path to the written JSON file.
    """
    all_drugs = registry_df[REGISTRY_COL].tolist()
    drug_soundex_map = {drug: _soundex_code(drug) for drug in all_drugs}
    results = []
    total = len(seed_pairs)

    for i, (anchor, known) in enumerate(zip(seed_pairs[COL_X1], seed_pairs[COL_X2])):
        candidates = extract_candidates(
            anchor,
            known,
            all_drugs,
            drug_soundex_map=drug_soundex_map,
            limit=_CANDIDATE_LIMIT,
        )

        proposed: list[str] = []
        reasoning = ""
        if candidates:
            user_prompt = construct_user_prompt(
                anchor, "\n".join(candidates), n_proposals
            )
            if USE_API_MODEL:
                proposed, reasoning = api_response(
                    user_prompt,
                    candidates=candidates,
                    system_prompt=SYSTEM_PROMPT,
                    return_reasoning=True,
                )
            else:
                proposed = response(
                    user_prompt,
                    model=model_choice,
                    candidates=candidates,
                    system_prompt=SYSTEM_PROMPT,
                    new_toks_len=64,
                )

        entry = {
            "run": i + 1,
            COL_X1: anchor,
            "seed_x_2": known,
            "candidates": candidates,
            COL_X2: proposed,
        }
        if reasoning:
            entry["reasoning"] = reasoning
        results.append(entry)

        proposed_str = ", ".join(proposed) if proposed else "(none proposed)"
        print(f"[inference] {i + 1}/{total}: {anchor!r} + {known!r} → {proposed_str}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[inference] Results saved → {output_path.resolve()}")
    return output_path


def load_inference(json_path: Path | str) -> pd.DataFrame:
    """
    Parse the JSON written by run_inference() into a pairs DataFrame with
    columns [COL_X1, COL_X2].

    Both the seed pair and the LLM's additions are emitted, so the result is
    the augmented P: every predefined pair is present whether or not that
    entry produced any proposals.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = []
    for entry in data:
        drug = entry[COL_X1]
        seed = entry.get("seed_x_2")
        if seed:
            rows.append({COL_X1: drug, COL_X2: seed})
        for confusible in entry.get(COL_X2, []):
            rows.append({COL_X1: drug, COL_X2: confusible})
    pairs = pd.DataFrame(rows, columns=[COL_X1, COL_X2])
    return pairs.drop_duplicates().reset_index(drop=True)
