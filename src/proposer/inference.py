"""Proposer: augments predefined LASA pairs with LLM-selected registry candidates."""

import json
import re
from pathlib import Path

import jellyfish
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import (
    COL_X1,
    COL_X2,
    LLM_N_PROPOSALS,
    LLM_OUTPUT_JSON,
    P_INPUT_COLS,
    REGISTRY_COL,
    USE_API_MODEL,
)
from src.adapters.llm.api import api_response
from src.adapters.llm.local import LocalModel, response
from src.adapters.tbb import nativize as _nativize
from src.proposer.prompt import SYSTEM_PROMPT, construct_user_prompt

_CANDIDATE_LIMIT = 20
_MIN_EDIT_DISTANCE = 1
_MIN_LENGTH_RATIO = 0.5
_RATIO_THRESHOLD = 50
_SOUNDEX_THRESHOLD = 0.75


def _metaphone_code(name: str) -> str:
    """Compute a Metaphone code for a drug name using its Filipino pronunciation.

    Args:
        name: The drug name string.

    Returns:
        A Metaphone code string, or an empty string if invalid.

    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    try:
        return jellyfish.metaphone(_nativize(name))
    except Exception:
        return ""


def _is_fragment(a: str, b: str) -> bool:
    """Check whether the shorter name is only a fragment of a much longer name.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        True if the shorter name is less than `_MIN_LENGTH_RATIO` of the longer
        name's length and is not the longer name's leading word(s).

    """
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= _MIN_LENGTH_RATIO * len(long):
        return False
    short_words = short.split()
    return long.split()[: len(short_words)] != short_words


def _is_strength_variant(a: str, b: str) -> bool:
    """Check whether two names differ only in their numbers (e.g. strengths).

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        True if the names are identical once digits are removed.

    """
    return re.sub(r"\d+", "", a).split() == re.sub(r"\d+", "", b).split()


def _spelling_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity of two names, also comparing each with the other's leading words.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        Similarity score between 0.0 and 1.0.

    """
    a_lead = " ".join(a.split()[: len(b.split())])
    b_lead = " ".join(b.split()[: len(a.split())])
    return max(
        JaroWinkler.normalized_similarity(a, b),
        JaroWinkler.normalized_similarity(a, b_lead),
        JaroWinkler.normalized_similarity(a_lead, b),
    )


def _soundex_code(name: str) -> str:
    """Compute a 4-character Soundex code for a drug name using its Filipino pronunciation.

    Args:
        name: The drug name string.

    Returns:
        A 4-character Soundex code string, or an empty string if invalid.

    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if not name:
        return ""
    try:
        return jellyfish.soundex(_nativize(name))
    except Exception:
        return ""


def _soundex_similarity(code_a: str, code_b: str) -> float:
    """Calculate normalized similarity between two Soundex codes.

    Args:
        code_a: First 4-character Soundex code.
        code_b: Second 4-character Soundex code.

    Returns:
        Similarity score between 0.0 and 1.0.

    """
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
    drug_metaphone_map: dict[str, str] | None = None,
) -> list[str]:
    """Extract and rank confusable candidate drugs for an anchor drug.

    A candidate is kept if it differs from the anchor by more than its numbers,
    is not a fragment of a much longer name (or vice versa), and either its
    spelling similarity (fuzz.ratio) is above `_RATIO_THRESHOLD` or its Soundex
    similarity is at least `_SOUNDEX_THRESHOLD`. Kept candidates are ranked by
    the mean of spelling similarity (Jaro-Winkler on the names or their leading
    words) and sound similarity (Jaro-Winkler on the Metaphone codes).

    Args:
        anchor: Target drug name.
        known: Known paired drug name to exclude.
        all_drugs: List of all available drug names in the registry.
        drug_soundex_map: Optional precomputed map of drug names to Soundex codes.
        limit: Maximum number of candidate names to return.
        drug_metaphone_map: Optional precomputed map of drug names to Metaphone codes.

    Returns:
        List of up to `limit` confusable candidate drug names.

    """
    if drug_soundex_map is None:
        drug_soundex_map = {d: _soundex_code(d) for d in all_drugs}
    if drug_metaphone_map is None:
        drug_metaphone_map = {d: _metaphone_code(d) for d in all_drugs}

    s_anchor = _soundex_code(anchor)
    m_anchor = _metaphone_code(anchor)
    scored: list[tuple[float, float, str]] = []

    for candidate in all_drugs:
        if candidate == anchor or candidate == known:
            continue
        if Levenshtein.distance(anchor, candidate) < _MIN_EDIT_DISTANCE:
            continue
        if _is_strength_variant(anchor, candidate):
            continue
        if _is_fragment(anchor, candidate):
            continue

        s_candidate = drug_soundex_map.get(candidate, "")
        s_sim = _soundex_similarity(s_anchor, s_candidate)
        f_score = fuzz.ratio(anchor, candidate)

        if f_score > _RATIO_THRESHOLD or s_sim >= _SOUNDEX_THRESHOLD:
            spelling = _spelling_similarity(anchor, candidate)
            sound = JaroWinkler.normalized_similarity(
                m_anchor, drug_metaphone_map.get(candidate, "")
            )
            scored.append((0.5 * spelling + 0.5 * sound, f_score, candidate))

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
    """Load and validate predefined confusable drug pairs from a CSV file.

    Args:
        seed_csv: Path to the CSV file containing predefined pairs.

    Returns:
        A DataFrame containing the cleaned predefined pairs.

    Raises:
        ValueError: If required columns are missing or if the file contains no pairs.

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
    """Augment predefined confusable drug pairs with additional AI proposals.

    Args:
        registry_df: Cleaned drug registry DataFrame.
        model_choice: Which LocalModel to use if running locally.
        seed_pairs: Predefined confusable pairs DataFrame.
        n_proposals: Number of extra confusable candidates to request per seed.
        output_path: Destination path for saving JSON results.

    Returns:
        Path to the saved JSON file.

    """
    all_drugs = registry_df[REGISTRY_COL].tolist()
    drug_soundex_map = {drug: _soundex_code(drug) for drug in all_drugs}
    drug_metaphone_map = {drug: _metaphone_code(drug) for drug in all_drugs}
    results = []
    total = len(seed_pairs)

    for i, (anchor, known) in enumerate(zip(seed_pairs[COL_X1], seed_pairs[COL_X2])):
        candidates = extract_candidates(
            anchor,
            known,
            all_drugs,
            drug_soundex_map=drug_soundex_map,
            limit=_CANDIDATE_LIMIT,
            drug_metaphone_map=drug_metaphone_map,
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
    """Parse proposer JSON output into a DataFrame of confirmed and proposed pairs.

    Args:
        json_path: Path to the JSON file generated by run_inference.

    Returns:
        A DataFrame containing unique drug pairs with columns x_1 and x_2.

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
