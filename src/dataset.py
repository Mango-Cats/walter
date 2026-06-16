"""
Assembles P and U into the final dataset D, then saves all three as CSVs.

Output schema for all three files:
    x_1    : drug name A
    x_2    : drug name B
    t_1    : IPA transcription of x_1
    t_2    : IPA transcription of x_2
    label  : 1 = confirmed LASA (P rows), 0 = unlabeled (U rows)

P.csv and U.csv also carry this schema so they are self-contained and
can be inspected or reloaded independently.

D.csv is the shuffled union of P and U, deduplicated.
"""

import re
import unicodedata

import pandas as pd

from config import (
    COL_X1,
    COL_X2,
    COL_T1,
    COL_T2,
    COL_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
    RESULTS_DIR,
    D_OUT_CSV,
    LASA_RUN_U_CSV,
    SHUFFLE_SEED,
)
from src.phonemes import transcribe_dataframe


def _clean_name(name: str) -> str:
    """
    Normalize a drug name for deduplication purposes.
    Lowercase, strip, diacritics removed, symbols → space, collapse spaces.
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    nfd = unicodedata.normalize("NFD", name)
    name = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    name = re.sub(r"[-/']", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _canonical_key(a: str, b: str) -> tuple[str, str]:
    """Sorted pair so (A, B) and (B, A) are treated as duplicates."""
    return tuple(sorted([a, b]))  # type: ignore[return-value]


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean drug names in COL_X1 / COL_X2, remove self-pairs,
    and drop canonical duplicates. Keeps the first occurrence.
    """
    df = df.copy()
    df[COL_X1] = df[COL_X1].apply(_clean_name)
    df[COL_X2] = df[COL_X2].apply(_clean_name)

    before = len(df)
    df = df[df[COL_X1] != df[COL_X2]]
    self_pairs = before - len(df)
    if self_pairs:
        print(f"[dataset] Removed {self_pairs:,} self-pairs")

    df["_key"] = df.apply(lambda r: _canonical_key(r[COL_X1], r[COL_X2]), axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["_key"], keep="first")
    dupes = before - len(df)
    if dupes:
        print(f"[dataset] Removed {dupes:,} duplicate pairs")

    df = df.drop(columns=["_key"]).reset_index(drop=True)
    return df


def _normalize_pairs(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """
    Ensure a pairs DataFrame has exactly [COL_X1, COL_X2, COL_LABEL].
    Drops any extra columns (similarity, tier, etc.) from noise output.
    """
    df = df.copy()

    # Handle noise output column naming (COL_X1 may already be correct,
    # but noise.py might have used legacy col names — guard here)
    for old, new in [("Brand Name", COL_X1), ("Confusible", COL_X2)]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    df[COL_LABEL] = label
    return df[[COL_X1, COL_X2, COL_LABEL]]


def assemble_and_save(
    P: pd.DataFrame,
    U: pd.DataFrame,
    add_phonemes: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean, deduplicate, transcribe, and save P, U, and D.

    Steps:
      1. Normalize column names and labels for both P and U
      2. Clean and deduplicate each independently
      3. Concatenate into D, deduplicate again across the union
      4. Add IPA transcriptions (t_1, t_2)
      5. Reorder columns to [x_1, t_1, x_2, t_2, label]
      6. Shuffle D
      7. Save P.csv, U.csv, D.csv to RESULTS_DIR

    Returns the assembled D DataFrame.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    P_norm = _normalize_pairs(P, POSITIVE_LABEL)
    U_norm = _normalize_pairs(U, UNLABELED_LABEL)

    P_clean = clean_and_deduplicate(P_norm)
    U_clean = clean_and_deduplicate(U_norm)

    if verbose:
        print(f"\n[dataset] P (clean): {len(P_clean):,} pairs")
        print(f"[dataset] U (clean): {len(U_clean):,} pairs")

    D = pd.concat([P_clean, U_clean], ignore_index=True)
    D = clean_and_deduplicate(D)

    if verbose:
        print(f"[dataset] D (union, deduped): {len(D):,} pairs")
        print(f"[dataset]   label=1 (P): {(D[COL_LABEL] == POSITIVE_LABEL).sum():,}")
        print(f"[dataset]   label=0 (U): {(D[COL_LABEL] == UNLABELED_LABEL).sum():,}")

    if add_phonemes:
        if verbose:
            print("\n[dataset] Adding IPA transcriptions...")
        D = transcribe_dataframe(D, verbose=verbose)
        trans_map_1 = dict(zip(D[COL_X1], D[COL_T1]))
        trans_map_2 = dict(zip(D[COL_X2], D[COL_T2]))

        for df_ in [P_clean, U_clean]:
            df_[COL_T1] = df_[COL_X1].map(trans_map_1).fillna("")
            df_[COL_T2] = df_[COL_X2].map(trans_map_2).fillna("")
    else:
        for df_ in [D, P_clean, U_clean]:
            df_[COL_T1] = ""
            df_[COL_T2] = ""

    final_cols = [COL_X1, COL_T1, COL_X2, COL_T2, COL_LABEL]
    D = D[final_cols]
    P_clean = P_clean.reindex(columns=final_cols, fill_value="")
    U_clean = U_clean.reindex(columns=final_cols, fill_value="")

    D = D.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    D.to_csv(D_OUT_CSV, index=False)

    if verbose:
        print("\n[dataset] Saved:")
        print(f"  D → {D_OUT_CSV}  ({len(D):,} rows)")

    return D


def _unselected_candidate_pairs(lasa_data: list[dict]) -> pd.DataFrame:
    """
    Build a (x_1, x_2, label=0) DataFrame from the candidates each entry
    in lasa_run.json was shown but did NOT propose (candidates - x_2).

    Args:
        lasa_data: Parsed JSON from LLM_OUTPUT_JSON / LASA_RUN_JSON —
                   a list of {"x_1": ..., "candidates": [...], "x_2": [...]}.

    Returns:
        DataFrame with columns [COL_X1, COL_X2, COL_LABEL], label=0.
    """
    rows = []
    for entry in lasa_data:
        x1 = entry.get(COL_X1)
        if not x1:
            continue
        candidates = entry.get("candidates", [])
        selected = {c.lower() for c in entry.get(COL_X2, [])}
        for cand in candidates:
            if cand.lower() not in selected:
                rows.append({COL_X1: x1, COL_X2: cand, COL_LABEL: UNLABELED_LABEL})

    return pd.DataFrame(rows, columns=[COL_X1, COL_X2, COL_LABEL])


def write_lasa_run_unlabeled_csv(
    lasa_data: list[dict],
    output_path=LASA_RUN_U_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Write the (x_1, unselected candidate) pairs from lasa_run.json to
    their own CSV, in isolation, for inspection.

    Returns the cleaned, deduplicated DataFrame that was written.
    """
    df = _unselected_candidate_pairs(lasa_data)
    df = clean_and_deduplicate(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"[dataset] Saved unselected-candidate pairs → {output_path}  ({len(df):,} rows)")

    return df


def add_lasa_run_unlabeled(
    lasa_data: list[dict],
    D: pd.DataFrame,
    add_phonemes: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extend D with (x_1, unselected candidate) pairs as label=0 rows.

    Cleans and deduplicates the new pairs, drops any that already exist
    in D (in either direction), optionally adds IPA transcriptions for
    any new drug names, and returns the extended D.

    Args:
        lasa_data:    Parsed JSON from LLM_OUTPUT_JSON / LASA_RUN_JSON.
        D:            Existing assembled dataset (columns x_1, t_1, x_2, t_2, label).
        add_phonemes: If True, transcribe any new unique drug names to IPA.
        verbose:      Print progress.

    Returns:
        Extended copy of D, NOT yet saved to disk.
    """
    new_pairs = _unselected_candidate_pairs(lasa_data)
    new_pairs = clean_and_deduplicate(new_pairs)

    # Drop new pairs that duplicate an existing row in D (either order)
    existing_keys = {
        _canonical_key(a, b) for a, b in zip(D[COL_X1], D[COL_X2])
    }
    new_pairs["_key"] = new_pairs.apply(
        lambda r: _canonical_key(r[COL_X1], r[COL_X2]), axis=1
    )
    before = len(new_pairs)
    new_pairs = new_pairs[~new_pairs["_key"].isin(existing_keys)]
    new_pairs = new_pairs.drop(columns=["_key"]).reset_index(drop=True)
    dupes = before - len(new_pairs)

    if verbose:
        print(f"\n[dataset] Unselected-candidate pairs: {before:,}")
        if dupes:
            print(f"[dataset]   {dupes:,} already present in D — dropped")
        print(f"[dataset]   {len(new_pairs):,} new label=0 rows to add")

    if add_phonemes and len(new_pairs):
        new_pairs = transcribe_dataframe(new_pairs, verbose=verbose)
    else:
        new_pairs[COL_T1] = ""
        new_pairs[COL_T2] = ""

    final_cols = [COL_X1, COL_T1, COL_X2, COL_T2, COL_LABEL]
    new_pairs = new_pairs.reindex(columns=final_cols, fill_value="")

    D_extended = pd.concat([D, new_pairs], ignore_index=True)

    if verbose:
        print(f"\n[dataset] D extended: {len(D):,} → {len(D_extended):,} rows")
        print(D_extended[COL_LABEL].value_counts().to_string())

    return D_extended
