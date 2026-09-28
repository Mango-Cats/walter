"""Combines confusable and non-confusable drug pairs into the main dataset (D.csv).

This module handles:
    1. Merging confirmed confusable pairs (P), sampled non-confusable pairs (U),
       and optionally rejected pairs (N).
    2. Deduplicating pairs regardless of word order (e.g. A-B is the same as B-A).
    3. Generating spoken pronunciations in the International Phonetic Alphabet (IPA)
       for both English and Filipino.
    4. Saving the assembled dataset to disk.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import (
    COL_LABEL,
    COL_T_ENG_1,
    COL_T_ENG_2,
    COL_T_FIL_1,
    COL_T_FIL_2,
    COL_X1,
    COL_X2,
    D_CSV,
    LASA_RUN_U_CSV,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RESULTS_DIR,
    SHUFFLE_SEED,
    UNLABELED_LABEL,
)
from src.adapters.g2p.transcribe import transcribe_all

_T1_COLS: list[str] = [COL_T_ENG_1, COL_T_FIL_1]
_T2_COLS: list[str] = [COL_T_ENG_2, COL_T_FIL_2]


def _clean_name(name: str) -> str:
    """Normalize a drug name for deduplication.

    Converts the name to lowercase, removes accents, strips punctuation, and
    collapses extra whitespace.

    Args:
        name: The raw drug name.

    Returns:
        The normalized drug name string.

    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    nfd = unicodedata.normalize("NFD", name)
    name = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    name = re.sub(r"[-/']", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def canonical_key(a: str, b: str) -> tuple[str, str]:
    """Return a sorted pair of names so (A, B) and (B, A) match as duplicates.

    Args:
        a: First drug name.
        b: Second drug name.

    Returns:
        Alphabetically sorted tuple of both drug names.

    """
    return (a, b) if a <= b else (b, a)


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Clean drug names in pair columns and remove duplicate or self-paired rows.

    Args:
        df: DataFrame containing pair columns COL_X1 and COL_X2.

    Returns:
        Cleaned and deduplicated DataFrame.

    """
    df = df.copy()
    df[COL_X1] = df[COL_X1].apply(_clean_name)
    df[COL_X2] = df[COL_X2].apply(_clean_name)

    before = len(df)
    df = df[df[COL_X1] != df[COL_X2]]
    self_pairs = before - len(df)
    if self_pairs:
        print(f"[dataset] Removed {self_pairs:,} self-pairs")

    df["_key"] = df.apply(lambda r: canonical_key(r[COL_X1], r[COL_X2]), axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["_key"], keep="first")
    dupes = before - len(df)
    if dupes:
        print(f"[dataset] Removed {dupes:,} duplicate pairs")

    df = df.drop(columns=["_key"]).reset_index(drop=True)
    return df


def _normalize_pairs(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Format pair DataFrame with standard columns and assign a label.

    Args:
        df: Input pairs DataFrame.
        label: Numeric label to assign to all rows.

    Returns:
        A DataFrame with standard columns [COL_X1, COL_X2, COL_LABEL].

    """
    df = df.copy()

    for old, new in [("Brand Name", COL_X1), ("Confusible", COL_X2)]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    df[COL_LABEL] = label
    return df[[COL_X1, COL_X2, COL_LABEL]]


def assemble_and_save(
    P: pd.DataFrame,
    U: pd.DataFrame,
    N: pd.DataFrame | None = None,
    add_phonemes: bool = True,
    verbose: bool = True,
    output_csv: Path = D_CSV,
) -> pd.DataFrame:
    """Combine confusable, non-confusable, and rejected pairs into the dataset CSV.

    Merges all pair sets, removes duplicates, adds English and Filipino IPA
    pronunciations, shuffles the rows, and writes the final dataset to disk.

    Args:
        P: Confirmed confusable drug pairs (label = 1).
        U: Sampled non-confusable drug pairs (label = 0).
        N: Optional rejected pairs for soft labeling (label = -1).
        add_phonemes: Whether to transcribe drug names into IPA pronunciations.
        verbose: Whether to print progress messages.
        output_csv: Destination file path for D.csv.

    Returns:
        The assembled DataFrame.

    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    P_clean = clean_and_deduplicate(_normalize_pairs(P, POSITIVE_LABEL))
    U_clean = clean_and_deduplicate(_normalize_pairs(U, UNLABELED_LABEL))
    N_clean = (
        clean_and_deduplicate(_normalize_pairs(N, NEGATIVE_LABEL))
        if N is not None
        else None
    )

    parts = [P_clean, U_clean] if N_clean is None else [P_clean, N_clean, U_clean]

    if verbose:
        print(f"\n[dataset] P (clean): {len(P_clean):,} pairs")
        if N_clean is not None:
            print(f"[dataset] N (clean): {len(N_clean):,} pairs")
        print(f"[dataset] U (clean): {len(U_clean):,} pairs")

    D = pd.concat(parts, ignore_index=True)
    D = clean_and_deduplicate(D)

    if verbose:
        print(f"[dataset] D (union, deduped): {len(D):,} pairs")
        print(
            f"[dataset]   label={POSITIVE_LABEL} (P): "
            f"{(D[COL_LABEL] == POSITIVE_LABEL).sum():,}"
        )
        if N_clean is not None:
            print(
                f"[dataset]   label={NEGATIVE_LABEL} (N): "
                f"{(D[COL_LABEL] == NEGATIVE_LABEL).sum():,}"
            )
        print(
            f"[dataset]   label={UNLABELED_LABEL} (U): "
            f"{(D[COL_LABEL] == UNLABELED_LABEL).sum():,}"
        )

    if add_phonemes:
        D = transcribe_all(D, tag="dataset", verbose=verbose)

        for df_ in parts:
            for col in _T1_COLS:
                df_[col] = df_[COL_X1].map(dict(zip(D[COL_X1], D[col]))).fillna("")
            for col in _T2_COLS:
                df_[col] = df_[COL_X2].map(dict(zip(D[COL_X2], D[col]))).fillna("")
    else:
        for df_ in [D, *parts]:
            for col in _T1_COLS + _T2_COLS:
                df_[col] = ""

    final_cols = [COL_X1, *_T1_COLS, COL_X2, *_T2_COLS, COL_LABEL]
    D = D[final_cols]
    P_clean = P_clean.reindex(columns=final_cols, fill_value="")
    U_clean = U_clean.reindex(columns=final_cols, fill_value="")
    if N_clean is not None:
        N_clean = N_clean.reindex(columns=final_cols, fill_value="")

    D = D.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(output_csv, index=False)

    if verbose:
        print("\n[dataset] Saved:")
        print(f"  D -> {output_csv}  ({len(D):,} rows)")

    return D


def unselected_candidate_pairs(
    lasa_data: list[dict],
    label: int = UNLABELED_LABEL,
) -> pd.DataFrame:
    """Extract unselected candidate pairs from AI proposer output.

    Args:
        lasa_data: Parsed JSON data containing proposer runs and candidates.
        label: Label value to assign to these pairs (UNLABELED_LABEL or NEGATIVE_LABEL).

    Returns:
        A DataFrame with columns [COL_X1, COL_X2, COL_LABEL].

    """
    rows = []
    for entry in lasa_data:
        x1 = entry.get(COL_X1)
        if not x1:
            continue
        candidates = entry.get("candidates", [])
        selected = {c.lower() for c in entry.get(COL_X2, [])}
        seed = entry.get("seed_x_2")
        if seed:
            selected.add(seed.lower())
        for cand in candidates:
            if cand.lower() not in selected:
                rows.append({COL_X1: x1, COL_X2: cand, COL_LABEL: label})

    return pd.DataFrame(rows, columns=[COL_X1, COL_X2, COL_LABEL])


def write_lasa_run_unlabeled_csv(
    lasa_data: list[dict],
    output_path: Path = LASA_RUN_U_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """Write unselected candidate pairs from the proposer output to a separate CSV.

    Args:
        lasa_data: Parsed JSON data from the AI proposer.
        output_path: File path to save the unselected pairs CSV.
        verbose: Whether to print progress messages.

    Returns:
        Cleaned and deduplicated DataFrame that was written to disk.

    """
    df = unselected_candidate_pairs(lasa_data)
    df = clean_and_deduplicate(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(
            f"[dataset] Saved unselected-candidate pairs → {output_path}  ({len(df):,} rows)"
        )

    return df


def add_lasa_run_unlabeled(
    lasa_data: list[dict],
    D: pd.DataFrame,
    add_phonemes: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extend dataset D with unselected candidates from proposer output.

    Args:
        lasa_data: Parsed JSON data from the AI proposer.
        D: Existing assembled dataset DataFrame.
        add_phonemes: Whether to generate IPA pronunciations for new names.
        verbose: Whether to print progress messages.

    Returns:
        Extended copy of DataFrame D.

    """
    new_pairs = unselected_candidate_pairs(lasa_data)
    new_pairs = clean_and_deduplicate(new_pairs)

    existing_keys = {canonical_key(a, b) for a, b in zip(D[COL_X1], D[COL_X2])}
    new_pairs["_key"] = new_pairs.apply(
        lambda r: canonical_key(r[COL_X1], r[COL_X2]), axis=1
    )
    before = len(new_pairs)
    new_pairs = new_pairs[~new_pairs["_key"].isin(existing_keys)]
    new_pairs = new_pairs.drop(columns=["_key"]).reset_index(drop=True)
    dupes = before - len(new_pairs)

    if verbose:
        print(f"\n[dataset] Unselected-candidate pairs: {before:,}")
        if dupes:
            print(f"[dataset]   {dupes:,} already present in D, dropped")
        print(f"[dataset]   {len(new_pairs):,} new label=0 rows to add")

    if add_phonemes and len(new_pairs):
        new_pairs = transcribe_all(new_pairs, tag="dataset", verbose=verbose)
    else:
        for col in _T1_COLS + _T2_COLS:
            new_pairs[col] = ""

    final_cols = [COL_X1, *_T1_COLS, COL_X2, *_T2_COLS, COL_LABEL]
    new_pairs = new_pairs.reindex(columns=final_cols, fill_value="")

    D_extended = pd.concat([D, new_pairs], ignore_index=True)

    if verbose:
        print(f"\n[dataset] D extended: {len(D):,} -> {len(D_extended):,} rows")
        print(D_extended[COL_LABEL].value_counts().to_string())

    return D_extended
