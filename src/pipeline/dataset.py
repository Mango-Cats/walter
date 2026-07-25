"""
Assembles P, N and U into the final dataset D and saves it:

D.csv (classification) - x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label.
One IPA transcription per language per name (see config.TRANSCRIPTION_LANGS).

D is the shuffled union of its inputs, deduplicated. N (rejected pairs, label
-1) is optional: pass it only under soft labels, otherwise D is the two-value
P/U dataset. Where a pair appears in more than one input the stronger claim
wins - a confirmed positive over a rejection, either over an unlabeled pair.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import (
    COL_X1,
    COL_X2,
    COL_T_ENG_1,
    COL_T_ENG_2,
    COL_T_FIL_1,
    COL_T_FIL_2,
    COL_LABEL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
    RESULTS_DIR,
    D_CSV,
    LASA_RUN_U_CSV,
    SHUFFLE_SEED,
)
from src.adapters.g2p.transcribe import transcribe_all

_T1_COLS: list[str] = [COL_T_ENG_1, COL_T_FIL_1]
_T2_COLS: list[str] = [COL_T_ENG_2, COL_T_FIL_2]


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


def canonical_key(a: str, b: str) -> tuple[str, str]:
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

    df["_key"] = df.apply(lambda r: canonical_key(r[COL_X1], r[COL_X2]), axis=1)
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
    # but noise.py might have used legacy col names - guard here)
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
    """
    Clean, deduplicate, transcribe, and save D to output_csv.

    Steps:
      1. Normalize column names and labels for each input
      2. Clean and deduplicate each independently
      3. Concatenate into D, deduplicate again across the union
      4. Add IPA transcriptions, one pair per language (English + Filipino)
      5. Reorder to [x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label]
      6. Shuffle D
      7. Save D (classification) to output_csv

    N is the soft-label input: rejected pairs, labelled NEGATIVE_LABEL. Leave
    it None for the two-value P/U dataset. The concatenation order is P, N, U
    and deduplication keeps the first occurrence, so a pair claimed by two
    inputs takes the label of the strongest claim rather than the last one read.

    Returns the assembled D DataFrame (classification schema).
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

    # parts is P before N before U, and clean_and_deduplicate keeps the first
    # occurrence, so the concatenation order is what resolves a contested pair.
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
    """
    Build a (x_1, x_2, label) DataFrame from the candidates each entry in
    lasa_run.json was shown but did NOT propose (candidates - x_2).

    Args:
        lasa_data: Parsed JSON from LLM_OUTPUT_JSON / LASA_RUN_JSON -
                   a list of {"x_1": ..., "candidates": [...], "x_2": [...]}.
        label:     What to label these pairs. UNLABELED_LABEL treats "the LLM
                   passed over it" as no information; NEGATIVE_LABEL treats it
                   as the rejection it is, and is what soft labels pass.

    Returns:
        DataFrame with columns [COL_X1, COL_X2, COL_LABEL].
    """
    rows = []
    for entry in lasa_data:
        x1 = entry.get(COL_X1)
        if not x1:
            continue
        candidates = entry.get("candidates", [])
        selected = {c.lower() for c in entry.get(COL_X2, [])}
        # The seed pair is confirmed input, never a candidate the LLM judged,
        # but guard anyway: it must never come back out as a rejection.
        seed = entry.get("seed_x_2")
        if seed:
            selected.add(seed.lower())
        for cand in candidates:
            if cand.lower() not in selected:
                rows.append({COL_X1: x1, COL_X2: cand, COL_LABEL: label})

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
    new_pairs = unselected_candidate_pairs(lasa_data)
    new_pairs = clean_and_deduplicate(new_pairs)

    # Drop new pairs that duplicate an existing row in D (either order)
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
