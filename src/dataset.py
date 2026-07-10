"""
Assembles P and U into the final dataset D and saves it:

D.csv (classification) — x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label.
One IPA transcription per language per name (see config.TRANSCRIPTION_LANGS).

D is the shuffled union of P and U, deduplicated.
"""

import re
import unicodedata

import pandas as pd

from config import (
    COL_X1,
    COL_X2,
    COL_T_ENG_1,
    COL_T_ENG_2,
    COL_T_FIL_1,
    COL_T_FIL_2,
    COL_LABEL,
    POSITIVE_LABEL,
    UNLABELED_LABEL,
    RESULTS_DIR,
    D_OUT_CSV,
    SHUFFLE_SEED,
)
from src.eng_g2p import transcribe_dataframe as transcribe_eng
from src.fil_g2p import transcribe_dataframe as transcribe_fil

# x_1's transcriptions, then x_2's, in language order.
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
    Clean, deduplicate, transcribe, and save D.csv to RESULTS_DIR.

    Steps:
      1. Normalize column names and labels for both P and U
      2. Clean and deduplicate each independently
      3. Concatenate into D, deduplicate again across the union
      4. Add IPA transcriptions, one pair per language (English + Filipino)
      5. Reorder to [x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label]
      6. Shuffle D
      7. Save D.csv (classification) to RESULTS_DIR

    Returns the assembled D DataFrame (classification schema).
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
            print("\n[dataset] Adding English IPA transcriptions...")
        D = transcribe_eng(D, verbose=verbose)
        if verbose:
            print("\n[dataset] Adding Filipino IPA transcriptions...")
        D = transcribe_fil(D, verbose=verbose)

        for df_ in [P_clean, U_clean]:
            for col in _T1_COLS:
                df_[col] = df_[COL_X1].map(dict(zip(D[COL_X1], D[col]))).fillna("")
            for col in _T2_COLS:
                df_[col] = df_[COL_X2].map(dict(zip(D[COL_X2], D[col]))).fillna("")
    else:
        for df_ in [D, P_clean, U_clean]:
            for col in _T1_COLS + _T2_COLS:
                df_[col] = ""

    final_cols = [COL_X1, *_T1_COLS, COL_X2, *_T2_COLS, COL_LABEL]
    D = D[final_cols]
    P_clean = P_clean.reindex(columns=final_cols, fill_value="")
    U_clean = U_clean.reindex(columns=final_cols, fill_value="")

    D = D.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    D.to_csv(D_OUT_CSV, index=False)

    if verbose:
        print("\n[dataset] Saved:")
        print(f"  D → {D_OUT_CSV}  ({len(D):,} rows)")

    return D
