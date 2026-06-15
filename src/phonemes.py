"""
Adds IPA phonetic transcriptions to a drug-pair DataFrame using
phonemizer (eSpeak-NG backend).
Requirements:
    pip install phonemizer
    + espeak-ng installed on the system:
        Windows : https://github.com/espeak-ng/espeak-ng/releases  (.msi)
        Linux   : apt install espeak-ng
        macOS   : brew install espeak-ng
"""

import os
import time
import pandas as pd
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from config import COL_X1, COL_X2, COL_T1, COL_T2, IPA_BATCH_SIZE

_ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(_ESPEAK_DLL):
    EspeakBackend.set_library(_ESPEAK_DLL)


def _normalize_ipa(ipa: str) -> str:
    """
    Normalize IPA strings to the symbol set supported by the ALINE config.
    Handles:
      - Unicode lookalikes (e.g. ɡ U+0261 -> g U+0067)
      - Cover symbols (e.g. ᵻ -> ɪ)
      - Syllabic diacritics (e.g. n̩ -> n)
    """
    replacements = {
        "\u0261": "g",  # ɡ (IPA velar stop) -> g (ASCII lookalike already in config)
        "\u1d7b": "ɪ",  # ᵻ (cover symbol)   -> ɪ
        "\u0329": "",  # combining syllabic diacritic -> drop it (n̩ -> n)
    }
    for src, tgt in replacements.items():
        ipa = ipa.replace(src, tgt)
    return ipa


def _transcribe_batch(names: list[str]) -> list[str]:
    """
    Transcribe a batch of drug names to IPA using eSpeak-NG.
    Returns one IPA string per name, stripped of trailing whitespace
    and normalized to the ALINE config symbol set.
    """
    results = phonemize(
        names,
        backend="espeak",
        language="en-us",
        with_stress=False,
        njobs=1,
    )
    if isinstance(results, str):
        return [_normalize_ipa(results.strip())]
    return [_normalize_ipa(r.strip()) for r in results]


def transcribe_dataframe(
    df: pd.DataFrame,
    batch_size: int = IPA_BATCH_SIZE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add IPA transcription columns t_1 and t_2 to a drug-pair DataFrame.
    Deduplicates all unique drug names before transcription so each name
    is only transcribed once regardless of how many pairs it appears in.
    Args:
        df:         DataFrame with columns COL_X1 and COL_X2.
        batch_size: Number of unique names per eSpeak call.
        verbose:    Print progress.
    Returns:
        Copy of df with added columns COL_T1 and COL_T2.
    """
    df = df.copy()
    names_x1 = df[COL_X1].fillna("").tolist()
    names_x2 = df[COL_X2].fillna("").tolist()
    unique_names = list(set(names_x1 + names_x2))
    if verbose:
        print(f"[phonemes] Unique drug names: {len(unique_names):,}")
        print(f"[phonemes] Batch size       : {batch_size}")
    cache: dict[str, str] = {}
    total = len(unique_names)
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch = unique_names[i : i + batch_size]
        results = _transcribe_batch(batch)
        cache.update(zip(batch, results))
        if verbose:
            done = min(i + batch_size, total)
            print(
                f"  {done:>6,} / {total:,}  ({done / total * 100:.1f}%)  [{time.time() - t0:.1f}s]"
            )
    if verbose:
        print(f"[phonemes] Done in {time.time() - t0:.1f}s")
    df[COL_T1] = [cache.get(n, "") for n in names_x1]
    df[COL_T2] = [cache.get(n, "") for n in names_x2]
    empty_1 = (df[COL_T1] == "").sum()
    empty_2 = (df[COL_T2] == "").sum()
    if empty_1 or empty_2:
        print(
            f"[phonemes] WARNING: empty transcriptions — "
            f"{COL_T1}: {empty_1}, {COL_T2}: {empty_2}"
        )
    return df
