"""English (en-us) grapheme-to-phoneme (G2P) transcription.

This module uses eSpeak-NG via the `phonemizer` library to convert English drug
names into International Phonetic Alphabet (IPA) pronunciations.
"""

import os
import tempfile

os.environ.setdefault(
    "PULSE_RUNTIME_PATH", os.path.join(tempfile.gettempdir(), "pulse")
)

import pandas as pd
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

from config import COL_T_ENG_1, COL_T_ENG_2, IPA_BATCH_SIZE
from src.adapters.g2p.client import transcribe_dataframe as _transcribe_dataframe

_TAG = "eng_g2p"

_ESPEAK_DLL = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(_ESPEAK_DLL):
    EspeakBackend.set_library(_ESPEAK_DLL)


def _normalize_ipa(ipa: str) -> str:
    """Normalize IPA phonetic symbols to match standard ALINE phonetic configurations.

    Args:
        ipa: Raw IPA pronunciation string.

    Returns:
        Normalized IPA pronunciation string.

    """
    replacements = {
        "\u0261": "g",
        "\u1d7b": "ɪ",
        "\u0329": "",
    }
    for src, tgt in replacements.items():
        ipa = ipa.replace(src, tgt)
    return ipa


def _transcribe_batch(names: list[str]) -> list[str]:
    """Convert a batch of drug names into English IPA pronunciations using eSpeak-NG.

    Args:
        names: List of drug names.

    Returns:
        List of normalized IPA pronunciation strings.

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
    """Add English IPA pronunciation columns to a drug-pair DataFrame.

    Args:
        df: DataFrame containing drug pair columns COL_X1 and COL_X2.
        batch_size: Number of unique drug names per eSpeak batch.
        verbose: Whether to print progress messages.

    Returns:
        Copy of DataFrame with English pronunciation columns added.

    """
    return _transcribe_dataframe(
        df,
        batch_fn=_transcribe_batch,
        out_cols=(COL_T_ENG_1, COL_T_ENG_2),
        tag=_TAG,
        batch_size=batch_size,
        verbose=verbose,
    )
