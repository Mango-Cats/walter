"""
English (en-us) G2P: adds IPA phonetic transcriptions to a drug-pair
DataFrame using phonemizer (eSpeak-NG backend).
Requirements:
    pip install phonemizer
    + espeak-ng installed on the system:
        Windows : https://github.com/espeak-ng/espeak-ng/releases  (.msi)
        Linux   : apt install espeak-ng
        macOS   : brew install espeak-ng
"""

import os
import tempfile

# eSpeak-NG pulls in libpulse, which on init tries to create its audio
# runtime dir under $XDG_RUNTIME_DIR (e.g. /run/user/1000/pulse). On WSL
# that path often doesn't exist, so libpulse spams stderr with
# "Failed to create secure directory (...)". We never play audio (text
# -> IPA only), so redirect pulse's runtime dir somewhere writable to
# silence it. Must be set before phonemizer imports espeak below.
os.environ.setdefault("PULSE_RUNTIME_PATH", os.path.join(tempfile.gettempdir(), "pulse"))

import pandas as pd
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

from config import COL_T_ENG_1, COL_T_ENG_2, IPA_BATCH_SIZE
from src.g2p_common import transcribe_dataframe as _transcribe_dataframe

_TAG = "eng_g2p"

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
    Add English IPA transcription columns COL_T_ENG_1 / COL_T_ENG_2 to a
    drug-pair DataFrame.

    Args:
        df:         DataFrame with columns COL_X1 and COL_X2.
        batch_size: Number of unique names per eSpeak call.
        verbose:    Print progress.
    Returns:
        Copy of df with added columns COL_T_ENG_1 and COL_T_ENG_2.
    """
    return _transcribe_dataframe(
        df,
        batch_fn=_transcribe_batch,
        out_cols=(COL_T_ENG_1, COL_T_ENG_2),
        tag=_TAG,
        batch_size=batch_size,
        verbose=verbose,
    )
