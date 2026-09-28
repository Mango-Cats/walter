"""Central dispatcher for grapheme-to-phoneme (G2P) pronunciations across languages.

This module routes pronunciation requests to the appropriate language engines
(English via eSpeak-NG and Filipino via Phonetisaurus) and merges their results.
"""

import pandas as pd

from config import TRANSCRIPTION_LANGS
from src.adapters.g2p.eng import transcribe_dataframe as _transcribe_eng
from src.adapters.g2p.fil import transcribe_dataframe as _transcribe_fil

_TRANSCRIBERS = {
    "eng": _transcribe_eng,
    "fil": _transcribe_fil,
}


def transcribe_all(
    df: pd.DataFrame,
    *,
    langs: list[str] | None = None,
    skip_existing: bool = False,
    tag: str = "g2p",
    verbose: bool = True,
) -> pd.DataFrame:
    """Add IPA pronunciation columns for all requested languages to a drug-pair DataFrame.

    Args:
        df: DataFrame containing drug pair columns COL_X1 and COL_X2.
        langs: Optional list of language codes to transcribe (defaults to all configured).
        skip_existing: If True, skips languages whose columns already contain values.
        tag: Logging prefix for status output.
        verbose: Whether to print progress messages.

    Returns:
        Copy of DataFrame with pronunciation columns added.

    Raises:
        ValueError: If an unknown language is requested or lacks an active transcriber.

    """
    selected = list(TRANSCRIPTION_LANGS) if langs is None else list(langs)

    unknown = [lang for lang in selected if lang not in TRANSCRIPTION_LANGS]
    if unknown:
        raise ValueError(
            f"[{tag}] unknown language(s) {unknown}; "
            f"config.TRANSCRIPTION_LANGS defines {list(TRANSCRIPTION_LANGS)}"
        )

    missing = [lang for lang in selected if lang not in _TRANSCRIBERS]
    if missing:
        raise ValueError(
            f"[{tag}] no transcriber registered for {missing}; "
            f"add one to _TRANSCRIBERS in src/adapters/g2p/transcribe.py"
        )

    for lang in selected:
        cols = TRANSCRIPTION_LANGS[lang]
        if skip_existing and _already_transcribed(df, cols):
            if verbose:
                print(f"[{tag}] {lang}: {list(cols)} already present, reusing")
            continue
        if verbose:
            print(f"\n[{tag}] Transcribing: {lang}")
        df = _TRANSCRIBERS[lang](df, verbose=verbose)

    return df


def _already_transcribed(df: pd.DataFrame, cols: tuple[str, str]) -> bool:
    """Check whether both pronunciation columns exist and contain non-empty values.

    Args:
        df: Input DataFrame to check.
        cols: Tuple of two column names.

    Returns:
        True if both columns exist and have at least one non-empty value, False otherwise.

    """
    if not all(c in df.columns for c in cols):
        return False
    return all(df[c].fillna("").astype(str).str.strip().ne("").any() for c in cols)
