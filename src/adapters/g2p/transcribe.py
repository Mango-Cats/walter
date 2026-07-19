"""
Single dispatch point for per-language IPA transcription.

This lives above src/adapters/g2p/eng.py and src/adapters/g2p/fil.py rather than inside
src/adapters/g2p/client.py, which those two modules import. Holding the language
registry in g2p.client would make the import cycle.

Four call sites used to keep private copies of this dispatch (dataset
assembly, augmenter, featurize, g2p), and they had drifted: two always
re-transcribed, one skipped languages already present, and one had gone
stale against the per-language schema entirely.
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
    """
    Add every language's IPA transcription columns to a drug-pair DataFrame.

    Args:
        df:            DataFrame with columns COL_X1 and COL_X2.
        langs:         Subset of TRANSCRIPTION_LANGS to transcribe.
                        None means all of them.
        skip_existing: Leave a language alone when its columns are already
                        populated. Callers that build a DataFrame from
                        scratch want False; callers re-processing a CSV
                        that may already carry transcriptions want True,
                        since G2P is the expensive stage.
        tag:           Log prefix, e.g. "dataset".
        verbose:       Print progress.

    Returns:
        Copy of df with each selected language's column pair added.
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
    """
    True when every column exists and holds at least one non-empty value.

    A present-but-blank column means a previous run failed or was run with
    add_phonemes off, so treating it as done would silently keep the blanks.
    """
    if not all(c in df.columns for c in cols):
        return False
    return all(df[c].fillna("").astype(str).str.strip().ne("").any() for c in cols)
