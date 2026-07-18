"""
Shared batching / caching machinery for the per-language G2P modules
(src/eng_g2p.py, src/fil_g2p.py).

Each language module supplies a `batch_fn` that maps a list of names to a list
of IPA strings, and the pair of output column names it owns. Everything else —
deduplication, batching, progress, the empty-transcription warning — is the
same regardless of language and lives here.
"""

import time
from typing import Callable

import pandas as pd

from config import COL_X1, COL_X2


def transcribe_dataframe(
    df: pd.DataFrame,
    *,
    batch_fn: Callable[[list[str]], list[str]],
    out_cols: tuple[str, str],
    tag: str,
    batch_size: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Add a pair of IPA transcription columns to a drug-pair DataFrame.

    Deduplicates all unique drug names before transcription so each name is
    only transcribed once regardless of how many pairs it appears in.

    Args:
        df:         DataFrame with columns COL_X1 and COL_X2.
        batch_fn:   Transcribes a batch of names, one IPA string per name.
        out_cols:   (column for x_1's transcription, column for x_2's).
        tag:        Log prefix, e.g. "eng_g2p".
        batch_size: Number of unique names per batch_fn call.
        verbose:    Print progress.

    Returns:
        Copy of df with out_cols added.
    """
    df = df.copy()
    col_1, col_2 = out_cols

    names_x1 = df[COL_X1].fillna("").tolist()
    names_x2 = df[COL_X2].fillna("").tolist()
    # sorted() rather than list(set(...)) so batch composition — and thus the
    # transcriptions — are reproducible across runs.
    unique_names = sorted(set(names_x1 + names_x2))

    if verbose:
        print(f"[{tag}] Unique drug names: {len(unique_names):,}")
        print(f"[{tag}] Batch size       : {batch_size}")

    cache: dict[str, str] = {}
    total = len(unique_names)
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch = unique_names[i : i + batch_size]
        results = batch_fn(batch)
        if len(results) != len(batch):
            raise RuntimeError(
                f"[{tag}] transcriber returned {len(results)} results for "
                f"{len(batch)} names; cannot align them to their inputs"
            )
        cache.update(zip(batch, results))
        if verbose:
            done = min(i + batch_size, total)
            print(
                f"  {done:>6,} / {total:,}  ({done / total * 100:.1f}%)  [{time.time() - t0:.1f}s]"
            )

    if verbose:
        print(f"[{tag}] Done in {time.time() - t0:.1f}s")

    df[col_1] = [cache.get(n, "") for n in names_x1]
    df[col_2] = [cache.get(n, "") for n in names_x2]

    empty_1 = (df[col_1] == "").sum()
    empty_2 = (df[col_2] == "").sum()
    if empty_1 or empty_2:
        print(
            f"[{tag}] WARNING: empty transcriptions — "
            f"{col_1}: {empty_1}, {col_2}: {empty_2}"
        )
    return df
