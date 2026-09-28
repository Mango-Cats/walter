"""Shared batching and transcription helper for grapheme-to-phoneme (G2P) models.

This module provides common utilities for transcribing large datasets of drug names:
    - Deduplicates names so each word is only transcribed once.
    - Processes names in batches to keep memory usage low.
    - Tracks progress and flags any empty transcriptions.
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
    """Add spoken IPA pronunciation columns to a drug-pair DataFrame.

    Deduplicates all unique drug names before transcription so each name is
    transcribed only once, then maps the results back into the DataFrame.

    Args:
        df: Input DataFrame containing columns COL_X1 and COL_X2.
        batch_fn: Function that converts a list of drug names into IPA pronunciations.
        out_cols: Tuple of output column names for x_1 and x_2 pronunciations.
        tag: Logging prefix for status output (e.g. 'eng_g2p').
        batch_size: Maximum number of unique names to process per batch.
        verbose: Whether to print progress messages.

    Returns:
        A copy of the DataFrame with the new pronunciation columns added.

    Raises:
        RuntimeError: If the transcription function returns an unexpected number of results.

    """
    df = df.copy()
    col_1, col_2 = out_cols

    names_x1 = df[COL_X1].fillna("").tolist()
    names_x2 = df[COL_X2].fillna("").tolist()
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
            f"[{tag}] WARNING: empty transcriptions - "
            f"{col_1}: {empty_1}, {col_2}: {empty_2}"
        )
    return df
