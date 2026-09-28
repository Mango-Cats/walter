"""Load the annotated gold pairs (data/recentannotation_gold.csv)."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

GOLD_CSV = Path("data/recentannotation_gold.csv")

# label modes -> how the two annotators (A1, A2) are collapsed into one label.
#   consensus  A1 == A2 only; the ~5% they disagree on are dropped
#   any        positive if either annotator said 1
#   all        positive only if both said 1
LABEL_MODES = ("consensus", "any", "all")


def load_gold(path: Path = GOLD_CSV) -> pd.DataFrame:
    """Columns: x_1, x_2, llm, a1, a2 (ints). The sheet has a blank lead column
    and a blank first row, so find the header by content, not position."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr = next(i for i, r in enumerate(rows) if "Drug Name 1" in r)
    cols = rows[hdr]
    ix = {c: cols.index(c) for c in ("Drug Name 1", "Drug Name 2", "LLM", "A1", "A2")}
    out = []
    for r in rows[hdr + 1 :]:
        if len(r) <= max(ix.values()) or not r[ix["Drug Name 1"]].strip():
            continue
        out.append(
            {
                "x_1": r[ix["Drug Name 1"]].strip(),
                "x_2": r[ix["Drug Name 2"]].strip(),
                "llm": int(r[ix["LLM"]]),
                "a1": int(r[ix["A1"]]),
                "a2": int(r[ix["A2"]]),
            }
        )
    return pd.DataFrame(out)


def labels(gold: pd.DataFrame, mode: str) -> np.ndarray:
    """1.0 / 0.0 per row; NaN where the mode leaves the row unlabeled."""
    a1, a2 = gold["a1"].to_numpy(), gold["a2"].to_numpy()
    if mode == "consensus":
        return np.where(a1 == a2, a1, np.nan).astype(float)
    if mode == "any":
        return ((a1 + a2) > 0).astype(float)
    if mode == "all":
        return ((a1 + a2) == 2).astype(float)
    raise ValueError(f"unknown label mode {mode!r}; pick one of {LABEL_MODES}")
