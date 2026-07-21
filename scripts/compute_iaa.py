"""
Inter-annotator agreement for the LASA binary annotation task.

Input : one CSV holding the returned annotations, one column per annotator
        (e.g. "1", "2", "3"), one row per pair. The pair itself no longer has
        to be in the sheet -- rows are assumed to be aligned across columns.
Output: raw agreement, positive prevalence, Fleiss' kappa, Gwet's AC1.

Works for any number of annotators >= 2. Computes the coefficients directly
from their definitions so the arithmetic is transparent; no reliability
library is required.

    python scripts/compute_iaa.py annotations.csv
    python scripts/compute_iaa.py annotations.csv --cols 1 2 3
"""

import argparse
import sys

import numpy as np
import pandas as pd


def load_labels(path, cols=None):
    """Read the sheet and return an N x R array of 0/1 labels plus the column names."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing columns {missing}. Found {list(df.columns)}"
            )
        rater_cols = list(cols)
    else:
        rater_cols = []
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().any() and v.dropna().isin([0, 1]).all():
                rater_cols.append(c)
        skipped = [c for c in df.columns if c not in rater_cols]
        if skipped:
            print(f"Ignoring non-annotation column(s): {skipped}")

    if len(rater_cols) < 2:
        raise ValueError(
            f"{path}: need at least 2 annotator columns, found {rater_cols}. "
            "Pass them explicitly with --cols."
        )

    labels = df[rater_cols].apply(pd.to_numeric, errors="coerce")

    n_before = len(labels)
    labels = labels.dropna()
    dropped = n_before - len(labels)
    if dropped:
        print(f"WARNING: dropped {dropped} row(s) not labelled by all annotators.")

    bad = ~labels.isin([0, 1]).all(axis=1)
    if bad.any():
        raise ValueError(f"{bad.sum()} row(s) have labels other than 0/1.")
    if labels.empty:
        raise ValueError(f"{path}: no rows labelled by all annotators.")

    return labels.astype(int).to_numpy(), rater_cols


def iaa(labels):
    """labels: N x R array of 0/1. Returns a dict of agreement statistics."""
    N, R = labels.shape
    n_pos = labels.sum(axis=1)
    n_neg = R - n_pos

    P_i = (n_pos**2 + n_neg**2 - R) / (R * (R - 1))
    P_bar = P_i.mean()

    p_pos = n_pos.sum() / (N * R)
    p_neg = 1 - p_pos

    Pe_fleiss = p_pos**2 + p_neg**2
    kappa = (
        (P_bar - Pe_fleiss) / (1 - Pe_fleiss) if (1 - Pe_fleiss) > 0 else float("nan")
    )

    Pe_gwet = 2 * p_pos * p_neg
    ac1 = (P_bar - Pe_gwet) / (1 - Pe_gwet) if (1 - Pe_gwet) > 0 else float("nan")

    unanimous = np.mean((n_pos == R) | (n_pos == 0))  # share of fully-agreed pairs

    return {
        "n_pairs": N,
        "n_annotators": R,
        "positive_prevalence": p_pos,
        "mean_pairwise_agreement": P_bar,
        "unanimous_agreement_rate": unanimous,
        "fleiss_kappa": kappa,
        "gwet_ac1": ac1,
    }


def landis_koch(k):
    if k != k:
        return "undefined (no variability in labels)"
    bands = [
        (0.0, "poor"),
        (0.20, "slight"),
        (0.40, "fair"),
        (0.60, "moderate"),
        (0.80, "substantial"),
        (1.01, "almost perfect"),
    ]
    for hi, name in bands:
        if k < hi:
            return name
    return "almost perfect"


def main():
    ap = argparse.ArgumentParser(
        description="Inter-annotator agreement for the LASA task."
    )
    ap.add_argument("csv", help="CSV with one 0/1 column per annotator")
    ap.add_argument(
        "--cols",
        nargs="+",
        metavar="COL",
        help="annotator column names (default: auto-detect 0/1 columns)",
    )
    args = ap.parse_args()

    try:
        labels, rater_cols = load_labels(args.csv, args.cols)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    r = iaa(labels)

    print(f"\nAnnotator columns              : {', '.join(rater_cols)}")
    print(f"Pairs scored by all annotators : {r['n_pairs']}")
    print(f"Annotators                     : {r['n_annotators']}")
    print(f"Positive prevalence            : {r['positive_prevalence']:.3f}")
    print(f"Mean pairwise agreement        : {r['mean_pairwise_agreement']:.3f}")
    print(f"Unanimous-agreement rate       : {r['unanimous_agreement_rate']:.3f}")
    print(
        f"Fleiss' kappa                  : {r['fleiss_kappa']:.3f}  ({landis_koch(r['fleiss_kappa'])})"
    )
    print(
        f"Gwet's AC1                     : {r['gwet_ac1']:.3f}  ({landis_koch(r['gwet_ac1'])})"
    )

    gap = abs(r["fleiss_kappa"] - r["gwet_ac1"])
    if gap == gap and gap > 0.15:
        print(
            f"\nNote: kappa and AC1 differ by {gap:.2f}. With prevalence at "
            f"{r['positive_prevalence']:.2f}, this gap is the prevalence effect on kappa; "
            f"report both and let AC1 carry the reliability claim."
        )


if __name__ == "__main__":
    main()
