"""Small numpy-only metrics (no sklearn), tie-aware."""

import numpy as np
import pandas as pd


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC via rank-sum; ties get the average rank."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_curve(y: np.ndarray, s: np.ndarray):
    """(precision, recall, threshold) evaluated at every distinct score."""
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    last = np.r_[ss[1:] != ss[:-1], True]  # last index of each tie group
    tp, fp, thr = tp[last], fp[last], ss[last]
    n_pos = max(int((y == 1).sum()), 1)
    return tp / np.maximum(tp + fp, 1), tp / n_pos, thr


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    if (y == 1).sum() == 0:
        return float("nan")
    p, r, _ = _pr_curve(y, s)
    return float(np.sum(np.diff(np.r_[0.0, r]) * p))


def best_f1(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """(best F1 over all thresholds, the threshold achieving it)."""
    if (y == 1).sum() == 0:
        return float("nan"), float("nan")
    p, r, thr = _pr_curve(y, s)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    i = int(np.argmax(f))
    return float(f[i]), float(thr[i])


def prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
