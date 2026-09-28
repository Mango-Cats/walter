"""
Build score matrices for every candidate-gating strategy under test.

`build_scorers(anchors, cands)` returns one Scorer per strategy, each holding an
(n_anchors x n_cands) score matrix and a boolean gate saying "this candidate
would reach the LLM". The same function serves both eval modes: retrieval calls
it with (gold anchors x whole registry), pair-scoring with (gold x_1 x gold x_2)
and reads the diagonal.

Strategies
    base:*        what the proposer does today (src/proposer/inference.py)
    pho:<cfg>     pho, one per .toml in --pho-configs, on the raw spelling
    nat:<cfg>     same configs on the TagaBaybay-nativized spelling
    max:<cfg>     max(pho, nat)  -> passes if EITHER spelling matches
    mean:<cfg>    mean(pho, nat)
"""

from dataclasses import dataclass
from pathlib import Path

import jellyfish
import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from .pho import DEFAULT_PHO_CONF, nativize_all, phoc_features


@dataclass
class Scorer:
    name: str
    score: np.ndarray  # (n_anchors, n_cands) float32
    gate: np.ndarray  # (n_anchors, n_cands) bool
    family: str  # "base" | "pho" | "nat" | "combo"


def _soundex_codes(names: list[str]) -> list[str]:
    # Same helper the proposer uses, so the baseline is the proposer's own.
    from src.proposer.inference import _soundex_code

    return [_soundex_code(n) for n in names]


def build_scorers(
    anchors: list[str],
    cands: list[str],
    pho_conf: Path = DEFAULT_PHO_CONF,
    gate_threshold: float = 0.75,
    min_edit_distance: int = 2,
) -> tuple[list[Scorer], np.ndarray, np.ndarray]:
    """Returns (scorers, edit_distance_matrix, wratio_matrix)."""
    na, nc = len(anchors), len(cands)
    shape = (na, nc)

    ed = process.cdist(anchors, cands, scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    wr = process.cdist(anchors, cands, scorer=fuzz.WRatio, workers=-1, dtype=np.float32)
    # The proposer drops candidates within edit distance <= 2 of the anchor, and
    # the anchor itself. Apply the same rule to every strategy so only the
    # similarity signal differs between them.
    eligible = (ed > min_edit_distance) & (
        np.array(anchors, dtype=object)[:, None] != np.array(cands, dtype=object)[None, :]
    )

    scorers: list[Scorer] = []

    # ---- baseline: the proposer's current gate, reproduced -------------------
    ca, cc = _soundex_codes(anchors), _soundex_codes(cands)
    code_ed = process.cdist(ca, cc, scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    s_sim = np.maximum(0.0, 1.0 - code_ed / 4.0).astype(np.float32)
    empty = np.array([not a for a in ca])[:, None] | np.array([not c for c in cc])[None, :]
    s_sim[empty] = 0.0  # _soundex_similarity returns 0.0 for a missing code

    composite = 0.5 * (wr / 100.0) + 0.5 * s_sim
    prop_gate = ((wr > 60) | (s_sim >= 0.75)) & eligible
    # Score is left unmasked: retrieval only ranks within the gate anyway, and
    # masking gated-out pairs to -1 wrecked pair-mode AUC (0.605 masked vs 0.902 real).
    scorers.append(Scorer("base:proposer_gate", composite.astype(np.float32), prop_gate, "base"))
    scorers.append(Scorer("base:fuzzy_only", wr / 100.0, (wr > 60) & eligible, "base"))
    scorers.append(Scorer("base:jf_soundex_only", s_sim, (s_sim >= 0.75) & eligible, "base"))

    # ---- pho on raw spelling, and on nativized spelling ----------------------
    nat = nativize_all(list(dict.fromkeys(anchors + cands)))
    raw_f = phoc_features(anchors, cands, pho_conf)
    nat_f = phoc_features([nat[a] for a in anchors], [nat[c] for c in cands], pho_conf)
    for col in raw_f.columns:
        r = raw_f[col].to_numpy().reshape(shape)
        n = nat_f[col].to_numpy().reshape(shape)
        for fam, mat in (("pho", r), ("nat", n), ("max", np.maximum(r, n)), ("mean", (r + n) / 2)):
            scorers.append(
                Scorer(f"{fam}:{col}", mat, (mat >= gate_threshold) & eligible,
                       "combo" if fam in ("max", "mean") else fam)
            )
    return scorers, ed, wr


def fil_indicator_scores(x1: list[str], x2: list[str], pho_conf: Path) -> dict[str, np.ndarray]:
    """phoc's built-in nativization indicators, per gold pair (not a matrix).

    phoc runs tagabaybay in-process with G2P off (see the pho README), so a few
    ambiguous-vowel names can nativize slightly differently from the tbb-cli
    nativizer behind the nat:* scorers. Pair-wise only: it builds a fresh
    nativizer per word, far too slow over the whole registry.
    """
    from .pho import phoc_zipped

    feats = phoc_zipped(x1, x2, pho_conf, flags=("--include-fil-features",))
    fil = [c for c in feats.columns if c.startswith("fil_")]
    out = {f"fil:{c}": feats[c].to_numpy() for c in fil}
    out["fil:mean"] = feats[fil].to_numpy().mean(axis=1)
    return out
