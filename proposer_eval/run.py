"""
Orchestration: pair-scoring on the gold pairs + candidate-retrieval on the registry.

    uv run python -m proposer_eval                      # both modes
    uv run python -m proposer_eval --labels any         # A1 OR A2 counts as positive
    uv run python -m proposer_eval --pho-configs DIR    # score your own phoc configs
    uv run python -m proposer_eval --skip-retrieval     # 200 pairs only, seconds

See README.md in this folder for how to read the output.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


from .gold import GOLD_CSV, LABEL_MODES, labels, load_gold
from .metrics import auc, average_precision, best_f1, prf
from .pho import DEFAULT_PHO_CONF, PACKAGE_DIR
from .scorers import Scorer, build_scorers, fil_indicator_scores

BASELINE = "base:proposer_gate"
_WR: np.ndarray | None = None  # WRatio matrix, for the parity check's tie-break


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="proposer_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=GOLD_CSV)
    ap.add_argument("--registry", type=Path, default=Path("data/ph_GenBrn.csv"),
                    help="registry to retrieve candidates from (one-column CSV, header ignored, "
                         "first column used). Default: the registry the gold pairs were generated from")
    ap.add_argument("--pho-configs", type=Path, default=DEFAULT_PHO_CONF,
                    help="directory of phoc .toml configs; every one is scored on "
                         "raw and nativized text (default: soundex binary + soft)")
    ap.add_argument("--labels", choices=LABEL_MODES, default="consensus",
                    help="how A1/A2 collapse into one label (default: consensus = A1==A2 only)")
    ap.add_argument("--gate-threshold", type=float, default=0.75,
                    help="score >= this lets a candidate through the gate (default 0.75, "
                         "the proposer's own Soundex threshold)")
    ap.add_argument("--min-edit-distance", type=int, default=2,
                    help="candidates within this edit distance of the anchor are dropped, "
                         "for every strategy (the proposer hardcodes 2)")
    ap.add_argument("-k", type=int, default=20, help="candidate cap per anchor (proposer: 20)")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-fil", action="store_true", help="skip phoc's fil_* indicators")
    ap.add_argument("--limit-anchors", type=int, default=None, help="dev: cap anchors in retrieval")
    ap.add_argument("--out", type=Path, default=PACKAGE_DIR / "out")
    return ap.parse_args()


# --------------------------------------------------------------------------- pairs
def pair_mode(gold: pd.DataFrame, y_all: np.ndarray, args) -> pd.DataFrame:
    x1, x2 = gold["x_1"].tolist(), gold["x_2"].tolist()
    # No edit-distance rule here: pair mode asks "does the score separate known
    # positives from negatives", and the gold set was generated WITHOUT that rule
    # (26 of 83 positives are within edit distance 2), so applying it would cap
    # every scorer's gate recall at the same artificial ceiling.
    scorers, _, _ = build_scorers(x1, x2, args.pho_conf_dir, args.gate_threshold, min_edit_distance=-1)
    n = len(gold)
    idx = np.arange(n)
    cols: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
        s.name: (s.score[idx, idx], s.gate[idx, idx], s.family) for s in scorers
    }
    if not args.skip_fil:
        for name, sc in fil_indicator_scores(x1, x2, args.pho_conf_dir).items():
            cols[name] = (sc, sc >= args.gate_threshold, "fil")

    keep = ~np.isnan(y_all)
    y = y_all[keep]
    rows = []
    for name, (score, gate, fam) in cols.items():
        p, r, f = prf(y, gate[keep].astype(int))
        bf, bt = best_f1(y, score[keep])
        rows.append({"scorer": name, "AUC": auc(y, score[keep]), "AP": average_precision(y, score[keep]),
                     "bestF1": bf, "@thr": bt, "gate_P": p, "gate_R": r, "gate_F1": f})
    # context, not a candidate gate: how the LLM's own verdict scored vs the annotators
    p, r, f = prf(y, gold["llm"].to_numpy()[keep])
    rows.append({"scorer": "ref:llm_label", "AUC": np.nan, "AP": np.nan, "bestF1": np.nan, "@thr": np.nan,
                 "gate_P": p, "gate_R": r, "gate_F1": f})
    table = pd.DataFrame(rows)

    per_pair = gold.assign(y=y_all)
    for name, (score, gate, _) in cols.items():
        per_pair[name] = score
        per_pair[name + "|gate"] = gate.astype(int)
    args.out.mkdir(parents=True, exist_ok=True)
    per_pair.to_csv(args.out / "pair_scores.csv", index=False)
    table.to_csv(args.out / "pair_metrics.csv", index=False)
    print(f"\n=== PAIR SCORING  (n={int(keep.sum())} labeled pairs, {int(y.sum())} positive, labels={args.labels})")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


# ---------------------------------------------------------------------- retrieval
def _parity_check(anchors, registry, scorers, k, min_ed) -> None:
    """The baseline here is a vectorised reproduction of extract_candidates();
    confirm it against the real function on every anchor so the comparison is fair."""
    from src.proposer.inference import _soundex_code, extract_candidates

    if min_ed != 2:
        print("[parity] skipped: extract_candidates() hardcodes edit distance > 2")
        return
    base = next(s for s in scorers if s.name == BASELINE)
    smap = {d: _soundex_code(d) for d in registry}
    exact = 0
    for ai, a in enumerate(anchors):
        real = set(extract_candidates(a, "\0", registry, drug_soundex_map=smap, limit=k))
        row = np.where(base.gate[ai])[0]
        # extract_candidates breaks composite ties by WRatio, then input order
        order = sorted(row, key=lambda j: (-base.score[ai, j], -_WR[ai, j], j))
        exact += real == {registry[j] for j in order[:k]}
    print(f"[parity] vectorised baseline == extract_candidates() top-{k} on {exact}/{len(anchors)} anchors")


def retrieval_mode(gold: pd.DataFrame, y_all: np.ndarray, args) -> pd.DataFrame | None:
    # Registries are one-column CSVs, header ignored, first column used
    # (config/paths.py's own comment on R[...]) -- so read positionally rather
    # than by REGISTRY_COL name, since a hand-supplied file's header can differ
    # (e.g. data/ph_GenBrn.csv's "Drug Names" vs the pipeline's "drug_name").
    reg = pd.read_csv(args.registry, dtype=str, keep_default_na=False).iloc[:, 0]
    registry = list(dict.fromkeys(n.strip() for n in reg if n.strip()))
    rset = set(registry)
    pos = gold[(y_all == 1)].copy()
    pos["in_registry"] = pos["x_2"].isin(rset)
    print(f"\n=== CANDIDATE RETRIEVAL  ({len(registry)} registry names)")
    print(f"gold positives: {len(pos)}; target x_2 present in registry: {int(pos.in_registry.sum())}")
    pos = pos[pos.in_registry]
    if pos.empty:
        print("no positives whose x_2 is in the registry; nothing to retrieve.")
        return None
    anchors = list(dict.fromkeys(pos["x_1"]))
    if args.limit_anchors:
        anchors = anchors[: args.limit_anchors]
        pos = pos[pos["x_1"].isin(anchors)]
    a_ix = {a: i for i, a in enumerate(anchors)}
    r_ix = {c: i for i, c in enumerate(registry)}

    t0 = time.time()
    print(f"scoring {len(anchors)} anchors x {len(registry)} names = {len(anchors) * len(registry):,} pairs per scorer ...", flush=True)
    scorers, ed, wr = build_scorers(anchors, registry, args.pho_conf_dir, args.gate_threshold, args.min_edit_distance)
    print(f"[done in {time.time() - t0:.0f}s]")
    global _WR
    _WR = wr
    _parity_check(anchors, registry, scorers, args.k, args.min_edit_distance)

    ai = pos["x_1"].map(a_ix).to_numpy()
    ti = pos["x_2"].map(r_ix).to_numpy()
    reach = ed[ai, ti] > args.min_edit_distance
    print(f"{len(pos)} positives over {len(anchors)} anchors; {int((~reach).sum())} are within edit distance "
          f"<= {args.min_edit_distance} of their anchor, which the proposer's rule drops for EVERY strategy.\n"
          f"Recall below is over the {int(reach.sum())} reachable positives.")

    K = args.k
    rows, hits = [], {}
    for s in scorers:
        pool_hit = np.zeros(len(pos), bool)
        rec_rand = np.zeros(len(pos))
        rec_fuzz = np.zeros(len(pos), bool)
        for j, (a, t) in enumerate(zip(ai, ti)):
            g, row, wrow = s.gate[a], s.score[a], wr[a]
            if not g[t]:
                continue
            pool_hit[j] = True
            n_gt = int((g & (row > row[t])).sum())
            n_eq = int((g & (row == row[t])).sum())
            rec_rand[j] = min(1.0, max(0.0, (K - n_gt) / n_eq))
            rec_fuzz[j] = int((g & ((row > row[t]) | ((row == row[t]) & (wrow > wrow[t])))).sum()) < K
        hits[s.name] = pool_hit
        pool_sizes = s.gate.sum(axis=1)
        rows.append({"scorer": s.name, "pool_recall": pool_hit[reach].mean(),
                     f"recall@{K}(rand ties)": rec_rand[reach].mean(),
                     f"recall@{K}(fuzzy ties)": rec_fuzz[reach].mean(), "mean_pool": pool_sizes.mean(),
                     "median_pool": float(np.median(pool_sizes))})
    table = pd.DataFrame(rows)
    base_hit = hits[BASELINE]
    table["gained_vs_base"] = [int((hits[n] & ~base_hit & reach).sum()) for n in table.scorer]
    table["lost_vs_base"] = [int((~hits[n] & base_hit & reach).sum()) for n in table.scorer]

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "retrieval_metrics.csv", index=False)
    detail = pos[["x_1", "x_2"]].reset_index(drop=True).assign(
        edit_distance=ed[ai, ti], reachable=reach.astype(int), **{n + "|in_pool": h.astype(int) for n, h in hits.items()})
    detail.to_csv(args.out / "retrieval_pairs.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\npool_recall  = reachable gold positives that pass the gate at all (before the top-{K} cut)\n"
          f"recall@{K}    = ... that survive the top-{K} cut ('rand ties' = expected value if tied scores are ordered "
          f"randomly; 'fuzzy ties' = ties broken by WRatio)\n"
          "mean_pool    = candidates passing the gate per anchor, of the whole registry\n"
          "gained/lost  = positives this gate finds that the current proposer gate misses / vice versa (pool level)")
    return table


def main() -> None:
    args = parse_args()
    args.pho_conf_dir = args.pho_configs
    gold = load_gold(args.gold)
    y = labels(gold, args.labels)
    print(f"gold: {len(gold)} pairs from {args.gold}; labels={args.labels}: "
          f"{int(np.nansum(y))} positive / {int((y == 0).sum())} negative / {int(np.isnan(y).sum())} dropped")
    print(f"pho configs: {args.pho_configs}  ({', '.join(p.stem for p in sorted(args.pho_configs.glob('*.toml')))})")
    pair_mode(gold, y, args)
    if not args.skip_retrieval:
        retrieval_mode(gold, y, args)
    print(f"\nfull tables + per-pair detail written to {args.out}/")
