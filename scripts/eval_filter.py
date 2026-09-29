#!/usr/bin/env python3
"""Compare candidate-filter options for the proposer.

Two questions per option:
  1. Recall: for each known LASA pair (x_1, x_2) from an independent list (FDA/ISMP, data/P_us.csv),
     run the filter with x_1 as anchor against a registry and check whether x_2 lands in the
     top-K candidates the LLM would see. The known names are added to the PH registry so the
     filter has to find them among ~22k realistic distractors.
  2. Junk: for the anchors in the annotated sheet, run the filter against the PH registry and count
     how many annotated pairs survive: human-positive pairs kept (want high) and human-negative
     pairs kept (want low). Also reports a heuristic share of "obviously not" candidates in the
     new top-K lists (main names < 50% similar after stripping numbers/modifiers/single letters).

Caveat: Soundex here is plain jellyfish.soundex on the cleaned name, not the nativized (tbb-cli)
Soundex the pipeline uses.
"""

import argparse
import sys
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.preprocessing import clean_name  # noqa: E402

TOP_K = 20
MODIFIERS = {
    "plus", "forte", "iv", "im", "mr", "xr", "sr", "er", "cr", "la", "od", "odt", "ds",
    "tab", "tablet", "tablets", "cap", "capsule", "syrup", "susp", "suspension", "drops",
    "cream", "gel", "oint", "ointment", "inj", "injection", "kids", "junior", "adult",
    "mg", "ml", "mcg", "g",
}


def main_name(s: str) -> str:
    toks = [t for t in s.split() if not t.isdigit() and len(t) > 1 and t not in MODIFIERS]
    return " ".join(toks) if toks else s


def soundex(s: str) -> str:
    s = "".join(c for c in s if c.isalpha())
    try:
        return jellyfish.soundex(s) if s else ""
    except Exception:
        return ""


def soundex_sim(a_codes, b_codes):
    """Matrix of 1 - lev(code)/4 between two lists of Soundex codes."""
    d = process.cdist(a_codes, b_codes, scorer=Levenshtein.distance, workers=-1).astype(float)
    sim = np.clip(1.0 - d / 4.0, 0.0, 1.0)
    empty_a = np.array([c == "" for c in a_codes])[:, None]
    empty_b = np.array([c == "" for c in b_codes])[None, :]
    sim[empty_a | empty_b] = 0.0
    return sim


def build_matrices(anchors, registry):
    reg_main = [main_name(r) for r in registry]
    anc_main = [main_name(a) for a in anchors]
    m = {
        "wratio": process.cdist(anchors, registry, scorer=fuzz.WRatio, workers=-1).astype(float),
        "ratio": process.cdist(anchors, registry, scorer=fuzz.ratio, workers=-1).astype(float),
        "ratio_main": process.cdist(anc_main, reg_main, scorer=fuzz.ratio, workers=-1).astype(float),
        "edits": process.cdist(anchors, registry, scorer=Levenshtein.distance, workers=-1),
        "sx": soundex_sim([soundex(a) for a in anchors], [soundex(r) for r in registry]),
        "sx_main": soundex_sim([soundex(a) for a in anc_main], [soundex(r) for r in reg_main]),
    }
    lens_a = np.array([len(a) for a in anchors], float)[:, None]
    lens_r = np.array([len(r) for r in registry], float)[None, :]
    m["len_ratio"] = np.minimum(lens_a, lens_r) / np.maximum(lens_a, lens_r)
    # "long name starts with the short name's main part" (allegra -> allegra anti itch cream)
    pref = np.zeros((len(anchors), len(registry)), bool)
    for i, (a, am) in enumerate(zip(anchors, anc_main)):
        if len(am) >= 4:
            pref[i] = [r.startswith(am) for r in registry]
        pref[i] |= [len(rm) >= 4 and a.startswith(rm) for rm in reg_main]
    m["prefix"] = pref
    m["main_sim"] = m["ratio_main"]
    m["jw"] = process.cdist(anchors, registry, scorer=JaroWinkler.normalized_similarity, workers=-1) * 100
    return m


def options(m):
    """Each option returns (passes_gate, ranking_score). Higher ranking score = earlier in the list."""
    gate_w = (m["wratio"] > 60) | (m["sx"] >= 0.75)
    gate_r = (m["ratio"] > 60) | (m["sx"] >= 0.75)
    gate_main = (np.maximum(m["ratio"], m["ratio_main"]) > 60) | (np.maximum(m["sx"], m["sx_main"]) >= 0.75)
    return {
        "A current: WRatio>60|sx>=.75, rank .5 WRatio+.5 sx": (gate_w, 0.5 * m["wratio"] / 100 + 0.5 * m["sx"]),
        "D current + length ratio>=0.3": (gate_w & (m["len_ratio"] >= 0.3), 0.5 * m["wratio"] / 100 + 0.5 * m["sx"]),
        "B ratio>60|sx>=.75, rank .5 ratio+.5 sx": (gate_r, 0.5 * m["ratio"] / 100 + 0.5 * m["sx"]),
        "E main-name gate, rank .5 ratio+.5 sx": (gate_main, 0.5 * np.maximum(m["ratio"], m["ratio_main"]) / 100 + 0.5 * np.maximum(m["sx"], m["sx_main"])),
        "G ratio>60|sx>=.75, rank ratio only": (gate_r, m["ratio"] / 100),
        "H ratio>60|sx>=.75, rank Jaro-Winkler": (gate_r, m["jw"] / 100),
    }


def ranks_of(gate, rank_score, edits, self_mask, pairs_idx):
    ok = gate & (edits > 2) & ~self_mask
    comp = np.where(ok, rank_score, -np.inf)
    out = []
    for i, j in pairs_idx:
        out.append(np.inf if not np.isfinite(comp[i, j]) else int((comp[i] > comp[i, j]).sum()) + 1)
    return np.array(out), comp


def top_k(comp, k):
    out = []
    for i in range(comp.shape[0]):
        idx = np.argsort(-comp[i])[:k]
        out.append(set(idx[np.isfinite(comp[i, idx])].tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", default="data/P_us.csv")
    ap.add_argument("--registry", default="data/R_ph_clean.csv")
    ap.add_argument("--sheet", default="../walter-stats/.data/annotation_sheet.csv")
    ap.add_argument("--out", default="results/filter_eval.csv")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 30, 50], help="candidate caps to report recall at")
    args = ap.parse_args()

    reg = pd.read_csv(args.registry).iloc[:, 0].astype(str).map(clean_name)
    known = pd.read_csv(args.known)
    known = pd.DataFrame({"x1": known.x_1.astype(str).map(clean_name), "x2": known.x_2.astype(str).map(clean_name)})
    known = known[(known.x1 != "") & (known.x2 != "") & (known.x1 != known.x2)].drop_duplicates()
    n_le2 = int(sum(Levenshtein.distance(a, b) <= 2 for a, b in zip(known.x1, known.x2)))

    registry = sorted(set(reg) | set(known.x1) | set(known.x2))
    pos = {r: i for i, r in enumerate(registry)}

    # --- recall on known pairs
    anchors = sorted(set(known.x1))
    m = build_matrices(anchors, registry)
    selfm = np.array([[r == a for r in registry] for a in anchors])
    ai = {a: i for i, a in enumerate(anchors)}

    # --- junk on the annotated sheet (PH)
    sh = pd.read_csv(args.sheet)
    sh["x1"] = sh["Drug Name 1"].astype(str).map(clean_name)
    sh["x2"] = sh["Drug Name 2"].astype(str).map(clean_name)
    sh["mv"] = (sh[["A1", "A2", "A3"]].sum(axis=1) >= 2).astype(int)
    reg_ph = sorted(set(reg) | set(sh.x1) | set(sh.x2))
    pos_ph = {r: i for i, r in enumerate(reg_ph)}
    s_anchors = sorted(set(sh.x1))
    ms = build_matrices(s_anchors, reg_ph)
    selfs = np.array([[r == a for r in reg_ph] for a in s_anchors])
    si = {a: i for i, a in enumerate(s_anchors)}

    known = known[[Levenshtein.distance(a, b) > 2 for a, b in zip(known.x1, known.x2)]]
    kidx = [(ai[a], pos[b]) for a, b in zip(known.x1, known.x2)]
    elig = np.array([ms["edits"][si[a], pos_ph[b]] > 2 for a, b in zip(sh.x1, sh.x2)])
    hp, hn, fp = (sh.mv == 1) & elig, (sh.mv == 0) & elig, (sh.LLM == 1) & (sh.mv == 0) & elig

    rows = []
    for (name, (g, rs)), (_, (gs, rss)) in zip(options(m).items(), options(ms).items()):
        ranks, _ = ranks_of(g, rs, m["edits"], selfm, kidx)
        _, comp_s = ranks_of(gs, rss, ms["edits"], selfs, [])
        row = {"option": name, "known: pass gate": f"{np.isfinite(ranks).mean():.1%}"}
        for k in args.k:
            row[f"known: recall@{k}"] = f"{(ranks <= k).mean():.1%}"
        tks = top_k(comp_s, TOP_K)
        kept = np.array([pos_ph[b] in tks[si[a]] for a, b in zip(sh.x1, sh.x2)])
        row[f"PH obviously-not share @{TOP_K}"] = f"{np.mean([ms['main_sim'][i, j] < 50 for i, s in enumerate(tks) for j in s]):.1%}"
        row[f"sheet human+ kept @{TOP_K}"] = f"{kept[hp].sum()}/{hp.sum()}"
        row[f"sheet human- kept @{TOP_K}"] = f"{kept[hn].sum()}/{hn.sum()}"
        row[f"sheet LLM false+ kept @{TOP_K}"] = f"{kept[fp].sum()}/{fp.sum()}"
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"Known pairs with distinct names: {n_le2 + len(known)}; {n_le2} have <=2 edits and are excluded by the "
          f"edits>2 rule under every option, leaving {len(known)} for recall.\n"
          f"Distractor registry: {len(registry):,} names. Sheet columns use top {TOP_K}; sheet counts only pairs allowed by edits>2.\n")
    print(res.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
