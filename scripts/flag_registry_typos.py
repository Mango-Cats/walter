#!/usr/bin/env python3
"""List registry name pairs that are likely misspellings of each other, for manual review.

A pair is listed when the names are close (1-2 edits, or 3 for names of 12+ characters),
do not differ only in their numbers, and at least one record of each name has the same
active ingredients and dosage form. Tiers:
    A  same strength and same brand owner (distributor, else trader, else importer)
    B  same brand owner, strength differs or is written differently
    C  different brand owner, names 1 edit apart

Reviewers fill `is_typo` (y/n) and `canonical`; confirmed rows become the alias file
(config.ALIASES, columns `typo`, `canonical`).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.preprocessing import clean_name  # noqa: E402

CHUNK = 2000


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9;+ ]", " ", s)).strip()


def _ingredients(generic: str) -> frozenset[str]:
    parts = re.split(r";|\+|\band\b", re.sub(r"\d", "", _norm(generic)))
    return frozenset(p.strip() for p in parts if p.strip())


def _form(form: str) -> str:
    words = _norm(form).split()
    return words[0] if words else ""


def _strength(strength: str) -> tuple[str, ...]:
    s = re.sub(r"\(.*?\)", "", str(strength)).replace(",", "")
    return tuple(re.findall(r"\d+(?:\.\d+)?", s)[:3])


def _owner(row: pd.Series) -> str:
    for col in ("Distributor", "Trader", "Importer", "Manufacturer"):
        v = str(row.get(col, "")).strip()
        if v and v.lower() not in {"n/a", "na", "none", "-"}:
            v = re.sub(r"[^a-z ]", " ", v.lower())
            v = re.sub(r"\b(inc|corp|corporation|co|ltd|phils|philippines|the)\b", " ", v)
            return re.sub(r"\s+", " ", v).strip()
    return ""


def load_records(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype=str, encoding_errors="replace").fillna("")
    d["name"] = d["Brand Name"].map(clean_name)
    d = d[d["name"] != ""].copy()
    d["ingredients"] = d["Generic Name"].map(_ingredients)
    d["form"] = d["Dosage Form"].map(_form)
    d["strength_key"] = d["Dosage Strength"].map(_strength)
    d["owner"] = d.apply(_owner, axis=1)
    return d


def close_pairs(names: list[str]) -> list[tuple[int, int, int]]:
    lens = np.array([len(n) for n in names])
    out = []
    for start in range(0, len(names), CHUNK):
        block = names[start : start + CHUNK]
        dist = process.cdist(block, names, scorer=Levenshtein.distance, score_cutoff=3, workers=-1, dtype=np.int32)
        i, j = np.nonzero((dist >= 1) & (dist <= 3))
        i = i + start
        keep = i < j
        for a, b in zip(i[keep], j[keep]):
            e = int(dist[a - start, b])
            if e <= 2 or min(lens[a], lens[b]) >= 12:
                out.append((int(a), int(b), e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/raw/drug_products.csv")
    ap.add_argument("--registry", default="data/R_ph_clean.csv")
    ap.add_argument("--out", default="results/typo_candidates_ph.csv")
    args = ap.parse_args()

    recs = load_records(Path(args.records))
    registry = set(pd.read_csv(args.registry).iloc[:, 0].astype(str))
    recs = recs[recs["name"].isin(registry)]
    by_name = {n: g for n, g in recs.groupby("name")}
    names = sorted(by_name)
    counts = recs["name"].value_counts()

    rows = []
    for a, b, edits in close_pairs(names):
        na, nb = names[a], names[b]
        if re.sub(r"\d+", "", na).split() == re.sub(r"\d+", "", nb).split():
            continue
        best = None
        for _, ra in by_name[na].iterrows():
            for _, rb in by_name[nb].iterrows():
                if ra["ingredients"] != rb["ingredients"] or ra["form"] != rb["form"] or not ra["ingredients"]:
                    continue
                same_owner = bool(ra["owner"]) and fuzz.token_set_ratio(ra["owner"], rb["owner"]) >= 85
                same_strength = bool(ra["strength_key"]) and ra["strength_key"] == rb["strength_key"]
                tier = "A" if same_owner and same_strength else "B" if same_owner else "C" if edits == 1 else None
                if tier and (best is None or tier < best[0]):
                    best = (tier, ra, rb)
        if best is None:
            continue
        tier, ra, rb = best
        canonical = na if counts.get(na, 0) >= counts.get(nb, 0) else nb
        rows.append({
            "tier": tier, "name_a": na, "name_b": nb, "edits": edits,
            "reg_a": ra["Registration Number"], "reg_b": rb["Registration Number"],
            "generic": ra["Generic Name"].replace("\n", " ")[:120], "form": ra["Dosage Form"].replace("\n", " "),
            "strength_a": ra["Dosage Strength"].replace("\n", " ")[:60], "strength_b": rb["Dosage Strength"].replace("\n", " ")[:60],
            "owner_a": ra["owner"], "owner_b": rb["owner"],
            "suggested_canonical": canonical, "is_typo": "", "canonical": "",
        })

    out = pd.DataFrame(rows).sort_values(["tier", "edits", "name_a"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} candidate pairs -> {args.out}")
    print(out.groupby(["tier", "edits"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
