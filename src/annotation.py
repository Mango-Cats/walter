"""
Batch export for the 3-rater annotation round described in
docs/annotation_codebook.md.

Sample a stratified batch from D and write one blinded sheet per rater, plus a
key file. Agreement is computed outside this repo, from the sheets the raters
return; the key is what joins them back together.

walter.py owns the CLI; nothing here parses arguments.
"""

import hashlib
import random
from pathlib import Path

import pandas as pd

from config import (
    ANNOTATION_DIR,
    ANNOTATION_SEED,
    COL_LABEL,
    COL_PAIR_ID,
    COL_STRATUM,
    COL_X1,
    COL_X2,
    N_CANDIDATES,
    N_PLACEBO,
    NEG_PER_POSITIVE,
    POSITIVE_LABEL,
    RATER_FIELDS,
    RATER_IDS,
    STRATUM_CANDIDATE,
    STRATUM_NEGATIVE,
    STRATUM_PLACEBO,
    TRANSCRIPTION_LANGS,
    UNLABELED_LABEL,
)
from src.pipeline.dataset import canonical_key
from src.pipeline.noise import is_similar_enough, normalize

KEY_FILENAME = "_key.csv"

_ID_PREFIX = "p"
_ID_HEX_LEN = 10

_PLACEBO_ATTEMPT_FACTOR = 200

_SHEET_LANG = "fil"

FORBIDDEN_FROM_BATCH = (COL_LABEL, "similarity", "tier", COL_STRATUM)


def pair_id(x_1: str, x_2: str) -> str:
    """Identifier for one pair, order-independent."""
    a, b = canonical_key(str(x_1), str(x_2))
    digest = hashlib.sha1(f"{a}\x00{b}".encode("utf-8")).hexdigest()
    return f"{_ID_PREFIX}_{digest[:_ID_HEX_LEN]}"


def add_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with COL_PAIR_ID prepended."""
    missing = [c for c in (COL_X1, COL_X2) if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build pair ids, missing columns: {missing}")

    df = df.copy()
    df[COL_PAIR_ID] = [pair_id(a, b) for a, b in zip(df[COL_X1], df[COL_X2])]
    return df[[COL_PAIR_ID] + [c for c in df.columns if c != COL_PAIR_ID]]


def _name_transcriptions(D: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Map each drug name to its transcription columns, taken from either side."""
    out: dict[str, dict[str, str]] = {}
    for lang, (c1, c2) in TRANSCRIPTION_LANGS.items():
        if c1 not in D.columns or c2 not in D.columns:
            continue
        for name, value in zip(D[COL_X1], D[c1]):
            out.setdefault(name, {})[c1] = value
        for name, value in zip(D[COL_X2], D[c2]):
            out.setdefault(name, {})[c2] = value
    return out


def _placebo_pairs(D: pd.DataFrame, count: int, rng: random.Random) -> pd.DataFrame:
    """
    Pairs of unrelated names that no attentive rater should call positive.

    Drawn from names already in D so their transcriptions come along, rather
    than fresh registry names that would need another G2P pass.
    """
    if count <= 0:
        return pd.DataFrame(columns=[COL_X1, COL_X2])

    left = sorted(set(D[COL_X1]))
    right = sorted(set(D[COL_X2]))
    existing = {pair_id(a, b) for a, b in zip(D[COL_X1], D[COL_X2])}

    rows: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    limit = count * _PLACEBO_ATTEMPT_FACTOR

    while len(rows) < count and attempts < limit:
        attempts += 1
        a, b = rng.choice(left), rng.choice(right)
        if a == b:
            continue
        pid = pair_id(a, b)
        if pid in existing or pid in seen:
            continue
        qualifies, _ = is_similar_enough(normalize(a), normalize(b))
        if qualifies:
            continue
        seen.add(pid)
        rows.append({COL_X1: a, COL_X2: b})

    if len(rows) < count:
        raise ValueError(
            f"Could only build {len(rows)} placebo pairs out of {count} after "
            f"{attempts} attempts. The vocabulary may be too small or too "
            f"self-similar; lower --n-placebo."
        )
    return pd.DataFrame(rows)


def build_batch(
    D: pd.DataFrame,
    n_candidates: int = N_CANDIDATES,
    neg_per_positive: float = NEG_PER_POSITIVE,
    n_placebo: int = N_PLACEBO,
    seed: int = ANNOTATION_SEED,
) -> pd.DataFrame:
    """
    Draw a stratified batch and return it shuffled, with pair_id and stratum.

    Args:
        D:                Assembled dataset, needs COL_X1, COL_X2, COL_LABEL.
        n_candidates:     LLM-flagged positives to include.
        neg_per_positive: Sampled negatives per candidate.
        n_placebo:        Unrelated control pairs (Section 8.1).
        seed:             Draw seed.

    Returns:
        DataFrame with COL_PAIR_ID, the pair, its transcriptions, COL_STRATUM
        and the source label. Row order is the order raters will see.
    """
    missing = [c for c in (COL_X1, COL_X2, COL_LABEL) if c not in D.columns]
    if missing:
        raise ValueError(f"D is missing columns: {missing}")

    rng = random.Random(seed)

    positives = D[D[COL_LABEL] == POSITIVE_LABEL]
    negatives = D[D[COL_LABEL] == UNLABELED_LABEL]

    n_cand = min(n_candidates, len(positives))
    if n_cand < n_candidates:
        print(
            f"[annotation] Only {len(positives):,} positives available, "
            f"requested {n_candidates:,}"
        )
    n_neg = min(round(n_cand * neg_per_positive), len(negatives))

    cand = (
        positives.sample(n=n_cand, random_state=seed) if n_cand else positives.head(0)
    )
    neg = negatives.sample(n=n_neg, random_state=seed) if n_neg else negatives.head(0)

    cand = cand.assign(**{COL_STRATUM: STRATUM_CANDIDATE})
    neg = neg.assign(**{COL_STRATUM: STRATUM_NEGATIVE})

    placebo = _placebo_pairs(D, n_placebo, rng)
    if not placebo.empty:
        trans = _name_transcriptions(D)
        for lang, (c1, c2) in TRANSCRIPTION_LANGS.items():
            if c1 in D.columns:
                placebo[c1] = [trans.get(n, {}).get(c1, "") for n in placebo[COL_X1]]
            if c2 in D.columns:
                placebo[c2] = [trans.get(n, {}).get(c2, "") for n in placebo[COL_X2]]
        placebo[COL_LABEL] = UNLABELED_LABEL
        placebo[COL_STRATUM] = STRATUM_PLACEBO

    batch = pd.concat([cand, neg, placebo], ignore_index=True)
    batch = add_pair_ids(batch)

    batch = batch.sample(frac=1, random_state=seed).reset_index(drop=True)

    dupes = batch[COL_PAIR_ID].duplicated().sum()
    if dupes:
        batch = batch.drop_duplicates(subset=[COL_PAIR_ID]).reset_index(drop=True)
        print(f"[annotation] Dropped {dupes} pair(s) that appeared in two strata")

    print(
        f"[annotation] Batch: {len(batch):,} pairs "
        f"({n_cand:,} candidate, {n_neg:,} negative, {len(placebo):,} placebo)"
    )
    return batch


def sheet_columns(show_similarity: bool = False) -> list[str]:
    """The exact column list a rater sheet may contain."""
    t1, t2 = TRANSCRIPTION_LANGS[_SHEET_LANG]
    cols = [COL_PAIR_ID, COL_X1, t1, COL_X2, t2]
    if show_similarity:
        cols.append("similarity")
    return cols + list(RATER_FIELDS)


def round_dir(round_name: str, base: Path = ANNOTATION_DIR) -> Path:
    return Path(base) / round_name


def export(
    batch: pd.DataFrame,
    round_name: str,
    base: Path = ANNOTATION_DIR,
    raters: tuple[str, ...] = RATER_IDS,
    show_similarity: bool = False,
) -> list[Path]:
    """
    Write one blinded CSV per rater plus the key file, and return the sheet paths.

    The key retains stratum and source label and is never handed to a rater.
    """
    out_dir = round_dir(round_name, base)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = sheet_columns(show_similarity)
    available = [c for c in cols if c in batch.columns or c in RATER_FIELDS]
    missing = [c for c in cols if c not in available]
    if missing:
        raise ValueError(f"Batch cannot fill sheet columns: {missing}")

    sheet = pd.DataFrame({c: ("" if c in RATER_FIELDS else batch[c]) for c in cols})

    carried = [c for c in cols if c not in RATER_FIELDS]
    leaked = [c for c in carried if c in FORBIDDEN_FROM_BATCH]
    if not show_similarity:
        leaked += [c for c in carried if c == "similarity"]
    if leaked:
        raise AssertionError(f"Blinding violated, sheet would carry: {leaked}")
    for field in RATER_FIELDS:
        if sheet[field].astype(str).str.strip().ne("").any():
            raise AssertionError(
                f"Blinding violated, rater field '{field}' is prefilled"
            )

    paths = []
    for rater in raters:
        path = out_dir / f"{rater}.csv"
        sheet.to_csv(path, index=False)
        paths.append(path)
        print(f"[annotation] sheet -> {path}")

    key_path = out_dir / KEY_FILENAME
    batch.to_csv(key_path, index=False)
    print(f"[annotation] key   -> {key_path}  (do not give this to raters)")
    return paths
