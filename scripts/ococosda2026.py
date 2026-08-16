"""
ococosda2026: LASA dataset construction, seeded from data/P_us.csv, with an
equal-count, phoc-vetted hard-negative set.

Runs the same walter pipeline (preprocess -> [P, U] -> assemble -> phoc) with
two differences from `walter all`:

  1. P is data/P_us.csv verbatim (config.P[DataSource.US]) - never
     LLM-proposed, so `walter propose` is skipped entirely.

  2. U is exactly len(P) HARD negatives, not the usual ~1:450 ratio
     (config.POSITIVE_PREVALENCE). "Hard" is deliberately a band, not a
     one-sided cutoff:

       - Too LOW similarity is an easy negative - useless for training a
         classifier that has to separate genuine confusables from noise.
       - Too HIGH similarity risks being an undocumented real LASA pair
         mislabeled as a negative, which contaminates the dataset rather
         than hardening it.

     So candidates are drawn from src/pipeline/noise.py's existing
     WRatio/Soundex/Metaphone anchor mining (oversampled well past len(P),
     and already guaranteed disjoint from every known P pair), then
     re-scored with phoc's `levenshtein` and `aline_eng_kondrak` (English
     G2P) columns - both normalized similarity in [0, 1], verified by hand
     (identical strings score 1.0, unrelated ones score ~0). Only
     candidates whose average of the two falls inside
     [MIN_HARDNESS, MAX_HARDNESS] are eligible; the len(P) hardest
     (highest-scoring) survivors of that band become U.

     If the band can't supply len(P) candidates, this is a hard failure
     (SystemExit), not a silent widening of the band - forcing the count by
     admitting near-duplicate-of-a-positive pairs would skew the model
     rather than harden it.

Usage:
    python ococosda2026.py
    python ococosda2026.py --output results/ococosda2026 --oversample 5 \\
        --min-hardness 0.4 --max-hardness 0.8
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from config import (
    COL_T1,
    COL_T2,
    COL_T_ENG_1,
    COL_T_ENG_2,
    COL_X1,
    COL_X2,
    D_FILENAME,
    D_PHO_FILENAME,
    DataSource,
    P,
    P_INPUT_COLS,
    PHOC_CONFIG_DIR,
    RESULTS_DIR,
    SEED,
)
from src import stages
from src.adapters.g2p.transcribe import transcribe_all
from src.adapters.phoc import run_phoc
from src.pipeline.noise import make_noise
from walter import Spinner

# Similarity band a candidate negative must fall in to count as "hard".
# See the module docstring for why both ends matter.
MIN_HARDNESS: float = 0.4
MAX_HARDNESS: float = 0.8

# How large a candidate pool to mine before hardness-filtering, as a
# multiple of len(P). The band keeps only a fraction of what's mined, so
# this needs enough headroom to still yield len(P) survivors.
OVERSAMPLE: int = 5


def _score_hardness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a `hardness` column: mean of phoc's `levenshtein` and
    `aline_eng_kondrak` (English) similarity scores, each in [0, 1].

    df must already carry COL_T_ENG_1 / COL_T_ENG_2 (see transcribe_all).
    """
    with tempfile.TemporaryDirectory(prefix="ococosda_phoc_") as tmp:
        tmp = Path(tmp)
        staged = df[[COL_X1, COL_X2]].copy()
        staged[COL_T1] = df[COL_T_ENG_1].fillna("")
        staged[COL_T2] = df[COL_T_ENG_2].fillna("")
        staged_csv = tmp / "candidates.csv"
        scored_csv = tmp / "candidates_scored.csv"
        staged.to_csv(staged_csv, index=False)
        run_phoc(staged_csv, scored_csv, PHOC_CONFIG_DIR)
        scored = pd.read_csv(scored_csv)

    out = df.copy()
    out["hardness"] = (scored["levenshtein"] + scored["aline_eng_kondrak"]) / 2
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ococosda2026",
        description="LASA dataset from data/P_us.csv with equal-count, "
        "phoc-vetted hard negatives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "ococosda2026",
        help="Directory to write D.csv / D_pho.csv into "
        "(default: results/ococosda2026)",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=OVERSAMPLE,
        help=f"Candidate pool size as a multiple of len(P) (default: {OVERSAMPLE})",
    )
    parser.add_argument(
        "--min-hardness",
        type=float,
        default=MIN_HARDNESS,
        help=f"Lower bound of the hard-negative similarity band (default: {MIN_HARDNESS})",
    )
    parser.add_argument(
        "--max-hardness",
        type=float,
        default=MAX_HARDNESS,
        help="Upper bound of the hard-negative similarity band - keep below "
        f"1.0 to avoid picking undocumented real LASA pairs (default: {MAX_HARDNESS})",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.min_hardness >= args.max_hardness:
        raise SystemExit("error: --min-hardness must be < --max-hardness")

    p_csv = P[DataSource.US]
    P_df = pd.read_csv(p_csv)[list(P_INPUT_COLS)].dropna().reset_index(drop=True)
    n_pos = len(P_df)
    print(f"P (positives) <- {p_csv}: {n_pos:,} pairs")

    with Spinner(f"Preprocessing {DataSource.US.name} registry"):
        R_clean = stages.preprocess(source=DataSource.US)
    print(f"Cleaned registry: {len(R_clean):,} drug names")

    pool_target = args.oversample * n_pos
    prevalence = n_pos / (n_pos + pool_target)
    with Spinner(f"Mining {pool_target:,} candidate negatives ({args.oversample}x oversample)"):
        candidates = make_noise(
            pairs_df=P_df,
            registry_df=R_clean,
            positive_prevalence=prevalence,
            seed=args.seed,
        )
    print(f"Candidate pool: {len(candidates):,} plausible-confusable pairs")

    with Spinner("Transcribing candidates (English G2P)"):
        candidates = transcribe_all(
            candidates, langs=["eng"], tag="ococosda2026", verbose=False
        )

    with Spinner("Scoring candidate hardness (phoc: levenshtein + aline_eng_kondrak)"):
        candidates = _score_hardness(candidates)

    band = candidates[
        (candidates["hardness"] >= args.min_hardness)
        & (candidates["hardness"] <= args.max_hardness)
    ].sort_values("hardness", ascending=False)
    print(
        f"\nHardness band [{args.min_hardness}, {args.max_hardness}]: "
        f"{len(band):,} / {len(candidates):,} candidates qualify"
    )

    if len(band) < n_pos:
        raise SystemExit(
            f"error: only {len(band):,} candidates fall in the hardness band, "
            f"need {n_pos:,} to match |P|. Widening the band to force the count "
            "would risk admitting near-duplicate-of-a-positive pairs as "
            "negatives, which skews the model rather than hardening it - so "
            "this stops instead. Try a wider --min-hardness/--max-hardness "
            "band or a larger --oversample."
        )

    selected = band.head(n_pos)
    hard_negatives = selected[[COL_X1, COL_X2]].reset_index(drop=True)
    print(
        f"Selected {len(hard_negatives):,} hardest negatives "
        f"(hardness {selected['hardness'].min():.3f}-{selected['hardness'].max():.3f})"
    )

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    d_csv = out_dir / D_FILENAME
    d_pho_csv = out_dir / D_PHO_FILENAME

    with Spinner("Assembling and saving D (P + hard negatives)"):
        D = stages.assemble(P_df, hard_negatives, None, output_csv=d_csv)
    print(f"\n{stages.summarize(D)}")

    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(d_csv, d_pho_csv)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"D -> {d_csv}")
    print(f"D_pho -> {d_pho_csv}")


if __name__ == "__main__":
    main()
