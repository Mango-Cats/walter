"""
walter:
    LLM-assisted dataset construction for LASA drugs.

Usage:
    walter                      full pipeline
    walter all                  same thing, named explicitly
    walter propose              augment predefined LASA pairs into P
    walter noise                sample U from the registry
    walter assemble             merge P and U into D, transcribe
    walter phoc                 add phonetic-similarity features
    walter engineer             add the orthographic META_FEATURES
    walter annotate             export blinded rater sheets for IAA

--input and --output are directories, not files. Each artifact has one
canonical filename (config/paths.py): a stage reads that name out of its
input directory and writes that name into its output directory, so pointing
two stages at the same directory is all it takes to chain them, and stages
run in any combination.

    walter phoc --input results --output results
        results/D.csv  ->  results/D_pho.csv

`walter propose` is the exception. Its --input is a CSV file of predefined
LASA pairs, which it augments; no other stage produces that file, so there is
no directory to take it from.

A stage whose input is missing names the command that produces it.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from rich.console import Console

from config import (
    ANNOTATION_DIR,
    ANNOTATION_SEED,
    D_ENGI_FILENAME,
    D_FILENAME,
    D_PHO_FILENAME,
    DATA_SOURCE,
    FROM_FILE,
    LLM_OUTPUT_FILENAME,
    N_CANDIDATES,
    N_PLACEBO,
    NEG_PER_POSITIVE,
    P,
    POSITIVE_PREVALENCE,
    RESULTS_DIR,
    SEED,
    TIER_2_SAMPLE_SIZE,
    U_FILENAME,
)
from src import stages
from src.artifacts import in_file, out_file, seed_file

_console = Console()


class Spinner:
    """Per-stage loading indicator, backed by rich's Console.status()."""

    def __init__(self, label: str):
        self.label = label
        self._status = _console.status(f"[bold cyan]{label}...", spinner="dots")
        self._start = 0.0

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        self._status.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._status.__exit__(exc_type, exc, tb)
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            _console.print(f"[bold green]OK[/] {self.label} done in {elapsed:.1f}s")
        else:
            _console.print(f"[bold red]X[/] {self.label} FAILED in {elapsed:.1f}s")


def _banner() -> None:
    print(f"Data source    : {DATA_SOURCE.name}")
    print(f"Pos. prevalence: {POSITIVE_PREVALENCE:.6f}")
    print(f"Tier 2 sample  : {TIER_2_SAMPLE_SIZE:,}")
    print(f"Seed           : {SEED}")
    print(f"P source       : {'CSV' if FROM_FILE else 'LLM proposer'}")
    print()


def cmd_propose(args: argparse.Namespace) -> None:
    seed_csv = seed_file(args.input, "predefined LASA pairs CSV")
    out = out_file(args.output, LLM_OUTPUT_FILENAME)
    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    with Spinner("Augmenting predefined pairs (LLM)"):
        stages.propose(R_clean, seed_csv=seed_csv, output_path=out)
    print(f"\nP -> {out}")


def cmd_noise(args: argparse.Namespace) -> None:
    P_load = stages.load_positives(args.input)
    out = out_file(args.output, U_FILENAME)
    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    with Spinner("Sampling unlabeled pairs (U)"):
        U = stages.noise(P_load, R_clean, output_path=out)
    print(f"\nU: {len(U):,} pairs")
    print(f"U -> {out}")


def cmd_assemble(args: argparse.Namespace) -> None:
    # U first: it is what --input advertises for this stage, so a user who
    # has not run `walter noise` should hear about that before P.
    U = stages.load_noise(in_file(args.input, U_FILENAME, "walter noise"))
    P_load = stages.load_positives(args.input)
    out = out_file(args.output, D_FILENAME)
    with Spinner("Assembling D"):
        D = stages.assemble(P_load, U, output_csv=out)
    print(f"\nD -> {out}")
    print(stages.summarize(D))


def cmd_phoc(args: argparse.Namespace) -> None:
    src = in_file(args.input, D_FILENAME, "walter assemble")
    out = out_file(args.output, D_PHO_FILENAME)
    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(src, out)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"D_pho -> {out}")


def cmd_engineer(args: argparse.Namespace) -> None:
    src = in_file(args.input, D_PHO_FILENAME, "walter phoc")
    out = out_file(args.output, D_ENGI_FILENAME)
    with Spinner("Engineering meta-features"):
        meta = stages.engineer(src, out)
    print(f"\nMeta-features ({len(meta)}): {', '.join(meta)}")
    print(f"D_engi -> {out}")


def cmd_all(args: argparse.Namespace) -> None:
    _banner()

    d_csv = out_file(args.output, D_FILENAME)
    d_pho_csv = out_file(args.output, D_PHO_FILENAME)
    d_engi_csv = out_file(args.output, D_ENGI_FILENAME)
    print(f"Output: D -> {d_csv}")

    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    print(f"\nCleaned registry: {len(R_clean):,} drug names")

    if not FROM_FILE:
        seed_csv = seed_file(args.input, "predefined LASA pairs CSV")
        with Spinner("Augmenting predefined pairs (LLM)"):
            stages.propose(
                R_clean,
                seed_csv=seed_csv,
                output_path=out_file(args.output, LLM_OUTPUT_FILENAME),
            )
    P_load = stages.load_positives(args.output)

    with Spinner("Sampling unlabeled pairs (U)"):
        U = stages.noise(P_load, R_clean, output_path=out_file(args.output, U_FILENAME))

    with Spinner("Assembling and saving D"):
        D = stages.assemble(P_load, U, output_csv=d_csv)
    print(f"\n{stages.summarize(D)}")

    # D.csv is already on disk, so a phoc failure aborts the run but leaves
    # the base dataset intact.
    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(d_csv, d_pho_csv)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")

    with Spinner("Engineering meta-features"):
        meta = stages.engineer(d_pho_csv, d_engi_csv)
    print(f"\nMeta-features ({len(meta)}): {', '.join(meta)}")
    print(f"D_engi -> {d_engi_csv}")


def cmd_annotate(args: argparse.Namespace) -> None:
    from src import annotation

    D = pd.read_csv(in_file(args.input, D_FILENAME, "walter assemble"))
    batch = annotation.build_batch(
        D,
        n_candidates=args.n_candidates,
        neg_per_positive=args.neg_per_positive,
        n_placebo=args.n_placebo,
        seed=args.seed,
    )
    annotation.export(
        batch,
        round_name=args.round,
        base=args.output,
        show_similarity=args.show_similarity,
    )
    print(
        "\nThe batch is stratified, not the true class prevalence "
        f"({POSITIVE_PREVALENCE:.6f}), so agreement computed on it does not "
        "estimate agreement over D."
    )


def _add_annotate_command(sub) -> None:
    p = sub.add_parser("annotate", help="Export blinded rater sheets for IAA")
    p.add_argument("--round", default="r1", help="Round name (default: r1)")
    p.add_argument(
        "--input", type=Path, default=RESULTS_DIR, help=f"Directory holding {D_FILENAME}"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ANNOTATION_DIR,
        help="Directory to write the round's sheets into",
    )
    p.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    p.add_argument("--neg-per-positive", type=float, default=NEG_PER_POSITIVE)
    p.add_argument("--n-placebo", type=int, default=N_PLACEBO)
    p.add_argument("--seed", type=int, default=ANNOTATION_SEED)
    p.add_argument(
        "--show-similarity",
        action="store_true",
        help="Include the fuzzy score (codebook Section 8.2 leaves this open; "
        "it may anchor raters toward the orthographic channel)",
    )
    p.set_defaults(func=cmd_annotate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walter",
        description="LLM-assisted dataset construction for LASA drugs.",
    )
    sub = parser.add_subparsers(dest="stage")

    # Every stage but `propose` reads a directory and writes a directory; the
    # filename inside is fixed per artifact, so only the directory varies.
    def _dirs(p, reads: str | None, writes: str, produced_by: str = "") -> None:
        if reads is not None:
            p.add_argument(
                "--input",
                type=Path,
                default=RESULTS_DIR,
                help=f"Directory holding {reads}"
                + (f" (from `{produced_by}`)" if produced_by else ""),
            )
        p.add_argument(
            "--output",
            type=Path,
            default=RESULTS_DIR,
            help=f"Directory to write {writes} into",
        )

    p_all = sub.add_parser("all", help="Run every stage (default)")
    p_all.add_argument(
        "--input",
        type=Path,
        default=P[DATA_SOURCE],
        help="CSV of predefined LASA pairs to seed the proposer with",
    )
    p_all.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write every artifact into",
    )
    p_all.set_defaults(func=cmd_all)

    p_propose = sub.add_parser(
        "propose", help="Augment predefined LASA pairs into P"
    )
    # The one stage taking a file: its input is user-supplied, not produced by
    # an earlier stage, so there is no canonical filename to look up.
    p_propose.add_argument(
        "--input",
        type=Path,
        default=P[DATA_SOURCE],
        help="CSV of predefined LASA pairs to augment",
    )
    p_propose.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory to write {LLM_OUTPUT_FILENAME} into",
    )
    p_propose.set_defaults(func=cmd_propose)

    p_noise = sub.add_parser("noise", help="Sample the unlabeled set U")
    _dirs(p_noise, LLM_OUTPUT_FILENAME, U_FILENAME, "walter propose")
    p_noise.set_defaults(func=cmd_noise)

    p_assemble = sub.add_parser("assemble", help="Merge P and U into D")
    _dirs(p_assemble, U_FILENAME, D_FILENAME, "walter noise")
    p_assemble.set_defaults(func=cmd_assemble)

    p_phoc = sub.add_parser("phoc", help="Add phonetic-similarity features")
    _dirs(p_phoc, D_FILENAME, D_PHO_FILENAME, "walter assemble")
    p_phoc.set_defaults(func=cmd_phoc)

    p_engineer = sub.add_parser("engineer", help="Add the orthographic META_FEATURES")
    _dirs(p_engineer, D_PHO_FILENAME, D_ENGI_FILENAME, "walter phoc")
    p_engineer.set_defaults(func=cmd_engineer)

    _add_annotate_command(sub)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # No subcommand keeps the original `python walter.py` behavior. Re-parsing
    # as `all` rather than just setting func picks up that parser's --input /
    # --output defaults, which a bare namespace does not carry.
    if getattr(args, "func", None) is None:
        args = parser.parse_args(["all"])
    try:
        args.func(args)
    except FileNotFoundError as exc:
        # A missing upstream artifact is normal when running one stage, and
        # the message already names the command to run. A traceback would
        # bury it.
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
