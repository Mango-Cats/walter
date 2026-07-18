"""
walter:
    LLM-assisted dataset construction for LASA drugs.

Usage:
    walter                      full pipeline
    walter all                  same thing, named explicitly
    walter propose              generate P with the LLM proposer
    walter noise                sample U from the registry
    walter assemble             merge P and U into D, transcribe
    walter phoc                 add phonetic-similarity features
    walter engineer             add the orthographic META_FEATURES

Every stage reads and writes the paths in config, so stages run in any
combination. Pass --input / --output to point one at a different file.
"""

import argparse
import time
from pathlib import Path

from rich.console import Console

from config import (
    D_CSV,
    D_ENGI_CSV,
    D_PHO_CSV,
    DATA_SOURCE,
    FROM_FILE,
    LLM_OUTPUT_JSON,
    POSITIVE_PREVALENCE,
    SEED,
    TIER_2_SAMPLE_SIZE,
    U_CSV,
)
from src import stages

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
    if FROM_FILE:
        raise SystemExit(
            "FROM_FILE = True, so P is read from a CSV and there is nothing to "
            "propose. Set FROM_FILE = False in config to use the LLM proposer."
        )
    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    with Spinner("Proposing pairs (LLM)"):
        out = stages.propose(R_clean, output_path=args.output)
    print(f"\nP -> {out}")


def cmd_noise(args: argparse.Namespace) -> None:
    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    P_load = stages.load_positives()
    with Spinner("Sampling unlabeled pairs (U)"):
        U = stages.noise(P_load, R_clean, output_path=args.output)
    print(f"\nU: {len(U):,} pairs")


def cmd_assemble(args: argparse.Namespace) -> None:
    P_load = stages.load_positives()
    U = stages.load_noise(args.input)
    with Spinner("Assembling D"):
        D = stages.assemble(P_load, U, output_csv=args.output)
    print(f"\nD -> {args.output}")
    print(stages.summarize(D))


def cmd_phoc(args: argparse.Namespace) -> None:
    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(args.input, args.output)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"D_pho -> {args.output}")


def cmd_engineer(args: argparse.Namespace) -> None:
    with Spinner("Engineering meta-features"):
        meta = stages.engineer(args.input, args.output)
    print(f"\nMeta-features ({len(meta)}): {', '.join(meta)}")
    print(f"D_engi -> {args.output}")


def cmd_all(args: argparse.Namespace) -> None:
    _banner()
    print(f"Output: D -> {D_CSV}")

    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    print(f"\nCleaned registry: {len(R_clean):,} drug names")

    if not FROM_FILE:
        with Spinner("Proposing pairs (LLM)"):
            stages.propose(R_clean)
    P_load = stages.load_positives()

    with Spinner("Sampling unlabeled pairs (U)"):
        U = stages.noise(P_load, R_clean)

    with Spinner("Assembling and saving D"):
        D = stages.assemble(P_load, U)
    print(f"\n{stages.summarize(D)}")

    # D.csv is already on disk, so a phoc failure aborts the run but leaves
    # the base dataset intact.
    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(D_CSV, D_PHO_CSV)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")

    with Spinner("Engineering meta-features"):
        meta = stages.engineer(D_PHO_CSV, D_ENGI_CSV)
    print(f"\nMeta-features ({len(meta)}): {', '.join(meta)}")
    print(f"D_engi -> {D_ENGI_CSV}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walter",
        description="LLM-assisted dataset construction for LASA drugs.",
    )
    sub = parser.add_subparsers(dest="stage")

    sub.add_parser("all", help="Run every stage (default)").set_defaults(
        func=cmd_all
    )

    p_propose = sub.add_parser("propose", help="Generate P with the LLM proposer")
    p_propose.add_argument("--output", type=Path, default=LLM_OUTPUT_JSON)
    p_propose.set_defaults(func=cmd_propose)

    p_noise = sub.add_parser("noise", help="Sample the unlabeled set U")
    p_noise.add_argument("--output", type=Path, default=U_CSV)
    p_noise.set_defaults(func=cmd_noise)

    p_assemble = sub.add_parser("assemble", help="Merge P and U into D")
    p_assemble.add_argument("--input", type=Path, default=U_CSV, help="U CSV")
    p_assemble.add_argument("--output", type=Path, default=D_CSV)
    p_assemble.set_defaults(func=cmd_assemble)

    p_phoc = sub.add_parser("phoc", help="Add phonetic-similarity features")
    p_phoc.add_argument("--input", type=Path, default=D_CSV)
    p_phoc.add_argument("--output", type=Path, default=D_PHO_CSV)
    p_phoc.set_defaults(func=cmd_phoc)

    p_engineer = sub.add_parser("engineer", help="Add the orthographic META_FEATURES")
    p_engineer.add_argument("--input", type=Path, default=D_PHO_CSV)
    p_engineer.add_argument("--output", type=Path, default=D_ENGI_CSV)
    p_engineer.set_defaults(func=cmd_engineer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # No subcommand keeps the original `python walter.py` behavior.
    if getattr(args, "func", None) is None:
        args.func = cmd_all
    try:
        args.func(args)
    except FileNotFoundError as exc:
        # A missing upstream artifact is normal when running one stage, and
        # the message already names the command to run. A traceback would
        # bury it.
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
