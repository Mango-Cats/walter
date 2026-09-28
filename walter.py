"""Walter: Main command-line tool for building Look-Alike, Sound-Alike (LASA) drug datasets.

Look-Alike, Sound-Alike (LASA) drugs are medicines with similar names (such as Celebrex
and Celexa) that can be easily confused by healthcare workers. Walter creates datasets
of confusable and non-confusable drug pairs to train machine learning models to detect
these dangerous mix-ups.

Usage:
    walter               Run the full pipeline from start to finish.
    walter all           Same as running `walter`.
    walter propose       Use an AI (LLM) to find new confusable drug pairs (P).
    walter noise         Sample similar-sounding non-confusable drug pairs (U).
    walter assemble      Merge confusable and non-confusable pairs and generate
                         spoken pronunciations in English and Filipino (D).
    walter phoc          Calculate phonetic and spelling similarity scores (D_pho).
    walter featurize     Take an existing list of drug pairs and add pronunciations
                         and similarity scores without building a new dataset.

File and folder inputs/outputs:
    Most stages read and write files inside folders (directories). Each stage writes
    a file with a standard name that the next stage expects:
        - `propose` writes `lasa_run.json`
        - `noise` writes `U.csv`
        - `assemble` writes `D.csv`
        - `phoc` writes `D_pho.csv`
    By default, files are read from and written to the `results/` folder.

Exceptions:
    `walter propose` and `walter featurize` take a specific CSV file as `--input`
    instead of a folder.
"""

import argparse
import time
from pathlib import Path

from rich.console import Console

from config import (
    D_FILENAME,
    D_PHO_FILENAME,
    DATA_SOURCE,
    FROM_FILE,
    LLM_OUTPUT_FILENAME,
    P,
    POSITIVE_PREVALENCE,
    RESULTS_DIR,
    SEED,
    SOFT_LABELS,
    TIER_2_SAMPLE_SIZE,
    U_FILENAME,
)
from src import stages
from src.artifacts import in_file, out_file, seed_file

_console = Console()


class Spinner:
    """Per-stage terminal loading indicator backed by rich's Console.status."""

    def __init__(self, label: str) -> None:
        """Initialize the spinner with a progress label.

        Args:
            label: Description of the task in progress.

        """
        self.label = label
        self._status = _console.status(f"[bold cyan]{label}...", spinner="dots")
        self._start = 0.0

    def __enter__(self) -> "Spinner":
        """Start the progress spinner."""
        self._start = time.monotonic()
        self._status.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Stop the progress spinner and display elapsed time."""
        self._status.__exit__(exc_type, exc, tb)
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            _console.print(f"[bold green]OK[/] {self.label} done in {elapsed:.1f}s")
        else:
            _console.print(f"[bold red]X[/] {self.label} FAILED in {elapsed:.1f}s")


def _banner(soft_labels: bool) -> None:
    """Print active configuration parameters to the console.

    Args:
        soft_labels: Whether soft labels (-1 / 0 / 1) are active.

    """
    print(f"Data source    : {DATA_SOURCE.name}")
    print(f"Pos. prevalence: {POSITIVE_PREVALENCE:.6f}")
    print(f"Tier 2 sample  : {TIER_2_SAMPLE_SIZE:,}")
    print(f"Seed           : {SEED}")
    print(f"P source       : {'CSV' if FROM_FILE else 'LLM proposer'}")
    print(f"Labels         : {'soft (1 / -1 / 0)' if soft_labels else 'hard (1 / 0)'}")
    print()


def _soft_labels(args: argparse.Namespace) -> bool:
    """Determine whether soft labels are enabled based on CLI arguments and configuration.

    Args:
        args: Parsed command-line arguments.

    Returns:
        True if soft labeling is enabled, False otherwise.

    Raises:
        SystemExit: If contradictory flags are passed (e.g. --no-soft-labels with --rejected).

    """
    rejected = getattr(args, "rejected", None)
    if args.soft_labels is None:
        return SOFT_LABELS or rejected is not None
    if not args.soft_labels and rejected is not None:
        raise SystemExit(
            "error: --rejected supplies the label=-1 rows, which only exist "
            "under soft labels. Drop --no-soft-labels or drop --rejected."
        )
    return args.soft_labels


def _add_label_flags(p: argparse.ArgumentParser) -> None:
    """Add soft-label and rejection-file command-line options to a parser.

    Args:
        p: Argument parser or subparser instance.

    """
    p.add_argument(
        "--soft-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Label rejected pairs -1 (LLM-rejected candidates and --rejected) "
        "and keep 0 for the combinatorially sampled pairs of U "
        f"(default: {SOFT_LABELS}, from config/proposer.py)",
    )
    p.add_argument(
        "--rejected",
        type=Path,
        default=None,
        help="CSV of predefined rejected pairs (x_1, x_2) to label -1; implies "
        "--soft-labels. Defaults to N[DATA_SOURCE], used when it exists",
    )


def cmd_propose(args: argparse.Namespace) -> None:
    """Run the AI proposer stage to augment seed pairs into new confusable pairs.

    Args:
        args: Parsed command-line arguments.

    """
    seed_csv = seed_file(args.input, "predefined LASA pairs CSV")
    out = out_file(args.output, LLM_OUTPUT_FILENAME)
    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    with Spinner("Augmenting predefined pairs (LLM)"):
        stages.propose(R_clean, seed_csv=seed_csv, output_path=out)
    print(f"\nP -> {out}")


def cmd_noise(args: argparse.Namespace) -> None:
    """Run the noise stage to sample non-confusable negative pairs (U).

    Args:
        args: Parsed command-line arguments.

    """
    P_load = stages.load_positives(args.input)
    out = out_file(args.output, U_FILENAME)
    with Spinner("Preprocessing registry"):
        R_clean = stages.preprocess()
    with Spinner("Sampling unlabeled pairs (U)"):
        U = stages.noise(P_load, R_clean, output_path=out)
    print(f"\nU: {len(U):,} pairs")
    print(f"U -> {out}")


def cmd_assemble(args: argparse.Namespace) -> None:
    """Run the assembly stage to combine pairs and add IPA pronunciations (D).

    Args:
        args: Parsed command-line arguments.

    """
    soft = _soft_labels(args)
    U = stages.load_noise(in_file(args.input, U_FILENAME, "walter noise"))
    P_load = stages.load_positives(args.input)
    N_load = (
        stages.load_rejections(args.input, rejected_csv=args.rejected) if soft else None
    )
    out = out_file(args.output, D_FILENAME)
    with Spinner("Assembling D"):
        D = stages.assemble(P_load, U, N_load, output_csv=out)
    print(f"\nD -> {out}")
    print(stages.summarize(D))


def cmd_phoc(args: argparse.Namespace) -> None:
    """Run the phoc and feature-engineering stage on the assembled dataset (D_pho).

    Args:
        args: Parsed command-line arguments.

    """
    src = in_file(args.input, D_FILENAME, "walter assemble")
    out = out_file(args.output, D_PHO_FILENAME)
    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(src, out)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"D_pho -> {out}")


def cmd_featurize(args: argparse.Namespace) -> None:
    """Run G2P pronunciation and similarity scoring on an existing pair CSV.

    Args:
        args: Parsed command-line arguments.

    """
    src = seed_file(args.input, "pair CSV to featurize")
    out = args.output or src.parent / f"{src.stem}_pho.csv"
    if out.resolve() == src.resolve():
        raise SystemExit("error: --output must differ from --input")
    with Spinner("Featurizing (g2p -> phoc)"):
        feats = stages.featurize(src, out)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"\n{src} -> {out}")


def cmd_all(args: argparse.Namespace) -> None:
    """Run all pipeline stages in sequence from start to finish.

    Args:
        args: Parsed command-line arguments.

    """
    soft = _soft_labels(args)
    _banner(soft)

    d_csv = out_file(args.output, D_FILENAME)
    d_pho_csv = out_file(args.output, D_PHO_FILENAME)
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

    N_load = (
        stages.load_rejections(args.output, rejected_csv=args.rejected)
        if soft
        else None
    )

    with Spinner("Assembling and saving D"):
        D = stages.assemble(P_load, U, N_load, output_csv=d_csv)
    print(f"\n{stages.summarize(D)}")

    with Spinner("Adding phonetic features (phoc)"):
        feats = stages.phoc(d_csv, d_pho_csv)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"D_pho -> {d_pho_csv}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for Walter.

    Returns:
        Configured ArgumentParser instance.

    """
    parser = argparse.ArgumentParser(
        prog="walter",
        description="LLM-assisted dataset construction for LASA drugs.",
    )
    sub = parser.add_subparsers(dest="stage")

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
    _add_label_flags(p_all)
    p_all.set_defaults(func=cmd_all)

    p_propose = sub.add_parser("propose", help="Augment predefined LASA pairs into P")
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
    _add_label_flags(p_assemble)
    p_assemble.set_defaults(func=cmd_assemble)

    p_phoc = sub.add_parser("phoc", help="Add phonetic-similarity features")
    _dirs(p_phoc, D_FILENAME, D_PHO_FILENAME, "walter assemble")
    p_phoc.set_defaults(func=cmd_phoc)

    p_featurize = sub.add_parser(
        "featurize",
        help="Run g2p and phoc over an existing pair CSV",
    )
    p_featurize.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV of pairs to featurize; needs x_1 and x_2. label is "
        "preserved if present; every other column is dropped and rebuilt",
    )
    p_featurize.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <input>_pho.csv beside the input)",
    )
    p_featurize.set_defaults(func=cmd_featurize)

    return parser


def main() -> None:
    """Execute the Walter command-line interface."""
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        args = parser.parse_args(["all"])
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
