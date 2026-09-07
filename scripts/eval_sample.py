"""
scripts/eval_sample.py

Step 1 of the prompt-evaluation harness: builds the pool that later steps
sample from -- the drug names in a gold-standard confusable-pairs file
(P_us.csv / P_ph.csv), a lookup of which names are confirmed confusable with
which, a sampled set of target drugs to actually test, and (if a registry is
given) a pool of unrelated names to use as easy negatives.

What it does:
  1. Reads the gold-standard CSV of confirmed pairs (columns x_1, x_2) and
     builds a lookup from every drug name to the set of names it's confirmed
     confusable with. This is symmetric -- if A is paired with B, B is
     recorded as paired with A too, even though the CSV only lists it once.
     A name can have more than one confirmed partner (verified on
     P_us.csv: 73 of 446 names do).
  2. Samples a set of target drugs from that lookup, under a fixed random
     seed so the sample is reproducible. Every sampled target is guaranteed
     to have at least one confirmed partner, since it was drawn from the
     lookup in the first place.
  3. Optionally samples a pool of "easy negative" names from a full drug
     registry (R_us.csv / R_ph.csv) -- names that have nothing to do with
     the gold-standard pairs at all, for use as low-difficulty distractors
     later.

Writes all of this to a single JSON file, consumed by eval_candidates.py.

Targeted testing (--targets): instead of a random sample, pass a fixed list
of specific names to test -- e.g. a small curated set of items a prior run
got wrong, so you can cheaply check whether a prompt change fixes known
weak spots without paying for a fresh full-size random run every time. The
answer key for these names still comes straight out of --gold like any
other target -- this is just picking *which* names to test, not supplying
answers, so it carries no leakage risk on its own (that only happens if a
name's answer also ends up written into a system prompt file -- see
prompts/held_out_names.txt). Mutually exclusive with --sample-size. Since
this is meant to be a cheap, repeatable check, a promising result here
should still be confirmed with a full random run before you trust it --
a curated set only tells you about the specific weak spots you picked.

Usage:
    uv run python scripts/eval_sample.py \\
        --gold data/P_us.csv --seed 0 --sample-size 100 \\
        --registry data/R_us.csv --n-easy 200 \\
        --output results/eval/pool.json

    # targeted regression check on a small curated set instead
    uv run python scripts/eval_sample.py \\
        --gold data/P_us.csv --targets results/eval/hard_cases.txt \\
        --registry data/R_us.csv --n-easy 200 \\
        --output results/eval/pool_targeted.json
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src.pipeline.preprocessing import clean_name


def load_gold_pairs(path: Path) -> list[tuple[str, str]]:
    """Read (x_1, x_2) from a gold-standard CSV, cleaned and deduplicated."""
    df = pd.read_csv(path)
    missing = [c for c in ("x_1", "x_2") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    pairs = []
    for x1, x2 in zip(df["x_1"], df["x_2"]):
        a, b = clean_name(str(x1)), clean_name(str(x2))
        if a and b and a != b:
            pairs.append((a, b))
    return pairs


def build_adjacency(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """
    Build a lookup from every drug name to the set of names it's confirmed
    confusable with, in both directions -- if A is paired with B in the
    input, B is recorded as paired with A too, even though the CSV only
    lists the pair once. Every name that appears as x_1 or x_2 anywhere
    becomes a key.
    """
    adjacency: dict[str, set[str]] = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def load_excluded_names(path: Path) -> set[str]:
    """
    Names to hold out of the eval pool entirely (one per line, blank lines
    and '#' comments ignored) -- cleaned the same way gold names are, so
    they match regardless of formatting differences (slashes, casing, etc).

    This exists for names used as few-shot exemplars in a system prompt:
    if an exemplar's target and its correct answer are also sitting in the
    pool this script samples from, the prompt can hand the model the exact
    answer to a question the eval later asks it -- inflating that item's
    score without testing anything. Excluding those names here removes them
    as both possible targets and possible confirmed partners, so they can
    never end up seeding the answer to their own question.
    """
    if not path.exists():
        raise FileNotFoundError(f"Exclude file not found: {path}")
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(clean_name(line))
    return names


def filter_adjacency(
    adjacency: dict[str, set[str]], excluded: set[str]
) -> dict[str, set[str]]:
    """Drop excluded names as both keys and as partners of any remaining name."""
    return {
        name: {p for p in partners if p not in excluded}
        for name, partners in adjacency.items()
        if name not in excluded
    }


def load_target_list(path: Path) -> list[str]:
    """
    A fixed list of specific names to test instead of a random sample (one
    per line, blank lines and '#' comments ignored -- use comments to group
    names into categories for your own reading, they carry no meaning to
    this script). Order is preserved and duplicates are dropped.
    """
    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")
    seen: set[str] = set()
    ordered: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = clean_name(line)
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def sample_targets(names: list[str], seed: int, sample_size: int | None) -> list[str]:
    """
    Randomly sample `sample_size` names to use as targets, under a fixed
    seed so the sample is reproducible. Returns every name if sample_size
    is None or larger than the pool.
    """
    rng = random.Random(seed)
    pool = list(names)
    rng.shuffle(pool)
    if sample_size is None or sample_size >= len(pool):
        return pool
    return pool[:sample_size]


def load_registry_pool(path: Path) -> set[str]:
    """All (cleaned) names in a registry CSV, first column, header optional."""
    df = pd.read_csv(path, usecols=[0], header=None, names=["name"])
    if str(df.iloc[0, 0]).strip().lower() == "name":
        df = df.iloc[1:]
    return {clean_name(str(n)) for n in df["name"] if clean_name(str(n))}


def sample_easy_negatives(
    registry_pool: set[str], gold_names: set[str], seed: int, count: int
) -> list[str]:
    """
    Randomly sample `count` names from the registry that are NOT part of
    the gold-standard set at all, for use as easy (unrelated) negatives.
    """
    if count <= 0:
        return []
    rng = random.Random(seed + 1)  # offset so this doesn't repeat the target sample
    # sorted(), not list(): Python's per-process hash randomization means
    # list(some_set) has no fixed order across runs, so seeding rng.shuffle
    # on an unsorted set-derived list silently gave a different draw every
    # time despite the same --seed. Sorting first fixes the starting order
    # so the seed actually determines the output, run after run.
    outside = sorted(registry_pool - gold_names)
    rng.shuffle(outside)
    return outside[:count]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the eval sampling pool from a gold-standard "
        "confusable-pairs CSV."
    )
    p.add_argument("--gold", type=Path, required=True, help="Path to P_*.csv")
    p.add_argument(
        "--exclude", type=Path, default=None,
        help="Optional text file of names to hold out of the pool entirely "
        "(one per line) -- use this for any name that appears in a few-shot "
        "prompt's examples, so the eval never tests the model on an answer "
        "it was already shown.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sample-size", "-n", type=int, default=None,
        help="How many targets to sample. Omit to use every name in the "
        "gold-standard set. Mutually exclusive with --targets.",
    )
    p.add_argument(
        "--targets", type=Path, default=None,
        help="Optional text file of specific names to test instead of a "
        "random sample (one per line, '#' comments ignored) -- for cheap, "
        "repeatable checks against a curated set of known weak spots. "
        "Mutually exclusive with --sample-size.",
    )
    p.add_argument(
        "--registry", type=Path, default=None,
        help="Optional R_*.csv to draw easy negatives from.",
    )
    p.add_argument(
        "--n-easy", type=int, default=0,
        help="How many easy negatives to sample. Requires --registry.",
    )
    p.add_argument("--output", type=Path, default=Path("results/eval/pool.json"))
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.targets is not None and args.sample_size is not None:
        raise SystemExit("--targets and --sample-size are mutually exclusive -- pick one.")

    pairs = load_gold_pairs(args.gold)
    adjacency = build_adjacency(pairs)
    print(f"[sample] Gold standard: {len(pairs):,} pairs -> {len(adjacency):,} unique names")

    excluded: set[str] = set()
    if args.exclude is not None:
        excluded = load_excluded_names(args.exclude)
        adjacency = filter_adjacency(adjacency, excluded)
        adjacency = {name: partners for name, partners in adjacency.items() if partners}
        print(
            f"[sample] Excluded {len(excluded):,} name(s) from {args.exclude} -> "
            f"{len(adjacency):,} unique names remain"
        )

    gold_names = set(adjacency.keys())

    multi = sum(1 for name in adjacency if len(adjacency[name]) > 1)
    print(f"[sample] {multi:,}/{len(gold_names):,} names have more than one confirmed partner")

    if args.targets is not None:
        requested = load_target_list(args.targets)
        missing = [n for n in requested if n not in gold_names]
        if missing:
            note = (
                " (held out via --exclude)" if excluded and any(n in excluded for n in missing)
                else ""
            )
            raise SystemExit(
                f"[sample] {len(missing)} requested target(s) not found in the gold "
                f"standard{note}: {missing}\n"
                "Check spelling/normalization against data/P_*.csv, or drop them from "
                f"{args.targets}."
            )
        targets = requested
        print(f"[sample] Using {len(targets):,} explicitly requested target(s) from {args.targets}")
    else:
        targets = sample_targets(sorted(gold_names), args.seed, args.sample_size)
        print(f"[sample] Sampled {len(targets):,} targets (seed={args.seed})")

    easy_negatives: list[str] = []
    if args.registry is not None and args.n_easy > 0:
        registry_pool = load_registry_pool(args.registry)
        easy_negatives = sample_easy_negatives(
            registry_pool, gold_names | excluded, args.seed, args.n_easy
        )
        print(f"[sample] Sampled {len(easy_negatives):,} easy negatives from {args.registry}")

    out = {
        "seed": args.seed,
        "gold_source": str(args.gold),
        "targets": targets,
        "adjacency": {name: sorted(adjacency[name]) for name in gold_names},
        "easy_negatives": easy_negatives,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[sample] Saved -> {args.output}")


if __name__ == "__main__":
    main()
