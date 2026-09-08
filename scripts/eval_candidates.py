"""
scripts/eval_candidates.py

Step 2 of the prompt-evaluation harness: turns the sampled targets and
answer key from eval_sample.py into a questionnaire -- one candidate list
per target, guaranteed to contain every one of that target's true
confusable partners, mixed in with distractors. This is what actually gets
shown to the proposer.

For each target, the candidate list is built from four pieces:
  - The target's true confirmed partner(s) -- always included, since
    without them there's nothing to measure recall against.
  - "Fuzzy" distractors: real drug names from the gold standard that are
    NOT confirmed partners of this target, but were the closest string
    matches to it (same scorer production's candidate mining uses --
    rapidfuzz's WRatio). These are the ones that actually stress-test
    judgment, since they're the most orthographically similar-looking
    names available. Restricted to the gold-standard set rather than the
    full drug registry on purpose: searching a smaller pool caps how close
    the top matches can possibly be, which lowers (does not eliminate) the
    odds of surfacing a real, undocumented confusable pair by accident --
    checked empirically, and it does still happen occasionally even within
    the gold standard, so this tier's disagreements still deserve a manual
    look later, same as any other.
  - "Random" distractors: other real gold-standard names picked with no
    similarity check at all. These test something different from the fuzzy
    tier -- whether the model over-triggers on any plausible-looking drug
    name, not just genuinely close ones.
  - "Easy" distractors: names that aren't part of the gold standard at
    all, pulled from the easy-negative pool eval_sample.py built (if any).

The fuzzy tier is picked first, and the random tier is drawn from what's
left over, so the two never overlap with each other.

The candidate list is shuffled before being written, so its order carries
no information about which entries are correct.

IMPORTANT: the true answer is written into each entry purely so
eval_score.py can grade it later -- it must never be shown to the LLM.
Whatever builds the actual prompt from this file should only read the
target name (c_c) and its candidate list (p_c), never the answer or tiers.

Each entry also records which tier every non-answer candidate came from
("tiers": {"fuzzy": [...], "random": [...], "easy": [...]}). This is also
grading-only metadata, same as "answer" -- it doesn't touch p_c or its
order at all, so re-running this script with the same --pool/seed/tier-size
args reproduces byte-identical candidate lists (and prompts) even after
this field is added, meaning any proposer_output*.json already generated
against an older questionnaire.json without this field is still valid and
doesn't need to be re-run. eval_score.py uses it (via --tiers) to report a
second, stricter score restricted to fuzzy-tier candidates only -- a closer
proxy for production, whose real candidate lists are all fuzzy matches and
have no random/easy filler at all.

Usage:
    uv run python scripts/eval_candidates.py \\
        --pool results/eval/pool/pool.json \\
        --n-fuzzy 5 --n-random 10 --n-easy 5 --seed 0 \\
        --output results/eval/questionnaire/questionnaire.json
"""

import argparse
import json
import random
from pathlib import Path

from rapidfuzz import fuzz, process


def build_questionnaire(
    pool: dict, n_fuzzy: int, n_random: int, n_easy: int, seed: int
) -> list[dict]:
    """Build one candidate-list entry per target, mixing in fuzzy, random, and easy distractors."""
    targets: list[str] = pool["targets"]
    adjacency: dict[str, list[str]] = pool["adjacency"]
    easy_negative_pool: list[str] = pool["easy_negatives"]
    all_names = set(adjacency.keys())

    rng = random.Random(seed)
    items = []
    for run, target in enumerate(targets, start=1):
        answer = set(adjacency.get(target, []))
        remaining = all_names - answer - {target}

        # fuzzy distractors: the closest string matches to the target among
        # the other gold-standard names, using the same scorer production's
        # candidate mining uses.
        # sorted(), not list(): Python's per-process hash randomization means
        # list(some_set) has no fixed order across runs -- feeding that into
        # process.extract or a seeded shuffle silently gave a different
        # result every run despite the same --seed. Sorting first fixes the
        # starting order so the seed actually determines the output.
        fuzzy_matches = process.extract(
            target, sorted(remaining), scorer=fuzz.WRatio, limit=n_fuzzy
        )
        fuzzy = [match for match, _score, _idx in fuzzy_matches]

        # random distractors: drawn from whatever's left after the fuzzy
        # picks, so the two tiers never overlap
        random_pool = sorted(remaining - set(fuzzy))
        rng.shuffle(random_pool)
        random_distractors = random_pool[:n_random]

        # easy distractors: names outside the gold standard entirely
        # (already a plain list from eval_sample.py, not a set -- no
        # reordering needed, just a copy so shuffling doesn't mutate it)
        easy_pool = list(easy_negative_pool)
        rng.shuffle(easy_pool)
        easy = easy_pool[:n_easy]

        candidates = sorted(answer | set(fuzzy) | set(random_distractors) | set(easy))
        rng.shuffle(candidates)

        items.append({
            "run": run,
            "c_c": target,             # the target drug shown to the proposer
            "p_c": candidates,         # its candidate list, shuffled
            "answer": sorted(answer),  # true partners -- for scoring only,
                                        # never shown to the proposer
            "tiers": {                 # which tier each non-answer candidate
                "fuzzy": sorted(fuzzy),           # came from -- for scoring only,
                "random": sorted(random_distractors),  # same as "answer": never
                "easy": sorted(easy),             # shown to the proposer, and
            },                          # doesn't change p_c/order at all
        })
    return items


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a proposer-input questionnaire from an eval sampling pool."
    )
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument(
        "--n-fuzzy", type=int, default=5,
        help="How many fuzzy-matched (closest string match) distractors to add per target",
    )
    p.add_argument(
        "--n-random", type=int, default=10,
        help="How many randomly-picked gold-standard distractors to add per target",
    )
    p.add_argument("--n-easy", type=int, default=5, help="How many easy distractors to add per target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results/eval/questionnaire/questionnaire.json"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    pool = json.loads(args.pool.read_text(encoding="utf-8"))

    if args.n_easy > 0 and not pool.get("easy_negatives"):
        print(
            "[candidates] WARNING: --n-easy > 0 but the sampling pool has no "
            "easy negatives (eval_sample.py was run without --registry). "
            "Easy distractors will be empty for every target."
        )

    items = build_questionnaire(pool, args.n_fuzzy, args.n_random, args.n_easy, args.seed)

    sizes = [len(it["p_c"]) for it in items]
    print(
        f"[candidates] {len(items):,} targets, candidate-list size ranges "
        f"{min(sizes)}-{max(sizes)} "
        f"(true partner(s) + up to {args.n_fuzzy} fuzzy + up to {args.n_random} "
        f"random + up to {args.n_easy} easy)"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"[candidates] Saved -> {args.output}")


if __name__ == "__main__":
    main()
