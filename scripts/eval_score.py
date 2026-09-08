"""
scripts/eval_score.py

Step 4 of the prompt-evaluation harness: scores a proposer's output against
the ground truth carried in it, and reports precision, recall, and F1.

For each target, this compares what the proposer chose against the true
confirmed partners carried through from the questionnaire, and counts:
  - correct picks -- chosen, and actually a true partner
  - over-proposals -- chosen, but NOT a true partner (a false alarm)
  - misses -- a true partner the proposer didn't pick

Two ways of averaging across targets are reported:
  - Micro-averaged: pool the counts across every target first, then divide.
    This weights every proposed-or-true pair equally, regardless of which
    target it came from.
  - Macro-averaged: score each target on its own, then average those
    per-target scores. This weights every target equally, regardless of how
    many true partners it has.

Precision and recall are also combined into F1 (their harmonic mean) for
both averaging methods.

The report also lists, per target, the true answer alongside the raw
over-proposals and misses -- these are worth reviewing by hand, since the
gold standard isn't exhaustive: an "over-proposal" here might actually be a
real, undocumented confusable pair rather than a genuine mistake.

Expects the proposer's output to be a JSON list of entries, each carrying
at minimum a target name (c_c), the candidates it was shown (p_c), the true
answer carried through from the questionnaire (answer), and what it chose
(chosen). This script never re-derives the answer -- it has to already be
in the file.

Production-proxy scoring (--tiers-pool): the real proposer (src/proposer/
inference.py) only ever shows the model ~20 candidates that are ALL close
fuzzy matches -- no random or easy filler, and the seed's own known partner
isn't even in the list (it's added separately, unconditionally). This
eval's candidate lists are deliberately harder to build a gold answer key
from (mixing in random/easy distractors so recall is measurable at all),
which means precision here is inflated relative to production, since a
chunk of the candidate list is trivially rejectable filler that production
never actually shows the model. Passing --tiers-pool <pool.json used to
build this run's questionnaire> re-derives which tier every candidate came
from (recomputed fresh from c_c/p_c/answer -- this does NOT depend on
regenerating questionnaire.json, so it works on any run's output, including
ones generated before this script's tier support existed) and adds a
"production_proxy" section to the report: the same precision/recall/F1,
but computed as if the random/easy tiers were never in the candidate list
at all -- a closer estimate of how this prompt would actually do in
production. --n-fuzzy must match what was passed to eval_candidates.py for
that run (default 5, the default used everywhere in this harness so far).

Usage:
    uv run python scripts/eval_score.py \\
        --input results/eval/proposer/proposer_output.json \\
        --output results/eval/report/report.json

    # also report a production-proxy score restricted to the fuzzy tier
    uv run python scripts/eval_score.py \\
        --input results/eval/proposer/proposer_output.json \\
        --tiers-pool results/eval/pool/pool.json \\
        --output results/eval/report/report.json
"""

import argparse
import json
from pathlib import Path

from rapidfuzz import fuzz, process


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def classify_tiers(
    target: str, p_c: list[str], answer: set[str], gold_names: set[str], n_fuzzy: int
) -> dict[str, str]:
    """
    Recompute which tier every non-answer candidate in p_c came from, fresh
    from (target, p_c, answer) and the gold-standard adjacency graph -- does
    NOT rely on matching a regenerated questionnaire.json (Python's
    per-process hash randomization means list(some_set) has no fixed order
    across runs, so an independently regenerated candidate list can't be
    trusted to match an old one item-for-item; this recomputes tier
    membership directly instead, which only depends on set membership and
    WRatio score comparisons -- both order-independent).

    Returns {candidate_name: "answer" | "fuzzy" | "random" | "easy"}.
    """
    tiers: dict[str, str] = {}
    non_gold = [c for c in p_c if c not in gold_names]
    for c in non_gold:
        tiers[c] = "easy"  # not part of the gold standard at all

    gold_candidates = [c for c in p_c if c in gold_names and c not in answer]
    if gold_candidates:
        remaining = sorted(gold_names - answer - {target})
        fuzzy_matches = process.extract(target, remaining, scorer=fuzz.WRatio, limit=n_fuzzy)
        fuzzy_set = {match for match, _score, _idx in fuzzy_matches}
        for c in gold_candidates:
            tiers[c] = "fuzzy" if c in fuzzy_set else "random"

    for c in answer:
        tiers[c] = "answer"
    return tiers


def _aggregate(per_target_counts: list[tuple[int, int, int]]) -> dict:
    """Turn a list of (correct, extra, missed) per-target counts into the
    micro/macro precision/recall/F1 block."""
    correct_total = sum(c for c, _, _ in per_target_counts)
    extra_total = sum(x for _, x, _ in per_target_counts)
    missed_total = sum(m for _, _, m in per_target_counts)

    precision_micro = correct_total / (correct_total + extra_total) if (correct_total + extra_total) else 0.0
    recall_micro = correct_total / (correct_total + missed_total) if (correct_total + missed_total) else 0.0

    precision_vals, recall_vals = [], []
    for c, x, m in per_target_counts:
        if c + x:
            precision_vals.append(c / (c + x))
        if c + m:
            recall_vals.append(c / (c + m))
    precision_macro = sum(precision_vals) / len(precision_vals) if precision_vals else 0.0
    recall_macro = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0

    return {
        "micro": {
            "precision": precision_micro,
            "recall": recall_micro,
            "f1": _f1(precision_micro, recall_micro),
        },
        "macro": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": _f1(precision_macro, recall_macro),
        },
    }


def score(entries: list[dict], gold_names: set[str] | None = None, n_fuzzy: int = 5) -> dict:
    """
    Score every entry and return micro/macro precision, recall, F1, and the
    list of per-target disagreements worth a manual look.

    If gold_names is given (the adjacency keys from the pool.json used to
    build this run's questionnaire), each disagreement's over-proposals are
    also tagged with which tier they came from (fuzzy/random/easy), and a
    second "production_proxy" score block is included -- the same metric,
    but computed as if random/easy-tier candidates were never shown at all,
    a closer estimate of real production behavior (see module docstring).
    """
    counts = []
    proxy_counts = []
    per_target = []
    disagreements = []

    for e in entries:
        target = e["c_c"]
        answer = set(e["answer"])
        chosen = set(e.get("chosen", []))

        correct = len(chosen & answer)
        extra = len(chosen - answer)
        missed = len(answer - chosen)
        counts.append((correct, extra, missed))

        precision = correct / (correct + extra) if (correct + extra) else None
        recall = correct / (correct + missed) if (correct + missed) else None
        per_target.append({"c_c": target, "precision": precision, "recall": recall})

        over_proposed = sorted(chosen - answer)
        missed_names = sorted(answer - chosen)

        tiers = None
        if gold_names is not None:
            tiers = classify_tiers(target, e.get("p_c", []), answer, gold_names, n_fuzzy)
            # production-proxy: as if the model had only ever been able to
            # choose from (answer | fuzzy) -- random/easy picks are dropped
            # entirely, not counted as either correct or a false alarm
            chosen_tiered = {c for c in chosen if tiers.get(c) in ("answer", "fuzzy")}
            p_correct = len(chosen_tiered & answer)
            p_extra = len(chosen_tiered - answer)
            p_missed = len(answer - chosen_tiered)
            proxy_counts.append((p_correct, p_extra, p_missed))

        if over_proposed or missed_names:
            entry = {
                "c_c": target,
                "answer": sorted(answer),         # the true partner(s), for reference
                "over_proposed": over_proposed,   # flagged, not a known pair
                "missed": missed_names,           # known pair, not flagged
            }
            if tiers is not None:
                entry["over_proposed_tiers"] = {c: tiers.get(c, "?") for c in over_proposed}
            disagreements.append(entry)

    report = {
        "n_items": len(entries),
        **_aggregate(counts),
        "per_target": per_target,
        "disagreements": disagreements,
    }
    if gold_names is not None:
        report["production_proxy"] = _aggregate(proxy_counts)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score a proposer's output against the ground truth carried in it."
    )
    p.add_argument("--input", type=Path, required=True, help="Proposer output JSON")
    p.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the full report (default: print summary only)",
    )
    p.add_argument(
        "--tiers-pool", type=Path, default=None,
        help="pool.json used to build this run's questionnaire -- when given, "
        "also reports a 'production_proxy' score restricted to fuzzy-tier "
        "candidates only (see module docstring). Tiers are recomputed fresh "
        "from each entry's own c_c/p_c/answer, so this works on any run's "
        "output regardless of when it was generated.",
    )
    p.add_argument(
        "--n-fuzzy", type=int, default=5,
        help="Must match --n-fuzzy passed to eval_candidates.py for this run "
        "(default: 5, used everywhere in this harness so far). Only matters "
        "with --tiers-pool.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    entries = json.loads(args.input.read_text(encoding="utf-8"))

    gold_names = None
    if args.tiers_pool is not None:
        pool = json.loads(args.tiers_pool.read_text(encoding="utf-8"))
        gold_names = set(pool["adjacency"].keys())

    report = score(entries, gold_names=gold_names, n_fuzzy=args.n_fuzzy)

    print(f"[score] {report['n_items']:,} targets scored")
    print(
        f"[score] micro  precision={report['micro']['precision']:.3f}  "
        f"recall={report['micro']['recall']:.3f}  f1={report['micro']['f1']:.3f}"
    )
    print(
        f"[score] macro  precision={report['macro']['precision']:.3f}  "
        f"recall={report['macro']['recall']:.3f}  f1={report['macro']['f1']:.3f}"
    )
    if "production_proxy" in report:
        pp = report["production_proxy"]
        print(
            f"[score] production_proxy micro  precision={pp['micro']['precision']:.3f}  "
            f"recall={pp['micro']['recall']:.3f}  f1={pp['micro']['f1']:.3f}"
        )
        print(
            f"[score] production_proxy macro  precision={pp['macro']['precision']:.3f}  "
            f"recall={pp['macro']['recall']:.3f}  f1={pp['macro']['f1']:.3f}"
        )
    print(
        f"[score] {len(report['disagreements']):,} targets have at least one "
        f"disagreement (review these by hand)"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[score] Full report -> {args.output}")


if __name__ == "__main__":
    main()
