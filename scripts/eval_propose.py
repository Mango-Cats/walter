"""
scripts/eval_propose.py

Step 3 of the prompt-evaluation harness: runs a proposer over the
questionnaire from eval_candidates.py and records what it picks. This is
kept separate from src/proposer/inference.py on purpose -- it's for trying
out different prompts quickly, not for production dataset construction, and
it shouldn't risk touching that code while it's still being iterated on.

For each target in the questionnaire, this builds a prompt from the target
name and its candidate list only (never the hidden answer), sends it to the
DeepSeek API, and records what the model chose plus its reasoning. The
candidate list, the target, and the hidden answer are all carried through
unchanged into the output, so eval_score.py can grade it later without
needing the questionnaire file again.

Swapping prompts: pass --system-prompt with a path to a plain text file, and
that file's contents are used as the system prompt for the run. Leave it out
and prompts/baseline.txt is used -- a copy of the production system prompt
(SYSTEM_PROMPT in src/proposer/prompt.py) with one line changed: production
tells the model to return "exactly the requested number" of names, which
doesn't apply here since this script never asks for a fixed number (see
below). Everything else -- the persona, the judgment criteria, the
reasoning instruction -- is unchanged from production, so baseline.txt is a
fair floor to compare new prompt files against.

One thing any prompt file needs to preserve: the task still asks the model
to select from the given candidates only, since answers are matched back
against that list literally (case-insensitive, one per line) to filter out
anything the model didn't actually choose from the list -- see
src/adapters/llm/__init__.py's clean_output. The instruction to answer this
way is written into the (fixed, non-swappable) user prompt below, not the
system prompt, so this holds no matter which system prompt file you use --
a system prompt file only needs to carry the actual judgment criteria (what
counts as confusable, how to reason about it, any examples), not the output
format.

Also unlike the production proposer, this doesn't ask for a fixed number of
picks. Production always asks for an exact count because it's augmenting a
seed pair, but here we're scoring against a known answer set, and some
targets have more than one true partner -- forcing a fixed count would
silently cap recall or manufacture false positives that have nothing to do
with the model's judgment. So the model is told to pick however many
candidates it thinks are genuinely confusable, including none.

Model: defaults to deepseek-v4-flash rather than config.DEEPSEEK_MODEL
(deepseek-v4-pro), since eval runs are for iterating on prompts across many
items and pro's cost adds up fast at that volume. Pass --model
deepseek-v4-pro to compare against the production model directly.

Concurrency: every item's prompt is independent of every other item's, so
requests are sent in parallel worker threads rather than one at a time --
see --concurrency below. Threads (not raw asyncio) on purpose: the actual
DeepSeek call in src/adapters/llm/api.py is a normal blocking request, and
threads get the same overlap benefit here without needing a separate async
client or duplicating that call. DeepSeek's documented concurrency limit is
2500 requests account-wide for deepseek-v4-flash (500 for deepseek-v4-pro)
-- see https://api-docs.deepseek.com/quick_start/rate_limit -- so there's a
lot of headroom before --concurrency itself becomes the bottleneck.

Checkpointing and resuming: the output file is saved after every single
item completes, not just once at the end, so an interrupted run (Ctrl+C, a
crash, closing the terminal) never loses more than the one item that was
mid-flight. Re-running the same command afterward automatically resumes --
it reads whatever's already at --output, skips every item that already
completed successfully, and only pays for what's left. An item that
previously failed is retried automatically rather than skipped. Pass
--no-resume to ignore any existing output and start over from scratch.

One item failing (a timeout, a rate limit, whatever) doesn't take the whole
batch down with it -- that item is recorded with an "error" field and an
empty chosen list instead, and gets retried on the next run.

Note --debug's raw-message dump happens inside each worker thread, so with
--concurrency > 1 those dumps can interleave in the terminal; --verbose's
own printing does not, since it only happens in the main thread after a
result comes back.

Usage:
    uv run python scripts/eval_propose.py \\
        --input results/eval/questionnaire.json \\
        --output results/eval/proposer_output.json

    # try a different prompt against the same questionnaire
    uv run python scripts/eval_propose.py \\
        --input results/eval/questionnaire.json \\
        --system-prompt prompts/few_shot_v1.txt \\
        --output results/eval/proposer_output_few_shot_v1.json

    # cheap smoke test before spending a full run's worth of API calls
    uv run python scripts/eval_propose.py --input results/eval/questionnaire.json --limit 3

    # got interrupted? just run the same command again -- it resumes
    uv run python scripts/eval_propose.py --input results/eval/questionnaire.json --concurrency 50
"""

import argparse
import concurrent.futures
import json
from pathlib import Path

from src.adapters.llm.api import api_response


def load_system_prompt(path: Path) -> str:
    """Read a system prompt from a text file (defaults to prompts/baseline.txt)."""
    if not path.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {path}\n"
            "Point --system-prompt at a text file, or make sure prompts/baseline.txt exists."
        )
    return path.read_text(encoding="utf-8")


def build_user_prompt(target: str, candidates: list[str]) -> str:
    """
    The user-turn prompt. Fixed and not swappable, on purpose -- it's what
    keeps the output parseable regardless of which system prompt is loaded.
    Unlike the production prompt, this doesn't ask for a fixed number of
    picks (see the module docstring for why).
    """
    return (
        f"Target Drug:\n{target}\n\n"
        f"Candidate Drugs:\n" + "\n".join(candidates) + "\n\n"
        f"Task:\nSelect every candidate drug above that is genuinely likely "
        f"to be confused with the target drug. There is no fixed number to "
        f"return -- select as many or as few as are actually warranted, "
        f"including none at all if nothing on the list is a real risk.\n\n"
        f"Output:\nOne selected drug name per line, written exactly as it "
        f"appears in the candidate list above. No explanations, headers, "
        f"numbering, or extra text in this section.\n"
    )


def _propose_one(entry: dict, system_prompt: str, model: str | None, debug: bool) -> dict:
    """Run one questionnaire entry through the proposer. Runs inside a worker thread."""
    target = entry["c_c"]
    candidates = entry["p_c"]
    user_prompt = build_user_prompt(target, candidates)

    kwargs = {"model": model} if model else {}
    chosen, reasoning = api_response(
        user_prompt,
        candidates=candidates,
        system_prompt=system_prompt,
        return_reasoning=True,
        debug=debug,
        **kwargs,
    )
    return {**entry, "chosen": chosen, "reasoning": reasoning}


def _load_checkpoint(output_path: Path, questionnaire: list[dict]) -> tuple[list[dict], list[int]]:
    """
    Load a previous (possibly partial) output file to resume from.

    Returns (results, remaining_indices): `results` starts as a copy of the
    questionnaire itself (so it's always a full, valid, same-length list),
    with any previously-succeeded entries overlaid on top. `remaining_indices`
    is every position that still needs to be run -- either never attempted,
    or attempted and failed last time.
    """
    results = [dict(entry) for entry in questionnaire]
    remaining = list(range(len(questionnaire)))

    if not output_path.exists():
        return results, remaining

    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[propose] WARNING: {output_path} exists but isn't valid JSON -- starting fresh")
        return results, remaining

    if len(existing) != len(questionnaire):
        print(
            f"[propose] WARNING: {output_path} has {len(existing):,} items but this "
            f"questionnaire has {len(questionnaire):,} -- starting fresh"
        )
        return results, remaining

    done_count = 0
    still_remaining = []
    for i, prior in enumerate(existing):
        if prior.get("c_c") != questionnaire[i]["c_c"]:
            still_remaining.append(i)  # doesn't match this questionnaire's order, redo it
            continue
        if prior.get("chosen") is not None and not prior.get("error"):
            results[i] = prior
            done_count += 1
        else:
            still_remaining.append(i)  # never ran, or failed last time -- retry

    if done_count:
        print(
            f"[propose] Resuming from {output_path}: {done_count:,}/{len(questionnaire):,} "
            f"already done, {len(still_remaining):,} left"
        )
    return results, still_remaining


def run_proposer(
    questionnaire: list[dict],
    system_prompt: str,
    model: str | None,
    debug: bool,
    verbose: bool = False,
    concurrency: int = 5,
    output_path: Path | None = None,
    resume: bool = True,
) -> list[dict]:
    """
    Run every questionnaire entry through the proposer, up to `concurrency`
    at a time, and return the augmented list in the same order as the input.

    If output_path is given, the result is saved to it after every single
    item completes (not just once at the end), and -- unless resume=False --
    any already-successful items already sitting in that file are skipped
    rather than re-run.

    With verbose=True, prints the system prompt once up front, then the
    full user prompt and full reasoning for every item as its result comes
    back -- otherwise only a one-line summary per item is shown.

    An item whose request fails is recorded with an empty chosen list and
    an "error" field instead of stopping the whole run.
    """
    if verbose:
        print("[propose] ===== SYSTEM PROMPT =====")
        print(system_prompt)
        print("[propose] ===== END SYSTEM PROMPT =====\n")

    total = len(questionnaire)

    if resume and output_path is not None:
        results, remaining_indices = _load_checkpoint(output_path, questionnaire)
    else:
        results = [dict(entry) for entry in questionnaire]
        remaining_indices = list(range(total))

    def checkpoint() -> None:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if not remaining_indices:
        print("[propose] Nothing left to do -- every item already completed.")
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(_propose_one, questionnaire[i], system_prompt, model, debug): i
            for i in remaining_indices
        }

        done = 0
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            entry = questionnaire[i]
            target = entry["c_c"]
            done += 1

            try:
                result = future.result()
            except Exception as e:
                print(f"[propose] ERROR on {target!r}: {e}")
                result = {**entry, "chosen": [], "reasoning": "", "error": str(e)}

            results[i] = result
            checkpoint()

            if verbose:
                print(f"[propose] ----- INPUT (item {i + 1}/{total}) -----")
                print(build_user_prompt(target, entry["p_c"]))
                print(f"[propose] ----- REASONING (item {i + 1}/{total}) -----")
                print(result.get("reasoning") or "(no reasoning returned)")
                print(f"[propose] ----- CHOSEN (item {i + 1}/{total}) -----")
                chosen = result.get("chosen", [])
                print(", ".join(chosen) if chosen else "(none selected)")
                print(f"[propose] ----- END (item {i + 1}/{total}) -----\n")
            else:
                chosen = result.get("chosen", [])
                chosen_str = ", ".join(chosen) if chosen else "(none selected)"
                print(
                    f"[propose] {done}/{len(remaining_indices)} "
                    f"(item {i + 1}/{total}): {target!r} -> {chosen_str}"
                )

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a proposer over an eval questionnaire and record what it picks."
    )
    p.add_argument("--input", type=Path, required=True, help="questionnaire.json from eval_candidates.py")
    p.add_argument("--output", type=Path, default=Path("results/eval/proposer_output.json"))
    p.add_argument(
        "--system-prompt", type=Path, default=Path("prompts/baseline.txt"),
        help="Path to a text file to use as the system prompt. Defaults to "
        "prompts/baseline.txt.",
    )
    p.add_argument(
        "--model", type=str, default="deepseek-v4-flash",
        help="DeepSeek model to use (default: deepseek-v4-flash -- much "
        "cheaper than deepseek-v4-pro, used here instead of "
        "config.DEEPSEEK_MODEL since eval runs are for iterating on prompts, "
        "not production dataset construction). Pass --model deepseek-v4-pro "
        "to use the production model instead.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N items -- useful for a cheap smoke test "
        "before spending a full run's worth of API calls.",
    )
    p.add_argument(
        "--concurrency", type=int, default=5,
        help="How many requests to run in parallel (default: 5). DeepSeek's "
        "account-wide concurrency limit is 2500 for deepseek-v4-flash (500 "
        "for deepseek-v4-pro), so this can usually go a lot higher than the "
        "default.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing --output file and start over from scratch, "
        "instead of resuming and skipping already-completed items.",
    )
    p.add_argument("--debug", action="store_true", help="Dump the raw LLM message for each item.")
    p.add_argument(
        "--verbose", action="store_true",
        help="Print the system prompt once, then the full user prompt and "
        "full reasoning for every item, instead of a one-line summary per item.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    questionnaire = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit:
        questionnaire = questionnaire[: args.limit]

    system_prompt = load_system_prompt(args.system_prompt)
    print(f"[propose] System prompt: {args.system_prompt}")
    print(f"[propose] Model: {args.model}")
    print(f"[propose] {len(questionnaire):,} targets in this run "
          f"(concurrency={args.concurrency})")

    results = run_proposer(
        questionnaire,
        system_prompt,
        args.model,
        args.debug,
        args.verbose,
        args.concurrency,
        output_path=args.output,
        resume=not args.no_resume,
    )

    print(f"[propose] Saved -> {args.output}")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"[propose] WARNING: {len(errors):,} item(s) failed -- rerun the same "
              f"command to retry just those")


if __name__ == "__main__":
    main()
