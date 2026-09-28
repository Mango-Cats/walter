# proposer_eval

Quizzes the proposer's candidate gate against `data/recentannotation_gold.csv`.

The proposer (`src/proposer/inference.py::extract_candidates`) pulls up to 20
candidates per anchor from the registry using RapidFuzz `WRatio > 60` or a
jellyfish-Soundex similarity `>= 0.75`, ranked by a 50/50 composite. This
harness swaps that signal for **pho's Soundex** (`phoc`) and/or the
**TagaBaybay nativizer** (`tbb-cli`) and measures whether the same gold
confusables still get through.

## Run

From the repo root (needs `bin/phoc`, `bin/tbb-cli`, and eSpeak-NG, same as `walter` itself):

```bash
uv run python -m proposer_eval                    # both modes
uv run python -m proposer_eval --skip-retrieval   # 200 pairs only, seconds
uv run python -m proposer_eval --labels any       # A1 OR A2 counts as positive
uv run python -m proposer_eval --pho-configs DIR  # your own phoc configs
```

The first retrieval run nativizes the whole registry (cached in `.cache/`).

## What is scored

Every `.toml` in `--pho-configs` (default: `pho_conf/`: Editex, Soundex binary + soft)
is scored three ways, so dropping in Zhean's weighted Soundex config, or
Editex / Metaphone, needs no code change (rebuild `phoc` if the algorithm
itself changed):

| prefix | meaning |
| --- | --- |
| `base:*` | the proposer today (`proposer_gate`), and its two halves on their own |
| `pho:<cfg>` | pho on the raw spelling |
| `nat:<cfg>` | pho on the TagaBaybay-nativized spelling |
| `max:` / `mean:<cfg>` | soundex + nativizer together: pass if either matches / average |
| `fil:*` | phoc's built-in nativization indicators (pair mode only) |

**Pair mode**: the 200 gold pairs scored directly: AUC, AP, best F1, and P/R/F1
at the gate (`--gate-threshold`, default 0.75).

**Retrieval mode** (closer to what the proposer really does): for each gold
positive, rank the whole registry (`--registry`, default `data/ph_GenBrn.csv`,
the file the gold pairs were generated from) from its anchor `x_1` and check whether `x_2`
makes it through the gate and the top-K cut. Reports `pool_recall`,
`recall@K`, and `mean_pool` (candidates per anchor; smaller = cheaper prompt).
`gained/lost_vs_base` lists which positives each gate finds or drops relative
to today's gate.

## Caveats

- Every gold pair came from WRatio's top 20 per anchor before the LLM and
  annotators saw it, so any true confusable WRatio misses is absent from the
  data, and WRatio's own retrieval recall is 100% by construction. This can
  tell you what a new gate **loses**; it cannot tell you what it would newly
  **find**.
- Gold negatives also come from WRatio's top 20 (candidates the annotators
  rejected), so every gold pair already scores high on WRatio. That is why
  WRatio barely separates positives from negatives in pair mode (AUC ~0.54):
  it is being asked to judge its own shortlist.
- Retrieval drops candidates within edit distance 2 of the anchor for every
  strategy (today's proposer rule; `--min-edit-distance 0` turns it off). The
  gold set was generated without it, so 26 of 83 positives are unreachable at
  the default. Pair mode never applies it.
- Ties matter for binary Soundex (hundreds of names share a code), so top-K
  recall is shown with random and WRatio tie-breaking.
- `base:proposer_gate` is a vectorised copy of `extract_candidates`; each run
  re-checks it against the real function on every anchor (`[parity]` line).

Outputs land in `out/` (git-ignored): `pair_metrics.csv`, `pair_scores.csv`,
`retrieval_metrics.csv`, `retrieval_pairs.csv`.
