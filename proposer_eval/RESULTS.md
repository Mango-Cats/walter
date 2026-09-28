# Results: Soundex / Editex / nativizer as the proposer's candidate gate

## Sprint summary

**Question:** if walter's candidate step used Editex, Soundex, or the nativizer
instead of WRatio, would the confusable pairs WRatio found (and annotators
confirmed) still reach the LLM?

The gold file is a random sample of 200 generated pairs (WRatio's top 20 per
anchor, then the LLM, then two annotators). Both annotators confirmed 83 as
confusable; 81 of those were proposed by the LLM and 2 were not. Walter
intentionally drops pairs within edit distance 2 (easy positives), and applying
that rule here leaves 57 non-trivial positives. These are 57 in the annotated
sample, not in the whole generated set. For
each, the check is whether the method puts it in its top 20 of 25,405 registry
names (pure swap, ties broken at random).

| method | non-trivial positives captured (top 20) |
|---|---|
| WRatio | 57/57 (100%) |
| today's walter gate (WRatio + Soundex mix) | ~38/57 (67%) |
| Editex | ~28/57 (49%) |
| Editex on nativized names | ~22/57 (39%) |
| Soundex (soft) | ~22/57 (39%) |
| Soundex (exact code) | ~21/57 (37%) |

1. Swapping WRatio out loses confirmed positives; the best alternative (Editex)
   misses about half.
2. Nativizing first makes it worse.
3. Soundex's weakness is ties: one true match is tied with over 3,000
   unrelated names.
4. Within WRatio's shortlist, Editex and Soundex judge pairs much better than
   WRatio (AUC ~0.90 vs 0.54): useful as re-rankers, not generators.
5. Today's walter gate keeps only 67% because Soundex gets a 0.5 weight in its
   ranking. Lowering that weight to 0.1 keeps 56/57, and using Soundex only to
   break ties keeps all 57 (keeping the edit-distance rule).

Caveat: the gold set was built from WRatio's top 20, so this measures recovery
of WRatio's positives, not discovery of confusables WRatio misses.

### Does a bigger candidate list fix it?

Same 57 positives, varying how many candidates per anchor would be sent to the
LLM (`-k`). The walter gate breaks ties by WRatio, as `extract_candidates`
does; Editex is a pure swap with random tie-breaking. WRatio is 100% at every
size by construction.

| candidates per anchor | walter gate | Editex | LLM prompt size vs today |
|---|---|---|---|
| top 20 (today) | ~38/57 (67%) | ~28/57 (49%) | 1x |
| top 30 | ~40/57 (70%) | ~30/57 (52%) | 1.5x |
| top 50 | ~42/57 (74%) | ~34/57 (60%) | 2.5x |
| top 100 | ~43/57 (75%) | ~37/57 (65%) | 5x |

Widening the list barely helps the walter gate. Every one of the 57 passes its
gate; the problem is ranking. It ranks by 0.5 x WRatio + 0.5 x Soundex
similarity, and Soundex similarity only takes the values 0, 0.25, 0.5, 0.75 or
1, so names that share the anchor's Soundex code get a large boost even with a
mediocre WRatio. About 14 of the 57 positives rank below 100th for this
reason. (At top 50 and top 100, the harness's copy of the walter gate differed
from the real `extract_candidates` on 1 and 4 of 48 anchors, from float
rounding in tied scores, so these counts may be off by about one. At top 20 and
30 they matched on all 48.)

### Is it Soundex, or the 50/50 split?

Both. Soundex similarity in the walter gate (Levenshtein on 4-character codes)
takes only 5 values: 0, 0.25, 0.5, 0.75, 1. Per anchor, a median of 11
registry names score 1.0 and 289 score 0.75, so a top-20 cut inside those ties
is close to random. That is why Soundex alone captures only 39% of the known
pairs.

The 0.5 weight lets that coarseness decide the walter gate's ranking. Each
Soundex step is worth 0.125 in the composite, the same as 12.5 WRatio points,
which is more than the WRatio gap between a true pair and the names ranked
around it. Re-weighting the composite (same gate, same 57 positives, top 20,
WRatio tie-break):

| Soundex weight in the composite | known pairs captured (top 20) |
|---|---|
| 0.5 (today) | 38/57 |
| 0.3 | 46/57 |
| 0.2 | 50/57 |
| 0.1 | 56/57 |
| 0.05 | 57/57 |
| Soundex only breaks exact WRatio ties | 57/57 |

For proposing, WRatio should drive the ranking and Soundex should get a small
weight (<= 0.1) or only break ties. In `src/proposer/inference.py` this is the
`composite = 0.5 * (f_score / 100.0) + 0.5 * s_sim` line. At 0.5 it drops a
third of the known non-trivial true pairs.

Caveat: any weight near 0 scores well here, because weight 0 is pure WRatio,
which picked these pairs. The table shows a heavy Soundex weight loses known
pairs; it cannot show a small weight adds anything over pure WRatio. The only
route for Soundex to add new pairs is the gate's `soundex_similarity >= 0.75`
condition, which lets in names WRatio scores <= 60; whether those contain true
pairs needs annotation.

### Recommendation for future data construction

- **Keep WRatio as the candidate backbone.** It is the only method tested that
  does not lose known positives. This does not show WRatio finds the *most*
  positives: nothing outside its top 20 was ever annotated, so its own blind
  spot is unmeasured.
- **Add to it rather than replace it.** For example, WRatio's top 20 plus
  Editex's top 5 that are not already in WRatio's list. Nothing WRatio would
  find is lost, and the extra candidates are the only place new positives can
  come from, at the cost of slightly larger prompts.
- **Use the phonetic scores as judges.** Within WRatio's shortlist, Editex and
  soft Soundex separate confusable from non-confusable pairs far better than
  WRatio (AUC ~0.90 vs 0.54).
- **Next experiment:** annotate the extra (non-WRatio) candidates from such a
  run. The share that come out positive measures WRatio's blind spot, which is
  the one thing this data cannot.

## Setup

Run: `uv run python -m proposer_eval` (defaults below).

- Gold: `data/recentannotation_gold.csv`, 200 pairs, `--labels consensus`
  (A1 == A2 only; 11 dropped on disagreement), leaving 189 labeled pairs, 83 positive.
- Registry: `data/ph_GenBrn.csv`, 25,405 names, the file the gold pairs were
  generated from. All 83 positives' `x_2` are in it.
- pho configs (`pho_conf/`): `editex.toml` (unlocalized English letter groups,
  copied from pho's `algorithm_configs/eng/editex.toml`), `soundex_binary.toml`,
  `soundex_soft.toml`. Each is scored on raw spelling (`pho:`), on the
  TagaBaybay-nativized spelling (`nat:`), and both combined (`max:`, `mean:`).
- Nativizer: `tbb-cli` with its eSpeak-NG G2P backend active (it changed the
  spelling of 86% of names).

## 1. Pair scoring (189 labeled pairs, 83 positive)

Each gold pair is scored directly; there is no registry or candidate ranking.
The question is whether the score separates the known positives from the known
negatives. The edit-distance rule is not applied here.

| scorer | AUC | AP | best F1 | gate P | gate R | gate F1 |
|---|---|---|---|---|---|---|
| base:proposer_gate (today's composite) | 0.902 | **0.911** | **0.827** | 0.439 | 1.000 | 0.610 |
| base:fuzzy_only (WRatio) | 0.544 | 0.599 | 0.620 | 0.439 | 1.000 | 0.610 |
| base:jf_soundex_only | 0.883 | 0.865 | 0.813 | 0.910 | 0.735 | 0.813 |
| pho:editex | **0.905** | 0.904 | 0.809 | 0.812 | 0.783 | 0.798 |
| nat:editex | 0.880 | 0.873 | 0.786 | 0.803 | 0.735 | 0.767 |
| max:editex | 0.899 | 0.901 | 0.810 | 0.798 | 0.807 | 0.802 |
| mean:editex | 0.896 | 0.894 | 0.819 | 0.803 | 0.735 | 0.767 |
| pho:soundex_binary | 0.717 | 0.682 | 0.610 | 1.000 | 0.434 | 0.605 |
| nat:soundex_binary | 0.711 | 0.676 | 0.610 | 1.000 | 0.422 | 0.593 |
| pho:soundex_soft | 0.843 | 0.847 | 0.813 | 0.910 | 0.735 | 0.813 |
| nat:soundex_soft | 0.857 | 0.855 | 0.819 | 0.924 | 0.735 | 0.819 |
| fil:fil_onset_match | 0.789 | 0.690 | 0.761 | 0.775 | 0.747 | 0.761 |
| ref:llm_label (context only) | | | | 0.862 | 0.976 | 0.915 |

"Gate" columns use a threshold of 0.75 for pho/nat/fil rows. The base rows
use today's rules: WRatio > 60 for fuzzy_only, jellyfish-Soundex >= 0.75 for
jf_soundex_only, and either one for proposer_gate.
WRatio's gate recall of 1.000 is by construction: every gold pair has
WRatio >= 62.5 (see "What cannot be concluded"). The full table, including the
max/mean Soundex rows and every fil_* indicator, is in `out/pair_metrics.csv`.

## 2. Candidate retrieval (57 reachable positives, 34 anchors)

For each gold positive, the anchor `x_1` is scored against all 25,405 registry
names. The check is whether `x_2` passes the gate and then survives the
proposer's top-20 cut. By default, candidates within edit distance 2 of the
anchor are dropped for every strategy, because today's proposer does this.
That makes 26 of the 83 positives unreachable, leaving the 57 scored below
(over 34 anchors).
The vectorised baseline matched the real `extract_candidates()` top-20 on 48 of
48 anchors.

| scorer | pass gate | top-20 (WRatio ties) | top-20 (random ties) | candidates/anchor |
|---|---|---|---|---|
| base:proposer_gate (today) | 1.000 | 0.667 | 0.667 | 455 |
| base:fuzzy_only | 1.000 | **1.000** | 0.960 | 160 |
| base:jf_soundex_only | 0.702 | 0.614 | 0.394 | 314 |
| pho:editex | 0.684 | 0.509 | 0.486 | 221 |
| nat:editex | 0.649 | 0.386 | 0.386 | 223 |
| max:editex | 0.719 | 0.439 | 0.410 | 323 |
| mean:editex | 0.649 | 0.474 | 0.465 | 162 |
| pho:soundex_binary | 0.439 | 0.439 | 0.377 | 19 |
| pho:soundex_soft | 0.702 | 0.614 | 0.392 | 315 |
| nat:soundex_soft | 0.702 | 0.596 | 0.357 | 312 |

**How to read this table.** Every gold positive is a true confusable pair
(both annotators confirmed it), so a method that fails to capture one misses a
real confusable pair. The table is a valid measure of how many known true
pairs each method captures. Two limits apply:

- WRatio's 100% is by construction, not an achievement: these pairs are in
  the gold set because WRatio's top 20 contained them.
- It counts only what a method misses, not what it finds instead. When a
  method misses one of these pairs, something else takes that slot in its
  top 20. That candidate was never annotated, so it could be another true
  pair that makes up for the loss. The table cannot say which method finds the
  most true pairs overall.

This answers the question directly: **if WRatio were replaced by another
method, would the positives WRatio surfaced (and annotators confirmed) still
reach the LLM?** No method does. The same holds with the edit-distance rule
off (`--min-edit-distance 0`, all 83 positives reachable), where the share
landing in each method's top 20 as a pure swap (random tie-break) is: today's
composite 72%, pho:editex 63%, mean:editex 60%, max:editex 58%, nat:editex
50%, soft Soundex 39%, binary Soundex 37%. Soundex recovers to 66–68% only
when WRatio breaks its ties, because its scores tie heavily.

Per-anchor detail and the positives each gate
gains or loses against the baseline are in `out/retrieval_metrics.csv` and
`out/retrieval_pairs.csv`.

## Conclusion

1. **Editex and soft Soundex score known pairs about as well as today's
   composite gate, and far better than WRatio alone.** On the 189 labeled
   pairs, AUC is 0.84–0.91 for these three against 0.54 for WRatio alone.
   Adding Editex does not improve on the composite: 0.905 vs 0.902 AUC is a tie
   at this sample size. The jellyfish-Soundex half of the composite is what
   makes it a good pair scorer.
2. **Replacing WRatio would lose known true pairs.** Of the 57 non-trivial
   confirmed pairs, today's composite captures 67% in its top 20, Editex 49%,
   soft Soundex 39% (pure swap). WRatio's 100% is by construction. What the
   other methods would find in place of the missed pairs was never annotated,
   so this shows what they lose, not which method finds the most true pairs
   overall.
3. **Soundex scores are coarse, which is a known risk for ranking.** For
   example, under soft Soundex `micoson` (a positive for `mcson scent`) is tied
   with 3,011 other registry names, with 399 above it. Large tie blocks make a
   top-K cut close to arbitrary, which is why Soundex captures only 39% of
   the known pairs as a pure swap.
4. **Nativizing first does not help.** For Editex it is worse on both tests
   (pair AUC 0.905 to 0.880, top-20 49% to 39%). For soft Soundex it is a wash
   on pair scoring (0.843 to 0.857) and slightly worse on retrieval. Stripping
   digits and spaces alone changes nothing (raw and stripped scores are
   identical), so these differences come from nativization itself.
   Binary (exact-code) Soundex is too coarse either way.

In practice: keep WRatio as the candidate generator, since swapping it out
loses a large share of known true pairs, and use Editex or soft Soundex to
judge pairs inside WRatio's shortlist, where WRatio itself is nearly useless
as a judge (AUC 0.54). Whether adding phonetic candidates on top of WRatio's
would find new true pairs needs annotations on candidates outside WRatio's
top 20.

### What this data supports

- Editex and soft Soundex are strong pair-level similarity scores, on par with
  today's composite and much better than WRatio alone.
- Within WRatio's shortlist, WRatio cannot tell confusable pairs from
  non-confusable ones (AUC 0.54), while the phonetic scores can.
- Replacing WRatio with Editex, Soundex, the nativizer, or today's composite
  would miss a large share (33–63%) of the known non-trivial true pairs, and
  widening the list to top 100 does not close the gap.
- Nativizing names before Editex or Soundex does not help on this data.

### What cannot be concluded

- **Which method finds the most true pairs overall.** The retrieval test
  counts known true pairs each method misses, but not the unannotated
  candidates it would propose instead, some of which could also be true pairs.

- **Whether Editex, Soundex or the nativizer would find LASA pairs that WRatio
  misses entirely.** Every gold pair has WRatio >= 62.5. The generator was
  WRatio's top 20 per anchor, then the LLM, then the annotators. A true confusable that
  WRatio scores low never became a candidate, so it is absent from the data,
  not labeled negative. This is why WRatio's 100% retrieval and gate recall
  hold almost by construction: they measure recall on pairs WRatio itself
  surfaced, not on real-world LASA pairs. Answering this needs fresh
  annotations on candidates that Editex or Soundex surface and WRatio does not.
- **Whether a Filipino-localized Editex would change these results.** No
  localized weights exist in the repo. Checked: pho `main`, `aline-test`,
  `copilot/explain-repository-structure`, and PR #8. Soundex has no
  per-letter weights anywhere, and the Editex config used here has English
  groups.
- **How the combined system would perform**, with WRatio for retrieval and
  Editex for re-ranking. This was not tested.
- **Whether the numbers are stable.** 189 labeled pairs (83 positive) is small
  for AUC and AP, and differences of about 0.02–0.03 (for example 0.905 vs
  0.902) are within noise. The gold set also predates today's edit-distance
  rule: 26 positives are within edit distance 2 of their anchor, which the
  current proposer could never surface.
