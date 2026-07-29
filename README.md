# walter

LLM-assisted dataset construction for LASA drugs. *"Walter, are these two LASA drugs? Make no mistakes."*

## Setup

Simply run

```bash
uv sync
```

this will add all required dependencies of the project.

Then, running

```bash
uv sync --extra llm
```

will add all dependencies for the Decoder-only transformers (GPTs; see how it's used in [src.adapters.llm](/src/adapters/llm/)). Without this extra, the API backend is still available — set `USE_API_MODEL = True` in `config/proposer.py`.

**eSpeak-NG**. English IPA transcription requires eSpeak-NG installed at the system level:

- **Windows**: download the `.msi` from <https://github.com/espeak-ng/espeak-ng/releases>
- **Linux**: `apt install espeak-ng`
- **macOS**: `brew install espeak-ng`

**WFST-based Filipino G2P**. Filipino IPA transcriptions require a Phonetisaurus-based G2P model. This project expects `cwik_model.fst` at `bin/`, copied verbatim from Taglog-G2P (`notebook/train/cwik_model.fst`). The decoder itself comes from the `phonetisaurus` wheel and needs nothing on your PATH; override the model path with `FIL_G2P_MODEL` in `config/phonetics.py`.

**Pho (phoc)**. Similarity features are computed by [`phoc`](https://github.com/Mango-Cats/pho),
a Rust CLI. This project expects the `phoc` executable at `bin/` and the other prerequisites expected by `phoc`.

**TagaBaybay (tbb-cli)** [`tbb-cli`](https://github.com/Mango-Cats/tagabaybay/), the TagaBaybay loanword-adaptation worker, is used for Filipino nativization. This project expects the `tbb-cli` executable at `bin/` and the other prerequisites expected by `tbb-cli`.

## Running

All parameters, paths, and column names are in **`config/`**.

```bash
uv run walter
```

Builds the full dataset per `config/` and saves the CSVs listed below.

### Running one stage

The pipeline is also split into stages that each read and write the paths in `config/`, so a run can start from any point. This matters because the stages differ enormously in cost: `propose` is one LLM call per predefined pair and `assemble` transcribes the whole registry, while `noise` is cheap. Retuning a sampling knob costs one `walter noise` instead of a full rebuild.

```bash
uv run walter propose     # augment predefined P    -> results/lasa_run.json
uv run walter noise       # sample U                -> results/U.csv
uv run walter assemble    # merge P + U, transcribe -> results/D.csv
uv run walter phoc        # phonetic features       -> results/D_pho.csv
uv run walter all         # every stage (same as bare `uv run walter`)
```

Plus `featurize`, which runs the feature stages over pairs you already
have instead of pairs this pipeline builds (see below).

#### Directories in, directories out

`--input` and `--output` are **directories**, not files. Every artifact has one canonical filename (`config/paths.py`), and a stage always reads that name out of its input directory and writes that name into its output directory:

| stage | reads | writes |
| --- | --- | --- |
| `propose` | *(a pairs CSV - see below)* | `lasa_run.json` |
| `noise` | `lasa_run.json` | `U.csv` |
| `assemble` | `U.csv` + `lasa_run.json` | `D.csv` |
| `phoc` | `D.csv` | `D_pho.csv` |
| `featurize` | *(any pairs CSV - see below)* | `<input>_pho.csv` |
| `annotate` | `D.csv` | `<round>/R*.csv`, `<round>/_key.csv` |

Because the name a stage writes is exactly the name the next one looks for,
chaining stages is just pointing them at the same directory:

```bash
uv run walter noise    --input /tmp/run7 --output /tmp/run7   # -> /tmp/run7/U.csv
uv run walter assemble --input /tmp/run7 --output /tmp/run7   # -> /tmp/run7/D.csv
uv run walter phoc     --input /tmp/run7 --output /tmp/run7   # -> /tmp/run7/D_pho.csv
```

Both default to `results/` (`annotate`'s `--output` defaults to `annotation/`),
so the bare commands above all work with no arguments. Output directories are
created if missing.

Running a stage whose input is missing names the command that produces it:

```
$ uv run walter phoc
error: results/D.csv not found. Run `walter assemble` first.
```

#### The proposer is the exception

`walter propose` takes a **file**: a CSV of predefined LASA pairs
(columns `x_1, x_2`) that it augments. That file is yours to supply and no
stage produces it, so there is no directory to look it up in.

```bash
uv run walter propose --input data/P_ph.csv --output results
```

It defaults to `P[DATA_SOURCE]` (`data/P_ph.csv`). For each predefined pair it
anchors on `x_1`, pulls fuzzy-matched candidates from the registry, and asks
the LLM which are true confusibles. The predefined pair is carried into
`lasa_run.json` verbatim and is never subject to the LLM's judgement, so the
output is a superset of the input. `walter all` takes the same `--input`, since
`propose` is its first stage.

##### Skipping the proposer entirely

`FROM_FILE` in `config/proposer.py` decides where the later stages read P from,
and the table above assumes the default (`False`):

| `FROM_FILE` | `noise` and `assemble` read P from | `propose` |
| --- | --- | --- |
| `False` *(default)* | `lasa_run.json` in `--input` | must have been run first |
| `True` | `P[DATA_SOURCE]` (`data/P_ph.csv`), ignoring `--input` | skipped by `walter all` |

Set it to `True` when your LASA pairs are already confirmed and you want no LLM
in the pipeline at all. Loading P never triggers a proposal as a side effect, so
a missing `lasa_run.json` is reported rather than silently regenerated at a cost
of one LLM call per pair.

#### Soft labels

By default `label` is binary: `1` for a pair P asserts, `0` for everything else.
`--soft-labels` splits that `0` in two, so a pair that was *judged and rejected*
is no longer indistinguishable from one nobody ever looked at:

| `label` | pairs | source |
| --- | --- | --- |
| `1` | proposed by the LLM, or predefined | `lasa_run.json` / `P[DATA_SOURCE]` |
| `-1` | shown to the LLM and not proposed, or predefined as rejected | `lasa_run.json` candidates minus `x_2` / `N[DATA_SOURCE]` |
| `0` | combinatorially induced | `U.csv` |

```bash
uv run walter assemble --soft-labels
uv run walter assemble --rejected data/N_ph.csv     # implies --soft-labels
uv run walter all --soft-labels
```

Both sources of `-1` are read and unioned, and each is optional: the LLM's
rejections are the candidates it was shown and passed over (free - they are
already in `lasa_run.json`, no extra call), and the predefined ones come from
`--rejected`, or from `N[DATA_SOURCE]` (`data/N_ph.csv`, same `x_1, x_2`
columns as P) when that file happens to exist. A named `--rejected` that is
missing is an error; the configured default merely being absent is not. With
`FROM_FILE = True` there is no proposal to mine, so a predefined file is the
only source.

`SOFT_LABELS` in `config/proposer.py` sets the default for both commands, and
`--no-soft-labels` overrides it back for one run. Off, nothing is read and `D`
is byte-for-byte the dataset it was before.

A pair can be claimed by more than one input - the LLM rejects it and the
sampler happens to draw it - so the union resolves in the order P, N, U and the
strongest claim wins: confirmed positive over rejection, either over unlabeled.
The `annotate` stage is unaffected; its negative stratum still draws from the
sampled `0` pairs only, so rejections never reach a rater sheet.

#### Featurizing a dataset you already have

`walter featurize` runs the feature stages - G2P, then `phoc` - over an
existing CSV of pairs, skipping pair construction entirely: no LLM proposal,
no predefined positive set, no sampled U, no assembly.

```bash
uv run walter featurize --input pairs.csv                    # -> pairs_pho.csv
uv run walter featurize --input pairs.csv --output feats.csv
```

The input needs `x_1` and `x_2`. `label` is preserved if present (unlike
`assemble`, which sets `label` from which of P or U each row came from), and
so is row order - but every other column is dropped and rebuilt from scratch,
whatever it's named: stale transcriptions, phoc features, old META_FEATURES,
unrelated metadata, all of it. `featurize` always recomputes; it never trusts
columns already in the input. Like `propose`, `--input` is a file rather than
a directory; `--output` is one too, defaulting to `<input>_pho.csv` beside the
input, so a run cannot overwrite the pipeline's own `results/D_pho.csv` by
accident.

Each step writes its own CSV next to the output, so a failure partway through
keeps the work already paid for:

| file | contents |
| --- | --- |
| `<input>_t.csv` | + the IPA transcriptions |
| `<input>_pho.csv` | + the phonetic-similarity columns *(the default `--output`)* |

### Annotation sheets

Draws a batch from `D.csv` and writes one blinded CSV per rater, for the human
round described in `docs/annotation_codebook.md`. Agreement itself is computed
outside this pipeline, from the sheets the raters hand back.

```bash
uv run walter annotate   # -> annotation/r1/R1.csv, R2.csv, R3.csv, _key.csv
```

`--input` is the directory holding `D.csv` (default `results/`) and `--output`
the directory the round is written under (default `annotation/`); `--round`
names the subdirectory within it.

Sheets are blinded by construction: they carry only `pair_id`, the pair, its
Filipino IPA, and empty `label` / `channel` / `confidence` / `notes` columns.
The true label, the fuzzy score, the stratum, and the English transcription are
all withheld. `--show-similarity` opts the fuzzy score in (codebook Section 8.2
leaves that open, and it may anchor raters toward the orthographic channel).
All three raters get identical rows in identical order, so position cannot be a
source of systematic difference between them.

The `_key.csv` written alongside maps each `pair_id` back to its stratum and
source label. It is what joins the returned sheets together for the agreement
calculation, and must not be given to raters.

Batch composition is configurable, because a uniform sample of `D` is almost
entirely negative and leaves an agreement statistic nothing to estimate:

```bash
uv run walter annotate --n-candidates 120 --neg-per-positive 0.5 --n-placebo 20
```

Placebo pairs are unrelated names no attentive rater should mark positive, so
agreement on them is the straight-lining check from codebook Section 8.1.
Because the batch is stratified this way, agreement measured on it is not an
estimate of agreement over `D`.

### Outputs

Saved to `results/`:

- `D.csv` - classification: `x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label`
  (`label` is `1`/`0`, or `1`/`-1`/`0` under `--soft-labels`)
- `D_pho.csv` - `D` with one phonetic-similarity feature column per `bin/pho_conf/*.toml`.
  Transcription-dependent configs (`aline`) yield one column per language
  (`aline_ph_mc_eng`, `aline_ph_mc_fil`); the rest score `x_1`/`x_2` only and appear once.
  An aline column reads `aline_<config phonology>_<variant>_<transcription>`, so
  `aline_eng_kondrak_fil` is the English segment table scored over the Filipino
  IPA. The `_ph_` and `_eng_` configs share their costs, salience and `[values.*]`
  scales and differ only in how each segment is described, so a gap between them
  is attributable to phonology rather than to a re-tuned weighting.

`D.csv` is always written before `phoc` runs, so a `phoc` failure leaves it intact.
