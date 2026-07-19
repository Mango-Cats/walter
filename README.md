# walter

LLM-assisted dataset construction for LASA drugs. *"Walter, are these two LASA drugs? Make no mistakes."*

All parameters, paths, and column names are in **`config/`**.

## Nomenclature and Terminology

**Raw registries** $\mathcal{R}^{\text{PH}}_{\text{raw}}$ and $\mathcal{R}^{\text{US}}_{\text{raw}}$ refer to the one-column drug name CSVs for the Philippine FDA and US FDA sources, stored at `data/R_ph.csv` and `data/R_us.csv` respectively. Only one is active per run, selected via `DATA_SOURCE` in `config/`. Their cleaned counterparts are cached alongside as `data/R_ph_clean.csv` and `data/R_us_clean.csv`.

**$\mathcal{R}_{\text{clean}}$** is the result of passing the active raw registry through the preprocessing pipeline: lowercased, symbols stripped, duplicates removed. It is an in-memory intermediate only and is not saved to disk. It serves as the sampling pool from which $U$ is drawn.

The final dataset $D$ has columns `x_1, t_1, x_2, t_2, label`, where `x_1` and `x_2` are drug names, `t_1` and `t_2` are their IPA transcriptions, and `label` is the pair's class. $D$ is the union of:

- **$P$** — confirmed LASA pairs. Always seeded from a pre-existing file of predefined pairs (`data/P_ph.csv`, columns `x_1, x_2`), used either as-is or *augmented* by the LLM proposer, which takes each predefined pair's `x_1` as an anchor and asks for further confusibles from $\mathcal{R}_{\text{clean}}$. The predefined pairs always survive augmentation. All $P$ rows carry `label = 1`.
- **$U$** — similarity-filtered unlabeled pairs drawn from $\mathcal{R}_{\text{clean}}$ via two-tier sampling. $U$ may contain undetected true LASA pairs. All $U$ rows carry `label = 0` (unlabeled, **not** confirmed negative).

$|U| \gg |P|$, $\quad P \cap U = \emptyset$.

Both $P$ and $U$ are recoverable from $D$ by filtering on `label`. $D$ is saved to `results/D.csv`, and $U$ is checkpointed to `results/U.csv` so `walter assemble` can run on its own.

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

will add all dependencies for the Decoder-only transformers (GPTs;
see how it's used in [src.proposer](/src/proposer/)).

### eSpeak-NG

IPA transcription requires eSpeak-NG installed at the system level:

- **Windows**: download the `.msi` from <https://github.com/espeak-ng/espeak-ng/releases>
- **Linux**: `apt install espeak-ng`
- **macOS**: `brew install espeak-ng`

### Pho (phoc)

Similarity features are computed by [`phoc`](https://github.com/Mango-Cats/pho),
a Rust CLI. This project expects:

- the `phoc` executable at `bin/phoc` (git-ignored — it's ~186MB, over
  GitHub's file limit; build it from the `pho` repo and drop it here);
- the config TOMLs at `bin/pho_conf/` (committed). Each `.toml` becomes one
  feature column named after the file; the file's `algorithm` key selects the
  algorithm.

See the `pho` docs for how the configs work.

### TagaBaybay (tbb-cli)

The Filipino-nativization features in [`src.feature_engineering`](/src/feature_engineering.py)
are computed by [`tbb-cli`](https://github.com/Mango-Cats/tagabaybay/), the TagaBaybay
loanword-adaptation worker.

- the `tbb-cli` executable at `bin/tbb-cli`.

## Running

```bash
uv run walter
```

Builds the full dataset per `config/` and saves the CSVs listed below.

### Running one stage

The pipeline is also split into stages that each read and write the paths in
`config/`, so a run can start from any point. This matters because the stages
differ enormously in cost: `propose` is one LLM call per predefined pair and
`assemble` transcribes the whole registry, while `noise` is cheap. Retuning a
sampling knob costs one `walter noise` instead of a full rebuild.

```bash
uv run walter propose     # augment predefined P    -> results/lasa_run.json
uv run walter noise       # sample U                -> results/U.csv
uv run walter assemble    # merge P + U, transcribe -> results/D.csv
uv run walter phoc        # phonetic features       -> results/D_pho.csv
uv run walter engineer    # META_FEATURES           -> results/D_engi.csv
uv run walter all         # every stage (same as bare `uv run walter`)
```

#### Directories in, directories out

`--input` and `--output` are **directories**, not files. Every artifact has one
canonical filename (`config/paths.py`), and a stage always reads that name out
of its input directory and writes that name into its output directory:

| stage | reads | writes |
| --- | --- | --- |
| `propose` | *(a pairs CSV — see below)* | `lasa_run.json` |
| `noise` | `lasa_run.json` | `U.csv` |
| `assemble` | `U.csv` + `lasa_run.json` | `D.csv` |
| `phoc` | `D.csv` | `D_pho.csv` |
| `engineer` | `D_pho.csv` | `D_engi.csv` |
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

- `D.csv` — classification: `x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label`
- `D_pho.csv` — `D` with one phonetic-similarity feature column per `bin/pho_conf/*.toml`.
  Transcription-dependent configs (`aline`) yield one column per language
  (`aline_ph_mc_eng`, `aline_ph_mc_fil`); the rest score `x_1`/`x_2` only and appear once.
- `D_engi.csv` — `D_pho` with the META_FEATURES from `src/feature_engineering.py`
  appended (structural / prosodic + the Filipino-nativization features from `bin/tbb-cli`)

`D.csv` is always written before `phoc` runs, so a `phoc` failure leaves it intact.
