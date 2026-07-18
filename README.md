# walter

LLM-assisted dataset construction for LASA drugs. *"Walter, are these two LASA drugs? Make no mistakes."*

All parameters, paths, and column names are in **`config.py`** — that is the single source of truth. Edit there, not here.

## Nomenclature and Terminology

**Raw registries** $\mathcal{R}^{\text{PH}}_{\text{raw}}$ and $\mathcal{R}^{\text{US}}_{\text{raw}}$ refer to the one-column drug name CSVs for the Philippine FDA and US FDA sources, stored at `_data/R_ph.csv` and `_data/R_us.csv` respectively. Only one is active per run, selected via `DATA_SOURCE` in `config.py`.

**$\mathcal{R}_{\text{clean}}$** is the result of passing the active raw registry through the preprocessing pipeline: lowercased, symbols stripped, duplicates removed. It is an in-memory intermediate only and is not saved to disk. It serves as the sampling pool from which $U$ is drawn.

The final dataset $D$ has columns `x_1, t_1, x_2, t_2, label`, where `x_1` and `x_2` are drug names, `t_1` and `t_2` are their IPA transcriptions, and `label` is the pair's class. $D$ is the union of:

- **$P$** — confirmed LASA pairs. Registry-independent: sourced either from a pre-existing file (`_data/P.csv`, columns `x_1, x_2`) or proposed by a local LLM using $\mathcal{R}_{\text{clean}}$ as its candidate pool. All $P$ rows carry `label = 1`.
- **$U$** — similarity-filtered unlabeled pairs drawn from $\mathcal{R}_{\text{clean}}$ via two-tier sampling. $U$ may contain undetected true LASA pairs. All $U$ rows carry `label = 0` (unlabeled, **not** confirmed negative).

$|U| \gg |P|$, $\quad P \cap U = \emptyset$.

Both $P$ and $U$ are recoverable from $D$ by filtering on `label`. Only $D$ is saved to disk (`_results/D.csv`).

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
differ enormously in cost: `propose` is 400 LLM calls and `assemble` transcribes
the whole registry, while `noise` is cheap. Retuning a sampling knob costs one
`walter noise` instead of a full rebuild.

```bash
uv run walter propose     # P via the LLM proposer  -> results/lasa_run.json
uv run walter noise       # sample U                -> results/U.csv
uv run walter assemble    # merge P + U, transcribe -> results/D.csv
uv run walter phoc        # phonetic features       -> results/D_pho.csv
uv run walter engineer    # META_FEATURES           -> results/D_engi.csv
uv run walter all         # every stage (same as bare `uv run walter`)
```

Each stage takes `--input` / `--output` to point it at a different file:

```bash
uv run walter phoc --input results/D.csv --output /tmp/scratch_pho.csv
```

Running a stage whose input is missing names the command that produces it:

```
$ uv run walter phoc
error: results/D.csv not found. Run `walter assemble` first.
```

### Outputs

Saved to `results/`:

- `D.csv` — classification: `x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label`
- `D_pho.csv` — `D` with one phonetic-similarity feature column per `bin/pho_conf/*.toml`.
  Transcription-dependent configs (`aline`) yield one column per language
  (`aline_ph_mc_eng`, `aline_ph_mc_fil`); the rest score `x_1`/`x_2` only and appear once.
- `D_engi.csv` — `D_pho` with the META_FEATURES from `src/feature_engineering.py`
  appended (structural / prosodic + the Filipino-nativization features from `bin/tbb-cli`)

`D.csv` is always written before `phoc` runs, so a `phoc` failure leaves it intact.
