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
uv run walter all         # every stage
```

Plus `featurize`, which runs the feature stages over pairs you already
have instead of pairs this pipeline builds (see below).

#### Directories in, directories out

`--input` and `--output` are **directories**. Every artifact has one canonical filename (`config/paths.py`), and a stage always reads that name out of its input directory and writes that name into its output directory:

| stage | reads | writes |
| --- | --- | --- |
| `propose` | *(a pairs CSV - see below)* | `lasa_run.json` |
| `noise` | `lasa_run.json` | `U.csv` |
| `assemble` | `U.csv` + `lasa_run.json` | `D.csv` |
| `phoc` | `D.csv` | `D_pho.csv` |
| `featurize` | *(any pairs CSV - see below)* | `<input>_pho.csv` |
| `relabel` | *(a dataset CSV and a changes CSV - see below)* | the CSV named by `--output` |
| `lexicon` | *(a held-out pair CSV - see below)* | the Parquet file named by `--output` |

Because the name a stage writes is exactly the name the next one looks for,
chaining stages is just pointing them at the same directory:

```bash
uv run walter noise    --input /my_dir --output /my_dir   # -> /my_dir/U.csv
uv run walter assemble --input /my_dir --output /my_dir   # -> /my_dir/D.csv
uv run walter phoc     --input /my_dir --output /my_dir   # -> /my_dir/D_pho.csv
```

Both default to `results/`, so the bare commands above all work with no
arguments. Output directories are
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

#### Core-word features

Registry names can carry salts, dosage forms, strengths, and qualifiers. The
core word is the first word left after skipping those (`CORE_NAME_STOPWORDS` in
`config/phonetics.py`): `sevelamer hydrochloride` becomes `sevelamer`, and
`tored 10` becomes `tored`. A name opening with a word that identifies little
alone (`CORE_LEADING_WORDS`) keeps the next usable word too, so `neo penotran`
and `sodium valproate` stay whole. Core words are scored in addition to the full
names, which keep all their own columns.

`phoc` and `featurize` add nine `core_*` columns: eight spelling measures
(`CORE_ORTHOGRAPHIC_CONFIGS`) and `core_aline_var:cangu_g2p:fil`, which scores the
Filipino transcription of the core word.

#### Starting from preconstructed positives and negatives

When LASA pairs and rejected pairs already exist, one command runs the whole pipeline
and skips the LLM. The input has `x_1`, `x_2`, and `label` (1 or -1):

```bash
uv run walter all --labelled data/ths2/D.csv --registry data/registries/R_ph.csv \
    --target 19928 --seed 42 --output results/seed42
```

It samples exactly `--target` noise pairs labelled 0, leaving out any pair already
labelled. By default, 10% are unfiltered random registry pairs (`--random-share`), and
pairs that differ only in strength, salt, form, or qualifier are left out
(`--no-skip-variants` keeps them). It then merges everything, transcribes it, and computes every feature, ending
in `D_pho.csv`. `U_report.txt` shows how evenly the noise uses names.

#### Changing labels in a finished dataset

Apply reviewed changes, then recompute the features:

```bash
uv run walter relabel --input D.csv --changes changes.csv --output pairs.csv
uv run walter featurize --input pairs.csv --output D_new.csv
```

`--allow-missing` applies changes made on another version of the dataset, and writes
what happened to each one to `<output>_changes.csv`.

#### Scoring a lexicon pool

`walter lexicon` supports a ranking evaluation. Each name in a LASA pair becomes
a query, scored against every other name in the input CSV:

```bash
uv run walter lexicon --input test_pairs.csv --output pool.parquet
```

The input needs `x_1`, `x_2`, `label`, and the transcription columns. The pool
holds the similarity and core columns, but not the engineered features.

### Outputs

Saved to `results/`:

- `D.csv` - classification: `x_1, t_eng_1, t_fil_1, x_2, t_eng_2, t_fil_2, label`
  (`label` is `1`/`0`, or `1`/`-1`/`0` under `--soft-labels`)
- `D_pho.csv` - `D` with one phonetic-similarity feature column per `bin/pho_conf/*.toml`.
  Transcription-dependent configs (`aline`) yield one column per transcription they
  are scored over; the rest score `x_1`/`x_2` only and appear once. An aline column
  reads `aline_var:<variant>_g2p:<transcription>`:

  | column | ALINE weights | transcription |
  | --- | --- | --- |
  | `aline_var:kondrak_g2p:eng` | Kondrak (`aline_kondrak.toml`) | English |
  | `aline_var:kondrak_g2p:fil` | Kondrak (`aline_kondrak.toml`) | Filipino |
  | `aline_var:cangu_g2p:fil` | learned (`aline_cangu.toml`) | Filipino |

  `config.PHONETIC_CONFIG_LANGS` limits a config to specific transcriptions.

  ALINE scores Filipino with one `ˈ` per word: on the vowel `tbb-cli` capitalizes,
  otherwise the penult. Only `mango_cats` configs use it; `kondrak` ignores stress
  marks. The marks are added for scoring, so `t_fil_*` stays unmarked.

`D.csv` is always written before `phoc` runs, so a `phoc` failure leaves it intact.
