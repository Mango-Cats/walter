# walter

LLM-assisted dataset construction for LASA drugs. "Walter, are these two LASA drugs? Make no mistakes."

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

### Pho

Similarity features are computed by [`phoc`](https://github.com/Mango-Cats/pho),
a Rust CLI. This project expects:

- the `phoc` executable at `bin/phoc` (git-ignored — it's ~186MB, over
  GitHub's file limit; build it from the `pho` repo and drop it here);
- the config TOMLs at `bin/pho_conf/` (committed). Each `.toml` becomes one
  feature column named after the file; the file's `algorithm` key selects the
  algorithm.

See the `pho` docs for how the configs work.

## Running

```bash
python walter.py
```

Builds the full dataset per `config.py` and saves four CSVs to `_results/`:

- `D.csv` — classification: `x_1, t_1, x_2, t_2, label`
- `D_rank.csv` — same rows + a `group` id, for a downstream grouped/ranking split
- `D_pho.csv` — `D` with one phonetic-similarity feature column per `bin/pho_conf/*.toml`
- `D_rank_pho.csv` — `D_rank` with those same feature columns appended

`D.csv` / `D_rank.csv` are always written before `phoc` runs, so a `phoc`
failure leaves them intact.
