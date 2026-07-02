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

## Running

```bash
python walter.py
```

Builds the full dataset per `config.py` and saves `_results/D.csv`
(classification) and `_results/D_rank.csv` (same rows + a group id, for
a downstream grouped/ranking split).
