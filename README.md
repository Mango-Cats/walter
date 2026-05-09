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

On Colab this is handled automatically by the setup cell in `walter.ipynb`.
