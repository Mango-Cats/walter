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

will add all dependencies for the Decoder-only transformers (generative
LLMs). These dependencies are one the heavier-side and some of these
are already included in Google Colab which is why it's separated.
