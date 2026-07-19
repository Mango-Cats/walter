"""
LLM inference backends for the proposer.

Two interchangeable backends, selected by config.USE_API_MODEL:

    local   a transformers model loaded from config.MODELS_DIR
    api     DeepSeek over an OpenAI-compatible client

Both answer the same question -- given a user prompt and the candidate names
that prompt was built from, which candidates did the model pick -- and both
take the system prompt as an argument rather than importing the proposer's.
Validating a model's reply against the candidate list is shared, since a
backend that invented a name would otherwise corrupt P silently.
"""


def clean_output(text: str, candidates: list[str]) -> list[str]:
    """
    Keep only lines that exactly match a candidate (case-insensitive).

    The model is instructed to choose from the candidate list and nothing
    else, so anything it emits that is not on that list is discarded rather
    than trusted: a hallucinated drug name entering P would be indistinguishable
    from a confirmed pair by the time the dataset is assembled.
    """
    candidate_set = {c.lower() for c in candidates}
    seen: set[str] = set()
    valid: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line in candidate_set and line not in seen:
            valid.append(line)
            seen.add(line)
    return valid
