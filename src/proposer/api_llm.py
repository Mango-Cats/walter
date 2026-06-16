"""
DeepSeek API inference for the LASA proposer.
OpenAI-compatible client pointed at https://api.deepseek.com.
Used when USE_API_MODEL = True in config.py.
"""

import os

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from src.proposer.prompt import SYSTEM_PROMPT


def _get_client() -> OpenAI:
    key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError(
            "No DeepSeek API key found. Set DEEPSEEK_API_KEY in config.py "
            "or as an environment variable."
        )
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _clean_output(text: str, candidates: list[str]) -> list[str]:
    """Keep only lines that exactly match a candidate (case-insensitive)."""
    candidate_set = {c.lower() for c in candidates}
    seen: set[str] = set()
    valid: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line in candidate_set and line not in seen:
            valid.append(line)
            seen.add(line)
    return valid


def api_response(
    user_prompt: str,
    candidates: list[str],
    model: str = DEEPSEEK_MODEL,
    debug: bool = False,
    return_reasoning: bool = False,
):
    """
    Call the DeepSeek API and return cleaned proposed confusibles.

    Args:
        user_prompt:      The constructed user-turn prompt.
        candidates:       Valid drug names to validate output against.
        model:            DeepSeek model string (overrides config default).
        debug:            If True, dump the raw message object so we can see
                          where reasoning/CoT content lands (e.g. a separate
                          `reasoning_content` field vs. inline in `content`).
        return_reasoning: If True, return (proposed, reasoning_content) instead
                          of just proposed. reasoning_content may be "" if the
                          model/response didn't include it.

    Returns:
        List of validated confusible drug names from candidates, or
        (list, reasoning_content) if return_reasoning=True.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=False,
    )

    message = response.choices[0].message

    if debug:
        print("\n[api_llm] --- RAW MESSAGE DUMP ---")
        try:
            print(message.model_dump_json(indent=2))
        except AttributeError:
            print(repr(message))
        print("[api_llm] --- END RAW MESSAGE DUMP ---\n")

    text = message.content or ""
    proposed = _clean_output(text, candidates)

    if return_reasoning:
        reasoning = getattr(message, "reasoning_content", "") or ""
        return proposed, reasoning

    return proposed
