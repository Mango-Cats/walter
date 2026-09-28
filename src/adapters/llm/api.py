"""Remote API integration for finding confusable drug pairs using DeepSeek.

Connects to DeepSeek's API using an OpenAI-compatible interface to evaluate
drug candidate lists and select plausible confusable pairs.
"""

import os

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from src.adapters.llm import clean_output


def _get_client() -> OpenAI:
    """Initialize and return an OpenAI client connected to the DeepSeek API.

    Returns:
        Configured OpenAI client instance.

    Raises:
        ValueError: If the API key is not configured in config or environment variables.

    """
    key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError(
            "No DeepSeek API key found. Set DEEPSEEK_API_KEY in config.py "
            "or as an environment variable."
        )
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def api_response(
    user_prompt: str,
    candidates: list[str],
    system_prompt: str,
    model: str = DEEPSEEK_MODEL,
    debug: bool = False,
    return_reasoning: bool = False,
) -> list[str] | tuple[list[str], str]:
    """Send a prompt to the DeepSeek API and return validated confusable candidates.

    Args:
        user_prompt: Prompt text containing the target drug and candidate list.
        candidates: List of valid candidate drug names to validate against.
        system_prompt: Instructions guiding the model on clinical drug confusion.
        model: DeepSeek model identifier string.
        debug: Whether to print raw API response objects for debugging.
        return_reasoning: If True, returns a tuple of (proposed_drugs, reasoning_text).

    Returns:
        List of proposed confusable drug names, or a tuple of (proposed_drugs, reasoning_text).

    """
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        stream=False,
    )

    message = response.choices[0].message

    if debug:
        print("\n[llm.api] --- RAW MESSAGE DUMP ---")
        try:
            print(message.model_dump_json(indent=2))
        except AttributeError:
            print(repr(message))
        print("[llm.api] --- END RAW MESSAGE DUMP ---\n")

    text = message.content or ""
    proposed = clean_output(text, candidates)

    if return_reasoning:
        reasoning = getattr(message, "reasoning_content", "") or ""
        return proposed, reasoning

    return proposed
