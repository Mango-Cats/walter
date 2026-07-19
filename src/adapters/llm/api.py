"""
DeepSeek API inference, backing the LASA proposer.

OpenAI-compatible client pointed at https://api.deepseek.com. Used when
USE_API_MODEL = True in config.py; otherwise the local backend runs instead.
"""

import os

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from src.adapters.llm import clean_output


def _get_client() -> OpenAI:
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
):
    """
    Call the DeepSeek API and return cleaned proposed confusibles.

    Args:
        user_prompt:      The constructed user-turn prompt.
        candidates:       Valid drug names to validate output against.
        system_prompt:    The system-turn prompt. Passed in rather than
                          imported so this module stays free of
                          proposer-specific domain knowledge.
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
