"""
System prompt for LLM-assisted true LASA pairs generation.
"""

from os import environ
from typing import NoReturn

MODEL = "claude-haiku-4-5-20251001"


def response(user_prompt: str) -> NoReturn:
    raise NotImplementedError
    # url = "https://api.anthropic.com/v1/messages"

    # headers = {
    #     "x-api-key": api_key,
    #     "anthropic-version": "2023-06-01",
    #     "content-type": "application/json",
    # }

    # payload = {
    #     "model": "claude-3-opus-20240229",
    #     "max_tokens": 128,
    #     "system": SYSTEM_PROMPT,
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": user_prompt,
    #         }
    #     ],
    # }

    # response_obj = requests.post(url, headers=headers, json=payload)

    # if response_obj.status_code != 200:
    #     raise Exception(f"Claude API error: {response_obj.text}")

    # data = response_obj.json()
    # output = data["content"][0]["text"]


def get_api_key(env_var_name: str):

    if (key := environ.get(key=env_var_name)) is None:
        raise KeyError(
            f"The environment variable {env_var_name} is required but is not found."
        )

    return key
