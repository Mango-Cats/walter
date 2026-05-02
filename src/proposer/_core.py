"""
Function definitions for LLM-assisted true LASA pairs generation.
"""

from ._prompt import construct_user_prompt
from pathlib import Path
from enum import Enum
import time


RESULTS_DIR = Path("")
DELAY = 0.2


class Model(Enum):
    CLAUDE = 0
    HF = 1


def propose(
    drug_name: str,
    dataset: str,
    model: Model,
    n: int = 1,
    api_key: str | None = None,
    trace: bool = True,
):
    """
    Prompt a language model to generate `n` LASA candidates for `drug_name`
    given a `dataset`.
    """
    if trace:
        print(f"<walter> Proposing LASA for: {drug_name}.")
    
    user_prompt = construct_user_prompt(drug_name, dataset,n)

    if model is Model.CLAUDE and not api_key:
        raise Exception("Model is defined as CLAUDE but no provided `api_key`.")

    results: list[str] = []
    for _ in range(n):
        output: str
        match model:
            case Model.HF:
                from .hf import response

                output = response(user_prompt=user_prompt)

            case Model.CLAUDE:
                from .claude import response

                output = response(user_prompt=user_prompt)

                raise NotImplementedError

            case _:
                raise ValueError("Unsupported model")

        results.append(output.strip())
        time.sleep(DELAY)

    return results
