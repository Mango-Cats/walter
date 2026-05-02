import transformers
import torch

from src.proposer._core import SYSTEM_PROMPT

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
PIPELINE: transformers.TextGenerationPipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
MAX_NEW_TOKENS = 64


def response(user_prompt: str) -> str:
    """
    Prompt a Llama (open-weights) decoder-only language model using
    the `user_prompt`.

    Remark. I don't really care about the performance of this 🦙. This
    was just added to check the performance of the prompt itself before
    passing it onto the commercial models. Since this 🦙 is open-source
    it provides the best testing ground for the prompt.

    Reference. https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
    """
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    output = PIPELINE(message, max_new_tokens=MAX_NEW_TOKENS)

    return output[0]["generated_text"][-1]["content"]
