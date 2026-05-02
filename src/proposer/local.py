import os
import transformers
import torch
from enum import Enum
from .prompt import SYSTEM_PROMPT


class LocalModel(Enum):
    DEEPSEEK_8B = "deepseek_8b"
    DEEPSEEK_1_5B = "deepseek_1_5b"


MODEL_REGISTRY = {
    LocalModel.DEEPSEEK_8B: "./models/deepseek-r1-distill-llama-8b",
    LocalModel.DEEPSEEK_1_5B: "./models/deepseek-r1-distill-qwen-1.5b",
}

_PIPELINES = {}


def get_pipeline(model: LocalModel):
    if model in _PIPELINES:
        return _PIPELINES[model]

    model_path = MODEL_REGISTRY[model]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model folder not found at {model_path}. Did you download it?"
        )

    pipe = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    _PIPELINES[model] = pipe
    return pipe


def response(user_prompt: str, model: LocalModel, new_toks_len: int = 256) -> str:
    pipe = get_pipeline(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    output = pipe(
        messages,
        max_new_tokens=new_toks_len,
        truncation=True,
    )

    return output[0]["generated_text"][-1]["content"]
