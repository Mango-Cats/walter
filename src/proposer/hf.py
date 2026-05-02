from .prompt import SYSTEM_PROMPT
from enum import Enum
import transformers
import torch


class HFModel(Enum):
    DEEPSEEK = "deepseek"
    TINY_LLAMA = "tinyllama"


MODEL_REGISTRY = {
    HFModel.DEEPSEEK: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    HFModel.TINY_LLAMA: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

_PIPELINES = {}


def get_pipeline(model: HFModel):
    if model in _PIPELINES:
        return _PIPELINES[model]

    model_id = MODEL_REGISTRY[model]

    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        model_kwargs={"dtype": torch.float32},
    )

    _PIPELINES[model] = pipe
    return pipe


def response(user_prompt: str, model: HFModel, new_toks_len: int = 64) -> str:
    pipe = get_pipeline(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    output = pipe(
        messages,
        max_new_tokens=new_toks_len,
        do_sample=False,
        temperature=0.0,
    )

    return output[0]["generated_text"][-1]["content"]
