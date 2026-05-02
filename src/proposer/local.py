import os
import transformers
import torch
from enum import Enum
from .prompt import SYSTEM_PROMPT


n_threads = os.cpu_count() or 4

torch.set_num_threads(n_threads)
torch.set_num_interop_threads(max(1, n_threads // 2))
torch.backends.mkldnn.enabled = True


class LocalModel(Enum):
    DEEPSEEK_8B = "deepseek_8b"
    DEEPSEEK_1_5B = "deepseek_1_5b"


MODEL_REGISTRY = {
    LocalModel.DEEPSEEK_8B: "./models/deepseek-r1-distill-llama-8b",
    LocalModel.DEEPSEEK_1_5B: "./models/deepseek-r1-distill-qwen-1.5b",
}

_PIPELINES = {}


def _clean_output(text: str, candidates: list[str]) -> list[str]:
    lines = text.strip().split("\n")

    candidate_set = {c.lower() for c in candidates}
    seen = set()
    valid = []

    for line in lines:
        line = line.strip().lower()

        if line in candidate_set and line not in seen:
            valid.append(line)
            seen.add(line)

    return valid


def get_pipeline(model_choice: LocalModel):
    if model_choice in _PIPELINES:
        return _PIPELINES[model_choice]

    model_path = MODEL_REGISTRY[model_choice]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading tokenizer for {model_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")

    if torch.cuda.is_available():
        print("Using CUDA...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print("Using CPU...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )

    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    _PIPELINES[model_choice] = pipe
    return pipe


def response(
    user_prompt: str,
    model: LocalModel,
    candidates: list[str],
    new_toks_len: int = 64,
):
    pipe = get_pipeline(model)

    # Flat prompt for better CPU + small-model stability
    flat_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nOutput:"

    output = pipe(
        flat_prompt,
        max_new_tokens=new_toks_len,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )

    raw_text = output[0].get("generated_text", "")

    return _clean_output(raw_text, candidates)
