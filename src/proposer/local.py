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
    DEEPSEEK_8B = "deepseek-8b"
    DEEPSEEK_1_5B = "deepseek-1.5b"
    LLAMA_8B = "llama-3.1-8b"

    @property
    def path(self):
        return os.path.join(".", "models", self.value)


_PIPELINES = {}


def get_pipeline(model_choice: LocalModel):
    if model_choice in _PIPELINES:
        return _PIPELINES[model_choice]

    model_path = model_choice.path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder missing: {model_path}")

    print(f"<walter> Loading {model_choice.name} from {model_path}...")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    load_params = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }

    if torch.cuda.is_available():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **load_params
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, **load_params
        )

    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    _PIPELINES[model_choice] = pipe
    return pipe


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


def response(
    user_prompt: str,
    model: LocalModel,
    candidates: list[str],
    new_toks_len: int = 64,
):
    pipe = get_pipeline(model)

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
