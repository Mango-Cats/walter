"""
Local LLM loader and inference runner for the LASA proposer.
Models are downloaded by scripts/model_setup.py into config.MODELS_DIR.
"""

import os
from enum import Enum

from config import MODELS_DIR

try:
    import torch
    import transformers
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))


class LocalModel(Enum):
    """Supported local models. Values are HuggingFace repo IDs."""

    QWEN3_4B = "Qwen/Qwen3-4B-Instruct-2507"
    SMOLLM2 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    QWEN3_1_7B = "Qwen/Qwen3-1.7B"

    @property
    def path(self) -> str:
        return str(MODELS_DIR / self.value.split("/")[-1].lower())


_MODELS: dict[LocalModel, tuple] = {}


def get_model(model_choice: LocalModel) -> tuple:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "torch/transformers not installed. "
            "Run: uv pip install -e '.[llm]'  or set USE_API_MODEL = True in config.py"
        )
    if model_choice in _MODELS:
        return _MODELS[model_choice]

    model_path = model_choice.path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python scripts/model_setup.py"
        )

    print(f"[local] Loading {model_choice.name} from {model_path}...")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=False,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.eval()

    _MODELS[model_choice] = (model, tokenizer)
    return model, tokenizer


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


def response(
    user_prompt: str,
    model: LocalModel,
    candidates: list[str],
    new_toks_len: int = 64,
) -> list[str]:
    """Run a single inference call and return cleaned proposed confusibles."""
    model_obj, tokenizer = get_model(model)
    prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model_obj.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model_obj.generate(
            **inputs,
            max_new_tokens=new_toks_len,
            do_sample=False,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return _clean_output(decoded, candidates)
