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


def get_pipeline(model_choice: LocalModel):
    if model_choice in _PIPELINES:
        return _PIPELINES[model_choice]

    model_path = MODEL_REGISTRY[model_choice]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model folder not found at {model_path}. Did you download it?"
        )

    print(f"Loading tokenizer for {model_path}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    # The code below checks your hardware, I added this since I
    # (zhean) don't have a GPU but I have an NPU.
    if torch.cuda.is_available():
        print("Hardware detected: NVIDIA GPU (CUDA). Loading model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    # NPU
    else:
        npu_loaded = False
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            from openvino.runtime import Core

            core = Core()
            if "NPU" in core.available_devices:
                print("Hardware detected: Intel NPU. Compiling via OpenVINO...")
                model = OVModelForCausalLM.from_pretrained(
                    model_path,
                    export=True,
                    device="NPU"
                )
                pipe = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                npu_loaded = True
        except Exception as e:
            print(f"OpenVINO/NPU setup skipped: {e}")

        # CPU Fallback
        if not npu_loaded:
            print("No accelerator detected. Falling back to standard CPU...")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
            pipe = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )

    _PIPELINES[model_choice] = pipe
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