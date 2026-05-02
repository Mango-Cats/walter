from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    local_dir="./models/deepseek",
    local_dir_use_symlinks=False
)