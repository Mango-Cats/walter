import os
from huggingface_hub import snapshot_download

MODELS_TO_DOWNLOAD = {
    # "deepseek-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

BASE_DIR = "./models"


def setup_local_models():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"<walter> Created directory: {BASE_DIR}")

    for folder_name, repo_id in MODELS_TO_DOWNLOAD.items():
        target_path = os.path.join(BASE_DIR, folder_name)

        if os.path.exists(target_path) and any(os.scandir(target_path)):
            print(f"<walter> Skipping {folder_name} (Already exists in {target_path})")
            continue

        print(f"\n<walter> Preparing to download: {repo_id}")
        print(f"<walter> Destination: {target_path}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_path,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            print(f"<walter> Successfully downloaded {folder_name}!")
        except Exception as e:
            print(f"<walter> Error downloading {repo_id}: {e}")
            print(
                "<walter> Make sure you are logged in via 'huggingface-cli login' for gated models."
            )


if __name__ == "__main__":
    setup_local_models()
