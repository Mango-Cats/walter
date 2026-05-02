import os
from huggingface_hub import snapshot_download

MODELS_TO_DOWNLOAD = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ibm-granite/granite-4.1-8b",
]

BASE_DIR = "./models"


def setup_local_models():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"<walter> Created directory: {BASE_DIR}")

    for repo_id in MODELS_TO_DOWNLOAD:
        folder_name = repo_id.split("/")[-1].lower()
        target_path = os.path.join(BASE_DIR, folder_name)

        if os.path.exists(target_path) and any(os.scandir(target_path)):
            print(f"<walter> Skipping {folder_name} (Already exists)")
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
            if "meta-llama" in repo_id:
                print(
                    "<walter> Note: Llama models require 'huggingface-cli login' and approved access."
                )


if __name__ == "__main__":
    setup_local_models()
