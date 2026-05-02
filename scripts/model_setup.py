import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen3-4B-Instruct-2507",
    # "Qwen/Qwen3-8B-Instruct",
]

BASE_DIR = "./models"


def setup_local_models():
    os.makedirs(BASE_DIR, exist_ok=True)

    for repo_id in MODELS_TO_DOWNLOAD:
        folder_name = repo_id.split("/")[-1].lower()
        target_path = os.path.join(BASE_DIR, folder_name)

        if os.path.exists(target_path) and any(os.scandir(target_path)):
            print(f"Skipping {folder_name}")
            continue

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_path,
                resume_download=True,
                max_workers=4,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            print(f"Done: {folder_name}")

        except Exception as e:
            print(f"Failed {repo_id}: {e}")


if __name__ == "__main__":
    setup_local_models()