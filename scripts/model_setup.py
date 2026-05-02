import os
from huggingface_hub import snapshot_download
from src.proposer.local import LocalModel

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

MODELS_TO_DOWNLOAD = [model for model in LocalModel]
BASE_DIR = "./models"

BLUE = "\033[94m"
RESET = "\033[0m"
PREFIX = f"{BLUE}<scripts.model_setup>{RESET}"


def log(msg: str):
    print(f"{PREFIX} {msg}")


def setup_local_models():
    os.makedirs(BASE_DIR, exist_ok=True)

    for model in MODELS_TO_DOWNLOAD:
        repo_id = model.value
        target_path = model.path

        log(f"Preparing model: {repo_id}")

        if os.path.exists(target_path) and any(os.scandir(target_path)):
            log(f"Skipping: {repo_id}")
            continue

        try:
            log(f"Downloading: {repo_id}")

            snapshot_download(
                repo_id=repo_id,
                local_dir=target_path,
                max_workers=4,
                token=os.environ.get("HF_TOKEN"),
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )

            log(f"Done: {repo_id}")

        except Exception as e:
            log(f"Failed: {repo_id} | {e}")


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")

    if token:
        log(f"HF token detected (length={len(token)})")
    else:
        log("HF token not found (anonymous access may fail for private repos)")

    setup_local_models()
