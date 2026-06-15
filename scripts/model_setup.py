"""
Downloads all LocalModel weights into config.MODELS_DIR.
Run once before using FROM_FILE = False in the notebook.

Usage:
    HF_TOKEN=<your_token> python scripts/model_setup.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download

from config import MODELS_DIR
from src.proposer.llm import LocalModel

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

BLUE = "\033[94m"
RESET = "\033[0m"


def log(msg: str) -> None:
    print(f"{BLUE}[model_setup]{RESET} {msg}")


def download_all() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    if token:
        log(f"HF token detected (length={len(token)})")
    else:
        log("No HF_TOKEN found — anonymous access only (may fail for private repos)")

    for model in LocalModel:
        repo_id = model.value
        target_path = model.path

        log(f"Preparing: {repo_id}")

        if os.path.exists(target_path) and any(os.scandir(target_path)):
            log(f"Already present, skipping: {repo_id}")
            continue

        try:
            log(f"Downloading: {repo_id} → {target_path}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_path,
                max_workers=4,
                token=token,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            log(f"Done: {repo_id}")
        except Exception as exc:
            log(f"FAILED: {repo_id} | {exc}")


if __name__ == "__main__":
    download_all()
