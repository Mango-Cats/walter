"""
walter:
    LLM-assisted dataset construction for LASA drugs.

Usage:
    python walter.py
"""

import pandas as pd
from pandas import DataFrame

from config import (
    POSITIVE_PREVALENCE,
    DataSource,
    DATA_SOURCE,
    FROM_FILE,
    P,
    D_OUT_CSV,
    D_PHO_OUT_CSV,
    D_ENGI_OUT_CSV,
    SEED,
    TIER_2_SAMPLE_SIZE,
)

from src.dataset import assemble_and_save
from src.phoc_runner import run_phoc_multilingual
from src.feature_engineering import run_engineering
import src.noise as noise
import src.preprocessing as pre
from pathlib import Path
import time

from rich.console import Console

_console = Console()


class Spinner:
    """Per-stage loading indicator, backed by rich's Console.status()."""

    def __init__(self, label: str):
        self.label = label
        self._status = _console.status(f"[bold cyan]{label}...", spinner="dots")
        self._start = 0.0

    def __enter__(self) -> "Spinner":
        self._start = time.monotonic()
        self._status.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._status.__exit__(exc_type, exc, tb)
        elapsed = time.monotonic() - self._start
        if exc_type is None:
            _console.print(f"[bold green]✓[/] {self.label} done in {elapsed:.1f}s")
        else:
            _console.print(f"[bold red]✗[/] {self.label} FAILED in {elapsed:.1f}s")


def _load_positive_pairs(R_clean: pd.DataFrame, source: DataSource) -> pd.DataFrame:
    if FROM_FILE:
        P_file: Path = P[source]
        if not P_file.exists():
            raise FileNotFoundError(
                f"{P_file} not found. "
                "Place your confirmed LASA pairs CSV there, "
                "or set FROM_FILE = False in config.py to use the LLM proposer."
            )
        P_ = pd.read_csv(P_file)
        print(f"\nLoaded P from {P_file}: {len(P):,} pairs")
        return P_

    from config import LLM_OUTPUT_JSON
    from src.proposer.inference import load_inference, run_inference
    from src.proposer.llm import LocalModel

    run_inference(registry_df=R_clean, model_choice=LocalModel.QWEN3_1_7B)
    P_llm = load_inference(LLM_OUTPUT_JSON)
    print(f"\nGenerated P via LLM: {len(P_llm):,} pairs")
    return P_llm


def main() -> None:
    print(f"Data source    : {DATA_SOURCE.name}")
    print(f"Pos. prevalence: {POSITIVE_PREVALENCE:.6f}")
    print(f"Tier 2 sample  : {TIER_2_SAMPLE_SIZE:,}")
    print(f"Seed           : {SEED}")
    print()
    print(f"Output: D → {D_OUT_CSV}")

    # --- Preprocessing ---
    with Spinner("Preprocessing registry"):
        R_clean: DataFrame = pre.run(source=DATA_SOURCE)
    print(f"\nCleaned registry: {len(R_clean):,} drug names")
    print(R_clean.head(10))

    # --- Confirmed LASA pairs (P) ---
    with Spinner("Loading confirmed pairs (P)"):
        P_load: pd.DataFrame = _load_positive_pairs(R_clean, source=DATA_SOURCE)
    print(P_load.head())
    print("Columns:", list(P_load.columns))

    # --- Unlabeled pairs (U) ---
    with Spinner("Sampling unlabeled pairs (U)"):
        U = noise.make_noise(
            pairs_df=P_load,
            registry_df=R_clean,
            positive_prevalence=POSITIVE_PREVALENCE,
            tier_2_sample_size=TIER_2_SAMPLE_SIZE,
            seed=SEED,
        )
    print(U.head())

    # --- Assemble and save ---
    with Spinner("Assembling and saving D"):
        D = assemble_and_save(P_load, U, add_phonemes=True)
    print(D.head(10))
    print(f"\nD shape: {D.shape}")
    print(D["label"].value_counts().to_string())

    # --- Phonetic features (phoc) ---
    # D.csv already exists on disk at this point, so a phoc failure aborts
    # the run but leaves the base dataset intact.
    with Spinner("Adding phonetic features (phoc)"):
        feats = run_phoc_multilingual(D_OUT_CSV, D_PHO_OUT_CSV)
    print(f"\nPhonetic features ({len(feats)}): {', '.join(feats)}")
    print(f"  D_pho → {D_PHO_OUT_CSV}")

    # --- Feature engineering (META_FEATURES) ---
    # Engineers the string / edit-distance META_FEATURES onto the phoc
    # output, so _engi = engineered and pho'd (every phonetic column plus
    # the META_FEATURES). D_pho stays intact on disk.
    with Spinner("Engineering meta-features"):
        meta = run_engineering(D_PHO_OUT_CSV, D_ENGI_OUT_CSV)
    print(f"\nMeta-features ({len(meta)}): {', '.join(meta)}")
    print(f"  D_engi → {D_ENGI_OUT_CSV}")


if __name__ == "__main__":
    main()
