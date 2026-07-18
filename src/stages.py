"""
The pipeline as separately runnable stages.

Each stage reads its input from disk and writes its output there, so a run
can start from any point. The stages differ enormously in cost: proposing P
is 400 LLM calls and G2P transcribes the whole registry, while sampling U is
cheap. Retuning a sampling knob should not re-pay for the proposal.

Functions here take and return DataFrames and never parse arguments. walter.py
owns the CLI and decides which paths to hand them.

load_positives() is deliberately separate from propose(): loading a previous
proposal must never trigger a new one by accident.
"""

from pathlib import Path

import pandas as pd

from config import (
    COL_LABEL,
    D_CSV,
    D_ENGI_CSV,
    D_PHO_CSV,
    DATA_SOURCE,
    DataSource,
    FROM_FILE,
    LLM_OUTPUT_JSON,
    P,
    POSITIVE_PREVALENCE,
    SEED,
    TIER_2_SAMPLE_SIZE,
    U_CSV,
)
from src.dataset import assemble_and_save
from src.feature_engineering import run_engineering
from src.noise import make_noise
from src.phoc_runner import run_phoc_multilingual
from src.preprocessing import run as run_preprocessing


def preprocess(source: DataSource = DATA_SOURCE) -> pd.DataFrame:
    """Clean the drug registry R, or load the cached clean copy."""
    return run_preprocessing(source=source)


def propose(
    registry_df: pd.DataFrame,
    output_path: Path = LLM_OUTPUT_JSON,
) -> Path:
    """
    Generate confirmed LASA pairs with the LLM proposer and write them.

    Imports are local because the LLM extras are optional; a run that only
    touches later stages should not need transformers or openai installed.
    """
    from src.proposer.inference import run_inference
    from src.proposer.llm import LocalModel

    run_inference(
        registry_df=registry_df,
        model_choice=LocalModel.QWEN3_1_7B,
        output_path=output_path,
    )
    return output_path


def load_positives(source: DataSource = DATA_SOURCE) -> pd.DataFrame:
    """
    Load P from wherever config says it lives, without ever generating it.

    Raises with the command to run when the artifact is absent, since the
    fix differs: a missing CSV is the user's to supply, a missing proposal
    means `walter propose` has not run yet.
    """
    if FROM_FILE:
        p_file: Path = P[source]
        if not p_file.exists():
            raise FileNotFoundError(
                f"{p_file} not found. Place your confirmed LASA pairs CSV "
                f"there, or set FROM_FILE = False in config to generate them "
                f"with `walter propose`."
            )
        pairs = pd.read_csv(p_file)
        print(f"[stages] Loaded P from {p_file}: {len(pairs):,} pairs")
        return pairs

    from src.proposer.inference import load_inference

    if not Path(LLM_OUTPUT_JSON).exists():
        raise FileNotFoundError(
            f"{LLM_OUTPUT_JSON} not found. Run `walter propose` first, or set "
            f"FROM_FILE = True in config to read P from a CSV instead."
        )
    pairs = load_inference(LLM_OUTPUT_JSON)
    print(f"[stages] Loaded P from {LLM_OUTPUT_JSON}: {len(pairs):,} pairs")
    return pairs


def noise(
    pairs_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    output_path: Path | None = U_CSV,
) -> pd.DataFrame:
    """Sample the unlabeled set U. Writes a checkpoint unless output_path is None."""
    U = make_noise(
        pairs_df=pairs_df,
        registry_df=registry_df,
        positive_prevalence=POSITIVE_PREVALENCE,
        tier_2_sample_size=TIER_2_SAMPLE_SIZE,
        seed=SEED,
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        U.to_csv(output_path, index=False)
        print(f"[stages] U -> {output_path}")
    return U


def load_noise(input_path: Path = U_CSV) -> pd.DataFrame:
    """Load a previously sampled U."""
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run `walter noise` first."
        )
    U = pd.read_csv(input_path)
    print(f"[stages] Loaded U from {input_path}: {len(U):,} pairs")
    return U


def assemble(
    pairs_df: pd.DataFrame,
    U: pd.DataFrame,
    output_csv: Path = D_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge P and U into D, transcribe, and write it."""
    return assemble_and_save(
        pairs_df, U, add_phonemes=True, verbose=verbose, output_csv=output_csv
    )


def phoc(
    input_csv: Path = D_CSV,
    output_csv: Path = D_PHO_CSV,
) -> list[str]:
    """Append the phonetic-similarity feature columns."""
    _require(input_csv, "walter assemble")
    return run_phoc_multilingual(input_csv, output_csv)


def engineer(
    input_csv: Path = D_PHO_CSV,
    output_csv: Path = D_ENGI_CSV,
) -> list[str]:
    """Append the orthographic META_FEATURES."""
    _require(input_csv, "walter phoc")
    return run_engineering(input_csv, output_csv)


def summarize(D: pd.DataFrame) -> str:
    """One-line label breakdown for the CLI to print."""
    counts = D[COL_LABEL].value_counts().to_dict()
    return f"{len(D):,} rows  " + "  ".join(
        f"label={k}: {v:,}" for k, v in sorted(counts.items())
    )


def _require(path: Path, produced_by: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `{produced_by}` first.")
