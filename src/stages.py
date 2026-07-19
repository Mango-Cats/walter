"""
The pipeline as separately runnable stages.

Each stage reads its input from disk and writes its output there, so a run
can start from any point. The stages differ enormously in cost: proposing P
is 400 LLM calls and G2P transcribes the whole registry, while sampling U is
cheap. Retuning a sampling knob should not re-pay for the proposal.

Functions here take and return DataFrames and never parse arguments. walter.py
owns the CLI, resolves each stage's directory to the canonical filename inside
it (see src/artifacts.py), and hands the resulting paths down.

This module is the seam between the layers: it is the only place that composes
src/pipeline, src/proposer and src/adapters into something runnable, so a stage
can be reordered or re-pointed here without any of them knowing.

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
    LLM_OUTPUT_FILENAME,
    LLM_OUTPUT_JSON,
    P,
    POSITIVE_PREVALENCE,
    RESULTS_DIR,
    SEED,
    TIER_2_SAMPLE_SIZE,
    U_CSV,
)
from src.adapters.phoc import run_phoc_multilingual
from src.artifacts import in_file, require_file
from src.pipeline.dataset import assemble_and_save
from src.pipeline.features import run_engineering
from src.pipeline.noise import make_noise
from src.pipeline.preprocessing import run as run_preprocessing


def preprocess(source: DataSource = DATA_SOURCE) -> pd.DataFrame:
    """Clean the drug registry R, or load the cached clean copy."""
    return run_preprocessing(source=source)


def propose(
    registry_df: pd.DataFrame,
    seed_csv: Path,
    output_path: Path = LLM_OUTPUT_JSON,
) -> Path:
    """
    Augment the predefined LASA pairs in seed_csv with the LLM proposer.

    seed_csv is a file, not a directory: it is user-supplied input that no
    other stage produces.

    Imports are local because the LLM extras are optional; a run that only
    touches later stages should not need transformers or openai installed.
    """
    from src.adapters.llm.local import LocalModel
    from src.proposer.inference import load_seed_pairs, run_inference

    seed_pairs = load_seed_pairs(seed_csv)
    print(f"[stages] Seeding proposer from {seed_csv}: {len(seed_pairs):,} pairs")

    run_inference(
        registry_df=registry_df,
        model_choice=LocalModel.QWEN3_1_7B,
        seed_pairs=seed_pairs,
        output_path=output_path,
    )
    return output_path


def load_positives(
    input_dir: Path = RESULTS_DIR,
    source: DataSource = DATA_SOURCE,
) -> pd.DataFrame:
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
                f"there, or set FROM_FILE = False in config to augment them "
                f"with `walter propose`."
            )
        pairs = pd.read_csv(p_file)
        print(f"[stages] Loaded P from {p_file}: {len(pairs):,} pairs")
        return pairs

    from src.proposer.inference import load_inference

    path = in_file(input_dir, LLM_OUTPUT_FILENAME, "walter propose")
    pairs = load_inference(path)
    print(f"[stages] Loaded P from {path}: {len(pairs):,} pairs")
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
    require_file(input_path, "walter noise")
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
    require_file(input_csv, "walter assemble")
    return run_phoc_multilingual(input_csv, output_csv)


def engineer(
    input_csv: Path = D_PHO_CSV,
    output_csv: Path = D_ENGI_CSV,
) -> list[str]:
    """Append the orthographic META_FEATURES."""
    require_file(input_csv, "walter phoc")
    return run_engineering(input_csv, output_csv)


def summarize(D: pd.DataFrame) -> str:
    """One-line label breakdown for the CLI to print."""
    counts = D[COL_LABEL].value_counts().to_dict()
    return f"{len(D):,} rows  " + "  ".join(
        f"label={k}: {v:,}" for k, v in sorted(counts.items())
    )
