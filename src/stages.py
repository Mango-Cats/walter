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

import json
from pathlib import Path

import pandas as pd

from config import (
    COL_LABEL,
    COL_X1,
    COL_X2,
    D_CSV,
    D_PHO_CSV,
    DATA_SOURCE,
    DataSource,
    FROM_FILE,
    LLM_OUTPUT_FILENAME,
    LLM_OUTPUT_JSON,
    N,
    NEGATIVE_LABEL,
    P,
    P_INPUT_COLS,
    POSITIVE_PREVALENCE,
    RESULTS_DIR,
    SEED,
    TIER_2_SAMPLE_SIZE,
    U_CSV,
)
from src.adapters.g2p.transcribe import transcribe_all
from src.adapters.phoc import run_phoc_multilingual
from src.artifacts import in_file, require_file, seed_file
from src.pipeline.dataset import assemble_and_save, unselected_candidate_pairs
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


def load_rejections(
    input_dir: Path = RESULTS_DIR,
    source: DataSource = DATA_SOURCE,
    rejected_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Load N, the rejected pairs, for a soft-labelled assembly. The mirror of
    load_positives(), and like it, it never generates anything.

    A rejection has two possible sources and both are read, since either can be
    absent:

      * the LLM's - every candidate an entry in lasa_run.json was shown and did
        not propose. Read whenever FROM_FILE is False, from the same file P
        comes from, so no extra stage or LLM call is involved.
      * a predefined file - rejected_csv when given, otherwise N[source] if it
        happens to exist. An explicitly named file that is missing is an error;
        the configured default simply being absent is not.

    Returns a [COL_X1, COL_X2] DataFrame, empty when neither source yielded
    anything. Callers only reach here under soft labels, so an empty N means
    "nothing was rejected", not "rejections were not asked for".
    """
    frames: list[pd.DataFrame] = []

    n_file = seed_file(rejected_csv, "rejected pairs CSV") if rejected_csv else N[source]
    if n_file.exists():
        pairs = pd.read_csv(n_file)
        missing = [c for c in P_INPUT_COLS if c not in pairs.columns]
        if missing:
            raise ValueError(
                f"{n_file} is missing column(s) {missing}. Predefined rejected "
                f"pairs need {list(P_INPUT_COLS)}."
            )
        frames.append(pairs[list(P_INPUT_COLS)].dropna())
        print(f"[stages] Loaded rejections from {n_file}: {len(frames[-1]):,} pairs")
    else:
        print(f"[stages] No predefined rejections at {n_file}, skipping")

    if not FROM_FILE:
        path = in_file(input_dir, LLM_OUTPUT_FILENAME, "walter propose")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        llm = unselected_candidate_pairs(data, label=NEGATIVE_LABEL)
        frames.append(llm[[COL_X1, COL_X2]])
        print(f"[stages] Loaded rejections from {path}: {len(llm):,} unselected pairs")

    if not frames:
        return pd.DataFrame(columns=list(P_INPUT_COLS))

    N_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    print(f"[stages] N: {len(N_df):,} rejected pairs")
    return N_df.reset_index(drop=True)


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
    N_df: pd.DataFrame | None = None,
    output_csv: Path = D_CSV,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge P, U and (under soft labels) N into D, transcribe, and write it.

    N_df is None for the two-value dataset; pass load_rejections() for the
    three-value one.
    """
    return assemble_and_save(
        pairs_df,
        U,
        N=N_df,
        add_phonemes=True,
        verbose=verbose,
        output_csv=output_csv,
    )


def phoc(
    input_csv: Path = D_CSV,
    output_csv: Path = D_PHO_CSV,
) -> list[str]:
    """Append the phonetic-similarity columns (phoc), then the engineered
    structural/prosodic/Filipino-nativization columns (features.py) on top."""
    require_file(input_csv, "walter assemble")
    feats = run_phoc_multilingual(input_csv, output_csv)
    engineered = run_engineering(output_csv, output_csv)
    return feats + engineered


def featurize(
    input_csv: Path,
    output_csv: Path,
    verbose: bool = True,
) -> list[str]:
    """
    Run the feature half of the pipeline over an already-built pair CSV:
    G2P, then phoc, then the engineered (features.py) columns.

    This is the tail of `all` with the pair-construction head removed - no LLM
    proposal, no predefined positive set, no sampled U, no assembly - for a
    dataset whose pairs already exist. A label column is carried along and
    never rewritten, unlike assemble(), which relabels every row by which of
    P or U it came from.

    Every column other than x_1, x_2 and label is dropped up front and
    rebuilt from scratch, whatever it's named - transcriptions, phoc
    features, old engineered features, unrelated metadata, all of it.
    featurize is for (re)computing features, not for carrying passengers.

    Each step writes its own CSV next to output_csv, named for the input, so a
    failure halfway through leaves the work already paid for on disk:

        <stem>_t.csv      transcriptions
        output_csv        + the phonetic-similarity columns, then the
                            engineered columns on top

    Returns the phonetic and engineered feature columns added, in that order.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    df = pd.read_csv(input_csv)

    missing = [c for c in (COL_X1, COL_X2) if c not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    keep = [c for c in (COL_X1, COL_X2, COL_LABEL) if c in df.columns]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        df = df[keep]
        if verbose:
            print(f"[stages] Overwriting existing columns: {', '.join(dropped)}")

    stage_dir = output_csv.parent
    stage_dir.mkdir(parents=True, exist_ok=True)
    t_csv = stage_dir / f"{input_csv.stem}_t.csv"

    # phoc reads t_csv; if the chosen output collides with it, phoc would
    # overwrite its own input mid-run.
    if output_csv.resolve() == t_csv.resolve():
        raise ValueError(
            f"output {output_csv} collides with the transcription intermediate "
            f"this stage writes for input {input_csv.name}. Pick another "
            f"output name or directory."
        )

    if verbose:
        print(f"[stages] Featurizing {input_csv}: {len(df):,} rows")

    df = transcribe_all(df, skip_existing=False, tag="featurize", verbose=verbose)
    df.to_csv(t_csv, index=False)
    if verbose:
        print(f"[stages] Transcriptions -> {t_csv}")

    feats = run_phoc_multilingual(t_csv, output_csv)
    if verbose:
        print(f"[stages] Phonetic features -> {output_csv}")

    engineered = run_engineering(output_csv, output_csv)
    if verbose:
        print(f"[stages] Engineered features -> {output_csv}")

    return feats + engineered


def summarize(D: pd.DataFrame) -> str:
    """One-line label breakdown for the CLI to print."""
    counts = D[COL_LABEL].value_counts().to_dict()
    return f"{len(D):,} rows  " + "  ".join(
        f"label={k}: {v:,}" for k, v in sorted(counts.items())
    )
