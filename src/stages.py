"""
Separately runnable stages of the Walter pipeline.

This module coordinates each major step of building LASA drug datasets:
    - preprocess: Cleans raw drug names from national registries.
    - propose: Uses an AI model to find new confusable drug pairs (P).
    - noise: Samples similar non-confusable drug pairs (U).
    - assemble: Combines pairs and adds spoken pronunciations (D).
    - phoc: Adds phonetic and orthographic similarity scores (D_pho).
    - featurize: Adds pronunciations and similarity scores to an existing pair CSV.
    - relabel: Applies reviewed label changes to a dataset's pairs.
    - lexicon: Scores LASA names against all held-out names, for a ranking test.
"""

import json
from pathlib import Path

import pandas as pd

from config import (
    POSITIVE_LABEL,
    NOISE_OVERSAMPLE,
    REGISTRY_COL,
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
from src.adapters.phoc import run_core_features, run_phoc_multilingual, score_lexicon
from src.artifacts import in_file, require_file, seed_file
from src.pipeline.core_name import is_variant_pair
from src.pipeline.dataset import assemble_and_save, unselected_candidate_pairs
from src.pipeline.features import run_engineering
from src.pipeline.noise import (
    get_positive_pairs,
    make_noise,
    noise_report,
    normalize,
    random_registry_pairs,
)
from src.pipeline.preprocessing import clean_registry
from src.pipeline.preprocessing import run as run_preprocessing


def preprocess(source: DataSource = DATA_SOURCE) -> pd.DataFrame:
    """Clean the drug registry, or load the cached clean copy.

    Args:
        source: The drug registry data source (PH or US).

    Returns:
        A DataFrame containing cleaned, unique drug names.
    """
    return run_preprocessing(source=source)


def propose(
    registry_df: pd.DataFrame,
    seed_csv: Path,
    output_path: Path = LLM_OUTPUT_JSON,
) -> Path:
    """Augment predefined confusable drug pairs using an AI model.

    Args:
        registry_df: Cleaned drug registry DataFrame.
        seed_csv: CSV file containing predefined confusable drug pairs.
        output_path: Output file path for the AI proposals JSON.

    Returns:
        Path to the saved JSON file.
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
    """Load confirmed confusable drug pairs (P) from disk without generating them.

    Args:
        input_dir: Directory containing previous proposal outputs.
        source: Active data source (PH or US).

    Returns:
        A DataFrame containing confirmed confusable pairs.

    Raises:
        FileNotFoundError: If the required pairs file or proposal output does not exist.
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
    """Load rejected drug pairs (N) for soft-label dataset creation.

    Args:
        input_dir: Directory containing proposer JSON outputs.
        source: Active data source (PH or US).
        rejected_csv: Optional path to a CSV file of predefined rejected pairs.

    Returns:
        A DataFrame containing rejected pairs with columns x_1 and x_2.

    Raises:
        ValueError: If the predefined rejected CSV is missing required columns.
    """
    frames: list[pd.DataFrame] = []

    n_file = (
        seed_file(rejected_csv, "rejected pairs CSV") if rejected_csv else N[source]
    )
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


def load_registry_file(path: Path) -> pd.DataFrame:
    """Load and clean a one-column registry CSV without using the registry cache.

    Args:
        path: CSV whose first column holds drug names.

    Returns:
        A cleaned single-column DataFrame of unique drug names.
    """
    registry = pd.read_csv(path, usecols=[0], dtype=str)
    registry.columns = [REGISTRY_COL]
    return clean_registry(registry.dropna())


def load_labelled(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a preconstructed CSV of positive and rejected pairs.

    Args:
        path: CSV with x_1, x_2, and label (or y), where the label is 1 or -1.

    Returns:
        The confirmed pairs (P) and the rejected pairs (N).

    Raises:
        ValueError: If a column is missing or a label is not 1 or -1.
    """
    labelled = pd.read_csv(path, keep_default_na=False).rename(columns={"y": COL_LABEL})
    missing = [c for c in (COL_X1, COL_X2, COL_LABEL) if c not in labelled.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    bad = set(labelled[COL_LABEL]) - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if bad:
        raise ValueError(f"{path} has labels other than 1 and -1: {sorted(bad)}")
    columns = [COL_X1, COL_X2]
    positives = labelled[labelled[COL_LABEL] == POSITIVE_LABEL][columns]
    rejected = labelled[labelled[COL_LABEL] == NEGATIVE_LABEL][columns]
    return positives.reset_index(drop=True), rejected.reset_index(drop=True)


def _pair_keys(df: pd.DataFrame) -> list[frozenset]:
    """Key each pair regardless of name order.

    Args:
        df: DataFrame with x_1 and x_2.

    Returns:
        One normalized key per row.
    """
    return [
        frozenset([normalize(a), normalize(b)]) for a, b in zip(df[COL_X1], df[COL_X2])
    ]


def noise(
    pairs_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    output_path: Path | None = U_CSV,
    target: int | None = None,
    seed: int | None = None,
    rejected_df: pd.DataFrame | None = None,
    random_share: float = 0.0,
    skip_variants: bool = False,
) -> pd.DataFrame:
    """Sample non-confusable drug pairs (U) and optionally save a checkpoint.

    With a target, the sampler over-generates, any pair already labelled is removed,
    and the rest is sampled down to exactly that many pairs. A random share of that
    target comes from unfiltered random registry pairs in place of similar ones.

    Args:
        pairs_df: Confirmed confusable drug pairs.
        registry_df: Cleaned drug registry DataFrame.
        output_path: Optional CSV destination for saving sampled pairs.
        target: Exact number of pairs wanted. Defaults to the configured prevalence.
        seed: Random seed. Defaults to the configured seed.
        rejected_df: Rejected pairs, which may not appear in the sample.
        random_share: Fraction of the target drawn as unfiltered random pairs.
        skip_variants: Leave out similar pairs that differ only in strength, salt,
            form, or qualifier.

    Returns:
        A DataFrame of sampled non-confusable drug pairs.

    Raises:
        ValueError: If fewer than target pairs remain after filtering.
    """
    seed = SEED if seed is None else seed
    prevalence = POSITIVE_PREVALENCE
    if target is not None:
        n_positive = len(get_positive_pairs(pairs_df))
        requested = target * NOISE_OVERSAMPLE
        prevalence = n_positive / (n_positive + requested)
    U = make_noise(
        pairs_df=pairs_df,
        registry_df=registry_df,
        positive_prevalence=prevalence,
        tier_2_sample_size=TIER_2_SAMPLE_SIZE,
        seed=seed,
    )
    if target is not None:
        labelled = get_positive_pairs(pairs_df)
        if rejected_df is not None:
            labelled = labelled | get_positive_pairs(rejected_df)
        keys = [
            frozenset([normalize(a), normalize(b)])
            for a, b in zip(U[COL_X1], U[COL_X2])
        ]
        U = U[[key not in labelled for key in keys]]
        if skip_variants:
            variant = [is_variant_pair(a, b) for a, b in zip(U[COL_X1], U[COL_X2])]
            print(f"[stages] Skipped {sum(variant):,} variant pairs")
            U = U[[not v for v in variant]]
        if len(U) < target:
            raise ValueError(
                f"only {len(U):,} of the {target:,} pairs needed remain after "
                "removing labelled pairs."
            )
        n_random = round(target * random_share)
        U = U.sample(target - n_random, random_state=seed)
        random_pairs = random_registry_pairs(registry_df, n_random, labelled, seed)
        U = pd.concat([U, random_pairs], ignore_index=True)
        U = U[~pd.Series(_pair_keys(U)).duplicated().to_numpy()]
        if len(U) != target:
            raise ValueError(f"expected {target:,} noise pairs, found {len(U):,}.")
        U = U.sample(frac=1, random_state=seed).reset_index(drop=True)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        U.to_csv(output_path, index=False)
        print(f"[stages] U -> {output_path}")
        report = noise_report(U, pairs_df)
        output_path.with_name(f"{output_path.stem}_report.txt").write_text(
            report + "\n"
        )
        print(report)
    return U


def load_noise(input_path: Path = U_CSV) -> pd.DataFrame:
    """Load previously sampled non-confusable drug pairs (U) from disk.

    Args:
        input_path: Path to the sampled U.csv file.

    Returns:
        A DataFrame of sampled non-confusable drug pairs.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
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
    """Merge confusable, non-confusable, and rejected pairs into the final dataset (D).

    Args:
        pairs_df: Confirmed confusable drug pairs (P).
        U: Sampled non-confusable drug pairs (U).
        N_df: Optional rejected pairs for soft labeling (N).
        output_csv: Path to save the assembled dataset (D.csv).
        verbose: Whether to print progress messages.

    Returns:
        The assembled dataset DataFrame.
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
    """Calculate phonetic and engineered similarity features for assembled pairs.

    Args:
        input_csv: Path to the assembled dataset CSV (D.csv).
        output_csv: Path to save the dataset with added features (D_pho.csv).

    Returns:
        A list of newly added feature column names.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
    """
    require_file(input_csv, "walter assemble")
    feats = run_phoc_multilingual(input_csv, output_csv)
    engineered = run_engineering(output_csv, output_csv)
    core = run_core_features(output_csv, output_csv)
    return feats + engineered + core


def relabel(
    input_csv: Path,
    changes_csv: Path,
    output_csv: Path,
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply reviewed label changes to a dataset and write its pairs.

    Only x_1, x_2, and label are written, ready for `walter featurize`. A label column
    named y is read as label.

    With allow_missing, the changes may come from a different version of the dataset:
    removing an absent pair does nothing, relabelling an absent pair adds it, and adding
    a present pair relabels it.

    Args:
        input_csv: Dataset CSV with x_1, x_2, and label.
        changes_csv: CSV with action (remove, relabel, or add), x_1, x_2, and new_label.
        output_csv: Path to write the changed pairs.
        allow_missing: Whether changes may refer to pairs that do not fit the dataset.

    Returns:
        The changed pairs, and each change with what happened to it.

    Raises:
        ValueError: If a change does not fit the dataset and allow_missing is False.
    """
    columns = [COL_X1, COL_X2, COL_LABEL]
    pairs = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    pairs = pairs.rename(columns={"y": COL_LABEL})[columns]
    changes = pd.read_csv(changes_csv, dtype=str, keep_default_na=False)
    keys = [frozenset(p) for p in zip(pairs[COL_X1], pairs[COL_X2])]
    labels = dict(zip(keys, pairs[COL_LABEL]))
    names = dict(zip(keys, zip(pairs[COL_X1], pairs[COL_X2])))

    outcomes = []
    for row in changes.itertuples():
        key = frozenset([getattr(row, COL_X1), getattr(row, COL_X2)])
        present = key in labels
        if not allow_missing and present == (row.action == "add"):
            raise ValueError(f"{row.action} does not fit the dataset: {sorted(key)}")
        if row.action == "remove":
            outcome = "removed" if present else "already absent"
            labels.pop(key, None)
            names.pop(key, None)
        else:
            if not present:
                outcome = "added"
                names[key] = (getattr(row, COL_X1), getattr(row, COL_X2))
            elif labels[key] == row.new_label:
                outcome = "already correct"
            else:
                outcome = f"relabelled from {labels[key]}"
            labels[key] = row.new_label
        outcomes.append(outcome)

    out = pd.DataFrame(
        [(*names[key], label) for key, label in labels.items()], columns=columns
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out, changes.assign(outcome=outcomes)


def lexicon(input_csv: Path, output_path: Path) -> list[str]:
    """Score every LASA name against all names in a held-out pair CSV.

    Args:
        input_csv: CSV with x_1, x_2, label, and transcription columns.
        output_path: Path to write the Parquet pool.

    Returns:
        A list of feature column names in the pool.

    """
    return score_lexicon(Path(input_csv), Path(output_path))


def featurize(
    input_csv: Path,
    output_csv: Path,
    verbose: bool = True,
) -> list[str]:
    """Add pronunciations and similarity scores to an existing drug-pair CSV.

    Args:
        input_csv: Path to an existing CSV file containing drug pairs (x_1 and x_2).
        output_csv: Path to write the feature-enriched CSV.
        verbose: Whether to print progress messages.

    Returns:
        A list of added phonetic and engineered feature column names.

    Raises:
        ValueError: If required columns are missing or if output collides with intermediates.
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

    core = run_core_features(output_csv, output_csv)
    if verbose:
        print(f"[stages] Core features -> {output_csv}")

    return feats + engineered + core


def summarize(D: pd.DataFrame) -> str:
    """Return a summary string showing row counts broken down by label.

    Args:
        D: Assembled dataset DataFrame containing a 'label' column.

    Returns:
        A formatted string showing total row counts and counts per label.
    """
    counts = D[COL_LABEL].value_counts().to_dict()
    return f"{len(D):,} rows  " + "  ".join(
        f"label={k}: {v:,}" for k, v in sorted(counts.items())
    )
