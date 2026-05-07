import pandas as pd
import unicodedata
import re
from rapidfuzz import fuzz

from src.noise import (
    normalize_name,
    make_noise,
    LABEL_COL,
    TARGET_COL,
    POSITIVE_LABEL,
    DEFAULT_UNLABELED_TO_POSITIVE_RATIO,
    SIMILARITY_THRESHOLD,
    TIER_2_SAMPLE_SIZE,
)


def clean_drug_name(name: str) -> str:
    """
    Comprehensive drug name cleaning:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    - Remove diacritics (accents, umlauts, etc.)
    - Normalize punctuation (hyphens, apostrophes, slashes → spaces)
    - Keep digits (B12, etc.)
    """
    if not isinstance(name, str):
        return ""

    name = name.lower().strip()

    name = re.sub(r"\s+", " ", name)

    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    name = re.sub(r"[-'/]", " ", name)

    name = re.sub(r"\s+", " ", name).strip()

    return name


def clean_and_deduplicate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans drug names and deduplicates pairs:
    - Cleans Brand Name and Confusible columns using clean_drug_name
    - Canonicalizes pairs to sorted order (A, B) vs (B, A)
    - Removes self-pairs (X, X)
    - Deduplicates based on cleaned canonical pairs
    - Updates the original columns with cleaned names

    Returns cleaned, deduplicated DataFrame.
    """
    df = df.copy()

    df[TARGET_COL] = df[TARGET_COL].apply(clean_drug_name)
    df[LABEL_COL] = df[LABEL_COL].apply(clean_drug_name)

    initial_count = len(df)
    df = df[df[TARGET_COL] != df[LABEL_COL]]
    self_pair_count = initial_count - len(df)
    if self_pair_count > 0:
        print(f"<walter> Removed {self_pair_count} self-pairs")

    df["_canonical_pair"] = df.apply(
        lambda r: tuple(sorted([r[TARGET_COL], r[LABEL_COL]])), axis=1
    )

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_canonical_pair"], keep="first")
    dedup_count = before_dedup - len(df)
    if dedup_count > 0:
        print(f"<walter> Removed {dedup_count} duplicate pairs")

    df = df.drop(columns=["_canonical_pair"])

    return df.reset_index(drop=True)


def build_pu_dataset(
    true_df: pd.DataFrame,
    full_registry: pd.DataFrame,
    ratio: int = DEFAULT_UNLABELED_TO_POSITIVE_RATIO,
    similarity_threshold: int = SIMILARITY_THRESHOLD,
    tier_2_sample_size: int = TIER_2_SAMPLE_SIZE,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Builds the full PU training dataset by combining P and U.

    P rows get label = 1 (known positive).
    U rows get label = 0 (unlabeled — NOT confirmed negative).

    The label column is named <LABEL_COL>_label to avoid confusion with
    the drug name in LABEL_COL.

    Args:
        true_df:        Confirmed LASA pairs [TARGET_COL, LABEL_COL].
        full_registry:  FDA registry [TARGET_COL].
        ratio:          |U| / |P| target ratio.
        similarity_threshold: Minimum similarity for inclusion in U.
        tier_2_sample_size:   Pre-sample size for Tier 2.
        seed:           Random seed.

    Returns:
        DataFrame with columns:
          TARGET_COL, LABEL_COL, similarity, tier, <LABEL_COL>_label
        where <LABEL_COL>_label is 1 for P rows and 0 for U rows.
    """
    p_df = true_df.copy()
    p_df["similarity"] = p_df.apply(
        lambda r: fuzz.WRatio(
            normalize_name(r[TARGET_COL]), normalize_name(r[LABEL_COL])
        ),
        axis=1,
    )
    p_df["tier"] = 0
    p_df[LABEL_COL + "_label"] = POSITIVE_LABEL

    u_df = make_noise(
        true_df=true_df,
        full_registry=full_registry,
        ratio=ratio,
        similarity_threshold=similarity_threshold,
        tier_2_sample_size=tier_2_sample_size,
        seed=seed,
    )

    pu_df = pd.concat([p_df, u_df], ignore_index=True)

    print(f"\n<walter> P size:     {len(p_df):,}")
    print(f"<walter> U size:     {len(u_df):,}")
    print(f"<walter> Total PU:   {len(pu_df):,}")

    return pu_df


def prepare_and_save_datasets(
    P: pd.DataFrame,
    U: pd.DataFrame,
    file_path,
    random_state=67,
    file_prefix="walter",
):
    """
    Cleans, deduplicates, combines, and shuffles P and U dataframes,
    then saves the combined dataset to CSV and Parquet formats.

    If U has a 'Confusible_label' column from the noise module,
    it will be renamed to 'label' for consistency.

    Removes auxiliary columns (similarity, tier) before saving.

    Cleaning includes:
    - Case and whitespace normalization
    - Diacritics and punctuation normalization
    - Pair canonicalization and deduplication
    - Self-pair removal
    """

    P = P.copy()
    U = U.copy()

    # Check if U already has a Confusible_label column (from make_noise)
    if "Confusible_label" in U.columns:
        U = U.rename(columns={"Confusible_label": "label"})

    # Add label column if not already present
    if "label" not in P.columns:
        P["label"] = 1
    if "label" not in U.columns:
        U["label"] = 0

    print("\n<walter> Combining P and U...")
    df_combined = pd.concat([P, U], ignore_index=True)
    print(
        f"<walter> Combined size before cleaning: {len(df_combined):,}"
    )

    print("\n<walter> Cleaning and deduplicating dataset...")
    df_combined = clean_and_deduplicate_dataset(df_combined)
    print(
        f"<walter> Combined size after cleaning: {len(df_combined):,}"
    )

    cols_to_drop = [col for col in ["similarity", "tier"] if col in df_combined.columns]
    if cols_to_drop:
        df_combined = df_combined.drop(columns=cols_to_drop)
        print(f"<walter> Removed auxiliary columns: {cols_to_drop}")

    df_combined = df_combined.sample(frac=1, random_state=random_state)

    dataset_csv = f"{file_path}{file_prefix}.csv"
    dataset_pq = f"{file_path}{file_prefix}.parquet"

    df_combined.to_csv(dataset_csv, index=False)
    df_combined.to_parquet(dataset_pq, index=False)

    print("\n<walter> Successfully saved dataset:")
    print(f"\t- CSV: {dataset_csv}")
    print(f"\t- Parquet: {dataset_pq}")
    print(f"\n<walter> Dataset size: {len(df_combined):,}")

    return df_combined
