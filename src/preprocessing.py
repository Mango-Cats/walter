"""
Preprocessing steps for the primary (uncleaned/unaltered) dataset.
Contains parsing and cleaning.
"""

import pandas as pd
from pathlib import Path
from random import randint

FILE_DIR = "data/"
PRIMARY_FNAME = "drug_products.csv"
CLEANED_FNAME = str(FILE_DIR + "cleaned_" + PRIMARY_FNAME.removesuffix(".csv"))
TARGET_COL = "Brand Name"


def pandafy(csv_file: str | Path) -> pd.DataFrame:
    """
    Constructs a Pandas DataFrame from the dataset while only
    including the brand name column. The brand name column is defined
    in the `TARGET_COL` constant.
    """
    return pd.read_csv(filepath_or_buffer=csv_file, usecols=[TARGET_COL])


def cleaner(df: pd.DataFrame, sort: bool = False) -> pd.DataFrame:
    """
    A multistep cleaning pipeline for the pandas DataFrame
    constructed by pandafy.
    """
    df = df.copy()

    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

    df[TARGET_COL] = df[TARGET_COL].str.replace(
        pat=r"[^\x00-\x7F]", repl="", regex=True
    )

    bad_values: list[str] = ["none", "nan", "n/a", ""]

    df = df[
        (df[TARGET_COL].str.strip().ne(""))
        & (~df[TARGET_COL].str.lower().isin(bad_values))
    ]

    df = df.drop_duplicates().dropna()

    if sort:
        df = df.sort_values(by=TARGET_COL, key=lambda col: col.str.lower())

    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Performs structural and basic sanity checks on the input DataFrame
    before processing. Returns None and will NEVER raise an exception.
    It allows the process to continue but will print warnings, if any.
    """

    weird_len = 1000
    expected_len = 22853

    if len(df) < weird_len:
        print(
            "<walter> Validation Warning: "
            f"Dataset is unusually small (< {weird_len}). "
            f"There should be about {expected_len}."
        )

    null_count = df[TARGET_COL].isna().sum()
    if null_count > 0:
        print(f"<walter> Validation Warning: Found {null_count} missing (NaN) values.")

    duplicate_count = df.duplicated(subset=[TARGET_COL]).sum()
    if duplicate_count > 0:
        print(
            f"<walter> Validation Warning: Found {duplicate_count} exact duplicate rows."
        )

    non_ascii_ctr = (
        df[TARGET_COL].astype(str).str.contains(pat=r"[^\x00-\x7F]", na=False).sum()
    )
    if non_ascii_ctr > 0:
        print(
            "<walter> Validation Warning: "
            "Dataset has entries that contain non-ASCII "
            f"characters. In total, there are {non_ascii_ctr} entries."
        )


def cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    """
    Compares raw and clean DataFrames to report on the exact transformations
    made during the text preprocessing pipeline.
    """
    print("<walter> Cleaning Report: ")

    row_diff = len(raw) - len(clean)
    print(f"\t- Total rows dropped during cleaning: {row_diff}")

    raw_non_ascii = (
        raw[TARGET_COL].astype(str).str.contains(r"[^\x00-\x7F]", na=False).sum()
    )
    clean_non_ascii = (
        clean[TARGET_COL].astype(str).str.contains(r"[^\x00-\x7F]", na=False).sum()
    )

    if raw_non_ascii > 0:
        print(
            f"\t- Non-ASCII entries sanitized/removed: {raw_non_ascii - clean_non_ascii} "
            f"(Out of {raw_non_ascii} original dirty entries)."
        )
    else:
        print("\t- No non-ASCII characters were found in the raw data.")

    common_idx = raw.index.intersection(clean.index)
    raw_has_text = ~raw[TARGET_COL].astype(str).str.fullmatch(r"\s*")
    clean_is_empty = clean[TARGET_COL].astype(str).str.fullmatch(r"\s*")

    newly_empty = (raw_has_text.loc[common_idx] & clean_is_empty.loc[common_idx]).sum()
    if newly_empty > 0:
        print(
            f"\t- <walter> Warning: {newly_empty} entries had text but were reduced to empty strings."
        )

    print("\t- Samples of modified text (repr format):")
    sample_count = 0
    for idx in common_idx:
        orig = str(raw.loc[idx, TARGET_COL])
        new = str(clean.loc[idx, TARGET_COL])

        if orig != new:
            print(f"\t\tOriginal: {repr(orig)}")
            print(f"\t\tCleaned:  {repr(new)}")
            print("\t\t---")
            sample_count += 1

        if sample_count >= 5:
            break

    if sample_count == 0:
        print("\t\t(No text modifications detected in remaining rows)")


def master_maker(sort: bool = False, save: bool = False) -> pd.DataFrame:
    """
    The preprocessing coordinator function.

    Always returns a cleaned DataFrame of the drug brand names. But
    will raise an error if a file with the filename `PRIMARY_FNAME`
    does not exist anywhere from the root folder (see how `finder`
    works).
    """
    path: Path = Path(FILE_DIR + PRIMARY_FNAME)

    raw: pd.DataFrame = pandafy(csv_file=path)

    validate_raw_data(df=raw)

    clean: pd.DataFrame = cleaner(df=raw, sort=sort)

    cleaning_report(raw=raw, clean=clean)

    clean = clean.reset_index(drop=True)

    if save:
        clean.to_parquet(path=CLEANED_FNAME + ".parquet", index=False)
        clean.to_csv(path_or_buf=CLEANED_FNAME + ".csv", index=False)
        print(f"<walter> Saved in {CLEANED_FNAME}")

    return clean


def get_rand_entries(df: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """
    Get's a random slice of `count` entries. The slice starts at
    some random number n and ends at n + `count`. This function is
    best used if `df` is sorted alphabetically since it can show
    potential duplicates or highly similar entries.
    """
    n: int = randint(a=0, b=df.shape[0] - count)
    return df.iloc[n : n + count]
