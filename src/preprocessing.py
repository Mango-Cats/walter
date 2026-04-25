"""
Preprocessing steps for the primary (uncleaned/unaltered) dataset.
Contains parsing and cleaning.
"""

import pandas as pd
from src.utils import finder

PRIMARY_FNAME = "drug_products.csv"
BRAND_COL = "Brand Name"


def pandafy(csv_file: str):
    """
    Constructs a Pandas DataFrame from the dataset while only
    including the brand name column. The brand name column is defined
    in the `BRAND_COL` constant.
    """

    return pd.read_csv(csv_file, usecols=[BRAND_COL])


def cleaner(df: pd.DataFrame, sort: bool = False):
    """
    A multistep cleaning pipeline for the pandas DataFrame
    constructed by pandafy.
    """
    df[BRAND_COL] = df[BRAND_COL].astype(str).str.strip()
    
    bad_values = ["none", "nan", "n/a", ""]
    df = df[
        df[BRAND_COL].str.strip().ne("") & 
        ~df[BRAND_COL].str.lower().isin(bad_values)
    ]

    df = df.drop_duplicates().dropna()

    if sort:
        df = df.sort_values(
                by=BRAND_COL, 
                key=lambda col: col.str.lower(),
                ignore_index=True
            )
    return df


def master_maker(sort: bool = False):
    """
    This is a coordinator function that calls the function above.
    Always returns a cleaned DataFrame of the drug brand names.
    """

    prim: str = str(finder(PRIMARY_FNAME))
    df: pd.DataFrame = pandafy(csv_file=prim)
    return cleaner(df=df, sort=sort)
