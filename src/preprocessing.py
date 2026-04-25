"""
Preprocessing steps for the primary (uncleaned/unaltered) dataset.
Contains parsing and cleaning.
"""

import pandas as pd
from src.utils import finder

PRIMARY_FNAME = "drug_products.csv"


def pandafy(csv_file: str):
    """
    Constructs a Pandas DataFrame from the dataset while only
    including the brand name column.
    """

    df = pd.read_csv(csv_file)
    return df[["Brand Name"]]


def cleaner(df: pd.DataFrame, sort: bool = False):
    """
    A multistep cleaning pipeline for the pandas DataFrame
    constructed by pandafy.
    """

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    empty_markers = ["none"]
    df = df[~df["Brand Name"].isin(empty_markers)]
    # if it's a whitespace string
    df = df[~df["Brand Name"].str.contains(r"^\s*$", regex=True, na=False)]

    if sort:
        df = df.sort_values(by="Brand Name")

    return df


def master_maker():
    """
    This is a coordinator function that calls the function above.
    Always returns a cleaned DataFrame of the drug brand names.
    """

    prim: str = str(finder(PRIMARY_FNAME))
    df: pd.DataFrame = pandafy(csv_file=prim)
    return cleaner(df=df)
