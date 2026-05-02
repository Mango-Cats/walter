import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_and_save_datasets(
    P: pd.DataFrame,
    U: pd.DataFrame,
    file_path,
    test_size=0.2,
    random_state=67,
    file_prefix="walter",
):
    """
    Labels, combines, shuffles, and splits P and U dataframes,
    then saves them to CSV and Parquet formats.
    """

    P = P.copy()
    U = U.copy()
    P["label"] = "p"
    U["label"] = "u"

    df_combined = pd.concat([P, U], ignore_index=True).sample(
        frac=1, random_state=random_state
    )

    train_df, test_df = train_test_split(
        df_combined,
        test_size=test_size,
        random_state=random_state,
        stratify=df_combined["label"],
    )

    train_csv = f"{file_path}train_{file_prefix}.csv"
    train_pq = f"{file_path}train_{file_prefix}.parquet"
    test_csv = f"{file_path}test_{file_prefix}.csv"
    test_pq = f"{file_path}test_{file_prefix}.parquet"

    train_df.to_csv(train_csv, index=False)
    train_df.to_parquet(train_pq, index=False)

    test_df.to_csv(test_csv, index=False)
    test_df.to_parquet(test_pq, index=False)

    print("<walter> Successfully saved split datasets:")
    print(f"\t- Train: {train_csv}, {train_pq}")
    print(f"\t- Test:  {test_csv}, {test_pq}")

    return train_df, test_df
