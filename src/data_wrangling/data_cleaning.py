from pathlib import Path
import pandas as pd

from src.config import CLEANED_DATA_FILE


def drop_irrelevant_columns(
    df: pd.DataFrame,
    columns_to_drop: list[str]
) -> pd.DataFrame:
    """
    Drop columns that are identifiers or not used as model features.
    """
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns).copy()


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the dataframe.
    """
    return df.drop_duplicates().copy()


def convert_last_review_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the last_review column to datetime if it exists.
    """
    df = df.copy()

    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    return df


def remove_invalid_numeric_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows with clearly invalid numeric values.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned dataframe and number of rows removed.
    """
    df = df.copy()

    valid_mask = (
        (df["price"] > 0) &
        (df["minimum_nights"] > 0) &
        (df["calculated_host_listings_count"] > 0) &
        (df["availability_365"] >= 0) &
        (df["availability_365"] <= 365)
    )

    cleaned_df = df.loc[valid_mask].copy()
    removed_rows_count = df.shape[0] - cleaned_df.shape[0]

    return cleaned_df, removed_rows_count


def clean_airbnb_data(
    df: pd.DataFrame,
    columns_to_drop: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full early-stage cleaning pipeline for the Airbnb dataset.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned dataframe and cleaning summary.
    """
    initial_rows = df.shape[0]

    df = drop_irrelevant_columns(df, columns_to_drop)
    after_drop_columns = df.shape[1]

    df_no_duplicates = remove_duplicate_rows(df)
    duplicate_rows_removed = df.shape[0] - df_no_duplicates.shape[0]
    df = df_no_duplicates

    df = convert_last_review_to_datetime(df)

    df, invalid_rows_removed = remove_invalid_numeric_rows(df)

    cleaning_summary = {
        "initial_rows": initial_rows,
        "final_rows": df.shape[0],
        "final_columns": after_drop_columns,
        "duplicate_rows_removed": duplicate_rows_removed,
        "invalid_rows_removed": invalid_rows_removed,
    }

    return df, cleaning_summary


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path = CLEANED_DATA_FILE
) -> None:
    """
    Save the cleaned dataframe to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)