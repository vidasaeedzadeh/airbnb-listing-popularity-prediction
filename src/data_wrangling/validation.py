import pandas as pd


def get_dataset_shape(df: pd.DataFrame) -> tuple[int, int]:
    """
    Return the number of rows and columns in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    tuple[int, int]
        Number of rows and columns.
    """
    return df.shape


def get_column_names(df: pd.DataFrame) -> list[str]:
    """
    Return the list of column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    list[str]
        List of dataframe column names.
    """
    return df.columns.tolist()


def summarize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize column data types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with column names and data types.
    """
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values
    })


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing value counts and percentages by column,
        sorted in descending order of missing count.
    """
    missing_count = df.isna().sum()
    missing_percent = (df.isna().mean() * 100).round(2)

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    })

    summary = summary.sort_values(
        by=["missing_count", "missing_percent"],
        ascending=False
    ).reset_index(drop=True)

    return summary


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """
    Count the number of duplicate rows in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    int
        Number of duplicate rows.
    """
    return int(df.duplicated().sum())


def summarize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Summary statistics for numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.describe().transpose()


def summarize_categorical_columns(
    df: pd.DataFrame,
    max_unique: int = 10
) -> dict[str, pd.Series]:
    """
    Summarize categorical columns with value counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    max_unique : int, optional
        Maximum number of top unique values to return per column.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary mapping column names to their top value counts.
    """
    categorical_df = df.select_dtypes(include=["object", "category"])
    summaries = {}

    for col in categorical_df.columns:
        summaries[col] = df[col].value_counts(dropna=False).head(max_unique)

    return summaries


def check_numeric_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for suspicious or invalid values in selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Summary table of suspicious value counts.
    """
    checks = {
        "price_less_equal_0": (df["price"] <= 0).sum(),
        "minimum_nights_less_equal_0": (df["minimum_nights"] <= 0).sum(),
        "number_of_reviews_less_0": (df["number_of_reviews"] < 0).sum(),
        "reviews_per_month_less_0": (df["reviews_per_month"] < 0).sum(),
        "calculated_host_listings_count_less_equal_0": (
            df["calculated_host_listings_count"] <= 0
        ).sum(),
        "availability_365_outside_range": (
            (df["availability_365"] < 0) | (df["availability_365"] > 365)
        ).sum()
    }

    return pd.DataFrame({
        "check": list(checks.keys()),
        "count": list(checks.values())
    })
