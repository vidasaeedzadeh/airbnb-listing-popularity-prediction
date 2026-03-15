from pathlib import Path
import pandas as pd

from src.config import TABLES_DIR
from src.data_wrangling.data_loader import load_raw_data
from src.data_wrangling.validation import (
    get_dataset_shape,
    summarize_data_types,
    summarize_missing_values,
    count_duplicate_rows,
    summarize_numeric_columns,
    check_numeric_constraints,
)


def ensure_output_directory(directory: Path) -> None:
    """
    Create the output directory if it does not already exist.

    Parameters
    ----------
    directory : Path
        Directory path to create.
    """
    directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Run data overview checks on the raw Airbnb dataset and save summary tables.
    """
    print("Loading raw dataset")
    df = load_raw_data()

    print("Running validation checks")
    n_rows, n_cols = get_dataset_shape(df)
    dtypes_summary = summarize_data_types(df)
    missing_summary = summarize_missing_values(df)
    duplicate_count = count_duplicate_rows(df)
    numeric_summary = summarize_numeric_columns(df)
    numeric_constraints = check_numeric_constraints(df)

    print("Dataset Overview")
    print(f"Shape: {n_rows} rows x {n_cols} columns")
    print(f"Duplicate rows: {duplicate_count}")

    print("Top missing values:")
    print(missing_summary.head(10).to_string(index=False))

    print("Numeric constraint checks:")
    print(numeric_constraints.to_string(index=False))

    print("Saving summary tables")
    ensure_output_directory(TABLES_DIR)

    dtypes_summary.to_csv(TABLES_DIR / "data_types_summary.csv", index=False)
    missing_summary.to_csv(TABLES_DIR / "missing_values_summary.csv", index=False)
    numeric_summary.to_csv(TABLES_DIR / "numeric_summary.csv")
    numeric_constraints.to_csv(TABLES_DIR / "numeric_constraint_checks.csv", index=False)

    overview_summary = {
        "n_rows": [n_rows],
        "n_columns": [n_cols],
        "duplicate_rows": [duplicate_count],
    }

    
    pd.DataFrame(overview_summary).to_csv(TABLES_DIR / "dataset_overview_summary.csv", index=False)

    print("Data overview complete.")
    print(f"Summary tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
