from pathlib import Path
import pandas as pd

from src.config import TABLES_DIR, CLEANED_DATA_FILE
from src.data_wrangling.data_loader import load_raw_data
from src.data_wrangling.data_cleaning import clean_airbnb_data, save_cleaned_data


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
    Run the data cleaning pipeline on the raw Airbnb dataset,
    save the cleaned dataset, and save a cleaning summary table.
    """
    print("Loading raw dataset")
    df_raw = load_raw_data()

    print("Preparing cleaning configuration")

    # Define columns to drop
    # These variables cannot be used as predictive features in a meaningful way. 
    columns_to_drop = ["id","name","host_id","host_name"]

    print("Running data cleaning pipeline")
    df_clean, cleaning_summary = clean_airbnb_data(df_raw,columns_to_drop=columns_to_drop)

    print("Saving cleaned dataset")
    save_cleaned_data(df_clean, CLEANED_DATA_FILE)

    print("Saving cleaning summary")
    ensure_output_directory(TABLES_DIR)

    cleaning_summary_df = pd.DataFrame([cleaning_summary])
    cleaning_summary_df.to_csv(TABLES_DIR / "data_cleaning_summary.csv", index=False)

    print("Data Cleaning Summary")
    print(f"Initial rows: {cleaning_summary['initial_rows']}")
    print(f"Final rows: {cleaning_summary['final_rows']}")
    print(f"Final columns: {cleaning_summary['final_columns']}")
    print(f"Duplicate rows removed: {cleaning_summary['duplicate_rows_removed']}")
    print(f"Invalid rows removed: {cleaning_summary['invalid_rows_removed']}")

    print("Data cleaning complete.")
    print(f"Cleaned dataset saved to: {CLEANED_DATA_FILE}")
    print(f"Cleaning summary saved to: {TABLES_DIR / 'data_cleaning_summary.csv'}")


if __name__ == "__main__":
    main()
