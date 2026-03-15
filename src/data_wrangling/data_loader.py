
from pathlib import Path
import pandas as pd

from src.config import RAW_DATA_FILE

def validate_file_exists(file_path: Path) -> None:
    """
    Check that the input data file exists.

    Parameters
    ----------
    file_path : Path
        Path to the input CSV file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")


def load_raw_data(file_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the raw Airbnb dataset from CSV.

    Parameters
    ----------
    file_path : Path, optional
        Path to the raw CSV file. Defaults to RAW_DATA_FILE from config.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.

    Raises
    ------
    ValueError
        If the loaded DataFrame is empty.
    """
    validate_file_exists(file_path)

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {file_path}")

    return df


