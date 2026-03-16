from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_DATA_FILE,
    METRICS_DIR,
    MODELS_DIR,
    PREPROCESSOR_FILE,
    RANDOM_SEED,
    TABLES_DIR,
    TEST_SIZE,
    X_TEST_FILE,
    X_TRAIN_FILE,
    Y_TEST_FILE,
    Y_TRAIN_FILE,
)
from src.data_wrangling.preprocessing import (
    build_preprocessing_pipeline,
    fill_na_in_target,
    split_features_and_target,
)


def ensure_output_directory(directory: Path) -> None:
    """
    Create output directory if it does not exist.

    Parameters
    ----------
    directory : Path
        Directory path to create.
    """
    directory.mkdir(parents=True, exist_ok=True)


def load_cleaned_data(file_path: Path = FEATURE_DATA_FILE) -> pd.DataFrame:
    """
    Load cleaned Airbnb dataset.

    Parameters
    ----------
    file_path : Path
        Path to cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    return pd.read_csv(file_path)


def main() -> None:
    """
    Run preprocessing on cleaned Airbnb data and save model-ready outputs.
    """
    print("Loading cleaned dataset")
    df = load_cleaned_data()

    print("Filling missing values in target")
    df = fill_na_in_target(df)

    print("Splitting features and target")
    X, y = split_features_and_target(df)

    print("Creating train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    print("Building preprocessing pipeline")
    preprocessor = build_preprocessing_pipeline()

    print("Fitting preprocessor on training data")
    X_train_processed = preprocessor.fit_transform(X_train)

    print("Transforming test data")
    X_test_processed = preprocessor.transform(X_test)

    print("Getting transformed feature names")
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    y_train_df = pd.DataFrame({"reviews_per_month": y_train}, index=y_train.index)
    y_test_df = pd.DataFrame({"reviews_per_month": y_test}, index=y_test.index)

    print("Saving processed datasets")
    ensure_output_directory(X_TRAIN_FILE.parent)
    ensure_output_directory(MODELS_DIR)
    ensure_output_directory(TABLES_DIR)
    ensure_output_directory(METRICS_DIR)

    X_train_processed_df.to_csv(X_TRAIN_FILE, index=False)
    X_test_processed_df.to_csv(X_TEST_FILE, index=False)
    y_train_df.to_csv(Y_TRAIN_FILE, index=False)
    y_test_df.to_csv(Y_TEST_FILE, index=False)

    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    preprocessing_summary = pd.DataFrame({
        "n_train_rows": [X_train_processed_df.shape[0]],
        "n_test_rows": [X_test_processed_df.shape[0]],
        "n_transformed_features": [X_train_processed_df.shape[1]],
        "test_size": [TEST_SIZE],
        "random_seed": [RANDOM_SEED],
    })
    preprocessing_summary.to_csv(
        TABLES_DIR / "preprocessing_summary.csv",
        index=False,
    )

    print("\nPreprocessing Summary")
    print(f"Training rows: {X_train_processed_df.shape[0]}")
    print(f"Test rows: {X_test_processed_df.shape[0]}")
    print(f"Transformed features: {X_train_processed_df.shape[1]}")
    print(f"Preprocessor saved to: {PREPROCESSOR_FILE}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
