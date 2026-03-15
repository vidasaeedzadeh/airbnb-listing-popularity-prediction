import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN


numeric_features = ["price","minimum_nights","calculated_host_listings_count","availability_365"]

categorical_features = ["neighbourhood_group","room_type"]

passthrough_features = ["latitude","longitude"]

drop_features = ["number_of_reviews","neighbourhood"]


def fill_na_in_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    """
    Fill missing values in the target column with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing target values filled.
    """
    df = df.copy()
    df[target_column] = df[target_column].fillna(0)

    return df


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into feature matrix and target vector.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : str
        Name of the target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X (features) and y (target).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build preprocessing pipeline for numeric and categorical features.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer.
    """

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("coords", "passthrough", passthrough_features),
            ("drop_cols", "drop", drop_features),
        ],
        remainder="drop",
    )

    return preprocessor
