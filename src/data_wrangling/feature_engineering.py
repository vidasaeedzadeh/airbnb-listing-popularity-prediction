import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

#### Log of skwede features
def add_log_transformed_features(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """
    Create log-transformed versions of skewed numeric features.
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df

### Distance from center of manhatten

def compute_distance_from_manhattan(df):

    center_lat = 40.7831
    center_lon = -73.9712

    lat_km = 111
    lon_km = 111 * np.cos(np.radians(center_lat))

    dlat = (df["latitude"] - center_lat) * lat_km
    dlon = (df["longitude"] - center_lon) * lon_km

    df["distance_from_manhattan_km"] = np.sqrt(dlat**2 + dlon**2)

    return df

### Sentiment for names

def add_listing_name_sentiment(
    df: pd.DataFrame,
    text_column: str = "name"
) -> pd.DataFrame:
    """
    Add sentiment score of listing title using VADER. 
    The score shows how positive/ negative the name is
    """
    df = df.copy()

    sia = SentimentIntensityAnalyzer()

    df["name_sentiment"] = df[text_column].fillna("").apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    return df

#### drop columns

def drop_leaky_features(
    df: pd.DataFrame,
    columns_to_drop: list[str]
) -> pd.DataFrame:
    """
    Remove features that may leak information or 
    will not exist or not useful at prediction time.
    """
    df = df.copy()

    existing = [c for c in columns_to_drop if c in df.columns]

    return df.drop(columns=existing)

def add_days_since_last_review(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    reference_date = df["last_review"].max()

    df["days_since_last_review"] = (
        reference_date - pd.to_datetime(df["last_review"])
    ).dt.days

    return df

def add_host_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["is_professional_host"] = (
        df["calculated_host_listings_count"] > 1
    ).astype(int)

    return df

### run

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """

    log_columns = [
        "price",
        "minimum_nights",
        "calculated_host_listings_count"
    ]

    df = add_log_transformed_features(df, log_columns)

    df = compute_distance_from_manhattan(df)

    df = add_listing_name_sentiment(df)

    df = drop_leaky_features(
        df,
        columns_to_drop=["number_of_reviews","latitude","longitude","name"]
    )

    df = add_days_since_last_review(df)
    df = add_host_features(df)

    return df


