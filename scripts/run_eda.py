import pandas as pd
import numpy as np

from src.config import CLEANED_DATA_FILE, TABLES_DIR, TARGET_COLUMN
from src.data_wrangling.eda import (
    plot_missingness_heatmap,
    analyze_missing_reviews_relationship,
    plot_missing_reviews_relationship,
    plot_numeric_histograms,
    plot_correlation_matrix,
    plot_target_correlations,
    plot_target_distribution,
    plot_target_boxplot,
    plot_log_target_distribution,
    plot_pairwise_numeric_relationships
)




def main():

    print("Loading cleaned dataset")
    df = pd.read_csv(CLEANED_DATA_FILE)

    print("Running EDA")

    # missing value correlation table
    missing_review_table = analyze_missing_reviews_relationship(df)

    missing_review_table.to_csv(
        TABLES_DIR / "missing_reviews_relationship.csv"
    )
    
    numeric_columns = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "latitude",
        "longitude"
    ]

    # visualizations

    plot_missingness_heatmap(df)
    plot_missing_reviews_relationship(df)
    plot_numeric_histograms(df, numeric_columns)
    plot_correlation_matrix(df, numeric_columns)
    plot_target_correlations(df, TARGET_COLUMN, numeric_columns)
    plot_target_distribution(df, TARGET_COLUMN)
    plot_target_boxplot(df, TARGET_COLUMN)
    plot_log_target_distribution(df, TARGET_COLUMN)
    plot_pairwise_numeric_relationships(df,numeric_columns)

    print("EDA complete.")


if __name__ == "__main__":
    main()

