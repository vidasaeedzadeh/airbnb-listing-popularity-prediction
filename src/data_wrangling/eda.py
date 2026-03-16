from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from src.config import FIGURES_DIR


def save_figure(output_path: Path) -> None:
    """
    Save the current matplotlib figure and close it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

####### Missing values corrolation
def plot_missingness_heatmap(
    df: pd.DataFrame,
) -> None:
    """

    Each column is converted to a binary variable indicating whether
    the value is missing (1) or present (0). Correlations between
    these indicators reveal whether missingness patterns are related.
    Plot heatmap showing missing value correlations between variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    output_path : Path
        Path where the figure will be saved.
    """

    #missing_df = df[df.columns[df.isna().any()]]
    missing_corr = df.isna().astype(int).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        missing_corr,
        annot=True,
        cmap="coolwarm",
        cbar=False,
        center=0,
        square=True
    )

    plt.title("Missing Value Correlation Matrix")

    save_figure(FIGURES_DIR / "missingness_heatmap.png")

###### Missing values corrolation with 0 in a column
def analyze_missing_reviews_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze whether missing values in review-related columns occur when
    number_of_reviews equals zero.

    Returns
    -------
    pd.DataFrame
        Crosstab showing relationship between number_of_reviews == 0
        and missing values in reviews_per_month and last_review.
    """

    review_zero = df["number_of_reviews"] == 0

    missing_reviews_per_month = df["reviews_per_month"].isna()
    missing_last_review = df["last_review"].isna()

    summary = pd.DataFrame({
        "number_of_reviews_zero": review_zero,
        "reviews_per_month_missing": missing_reviews_per_month,
        "last_review_missing": missing_last_review
    })

    return summary.groupby("number_of_reviews_zero").agg(
    count=("number_of_reviews_zero", "size"),
    reviews_per_month_missing_rate=("reviews_per_month_missing", "mean"),
    last_review_missing_rate=("last_review_missing", "mean")
)


def plot_missing_reviews_relationship(df: pd.DataFrame) -> None:
    """
    Visualize relationship between missing review variables and listings
    with zero reviews.
    """

    df_plot = pd.DataFrame({
        "number_of_reviews_zero": df["number_of_reviews"] == 0,
        "reviews_per_month_missing": df["reviews_per_month"].isna(),
        "last_review_missing": df["last_review"].isna()
    })

    summary = df_plot.groupby("number_of_reviews_zero").mean().reset_index()

    summary_melt = summary.melt(
        id_vars="number_of_reviews_zero",
        var_name="variable",
        value_name="missing_rate"
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=summary_melt,
        x="variable",
        y="missing_rate",
        hue="number_of_reviews_zero"
    )

    plt.title("Missing Review Variables vs Listings with Zero Reviews")
    plt.ylabel("Fraction Missing")
    plt.xlabel("Variable")

    save_figure(FIGURES_DIR / "missing_reviews_relationship.png")

#### Numeric columns histogram grid
def plot_numeric_histograms(
    df: pd.DataFrame,
    columns: list[str],
    bins: int = 30,
    n_cols: int = 4
) -> None:
    """
    Plot histograms for selected numeric columns in a grid layout.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Numeric columns to plot.
    bins : int, optional
        Number of histogram bins.
    n_cols : int, optional
        Number of subplot columns.
    """
    n_plots = len(columns)
    n_rows = ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col].dropna(), bins=bins, ax=axes[i], kde=False)
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    save_figure(FIGURES_DIR / "numeric_histograms.png")

#### Correlation matrix heatmap

def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str]
) -> None:
    """
    Plot correlation matrix heatmap for selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list[str]
        Numeric columns to include in the correlation matrix.
    """
    corr_matrix = df[columns].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        square=True,
        cbar=False,
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Matrix of Numeric Features")

    save_figure(FIGURES_DIR / "correlation_matrix.png")

#### Correlation with target plot

def plot_target_correlations(
    df: pd.DataFrame,
    target_col: str,
    columns: list[str]
) -> None:
    """
    Plot correlations between selected numeric features and the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str
        Target column name.
    columns : list[str]
        Numeric columns to consider.
    """
    corr_series = (
        df[columns]
        .corr()[target_col]
        .drop(target_col)
        .sort_values()
    )

    plt.figure(figsize=(8, 5))
    corr_series.plot(kind="barh")

    plt.title(f"Correlation of Numeric Features with {target_col}")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")

    save_figure(FIGURES_DIR / "target_correlations.png")

#### Target distribution plots - Histogram of raw target
def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str
) -> None:
    """
    Plot histogram of the target variable.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col].dropna(), bins=30, kde=False)

    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")

    save_figure(FIGURES_DIR / "target_distribution.png")

#### Boxplot of raw target

def plot_target_boxplot(
    df: pd.DataFrame,
    target_col: str
) -> None:
    """
    Plot boxplot of the target variable.
    """
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[target_col].dropna())

    plt.title(f"Boxplot of {target_col}")
    plt.xlabel(target_col)

    save_figure(FIGURES_DIR / "target_boxplot.png")

#### Histogram of log-transformed target

def plot_log_target_distribution(
    df: pd.DataFrame,
    target_col: str
) -> None:
    """
    Plot histogram of log-transformed target variable using log1p.
    """
    target_log = np.log1p(df[target_col].dropna())

    plt.figure(figsize=(8, 5))
    sns.histplot(target_log, bins=30, kde=False)

    plt.title(f"Log-Transformed Distribution of {target_col}")
    plt.xlabel(f"log1p({target_col})")
    plt.ylabel("Count")

    save_figure(FIGURES_DIR / "log_target_distribution.png")

#### Numerical corrolation
def plot_pairwise_numeric_relationships(
    df: pd.DataFrame,
    columns: list[str]
) -> None:
    """
    Create a pairplot for selected numeric columns.
    """
    plot_df = df[columns].dropna().copy()

    pair_grid = sns.pairplot(
        plot_df,
        diag_kind="hist",
        corner=True,
        plot_kws={"alpha": 0.3, "s": 18},
        diag_kws={"bins": 30}
    )

    pair_grid.fig.suptitle(
        "Pairwise Relationships of Selected Numeric Features",
        y=1.02,
        fontsize=16
    )

    output_path = FIGURES_DIR / "pairwise_numeric_relationships.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(pair_grid.fig)



