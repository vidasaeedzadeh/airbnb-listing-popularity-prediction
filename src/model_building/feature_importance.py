from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_feature_importance_table(
    model,
    feature_names,
) -> pd.DataFrame:
    """
    Build a dataframe of feature importances.

    Parameters
    ----------
    model : estimator
        Trained tree-based model with feature_importances_ attribute.
    feature_names : list-like
        Names of transformed features.

    Returns
    -------
    pd.DataFrame
        Dataframe with feature names and importance scores, sorted descending.
    """
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df


def save_feature_importance_table(
    importance_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save feature importance table to CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    top_n: int = 25
) -> None:
    """
    Plot top-N feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Dataframe with columns ['feature', 'importance'].
    output_path : Path
        Path to save the figure.
    model_name : str
        Name of the model for the title.
    top_n : int, default=15
        Number of top features to plot.
    """
    plot_df = importance_df.head(top_n).copy()
    plot_df = plot_df.sort_values(by="importance", ascending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
