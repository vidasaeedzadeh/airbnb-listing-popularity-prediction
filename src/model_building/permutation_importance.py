from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """
    Compute permutation feature importance.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring="r2",
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values(by="importance_mean", ascending=False)

    return importance_df


def plot_permutation_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
):
    """
    Plot top-N permutation importances.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_plot = importance_df.head(top_n).iloc[::-1]

    plt.figure()
    plt.barh(df_plot["feature"], df_plot["importance_mean"])
    plt.xlabel("Importance (decrease in R²)")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
