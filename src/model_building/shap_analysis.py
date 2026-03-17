from pathlib import Path

import shap
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for tree-based models.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer


def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    output_path: Path,
    plot_type: str = "dot",
):
    """
    Create SHAP summary plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type=plot_type,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_bar(
    shap_values,
    X: pd.DataFrame,
    output_path: Path,
):
    """
    Create SHAP bar plot (global importance).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        plot_type="bar",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
