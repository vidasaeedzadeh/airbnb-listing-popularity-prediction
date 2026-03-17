from pathlib import Path
import joblib
import pandas as pd

from src.config import (
    X_TEST_FILE,
    MODELS_DIR,
    FIGURES_DIR,
)

from src.model_building.shap_analysis import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
)


XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_regression_model.joblib"


def load_data():
    X_test = pd.read_csv(X_TEST_FILE)
    return X_test


def get_best_estimator(model):
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_
    return model


def main():
    print("Loading data...")
    X_test = load_data()

    print("Loading model...")
    model = joblib.load(XGBOOST_MODEL_FILE)
    model = get_best_estimator(model)

    print("Computing SHAP values...")
    shap_values, explainer = compute_shap_values(model, X_test)

    print("Plotting SHAP summary (dot)...")
    plot_shap_summary(
        shap_values,
        X_test,
        FIGURES_DIR / "shap_summary_dot.png",
        plot_type="dot",
    )

    print("Plotting SHAP summary (bar)...")
    plot_shap_bar(
        shap_values,
        X_test,
        FIGURES_DIR / "shap_summary_bar.png",
    )

    print("\nSHAP analysis complete.")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
