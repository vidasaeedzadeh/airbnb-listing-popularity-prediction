from pathlib import Path
import joblib
import pandas as pd

from src.config import (
    X_TEST_FILE,
    Y_TEST_FILE,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)

from src.model_building.permutation_importance import (
    compute_permutation_importance,
    plot_permutation_importance,
)


XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_regression_model.joblib"


def load_data():
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE).squeeze("columns")
    return X_test, y_test


def get_best_estimator(model):
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_
    return model


def main():
    print("Loading data...")
    X_test, y_test = load_data()

    print("Loading model...")
    model = joblib.load(XGBOOST_MODEL_FILE)
    model = get_best_estimator(model)

    print("Computing permutation importance...")
    importance_df = compute_permutation_importance(
        model,
        X_test,
        y_test,
    )

    print("Saving results...")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(
        TABLES_DIR / "permutation_importance.csv",
        index=False,
    )

    plot_permutation_importance(
        importance_df,
        FIGURES_DIR / "permutation_importance.png",
    )

    print("\nTop features:")
    print(importance_df.head(10))

    print("\nDone.")


if __name__ == "__main__":
    main()
