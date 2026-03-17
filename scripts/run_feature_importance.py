from pathlib import Path
import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    PREPROCESSOR_FILE,
    TABLES_DIR,
    FIGURES_DIR,
)

from src.model_building.feature_importance import (
    build_feature_importance_table,
    save_feature_importance_table,
    plot_feature_importance,
)


RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest_regression_model.joblib"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_regression_model.joblib"

RANDOM_FOREST_IMPORTANCE_TABLE_FILE = TABLES_DIR / "random_forest_feature_importance.csv"
XGBOOST_IMPORTANCE_TABLE_FILE = TABLES_DIR / "xgboost_feature_importance.csv"

RANDOM_FOREST_IMPORTANCE_FIGURE_FILE = FIGURES_DIR / "random_forest_feature_importance.png"
XGBOOST_IMPORTANCE_FIGURE_FILE = FIGURES_DIR / "xgboost_feature_importance.png"


def load_preprocessor(preprocessor_path: Path = PREPROCESSOR_FILE):
    """
    Load fitted preprocessing pipeline.
    """
    return joblib.load(preprocessor_path)


def load_model(model_path: Path):
    """
    Load trained model or GridSearchCV object.
    """
    return joblib.load(model_path)


def get_best_estimator(model):
    """
    Return best estimator if model is a GridSearchCV object,
    otherwise return the model itself.
    """
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_
    return model


def main() -> None:
    print("Loading preprocessor...")
    preprocessor = load_preprocessor()
    feature_names = preprocessor.get_feature_names_out()

    # --------------------------------------------------
    # Random Forest feature importance
    # --------------------------------------------------
    print("Loading Random Forest model...")
    rf_model = load_model(RANDOM_FOREST_MODEL_FILE)
    rf_estimator = get_best_estimator(rf_model)

    print("Computing Random Forest feature importance...")
    rf_importance_df = build_feature_importance_table(
        rf_estimator,
        feature_names
    )

    save_feature_importance_table(
        rf_importance_df,
        RANDOM_FOREST_IMPORTANCE_TABLE_FILE
    )

    plot_feature_importance(
        rf_importance_df,
        RANDOM_FOREST_IMPORTANCE_FIGURE_FILE,
        model_name="Random Forest",
        top_n=15
    )

    print("\nTop 10 Random Forest features:")
    print(rf_importance_df.head(10).to_string(index=False))

    # --------------------------------------------------
    # XGBoost feature importance
    # --------------------------------------------------
    print("\nLoading XGBoost model...")
    xgb_model = load_model(XGBOOST_MODEL_FILE)
    xgb_estimator = get_best_estimator(xgb_model)

    print("Computing XGBoost feature importance...")
    xgb_importance_df = build_feature_importance_table(
        xgb_estimator,
        feature_names
    )

    save_feature_importance_table(
        xgb_importance_df,
        XGBOOST_IMPORTANCE_TABLE_FILE
    )

    plot_feature_importance(
        xgb_importance_df,
        XGBOOST_IMPORTANCE_FIGURE_FILE,
        model_name="XGBoost",
        top_n=15
    )

    print("\nTop 10 XGBoost features:")
    print(xgb_importance_df.head(10).to_string(index=False))

    print("\nFeature importance analysis complete.")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
