import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression

from src.config import (
    MODELS_DIR,
    METRICS_DIR,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
)
from src.model_building.evaluate import (
    evaluate_model,
    cross_validate_model,
    extract_gridsearch_summary,
)
from src.model_building.train import (train_linear_regression,
                                      train_ridge_regression,
                                      train_decision_tree_regression,
                                      train_random_forest_regression,
                                      train_xgboost_regression
                                      )



LINEAR_MODEL_FILE = MODELS_DIR / "linear_regression_model.joblib"
LINEAR_METRICS_FILE = METRICS_DIR / "linear_regression_metrics.csv"

LOG_LINEAR_MODEL_FILE = MODELS_DIR / "log_target_linear_regression_model.joblib"
LOG_LINEAR_METRICS_FILE = METRICS_DIR / "log_target_linear_regression_metrics.csv"

RIDGE_MODEL_FILE = MODELS_DIR / "ridge_regression_model.joblib"
RIDGE_METRICS_FILE = METRICS_DIR / "ridge_regression_metrics.csv"

TREE_MODEL_FILE = MODELS_DIR / "decision_tree_regression_model.joblib"
TREE_METRICS_FILE = METRICS_DIR / "decision_tree_regression_metrics.csv"

FOREST_MODEL_FILE = MODELS_DIR / "random_forest_regression_model.joblib"
FOREST_METRICS_FILE = METRICS_DIR / "random_forest_regression_metrics.csv"

XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_regression_model.joblib"
XGBOOST_METRICS_FILE = METRICS_DIR / "xgboost_regression_metrics.csv"


def ensure_output_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def load_modeling_data():
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE).squeeze("columns")
    y_test = pd.read_csv(Y_TEST_FILE).squeeze("columns")

    return X_train, X_test, y_train, y_test


def main() -> None:
    print("Loading preprocessed training and test data...")
    X_train, X_test, y_train, y_test = load_modeling_data()

    ensure_output_directory(MODELS_DIR)
    ensure_output_directory(METRICS_DIR)

    
    # Baseline linear regression
    print("Training baseline linear regression model...")
    linear_model = train_linear_regression(X_train, y_train, log_target=False)

    print("Evaluating baseline linear regression model...")
    linear_metrics = evaluate_model(
        linear_model,
        X_train,
        X_test,
        y_train,
        y_test,
        log_target=False,
    )

    linear_cv_summary = cross_validate_model(
    LinearRegression(),
    X_train,
    y_train,
    cv=5,)

    joblib.dump(linear_model, LINEAR_MODEL_FILE)

    linear_metrics_df = pd.DataFrame([{
    "model": "linear_regression",
    "cv_r2_score": linear_cv_summary["cv_r2_score"],
    "mean_fit_time": linear_cv_summary["mean_fit_time"],
    "mean_score_time": linear_cv_summary["mean_score_time"],
    **linear_metrics,
    }])
    linear_metrics_df.to_csv(LINEAR_METRICS_FILE, index=False)

    print("Baseline Linear Regression Results")
    for key, value in linear_metrics.items():
        print(f"{key}: {value:.4f}")


    # Log-target linear regression
    print("Training log-target linear regression model...")
    log_linear_model = train_linear_regression(X_train, y_train, log_target=True)

    print("Evaluating log-target linear regression model...")
    log_linear_metrics = evaluate_model(
        log_linear_model,
        X_train,
        X_test,
        y_train,
        y_test,
        log_target=True,
    )

    log_linear_cv_summary = cross_validate_model(
    LinearRegression(),
    X_train,
    np.log1p(y_train),
    cv=5,
    )

    joblib.dump(log_linear_model, LOG_LINEAR_MODEL_FILE)

    log_linear_metrics_df = pd.DataFrame([{
    "model": "log_target_linear_regression",
    "cv_r2_score": log_linear_cv_summary["cv_r2_score"],
    "mean_fit_time": log_linear_cv_summary["mean_fit_time"],
    "mean_score_time": log_linear_cv_summary["mean_score_time"],
    **log_linear_metrics,
    }])
    log_linear_metrics_df.to_csv(LOG_LINEAR_METRICS_FILE, index=False)

    print("Log-Target Linear Regression Results")
    for key, value in log_linear_metrics.items():
        print(f"{key}: {value:.4f}")

    # Ridge regression
    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    print("Training ridge regression model")
    ridge_model = train_ridge_regression(
        X_train,
        y_train,
        alpha_grid=alpha_grid,
        log_target=False,
    )

    print("Evaluating ridge regression model")
    ridge_metrics = evaluate_model(
        ridge_model,
        X_train,
        X_test,
        y_train,
        y_test,
        log_target=False,
    )

    ridge_cv_summary = extract_gridsearch_summary(ridge_model)

    joblib.dump(ridge_model, RIDGE_MODEL_FILE)

    ridge_metrics_df = pd.DataFrame([{
    "model": "ridge_regression",
    "cv_r2_score": ridge_cv_summary["cv_r2_score"],
    "mean_fit_time": ridge_cv_summary["mean_fit_time"],
    "mean_score_time": ridge_cv_summary["mean_score_time"],
    **ridge_metrics,
    "best_alpha": ridge_model.best_params_["alpha"],
    }])
    ridge_metrics_df.to_csv(RIDGE_METRICS_FILE, index=False)

    print("Ridge Regression Results")
    print(f"best_alpha: {ridge_model.best_params_['alpha']}")
    for key, value in ridge_metrics.items():
        print(f"{key}: {value:.4f}")

   
    # Decision tree regression
    tree_param_grid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
    }

    print("\nTraining decision tree regression model...")
    tree_model = train_decision_tree_regression(
        X_train,
        y_train,
        param_grid=tree_param_grid,
    )

    print("Evaluating decision tree regression model...")
    tree_metrics = evaluate_model(
        tree_model,
        X_train,
        X_test,
        y_train,
        y_test,
        log_target=False,
    )

    tree_cv_summary = extract_gridsearch_summary(tree_model)

    joblib.dump(tree_model, TREE_MODEL_FILE)

    tree_metrics_df = pd.DataFrame([{
    "model": "decision_tree_regression",
    "cv_r2_score": tree_cv_summary["cv_r2_score"],
    "mean_fit_time": tree_cv_summary["mean_fit_time"],
    "mean_score_time": tree_cv_summary["mean_score_time"],
    **tree_metrics,
    "best_max_depth": tree_model.best_params_["max_depth"],
    "best_min_samples_split": tree_model.best_params_["min_samples_split"],
    "best_min_samples_leaf": tree_model.best_params_["min_samples_leaf"],
    }])
    tree_metrics_df.to_csv(TREE_METRICS_FILE, index=False)

    print("\nDecision Tree Regression Results")
    print(f"best params: {tree_model.best_params_}")
    for key, value in tree_metrics.items():
        print(f"{key}: {value:.4f}")

    
    # Random forest regression
    forest_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", None],
    }

    print("\nTraining random forest regression model...")
    forest_model = train_random_forest_regression(
        X_train,
        y_train,
        param_grid=forest_param_grid,
    )

    print("Evaluating random forest regression model...")
    forest_metrics = evaluate_model(
        forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
        log_target=False,
    )

    forest_cv_summary = extract_gridsearch_summary(forest_model)

    joblib.dump(forest_model, FOREST_MODEL_FILE)

    forest_metrics_df = pd.DataFrame([{
    "model": "random_forest_regression",
    "cv_r2_score": forest_cv_summary["cv_r2_score"],
    "mean_fit_time": forest_cv_summary["mean_fit_time"],
    "mean_score_time": forest_cv_summary["mean_score_time"],
    **forest_metrics,
    "best_n_estimators": forest_model.best_params_["n_estimators"],
    "best_max_depth": forest_model.best_params_["max_depth"],
    "best_min_samples_split": forest_model.best_params_["min_samples_split"],
    "best_min_samples_leaf": forest_model.best_params_["min_samples_leaf"],
    "best_max_features": forest_model.best_params_["max_features"],
    }])
    forest_metrics_df.to_csv(FOREST_METRICS_FILE, index=False)

    print("\nRandom Forest Regression Results")
    print(f"best params: {forest_model.best_params_}")
    for key, value in forest_metrics.items():
        print(f"{key}: {value:.4f}")

    # XGBoost regression
    xgb_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 1],
    }

    print("\nTraining XGBoost regression model...")
    xgb_model = train_xgboost_regression(
        X_train,
        y_train,
        param_grid=xgb_param_grid,
    )

    print("Evaluating XGBoost regression model...")
    xgb_metrics = evaluate_model(
        xgb_model,
        X_train,
        X_test,
        y_train,
        y_test,
        log_target=False,
    )
    xgb_cv_summary = extract_gridsearch_summary(xgb_model)
    joblib.dump(xgb_model, XGBOOST_MODEL_FILE)

    xgb_metrics_df = pd.DataFrame([{
    "model": "xgboost_regression",
    "cv_r2_score": xgb_cv_summary["cv_r2_score"],
    "mean_fit_time": xgb_cv_summary["mean_fit_time"],
    "mean_score_time": xgb_cv_summary["mean_score_time"],
    **xgb_metrics,
    "best_n_estimators": xgb_model.best_params_["n_estimators"],
    "best_max_depth": xgb_model.best_params_["max_depth"],
    "best_learning_rate": xgb_model.best_params_["learning_rate"],
    "best_subsample": xgb_model.best_params_["subsample"],
    "best_colsample_bytree": xgb_model.best_params_["colsample_bytree"],
    }])
    xgb_metrics_df.to_csv(XGBOOST_METRICS_FILE, index=False)

    print("\nXGBoost Regression Results")
    print(f"best params: {xgb_model.best_params_}")
    for key, value in xgb_metrics.items():
        print(f"{key}: {value:.4f}")

    
     # Combined metrics summary
    combined_metrics_df = pd.concat(
        [linear_metrics_df, log_linear_metrics_df,ridge_metrics_df,tree_metrics_df,
         forest_metrics_df,xgb_metrics_df],
        ignore_index=True
    )
    combined_metrics_df.to_csv(
        METRICS_DIR / "all_models_comparison.csv",
        index=False
    )

    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")



if __name__ == "__main__":
    main()