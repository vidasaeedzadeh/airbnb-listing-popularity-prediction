import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate

def predict(model, X, log_target=False):
    y_pred = model.predict(X)

    if log_target:
        y_pred = np.expm1(y_pred)

    return y_pred

def compute_regression_metrics(y_true, y_pred) -> dict:
    """
    Compute regression metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse/(y_true.max() - y_true.min())
    r2 = r2_score(y_true, y_pred)

    return {'nrmse':nrmse, 'rmse':rmse, 'r2':r2}


def evaluate_model(model, X_train, X_test, y_train, y_test, log_target: bool = False) -> dict:
    """
    Evaluate model on train and test data.
    """
    y_train_pred = predict(model, X_train, log_target=log_target)
    y_test_pred = predict(model, X_test, log_target=log_target)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    return {
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_nrmse": test_metrics["nrmse"],
    }

def extract_gridsearch_summary(grid_model) -> dict:
    """
    Extract score and timing information from a fitted GridSearchCV object.
    """
    best_idx = grid_model.best_index_

    return {
        "cv_r2_score": grid_model.best_score_,
        "mean_fit_time": grid_model.cv_results_["mean_fit_time"][best_idx],
        "mean_score_time": grid_model.cv_results_["mean_score_time"][best_idx],
    }




def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    """
    Compute cross-validated score and timing for a model.
    """
    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        return_train_score=False,
        n_jobs=-1,
    )

    return {
        "cv_r2_score": results["test_score"].mean(),
        "mean_fit_time": results["fit_time"].mean(),
        "mean_score_time": results["score_time"].mean(),
    }

