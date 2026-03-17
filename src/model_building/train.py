import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from src.config import RANDOM_SEED


#Base model: linear regression
def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log_target: bool = False
) -> tuple[LinearRegression, bool]:
    """
    Train a linear regression model.

    Parameters
    ----------
    log_target : bool
        If True, train on log1p(target)
    """
    model = LinearRegression()

    if log_target:
        y_train = np.log1p(y_train)

    model.fit(X_train, y_train)

    return model

# Base model: Ridge
def train_ridge_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha_grid: list[float],
    log_target: bool = False
) -> GridSearchCV:
    if log_target:
        y_train = np.log1p(y_train)

    ridge = Ridge()
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid={"alpha": alpha_grid},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search

#Desision Tree
def train_decision_tree_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict
) -> GridSearchCV:
    """
    Train a decision tree regressor with grid search.
    """
    tree = DecisionTreeRegressor(random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=tree,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search

# Ensemble model - random forest
def train_random_forest_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict
) -> GridSearchCV:
    """
    Train a random forest regressor with grid search.
    """
    forest = RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=forest,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search

#Ensemble model - XGBoost

def train_xgboost_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict
) -> GridSearchCV:
    """
    Train an XGBoost regressor with grid search.
    """
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return grid_search

