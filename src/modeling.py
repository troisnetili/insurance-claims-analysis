"""
Model training and evaluation for the claim amount regression model.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def train_test_split_data(
    model_df: pd.DataFrame, target: str = "CLAIM_AMOUNT",
    test_size: float = 0.2, random_state: int = 42,
) -> SplitData:
    """Split into train/test sets for a regression target.

    Unlike a classification split, there's no class balance to preserve
    here, so this is a plain random split.
    """
    X = model_df.drop(columns=[target])
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return SplitData(X_train, X_test, y_train, y_test)


def train_linear_regression(split: SplitData) -> tuple:
    """Train a linear regression baseline on scaled features.

    Returns (fitted_model, fitted_scaler) since the scaler is needed again
    at prediction time on new data. Scaling isn't required for linear
    regression's predictions, but it makes the fitted coefficients directly
    comparable to each other.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split.X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, split.y_train)
    return model, scaler


def train_random_forest(
    split: SplitData, n_estimators: int = 200, max_depth: int = 8
) -> RandomForestRegressor:
    """Train a Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(split.X_train, split.y_train)
    return model


def evaluate(model, X_test, y_test, scaler=None) -> dict:
    """Evaluate a fitted regression model on the test set.

    Returns RMSE, MAE, R^2, and the raw predictions/residuals for further
    diagnostic plotting (e.g. residuals vs. fitted, since checking the
    standard linear regression assumptions matters as much as the headline
    error metric).
    """
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    residuals = y_test.values - y_pred

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred,
        "residuals": residuals,
    }


def top_feature_importances(
    model: RandomForestRegressor, feature_names, top_n: int = 10
) -> pd.Series:
    """Return the top-N feature importances from a fitted tree ensemble."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)
