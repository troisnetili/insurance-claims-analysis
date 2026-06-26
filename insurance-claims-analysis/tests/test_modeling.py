import numpy as np
import pandas as pd

from src.modeling import (
    evaluate,
    top_feature_importances,
    train_random_forest,
    train_test_split_data,
)


def _make_toy_model_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    # Make the target genuinely depend on X1 so the model has real signal
    y = 50 + 10 * X1 + rng.normal(scale=2, size=n)
    return pd.DataFrame({"feat1": X1, "feat2": X2, "CLAIM_AMOUNT": y})


def test_train_test_split_sizes():
    df = _make_toy_model_df()
    split = train_test_split_data(df, test_size=0.25)
    assert len(split.X_train) == 150
    assert len(split.X_test) == 50


def test_random_forest_trains_and_evaluates():
    df = _make_toy_model_df()
    split = train_test_split_data(df)
    model = train_random_forest(split, n_estimators=20, max_depth=3)
    result = evaluate(model, split.X_test, split.y_test)
    # the model should explain most of the variance given the strong
    # synthetic signal -- a loose sanity check, not a precise bound
    assert result["r2"] > 0.5
    assert len(result["y_pred"]) == len(split.y_test)
    assert len(result["residuals"]) == len(split.y_test)


def test_top_feature_importances_returns_sorted_series():
    df = _make_toy_model_df()
    split = train_test_split_data(df)
    model = train_random_forest(split, n_estimators=20, max_depth=3)
    importances = top_feature_importances(model, split.X_train.columns, top_n=2)
    assert len(importances) == 2
    # should be sorted descending, and feat1 (the real signal) should win
    assert importances.iloc[0] >= importances.iloc[1]
    assert importances.index[0] == "feat1"
