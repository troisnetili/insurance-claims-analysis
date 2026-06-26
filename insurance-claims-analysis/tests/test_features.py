import pandas as pd
import pytest

from src.features import add_date_features, build_model_table


def test_add_date_features_computes_day_gaps():
    df = pd.DataFrame({
        "POLICY_EFF_DT": pd.to_datetime(["2023-01-01"]),
        "LOSS_DT": pd.to_datetime(["2023-01-11"]),
        "REPORT_DT": pd.to_datetime(["2023-01-15"]),
    })
    out = add_date_features(df)
    assert out["DAYS_POLICY_TO_LOSS"].iloc[0] == 10
    assert out["DAYS_LOSS_TO_REPORT"].iloc[0] == 4


def test_build_model_table_one_hot_encodes_and_keeps_target():
    df = pd.DataFrame({
        "PREMIUM_AMOUNT": [100, 200],
        "AGE": [30, 40],
        "TENURE": [1, 2],
        "NO_OF_FAMILY_MEMBERS": [2, 3],
        "INCIDENT_HOUR_OF_THE_DAY": [10, 20],
        "ANY_INJURY": [0, 1],
        "POLICE_REPORT_AVAILABLE": [1, 0],
        "DAYS_POLICY_TO_LOSS": [100, 200],
        "DAYS_LOSS_TO_REPORT": [1, 2],
        "INSURANCE_TYPE": ["Motor", "Health"],
        "RISK_SEGMENTATION": ["H", "L"],
        "MARITAL_STATUS": ["Y", "N"],
        "EMPLOYMENT_STATUS": ["Y", "Y"],
        "HOUSE_TYPE": ["Own", "Rent"],
        "SOCIAL_CLASS": ["MI", "LI"],
        "INCIDENT_SEVERITY": ["Minor Loss", "Total Loss"],
        "CLAIM_AMOUNT": [500, 600],
    })
    model_df = build_model_table(df)
    assert "CLAIM_AMOUNT" in model_df.columns
    # one-hot encoding should remove the original categorical columns
    assert "INSURANCE_TYPE" not in model_df.columns
    # and all remaining columns should be numeric (encodable for sklearn)
    assert all(pd.api.types.is_numeric_dtype(model_df[c]) or model_df[c].dtype == bool
               for c in model_df.columns)


def test_build_model_table_rejects_leaky_features():
    # LOSS_RATIO is derived from CLAIM_AMOUNT (the target) -- including it
    # as a feature would leak the target into the inputs.
    df = pd.DataFrame({
        "PREMIUM_AMOUNT": [100, 200],
        "LOSS_RATIO": [5, 3],
        "CLAIM_AMOUNT": [500, 600],
    })
    with pytest.raises(ValueError):
        build_model_table(
            df,
            numeric_features=["PREMIUM_AMOUNT", "LOSS_RATIO"],
            categorical_features=[],
        )
