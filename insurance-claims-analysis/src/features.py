"""
Feature engineering for the claim amount regression model.
"""
import pandas as pd

NUMERIC_FEATURES = [
    "PREMIUM_AMOUNT", "AGE", "TENURE", "NO_OF_FAMILY_MEMBERS",
    "INCIDENT_HOUR_OF_THE_DAY", "ANY_INJURY", "POLICE_REPORT_AVAILABLE",
    "DAYS_POLICY_TO_LOSS", "DAYS_LOSS_TO_REPORT",
]

CATEGORICAL_FEATURES = [
    "INSURANCE_TYPE", "RISK_SEGMENTATION", "MARITAL_STATUS",
    "EMPLOYMENT_STATUS", "HOUSE_TYPE", "SOCIAL_CLASS", "INCIDENT_SEVERITY",
]

TARGET = "CLAIM_AMOUNT"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add two time-gap features between policy, loss, and report dates.

    DAYS_POLICY_TO_LOSS: days between policy start and the loss event.
    DAYS_LOSS_TO_REPORT: days between the loss event and it being reported.

    Both are plausible drivers of claim size (e.g. a claim reported long
    after the loss may involve more accumulated damage) and are cheap to
    compute from columns already in the raw data.
    """
    out = df.copy()
    out["DAYS_POLICY_TO_LOSS"] = (out["LOSS_DT"] - out["POLICY_EFF_DT"]).dt.days
    out["DAYS_LOSS_TO_REPORT"] = (out["REPORT_DT"] - out["LOSS_DT"]).dt.days
    return out


def build_model_table(
    df: pd.DataFrame,
    numeric_features: list = None,
    categorical_features: list = None,
    target: str = TARGET,
) -> pd.DataFrame:
    """Select model features, one-hot encode categoricals, and attach the target.

    CLAIM_AMOUNT and any column derived from it (e.g. LOSS_RATIO) must stay
    out of the feature list -- including them would leak the target into
    the inputs and make the model look far better than it really is.

    Returns a numeric-only DataFrame ready to split into X / y.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    leak_cols = set(numeric_features) & {target, "LOSS_RATIO"}
    if leak_cols:
        raise ValueError(f"Feature list contains target-derived columns: {leak_cols}")

    cols = numeric_features + categorical_features + [target]
    model_df = df[cols].copy()
    model_df = pd.get_dummies(model_df, columns=categorical_features, drop_first=True)
    return model_df
