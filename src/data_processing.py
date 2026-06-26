"""
Data loading and cleaning utilities for the insurance fraud project.
"""
from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "insurance_data.csv"

DATE_COLS = ["TXN_DATE_TIME", "POLICY_EFF_DT", "LOSS_DT", "REPORT_DT"]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw insurance claims CSV.

    Parameters
    ----------
    path : Path
        Location of the CSV file. Defaults to data/insurance_data.csv.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified claims data.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning: parse dates, drop exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw claims data as returned by load_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input data.
    """
    out = df.copy()
    out = out.drop_duplicates()

    for col in DATE_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def add_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add a LOSS_RATIO column (CLAIM_AMOUNT / PREMIUM_AMOUNT).

    Rows with a zero or missing PREMIUM_AMOUNT get a NaN loss ratio rather
    than raising a divide-by-zero error.
    """
    out = df.copy()
    safe_premium = out["PREMIUM_AMOUNT"].replace(0, pd.NA)
    out["LOSS_RATIO"] = out["CLAIM_AMOUNT"] / safe_premium
    return out


def most_common_category(df: pd.DataFrame, column: str) -> str:
    """Return the most frequent value in a categorical column.

    Note: this exists because `.max()` on a string column returns the
    alphabetically last value, not the most frequent one -- a mistake worth
    avoiding deliberately.
    """
    return df[column].value_counts().idxmax()
