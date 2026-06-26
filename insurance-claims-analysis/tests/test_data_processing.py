import pandas as pd

from src.data_processing import add_loss_ratio, clean_data, most_common_category


def test_clean_data_drops_duplicates():
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "TXN_DATE_TIME": ["2023-01-01", "2023-01-01", "2023-01-02"],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2


def test_clean_data_parses_dates():
    df = pd.DataFrame({"TXN_DATE_TIME": ["2023-01-01"], "A": [1]})
    cleaned = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["TXN_DATE_TIME"])


def test_add_loss_ratio_basic():
    df = pd.DataFrame({"CLAIM_AMOUNT": [100.0, 200.0], "PREMIUM_AMOUNT": [50.0, 100.0]})
    out = add_loss_ratio(df)
    assert out["LOSS_RATIO"].tolist() == [2.0, 2.0]


def test_add_loss_ratio_handles_zero_premium():
    df = pd.DataFrame({"CLAIM_AMOUNT": [100.0], "PREMIUM_AMOUNT": [0.0]})
    out = add_loss_ratio(df)
    assert pd.isna(out["LOSS_RATIO"].iloc[0])


def test_most_common_category():
    df = pd.DataFrame({"TYPE": ["A", "B", "B", "B", "Z"]})
    # NOTE: regression test for the original notebook bug -- `.max()` would
    # incorrectly return "Z" here (alphabetically last), not "B" (most frequent).
    assert most_common_category(df, "TYPE") == "B"
