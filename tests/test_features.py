import pandas as pd
import pytest

from time_series_forecasting_service.features import (
    build_feature_frame,
    calendar_features_for_dates,
    make_product_code_map,
)


@pytest.fixture
def simple_series_df() -> pd.DataFrame:
    # sales[i] == i, which makes expected lag/rolling values easy to hand-compute.
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=40, freq="D"),
            "product_id": "p1",
            "sales": range(40),
        }
    )


@pytest.fixture
def simple_features_df(simple_series_df: pd.DataFrame) -> pd.DataFrame:
    product_code_map = make_product_code_map(["p1"])
    return build_feature_frame(
        simple_series_df,
        reference_date=simple_series_df["date"].min(),
        product_code_map=product_code_map,
    )


def test_lag_features_reference_only_past_actuals(simple_series_df, simple_features_df):
    target_date = simple_series_df["date"].iloc[30]
    row = simple_features_df.loc[simple_features_df["date"] == target_date].iloc[0]
    assert row["sales"] == 30
    assert row["lag_1"] == 29
    assert row["lag_7"] == 23
    assert row["lag_14"] == 16
    assert row["lag_28"] == 2
    # Critically, the lag must never equal the row's own (current-day) value.
    assert row["lag_1"] != row["sales"]


def test_rolling_mean_excludes_the_current_day(simple_series_df, simple_features_df):
    target_date = simple_series_df["date"].iloc[30]
    row = simple_features_df.loc[simple_features_df["date"] == target_date].iloc[0]
    # roll_mean_7 at day 30 = mean(sales[23..29]) = mean(23..29) = 26, never includes sales[30]=30.
    assert row["roll_mean_7"] == pytest.approx(26.0)
    assert row["roll_mean_14"] == pytest.approx(sum(range(16, 30)) / 14)


def test_warm_up_rows_without_enough_history_are_dropped(simple_series_df, simple_features_df):
    # The longest window is a 28-day rolling mean of the already-shifted
    # series, so the first 28 days (indices 0..27) cannot have a complete
    # feature vector and must be dropped.
    assert simple_features_df["date"].min() == simple_series_df["date"].iloc[28]
    assert not simple_features_df.isna().any().any()


def test_calendar_features_known_dates():
    dates = pd.Series(pd.to_datetime(["2023-01-01", "2023-07-04", "2023-06-14"]))  # Sun, Tue, Wed
    cal = calendar_features_for_dates(dates)
    assert cal.loc[0, "day_of_week"] == 6  # Sunday
    assert cal.loc[0, "is_weekend"] == 1
    assert cal.loc[0, "is_holiday"] == 1  # New Year's Day
    assert cal.loc[1, "is_holiday"] == 1  # Independence Day
    assert cal.loc[2, "is_holiday"] == 0
    assert cal.loc[2, "month"] == 6


def test_product_code_map_is_sorted_and_stable():
    mapping = make_product_code_map(["zebra", "apple", "mango"])
    assert mapping == {"apple": 0, "mango": 1, "zebra": 2}
