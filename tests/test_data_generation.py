import pandas as pd

from time_series_forecasting_service.data_generation import (
    PRODUCT_PROFILES,
    generate_synthetic_sales,
    product_ids,
)


def test_generation_is_deterministic_for_a_fixed_seed():
    df1 = generate_synthetic_sales(seed=42, n_days=120)
    df2 = generate_synthetic_sales(seed=42, n_days=120)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_noise():
    df1 = generate_synthetic_sales(seed=42, n_days=120)
    df2 = generate_synthetic_sales(seed=7, n_days=120)
    assert not df1["sales"].equals(df2["sales"])


def test_schema_and_value_ranges():
    df = generate_synthetic_sales(n_days=60)
    assert set(df.columns) == {"date", "product_id", "sales"}
    assert df["sales"].ge(0).all()
    assert pd.api.types.is_integer_dtype(df["sales"])
    assert df["product_id"].nunique() == len(PRODUCT_PROFILES)
    assert len(df) == 60 * len(PRODUCT_PROFILES)


def test_date_range_starts_where_requested():
    df = generate_synthetic_sales(start_date="2023-01-01", n_days=60)
    assert df["date"].min() == pd.Timestamp("2023-01-01")
    assert df["date"].max() == pd.Timestamp("2023-01-01") + pd.Timedelta(days=59)


def test_product_ids_matches_profiles_order():
    assert product_ids() == [p.product_id for p in PRODUCT_PROFILES]


def test_each_product_has_a_full_contiguous_daily_series():
    df = generate_synthetic_sales(n_days=90)
    for _, group in df.groupby("product_id"):
        dates = group.sort_values("date")["date"]
        expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(dates.to_numpy()), expected, check_names=False, check_freq=False
        )
