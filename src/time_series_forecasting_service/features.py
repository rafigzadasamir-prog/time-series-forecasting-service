"""Feature engineering shared by training, backtesting, and serving.

All lag/rolling features are computed with an explicit ``.shift(1)`` before
windowing so that the feature for day ``t`` only ever uses information from
strictly before day ``t`` - this is what lets the exact same
``build_feature_frame`` function be reused safely both for bulk, offline
training/backtesting and for the recursive one-day-at-a-time forecasting
used at serving time (see ``forecasting.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from time_series_forecasting_service.config import LAGS, ROLLING_WINDOWS, TARGET_COLUMN
from time_series_forecasting_service.holidays import is_holiday


def calendar_features_for_dates(dates: pd.Series) -> pd.DataFrame:
    """Vectorized calendar features for a column of dates.

    Kept separate from the lag/rolling logic because calendar features need
    no history at all - they depend only on the calendar date itself, so
    they are trivially identical whether computed in bulk or one date at a
    time during recursive forecasting.
    """
    dt = pd.DatetimeIndex(dates)
    doy = dt.day_of_year.to_numpy()
    return pd.DataFrame(
        {
            "day_of_week": dt.day_of_week.to_numpy(),
            "is_weekend": (dt.day_of_week >= 5).astype(int),
            "month": dt.month.to_numpy(),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_holiday": np.array([int(is_holiday(d.date())) for d in dt]),
        },
        index=dates.index if isinstance(dates, pd.Series) else None,
    )


def trend_index(dates: pd.Series, reference_date: pd.Timestamp) -> np.ndarray:
    """Integer days-since-``reference_date``, used as a simple trend feature.

    ``reference_date`` must be fixed across training, backtesting and
    serving (it is the minimum date of the full historical dataset) so the
    feature means the same thing no matter which slice of data it is
    computed on.
    """
    return (pd.DatetimeIndex(dates) - reference_date).days.to_numpy()


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    lags: tuple[int, ...] = LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
    group_col: str = "product_id",
) -> pd.DataFrame:
    """Add per-product lag and trailing rolling-mean features.

    ``df`` must already be sorted by (group_col, date). Rolling means are
    computed on the series shifted by one day first, so ``roll_mean_7`` for
    day t is the mean of days [t-7, t-1] and never includes day t itself.
    """
    df = df.copy()
    grouped_target = df.groupby(group_col)[target_col]

    for lag in lags:
        df[f"lag_{lag}"] = grouped_target.shift(lag)

    shifted = grouped_target.shift(1)
    for window in rolling_windows:
        df[f"roll_mean_{window}"] = shifted.groupby(df[group_col]).transform(
            lambda s, w=window: s.rolling(window=w, min_periods=w).mean()
        )

    return df


def build_feature_frame(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    product_code_map: dict[str, int],
    date_col: str = "date",
    target_col: str = TARGET_COLUMN,
    group_col: str = "product_id",
    lags: tuple[int, ...] = LAGS,
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Build the full feature set (calendar + trend + lag + rolling + product code).

    Rows are returned sorted by (product_id, date) - the same order
    ``add_lag_and_rolling_features`` relies on - and warm-up rows that lack
    enough history for the longest lag/rolling window are dropped, since a
    model cannot be trained or scored on a feature vector containing NaNs.
    """
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)
    calendar = calendar_features_for_dates(df[date_col])
    df = pd.concat([df.reset_index(drop=True), calendar.reset_index(drop=True)], axis=1)
    df["trend_idx"] = trend_index(df[date_col], reference_date)
    df["product_code"] = df[group_col].map(product_code_map).astype(int)

    df = add_lag_and_rolling_features(
        df, target_col=target_col, lags=lags, rolling_windows=rolling_windows, group_col=group_col
    )

    feature_cols = [f"lag_{lag}" for lag in lags] + [f"roll_mean_{w}" for w in rolling_windows]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def make_product_code_map(product_ids: list[str]) -> dict[str, int]:
    """Stable product_id -> integer code mapping, sorted for determinism."""
    return {pid: i for i, pid in enumerate(sorted(product_ids))}
