"""Recursive multi-step forecasting for the lag-feature ML model.

A model trained on lag features (lag_1, lag_7, ...) cannot see the true
future sales it would need to compute those lags for day t+2, t+3, ... of a
multi-day horizon - only day t+1 can be built from real history. The
standard way to forecast further out is *recursive* (a.k.a. "walk"):
predict one day, feed that prediction back in as if it were the actual
value, then predict the next day, and so on.

This function is used both by the backtest (so the reported ML metrics
reflect genuine H-day-ahead forecasting, not an optimistic version that
peeks at true future lags) and by the serving layer in ``service.py``, so
a backtest fold and a live API call exercise exactly the same forecasting
logic.
"""

from __future__ import annotations

import pandas as pd

from time_series_forecasting_service.config import TARGET_COLUMN
from time_series_forecasting_service.features import build_feature_frame


def recursive_forecast(
    model,
    history: pd.DataFrame,
    product_id: str,
    forecast_dates: list[pd.Timestamp],
    reference_date: pd.Timestamp,
    product_code_map: dict[str, int],
    feature_columns: list[str],
    date_col: str = "date",
    group_col: str = "product_id",
    target_col: str = TARGET_COLUMN,
) -> list[float]:
    """Forecast ``forecast_dates`` (must be consecutive days, immediately
    following the last date in ``history`` for this product) one day at a
    time, re-deriving lag/rolling features from the growing series of
    actual-plus-predicted values after every step.

    Recomputing the full feature frame on every step (rather than hand-
    rolling an incremental lag/rolling update) is deliberately the simple,
    obviously-correct option: horizons here are at most a few dozen days
    and product histories a few thousand rows, so the O(horizon x history)
    cost is negligible, and it guarantees the recursive features are
    computed by the exact same code path used for training.
    """
    columns = [date_col, group_col, target_col]
    product_history = history.loc[history[group_col] == product_id, columns]
    product_history = product_history.sort_values(date_col).reset_index(drop=True)

    predictions: list[float] = []
    for next_date in forecast_dates:
        placeholder = pd.DataFrame(
            {date_col: [next_date], group_col: [product_id], target_col: [float("nan")]}
        )
        extended = pd.concat([product_history, placeholder], ignore_index=True)
        features = build_feature_frame(
            extended,
            reference_date=reference_date,
            product_code_map=product_code_map,
            date_col=date_col,
            target_col=target_col,
            group_col=group_col,
        )
        # The row for `next_date` is guaranteed to survive build_feature_frame's
        # warm-up dropna because it only has NaN in `target_col` itself, not in
        # any lag/rolling feature column (those are computed from the rows
        # before it, which all have real values).
        last_row = features.loc[features[date_col] == next_date]
        prediction = float(model.predict(last_row[feature_columns])[0])

        predictions.append(prediction)
        new_row = pd.DataFrame(
            {date_col: [next_date], group_col: [product_id], target_col: [prediction]}
        )
        product_history = pd.concat([product_history, new_row], ignore_index=True)

    return predictions
