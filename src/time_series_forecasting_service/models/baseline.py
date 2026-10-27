"""Classical statistical baseline: Holt-Winters exponential smoothing.

One univariate model is fit per product directly on its raw sales series -
no engineered features are needed for this approach, which is the point of
including it: it is the simple, cheap thing a more expensive ML model has
to beat in order to justify its complexity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExponentialSmoothingBaseline:
    """Additive-trend, additive-weekly-seasonal Holt-Winters forecaster.

    Falls back to a seasonal-naive forecast (repeat the value from the same
    weekday in the most recent full seasonal cycle) if statsmodels fails to
    converge, which can happen on short or unusually flat training windows.
    This keeps the backtest loop from crashing across dozens of
    product/fold combinations without needing per-series manual tuning.
    """

    def __init__(self, seasonal_periods: int = 7):
        self.seasonal_periods = seasonal_periods
        self._train_tail: np.ndarray | None = None
        self._fitted = None

    def fit(self, train_series: pd.Series) -> ExponentialSmoothingBaseline:
        values = train_series.to_numpy(dtype=float)
        self._train_tail = values[-self.seasonal_periods :]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                )
                self._fitted = model.fit(optimized=True)
        except Exception:
            self._fitted = None
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._fitted is not None:
            try:
                forecast = np.asarray(self._fitted.forecast(horizon), dtype=float)
                if np.all(np.isfinite(forecast)):
                    return forecast
            except Exception:
                pass
        return self._seasonal_naive(horizon)

    def _seasonal_naive(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / self.seasonal_periods))
        return np.tile(self._train_tail, reps)[:horizon]
