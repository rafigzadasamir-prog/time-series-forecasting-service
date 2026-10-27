"""Trains and holds the production forecasting model(s) used by the API.

The walk-forward backtest (``scripts/backtest_report.py``, real numbers in
the README) does *not* show one model winning everywhere: the exponential
smoothing baseline wins on overall average MAPE, but the gradient boosting
model wins specifically on the holiday-driven product, where its
``is_holiday`` feature captures sharp Black Friday/Christmas spikes that a
7-day additive seasonal component structurally cannot. Rather than
hardcoding a single "winner", the service loads the committed backtest
results (``reports/backtest_metrics_by_product.csv``) and serves whichever
model actually backtested best *for that specific product* - this is the
literal meaning of "use the best-performing model" once you accept that
"best" can depend on the series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from time_series_forecasting_service.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from time_series_forecasting_service.evaluation import BASELINE_NAME, ML_MODEL_NAME
from time_series_forecasting_service.features import build_feature_frame, make_product_code_map
from time_series_forecasting_service.forecasting import recursive_forecast
from time_series_forecasting_service.models.baseline import ExponentialSmoothingBaseline
from time_series_forecasting_service.models.ml_model import GradientBoostingForecaster

BACKTEST_METRICS_PATH = REPORTS_DIR / "backtest_metrics_by_product.csv"

# Holdout window used only to estimate a per-product residual std for simple
# prediction intervals - this holdout never influences model selection or
# how the final, served model is trained.
INTERVAL_HOLDOUT_DAYS = 60
Z_SCORE_95 = 1.96


def load_model_selection(path=BACKTEST_METRICS_PATH) -> dict[str, str] | None:
    """Per-product {product_id: model_name}, picking the lower-MAPE model.

    Returns None if no backtest report has been generated yet (e.g. a very
    first checkout before ``scripts/backtest_report.py`` has run), in which
    case the caller falls back to always serving the ML model.
    """
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    best_rows = metrics.loc[metrics.groupby("product_id")["mape"].idxmin()]
    return dict(zip(best_rows["product_id"], best_rows["model"], strict=False))


class Forecaster:
    """Owns the trained model(s), the full history, and interval widths."""

    def __init__(self, history: pd.DataFrame, model_selection: dict[str, str] | None = None):
        self.history = history.sort_values(["product_id", "date"]).reset_index(drop=True)
        self.reference_date = self.history["date"].min()
        self.product_ids = sorted(self.history["product_id"].unique())
        self.product_code_map = make_product_code_map(self.product_ids)

        if model_selection is not None:
            loaded_selection = model_selection
        else:
            loaded_selection = load_model_selection()
        # No backtest report on disk yet: default every product to the ML model.
        self.model_selection = loaded_selection or {pid: ML_MODEL_NAME for pid in self.product_ids}

        self.ml_model = self._train_ml_model(self.history)
        self.residual_std = self._estimate_residual_std()

    def _train_ml_model(self, history: pd.DataFrame) -> GradientBoostingForecaster:
        features = build_feature_frame(
            history, reference_date=self.reference_date, product_code_map=self.product_code_map
        )
        return GradientBoostingForecaster().fit(features[FEATURE_COLUMNS], features[TARGET_COLUMN])

    def _baseline_forecast(
        self, product_id: str, train_df: pd.DataFrame, horizon: int
    ) -> np.ndarray:
        product_rows = train_df.loc[train_df["product_id"] == product_id]
        series = product_rows.sort_values("date")[TARGET_COLUMN]
        return ExponentialSmoothingBaseline().fit(series).predict(horizon)

    def _ml_forecast(
        self,
        product_id: str,
        history: pd.DataFrame,
        forecast_dates: list[pd.Timestamp],
        model: GradientBoostingForecaster,
    ) -> list[float]:
        return recursive_forecast(
            model,
            history=history,
            product_id=product_id,
            forecast_dates=forecast_dates,
            reference_date=self.reference_date,
            product_code_map=self.product_code_map,
            feature_columns=FEATURE_COLUMNS,
        )

    def _estimate_residual_std(self) -> dict[str, float]:
        max_date = self.history["date"].max()
        holdout_start = max_date - pd.Timedelta(days=INTERVAL_HOLDOUT_DAYS - 1)
        train_df = self.history[self.history["date"] < holdout_start]
        holdout_mask = self.history["date"] >= holdout_start
        unique_holdout_dates = sorted(self.history.loc[holdout_mask, "date"].unique())
        holdout_dates = list(pd.DatetimeIndex(unique_holdout_dates))

        needs_ml_holdout = any(m == ML_MODEL_NAME for m in self.model_selection.values())
        holdout_ml_model = self._train_ml_model(train_df) if needs_ml_holdout else None

        residual_std: dict[str, float] = {}
        for product_id in self.product_ids:
            product_holdout_mask = (self.history["product_id"] == product_id) & holdout_mask
            actuals = self.history.loc[product_holdout_mask, TARGET_COLUMN].to_numpy()
            if len(actuals) == 0:
                residual_std[product_id] = 1.0
                continue

            if self.model_selection.get(product_id) == BASELINE_NAME:
                preds = self._baseline_forecast(product_id, train_df, len(holdout_dates))
            else:
                preds = self._ml_forecast(product_id, train_df, holdout_dates, holdout_ml_model)
            residual_std[product_id] = float(np.std(actuals - np.asarray(preds)))
        return residual_std

    def forecast(self, product_id: str, horizon_days: int) -> tuple[list[dict], str]:
        if product_id not in self.product_ids:
            raise KeyError(product_id)

        model_name = self.model_selection.get(product_id, ML_MODEL_NAME)
        last_date = self.history.loc[self.history["product_id"] == product_id, "date"].max()
        forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

        if model_name == BASELINE_NAME:
            point_forecasts = self._baseline_forecast(product_id, self.history, horizon_days)
        else:
            point_forecasts = self._ml_forecast(
                product_id, self.history, forecast_dates, self.ml_model
            )

        std = self.residual_std.get(product_id, 1.0)
        results = []
        paired = zip(forecast_dates, point_forecasts, strict=False)
        for step, (forecast_date, point) in enumerate(paired, start=1):
            # Daily errors are treated as roughly independent, so the
            # uncertainty band widens with sqrt(step) across the horizon.
            width = Z_SCORE_95 * std * (step**0.5)
            point = max(float(point), 0.0)
            results.append(
                {
                    "date": forecast_date.date().isoformat(),
                    "point_forecast": round(point, 2),
                    "lower_bound": round(max(point - width, 0.0), 2),
                    "upper_bound": round(point + width, 2),
                }
            )
        return results, model_name


def load_history() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


_forecaster: Forecaster | None = None


def get_forecaster() -> Forecaster:
    """Lazily train and cache the single process-wide Forecaster instance."""
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster(load_history())
    return _forecaster


def reset_forecaster_cache() -> None:
    """Used by tests to force a fresh Forecaster (e.g. on a tiny fixture CSV)."""
    global _forecaster
    _forecaster = None
