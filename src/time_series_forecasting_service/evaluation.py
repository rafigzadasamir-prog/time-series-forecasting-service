"""Runs both forecasting approaches across every walk-forward fold.

Kept separate from ``backtest.py`` (which only generates the fold schedule
and has no model dependencies) so the split logic itself stays trivially
unit-testable in isolation from statsmodels/LightGBM.
"""

from __future__ import annotations

import pandas as pd

from time_series_forecasting_service.backtest import Fold
from time_series_forecasting_service.config import FEATURE_COLUMNS, TARGET_COLUMN
from time_series_forecasting_service.features import build_feature_frame
from time_series_forecasting_service.forecasting import recursive_forecast
from time_series_forecasting_service.metrics import compute_all_metrics
from time_series_forecasting_service.models.baseline import ExponentialSmoothingBaseline
from time_series_forecasting_service.models.ml_model import GradientBoostingForecaster

BASELINE_NAME = "exp_smoothing_baseline"
ML_MODEL_NAME = "gradient_boosting"


def run_backtest(
    df: pd.DataFrame,
    folds: list[Fold],
    reference_date: pd.Timestamp,
    product_code_map: dict[str, int],
    feature_columns: list[str] = FEATURE_COLUMNS,
    date_col: str = "date",
    group_col: str = "product_id",
    target_col: str = TARGET_COLUMN,
    collect_predictions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the baseline and the ML model on every fold and product.

    Returns one row per (fold, product, model) with mae/rmse/mape. When
    ``collect_predictions`` is True, also returns a second long-format
    DataFrame of every individual (date, product, model) prediction vs.
    actual, which the report script uses to plot forecast-vs-actual curves.
    """
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    product_ids = sorted(df[group_col].unique())

    for fold in folds:
        train_mask = df[date_col].between(fold.train_start, fold.train_end)
        test_mask = df[date_col].between(fold.test_start, fold.test_end)
        df_train = df.loc[train_mask].sort_values([group_col, date_col])
        df_test = df.loc[test_mask].sort_values([group_col, date_col])

        # --- Classical baseline: one univariate Holt-Winters fit per product ---
        for product_id in product_ids:
            train_series = df_train.loc[df_train[group_col] == product_id].sort_values(date_col)
            test_series = df_test.loc[df_test[group_col] == product_id].sort_values(date_col)
            if train_series.empty or test_series.empty:
                continue
            baseline = ExponentialSmoothingBaseline().fit(train_series[target_col])
            preds = baseline.predict(len(test_series))
            actual = test_series[target_col].to_numpy()
            rows.append(_result_row(fold, product_id, BASELINE_NAME, actual, preds))
            if collect_predictions:
                prediction_rows.extend(
                    _prediction_rows(
                        fold, product_id, BASELINE_NAME, test_series[date_col], actual, preds
                    )
                )

        # --- ML model: trained once per fold, pooled across products ---
        train_features = build_feature_frame(
            df_train, reference_date=reference_date, product_code_map=product_code_map,
            date_col=date_col, target_col=target_col, group_col=group_col,
        )
        if train_features.empty:
            continue
        ml_model = GradientBoostingForecaster().fit(
            train_features[feature_columns], train_features[target_col]
        )

        test_dates = sorted(df_test[date_col].unique())
        for product_id in product_ids:
            test_series = df_test.loc[df_test[group_col] == product_id].sort_values(date_col)
            if test_series.empty:
                continue
            preds = recursive_forecast(
                ml_model,
                history=df_train,
                product_id=product_id,
                forecast_dates=list(pd.DatetimeIndex(test_dates)),
                reference_date=reference_date,
                product_code_map=product_code_map,
                feature_columns=feature_columns,
                date_col=date_col,
                group_col=group_col,
                target_col=target_col,
            )
            actual = test_series[target_col].to_numpy()
            rows.append(_result_row(fold, product_id, ML_MODEL_NAME, actual, preds))
            if collect_predictions:
                prediction_rows.extend(
                    _prediction_rows(
                        fold, product_id, ML_MODEL_NAME, test_series[date_col], actual, preds
                    )
                )

    results = pd.DataFrame(rows)
    if collect_predictions:
        return results, pd.DataFrame(prediction_rows)
    return results


def _prediction_rows(
    fold: Fold, product_id: str, model_name: str, dates, y_true, y_pred
) -> list[dict]:
    return [
        {
            "fold_id": fold.fold_id,
            "product_id": product_id,
            "model": model_name,
            "date": d,
            "y_true": float(t),
            "y_pred": float(p),
        }
        for d, t, p in zip(dates, y_true, y_pred, strict=False)
    ]


def _result_row(fold: Fold, product_id: str, model_name: str, y_true, y_pred) -> dict:
    metrics = compute_all_metrics(y_true, y_pred)
    return {
        "fold_id": fold.fold_id,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "product_id": product_id,
        "model": model_name,
        "n_obs": len(y_true),
        **metrics,
    }


def summarize_by_model_and_product(results: pd.DataFrame) -> pd.DataFrame:
    """Average mae/rmse/mape across folds, per (model, product)."""
    return (
        results.groupby(["model", "product_id"])[["mae", "rmse", "mape"]]
        .mean()
        .reset_index()
        .sort_values(["product_id", "model"])
    )


def summarize_by_model(results: pd.DataFrame) -> pd.DataFrame:
    """Average mae/rmse/mape across folds and products, per model."""
    return (
        results.groupby("model")[["mae", "rmse", "mape"]]
        .mean()
        .reset_index()
        .sort_values("model")
    )
