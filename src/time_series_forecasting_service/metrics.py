"""Standard point-forecast accuracy metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, eps: float = 1.0) -> float:
    """Mean absolute percentage error, as a percentage.

    ``eps`` floors the denominator so that a handful of near-zero actuals
    (possible in the synthetic data, e.g. around a stockout shock) cannot
    make a single day dominate or blow up the whole metric.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_all_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
