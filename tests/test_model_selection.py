"""Unit tests for picking the per-product best model from a backtest report,
isolated from the real committed CSV so they don't depend on it changing."""

import pandas as pd
import pytest

from time_series_forecasting_service.service import load_model_selection


@pytest.fixture
def fixture_metrics_path(tmp_path) -> "pytest.fixture":
    # p1: the baseline has the lower MAPE. p2: the ML model has the lower MAPE.
    metrics = pd.DataFrame(
        {
            "model": ["exp_smoothing_baseline", "gradient_boosting"] * 2,
            "product_id": ["p1", "p1", "p2", "p2"],
            "mae": [10, 11, 5, 4],
            "rmse": [12, 13, 6, 5],
            "mape": [8.0, 9.0, 20.0, 15.0],
        }
    )
    path = tmp_path / "backtest_metrics_by_product.csv"
    metrics.to_csv(path, index=False)
    return path


def test_picks_the_lower_mape_model_per_product(fixture_metrics_path):
    selection = load_model_selection(fixture_metrics_path)
    assert selection == {"p1": "exp_smoothing_baseline", "p2": "gradient_boosting"}


def test_returns_none_when_no_report_exists(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    assert load_model_selection(missing_path) is None
