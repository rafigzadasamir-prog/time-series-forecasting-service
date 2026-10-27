import numpy as np
import pytest

from time_series_forecasting_service.metrics import compute_all_metrics, mae, mape, rmse


def test_mae_matches_hand_calculation():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7 -> mean = 7/3
    assert mae(y_true, y_pred) == pytest.approx(7 / 3)


def test_mae_is_zero_for_a_perfect_forecast():
    assert mae([5, 6, 7], [5, 6, 7]) == pytest.approx(0.0)


def test_rmse_matches_hand_calculation():
    y_true = [0, 0, 0]
    y_pred = [3, 4, 0]
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 10]
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_mape_matches_hand_calculation():
    y_true = [100, 200]
    y_pred = [110, 180]
    expected = ((10 / 100) + (20 / 200)) / 2 * 100
    assert mape(y_true, y_pred) == pytest.approx(expected)


def test_mape_floors_near_zero_actuals_with_eps():
    y_true = [0]
    y_pred = [5]
    assert mape(y_true, y_pred, eps=1.0) == pytest.approx(500.0)


def test_compute_all_metrics_matches_individual_functions():
    y_true = [10, 20, 30, 40]
    y_pred = [11, 19, 33, 36]
    combined = compute_all_metrics(y_true, y_pred)
    assert combined["mae"] == pytest.approx(mae(y_true, y_pred))
    assert combined["rmse"] == pytest.approx(rmse(y_true, y_pred))
    assert combined["mape"] == pytest.approx(mape(y_true, y_pred))
