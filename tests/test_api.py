"""Contract tests for the FastAPI /forecast endpoint.

These exercise the real, committed dataset (data/synthetic_sales.csv) via
the actual Forecaster/ML training path, not a mock - the ``service`` module
caches the trained Forecaster at module/process scope, so the (small) cost
of training is paid once for the whole test session.
"""

import pytest
from fastapi.testclient import TestClient

from time_series_forecasting_service import service
from time_series_forecasting_service.api import app

client = TestClient(app)


@pytest.fixture(scope="module")
def known_product_id() -> str:
    products = client.get("/products").json()["products"]
    assert products, "expected at least one known product"
    return sorted(products)[0]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "ml_backend" in body


def test_products_endpoint_lists_the_five_synthetic_products():
    response = client.get("/products")
    assert response.status_code == 200
    products = response.json()["products"]
    assert len(products) == 5
    assert all(isinstance(p, str) for p in products)


def test_forecast_returns_exactly_the_requested_horizon(known_product_id):
    response = client.post("/forecast", json={"product_id": known_product_id, "horizon_days": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["product_id"] == known_product_id
    assert "model_used" in body
    assert len(body["forecast"]) == 10


def test_forecast_response_contract(known_product_id):
    response = client.post("/forecast", json={"product_id": known_product_id, "horizon_days": 5})
    body = response.json()
    dates = [point["date"] for point in body["forecast"]]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)  # consecutive, non-repeating days
    for point in body["forecast"]:
        assert {"date", "point_forecast", "lower_bound", "upper_bound"} <= point.keys()
        assert point["lower_bound"] <= point["point_forecast"] <= point["upper_bound"]
        assert point["point_forecast"] >= 0


def test_forecast_unknown_product_returns_404():
    response = client.post("/forecast", json={"product_id": "does_not_exist", "horizon_days": 5})
    assert response.status_code == 404


@pytest.mark.parametrize("bad_horizon", [0, -1, 9999])
def test_forecast_invalid_horizon_returns_422(known_product_id, bad_horizon):
    payload = {"product_id": known_product_id, "horizon_days": bad_horizon}
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422


def test_forecast_missing_fields_returns_422():
    response = client.post("/forecast", json={"horizon_days": 5})
    assert response.status_code == 422


def test_get_forecaster_is_cached_across_calls():
    first = service.get_forecaster()
    second = service.get_forecaster()
    assert first is second


def test_model_used_matches_the_real_backtest_winner_per_product():
    """The service must serve whichever model actually backtested best for
    each product (see reports/backtest_metrics_by_product.csv), not always
    the same one - this is the whole point of per-product model selection."""
    forecaster = service.get_forecaster()
    for product_id, expected_model in forecaster.model_selection.items():
        response = client.post("/forecast", json={"product_id": product_id, "horizon_days": 3})
        assert response.json()["model_used"] == expected_model
