"""FastAPI serving layer.

The production model is trained once, lazily, on first use (see
``service.get_forecaster``) rather than at import time, which keeps
``import time_series_forecasting_service.api`` cheap for tooling/tests and
lets tests swap in a small fixture dataset before the first request.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from time_series_forecasting_service import service
from time_series_forecasting_service.models.ml_model import backend_name

app = FastAPI(
    title="Time Series Forecasting Service",
    description=(
        "Daily demand/sales forecasting API backed by walk-forward-validated "
        "per-product best models (classical exponential smoothing or gradient boosting)."
    ),
    version="0.1.0",
)

MAX_HORIZON_DAYS = 90


class ForecastRequest(BaseModel):
    product_id: str = Field(
        ..., description="Synthetic product identifier, e.g. 'product_A_grocery_staple'."
    )
    horizon_days: int = Field(
        ..., ge=1, le=MAX_HORIZON_DAYS, description="Number of days ahead to forecast."
    )


class ForecastPoint(BaseModel):
    date: str
    point_forecast: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    product_id: str
    model_used: str
    generated_at: str
    forecast: list[ForecastPoint]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ml_backend": backend_name()}


@app.get("/products")
def list_products() -> dict:
    forecaster = service.get_forecaster()
    return {"products": forecaster.product_ids}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    forecaster = service.get_forecaster()
    if request.product_id not in forecaster.product_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown product_id '{request.product_id}'. "
                f"Known products: {forecaster.product_ids}"
            ),
        )

    points, model_used = forecaster.forecast(request.product_id, request.horizon_days)
    return ForecastResponse(
        product_id=request.product_id,
        model_used=model_used,
        generated_at=datetime.now(UTC).isoformat(),
        forecast=[ForecastPoint(**p) for p in points],
    )
