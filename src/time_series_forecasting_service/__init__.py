"""Time-series demand/sales forecasting service.

Package layout:
    data_generation  - deterministic synthetic daily sales generator
    holidays         - fixed US holiday calendar (shared by generator + features)
    features         - calendar/lag/rolling feature engineering
    backtest         - walk-forward (rolling-origin) cross-validation
    metrics          - MAE / RMSE / MAPE
    forecasting      - recursive multi-step forecasting shared by backtest + API
    models/          - the two forecasting approaches being compared
    service          - trains/holds the production model(s) for serving
    api              - FastAPI app exposing POST /forecast
"""

__version__ = "0.1.0"
