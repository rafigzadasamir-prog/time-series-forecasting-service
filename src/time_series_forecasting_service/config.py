"""Central paths and shared constants.

Keeping these in one place means the data generator, the backtest script,
the tests and the API all agree on where the dataset lives and which
products/feature knobs to use - there is exactly one source of truth.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "synthetic_sales.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Random seed for the synthetic data generator. Fixed so the committed CSV
# is exactly reproducible by anyone who re-runs scripts/generate_data.py.
DATA_SEED = 42

# ~3 years of daily history.
DATA_START_DATE = "2023-01-01"
N_DAYS = 1096

# Lag / rolling-window feature engineering knobs, shared by training,
# backtesting and the recursive serving forecaster so that all three see
# an identical feature contract.
LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)

# Walk-forward backtest configuration.
# 730 days (two full yearly cycles) of initial training history is required
# before the first fold so that yearly seasonality has actually been
# observed at least twice by the time a model is scored on it.
INITIAL_TRAIN_DAYS = 730
# A 30-day test window mirrors a typical monthly operational forecasting
# horizon, and stepping forward by the same 30 days keeps folds
# non-overlapping so each fold scores an independent stretch of the future.
TEST_WINDOW_DAYS = 30
STEP_DAYS = 30

FEATURE_COLUMNS = [
    "day_of_week",
    "is_weekend",
    "month",
    "doy_sin",
    "doy_cos",
    "is_holiday",
    "trend_idx",
    "product_code",
    *[f"lag_{lag}" for lag in LAGS],
    *[f"roll_mean_{w}" for w in ROLLING_WINDOWS],
]

TARGET_COLUMN = "sales"
