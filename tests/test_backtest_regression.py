"""CI regression guard: a fast backtest on a small subset of synthetic data.

This is intentionally much smaller than the full backtest in
scripts/backtest_report.py (2 products, ~8 months of history instead of 5
products / 3 years) purely so it runs in a couple of seconds in CI. It is
not meant to be a tight accuracy benchmark - the threshold is deliberately
loose - its job is to catch a real regression (e.g. a feature-engineering
bug that silently breaks lag alignment) rather than to chase a specific
score.
"""

from __future__ import annotations

from time_series_forecasting_service.backtest import generate_walk_forward_folds
from time_series_forecasting_service.data_generation import (
    PRODUCT_PROFILES,
    generate_synthetic_sales,
)
from time_series_forecasting_service.evaluation import run_backtest, summarize_by_model
from time_series_forecasting_service.features import make_product_code_map

# Loose regression bound: well above the real, measured MAPE from the full
# backtest (see README), but tight enough to fail loudly if, say, lag
# features silently stopped aligning or a model started predicting garbage.
MAPE_REGRESSION_THRESHOLD = 60.0


def test_fast_backtest_on_small_subset_stays_within_sane_mape_bound():
    small_df = generate_synthetic_sales(n_days=220, seed=42, profiles=PRODUCT_PROFILES[:2])
    reference_date = small_df["date"].min()
    product_code_map = make_product_code_map(sorted(small_df["product_id"].unique()))

    folds = generate_walk_forward_folds(
        small_df["date"], initial_train_days=150, test_days=14, step_days=14
    )
    assert len(folds) >= 2, "expected at least two folds from the small subset"

    results = run_backtest(
        small_df, folds, reference_date=reference_date, product_code_map=product_code_map
    )
    assert not results.empty

    summary = summarize_by_model(results)
    assert (summary["mape"] < MAPE_REGRESSION_THRESHOLD).all(), summary.to_dict(orient="records")


def test_fast_backtest_has_no_leaked_or_missing_predictions():
    small_df = generate_synthetic_sales(n_days=220, seed=42, profiles=PRODUCT_PROFILES[:2])
    reference_date = small_df["date"].min()
    product_code_map = make_product_code_map(sorted(small_df["product_id"].unique()))
    folds = generate_walk_forward_folds(
        small_df["date"], initial_train_days=150, test_days=14, step_days=14
    )

    results = run_backtest(
        small_df, folds, reference_date=reference_date, product_code_map=product_code_map
    )
    # Every fold should have produced results for both models and both products.
    expected_rows = len(folds) * 2 * 2
    assert len(results) == expected_rows
    assert results["mae"].notna().all()
    assert (results["n_obs"] > 0).all()
