#!/usr/bin/env python
"""Run the full walk-forward backtest and save a report under reports/.

Produces:
  reports/backtest_metrics_by_product.csv - per (model, product) averages
  reports/backtest_metrics_summary.csv    - per-model averages across all products
  reports/forecast_vs_actual_<product>.png - a couple of forecast-vs-actual plots

Uses matplotlib's non-interactive Agg backend throughout since this is meant
to run headlessly (including in CI) and must never try to pop up a window.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from time_series_forecasting_service.backtest import generate_walk_forward_folds  # noqa: E402
from time_series_forecasting_service.config import (  # noqa: E402
    DATA_PATH,
    INITIAL_TRAIN_DAYS,
    REPORTS_DIR,
    STEP_DAYS,
    TEST_WINDOW_DAYS,
)
from time_series_forecasting_service.evaluation import (  # noqa: E402
    run_backtest,
    summarize_by_model,
    summarize_by_model_and_product,
)
from time_series_forecasting_service.features import make_product_code_map  # noqa: E402

# Plot one forecast-vs-actual chart for the most recent fold of each of
# these two products - picked to show a stable product and a strongly
# seasonal/shock-driven one side by side.
PLOT_PRODUCTS = ("product_A_grocery_staple", "product_E_holiday_gift")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    reference_date = df["date"].min()
    product_code_map = make_product_code_map(sorted(df["product_id"].unique()))

    folds = generate_walk_forward_folds(
        df["date"],
        initial_train_days=INITIAL_TRAIN_DAYS,
        test_days=TEST_WINDOW_DAYS,
        step_days=STEP_DAYS,
    )
    print(f"Running walk-forward backtest with {len(folds)} folds...")

    results, predictions = run_backtest(
        df, folds, reference_date=reference_date, product_code_map=product_code_map,
        collect_predictions=True,
    )

    by_product = summarize_by_model_and_product(results)
    by_model = summarize_by_model(results)

    by_product.to_csv(REPORTS_DIR / "backtest_metrics_by_product.csv", index=False)
    by_model.to_csv(REPORTS_DIR / "backtest_metrics_summary.csv", index=False)
    results.to_csv(REPORTS_DIR / "backtest_metrics_raw.csv", index=False)

    print("\n=== Per-model, per-product average metrics (across folds) ===")
    print(by_product.to_string(index=False))
    print("\n=== Overall per-model average metrics ===")
    print(by_model.to_string(index=False))

    last_fold_id = results["fold_id"].max()
    for product_id in PLOT_PRODUCTS:
        _plot_forecast_vs_actual(predictions, product_id, last_fold_id)

    print(f"\nReport written to {REPORTS_DIR}")


def _plot_forecast_vs_actual(predictions: pd.DataFrame, product_id: str, fold_id: int) -> None:
    mask = (predictions["product_id"] == product_id) & (predictions["fold_id"] == fold_id)
    subset = predictions[mask]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, group in subset.groupby("model"):
        group = group.sort_values("date")
        label = f"{model_name} (forecast)"
        ax.plot(group["date"], group["y_pred"], marker="o", markersize=3, label=label)

    actual = subset[subset["model"] == subset["model"].unique()[0]].sort_values("date")
    ax.plot(actual["date"], actual["y_true"], color="black", linewidth=2, label="actual")

    ax.set_title(f"Forecast vs. actual - {product_id} (backtest fold {fold_id})")
    ax.set_xlabel("date")
    ax.set_ylabel("sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = REPORTS_DIR / f"forecast_vs_actual_{product_id}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
