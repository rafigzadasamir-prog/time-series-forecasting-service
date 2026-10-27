#!/usr/bin/env python
"""Regenerate the committed synthetic dataset at data/synthetic_sales.csv.

Running this script is entirely optional - the CSV it produces is already
committed to the repo so the project is reproducible without re-running
anything - but doing so should always produce a byte-identical file, since
generation is fully seeded and deterministic (see
``time_series_forecasting_service.data_generation``).
"""

from __future__ import annotations

from time_series_forecasting_service.config import DATA_PATH, DATA_SEED, DATA_START_DATE, N_DAYS
from time_series_forecasting_service.data_generation import generate_synthetic_sales


def main() -> None:
    df = generate_synthetic_sales(start_date=DATA_START_DATE, n_days=N_DAYS, seed=DATA_SEED)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Wrote {len(df):,} rows to {DATA_PATH}")
    print(f"Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Products: {sorted(df['product_id'].unique().tolist())}")
    print(df.groupby("product_id")["sales"].agg(["mean", "std", "min", "max"]).round(1))


if __name__ == "__main__":
    main()
