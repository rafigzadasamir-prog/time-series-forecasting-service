"""Gradient-boosted ML forecaster.

Trained once, pooled across every product (with ``product_code`` as a
feature), on the engineered lag/rolling/calendar feature set from
``features.py``. Pooling across products lets the model share patterns
(e.g. the shape of weekly seasonality) across series instead of fitting
each product from scratch, which matters for the shorter series.

LightGBM is used when available since it is the de facto standard
gradient-boosting library for tabular/forecasting problems; if it cannot be
imported (e.g. no prebuilt wheel for a given platform), this module
transparently falls back to scikit-learn's HistGradientBoostingRegressor,
which has the same strengths (native NaN/categorical-ish handling, fast
histogram-based training) without an extra dependency.

Note: ``requirements.txt``/``pyproject.toml`` pin ``lightgbm<4.7`` - the
4.7.0 Windows wheel was found (via this project's own test suite) to crash
with a native access violation inside ``LGBM_DatasetSetField`` when paired
with recent numpy releases, so 4.4-4.6 is the known-good range.
"""

from __future__ import annotations

import pandas as pd

try:
    from lightgbm import LGBMRegressor

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only when lightgbm is absent
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_LIGHTGBM = False


class GradientBoostingForecaster:
    """Thin, backend-agnostic wrapper around LightGBM / HistGradientBoosting."""

    def __init__(self, random_state: int = 42):
        if _HAS_LIGHTGBM:
            self.model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                random_state=random_state,
                verbosity=-1,
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, random_state=random_state
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingForecaster:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)


def backend_name() -> str:
    return "lightgbm" if _HAS_LIGHTGBM else "sklearn_hist_gradient_boosting"
