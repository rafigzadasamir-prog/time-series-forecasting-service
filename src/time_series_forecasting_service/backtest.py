"""Walk-forward (rolling-origin) cross-validation.

This is the only backtesting strategy used anywhere in this project. A
random train/test split is never appropriate for time series: it would let
a model train on rows that happen to fall *after* a test row chronologically,
which is a leakage of future information into the past and produces
unrealistically optimistic accuracy estimates. Walk-forward validation
instead always expands the training window forward in time and evaluates
on a block of dates strictly after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_folds(
    dates: pd.Series | pd.DatetimeIndex,
    initial_train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[Fold]:
    """Build an expanding-window walk-forward schedule.

    Fold 0 trains on the first ``initial_train_days`` days and tests on the
    following ``test_days`` days. Each subsequent fold keeps the same
    ``train_start`` (the window only ever expands) and pushes ``train_end``
    forward by ``step_days``, so training data for fold k+1 is a strict
    superset of fold k's, and every fold's test block starts the day after
    its train block ends - by construction a fold's training window can
    never include a date after its own test window.
    """
    step_days = step_days or test_days
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    n = len(unique_dates)

    folds: list[Fold] = []
    train_end_idx = initial_train_days - 1
    fold_id = 0
    while train_end_idx + test_days < n:
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=unique_dates[0],
                train_end=unique_dates[train_end_idx],
                test_start=unique_dates[test_start_idx],
                test_end=unique_dates[test_end_idx],
            )
        )
        fold_id += 1
        train_end_idx += step_days

    return folds
