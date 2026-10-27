import pandas as pd

from time_series_forecasting_service.backtest import generate_walk_forward_folds


def _dates(n_days: int, start: str = "2023-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_days, freq="D")


def test_no_fold_trains_on_a_date_after_its_own_test_window():
    """The one non-negotiable invariant of walk-forward CV: a fold's training
    window must never include a date later than any date in its test window.
    This is the leakage guard the whole project depends on."""
    dates = _dates(400)
    folds = generate_walk_forward_folds(dates, initial_train_days=200, test_days=30, step_days=30)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.train_end < fold.test_end


def test_test_window_starts_the_day_immediately_after_training_ends():
    dates = _dates(400)
    folds = generate_walk_forward_folds(dates, initial_train_days=200, test_days=30, step_days=30)
    for fold in folds:
        assert fold.test_start == fold.train_end + pd.Timedelta(days=1)


def test_test_window_has_the_requested_length():
    dates = _dates(400)
    test_days = 15
    folds = generate_walk_forward_folds(
        dates, initial_train_days=200, test_days=test_days, step_days=test_days
    )
    for fold in folds:
        assert (fold.test_end - fold.test_start).days == test_days - 1


def test_training_window_expands_and_anchors_at_the_series_start():
    dates = _dates(400)
    folds = generate_walk_forward_folds(dates, initial_train_days=200, test_days=30, step_days=30)
    assert len(folds) >= 2
    for fold in folds:
        assert fold.train_start == dates.min()
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.train_end > earlier.train_end
        assert (later.train_end - earlier.train_end).days == 30


def test_no_data_produces_no_folds():
    dates = _dates(50)
    folds = generate_walk_forward_folds(dates, initial_train_days=200, test_days=30)
    assert folds == []


def test_step_days_defaults_to_test_days():
    dates = _dates(400)
    with_default = generate_walk_forward_folds(dates, initial_train_days=200, test_days=30)
    explicit = generate_walk_forward_folds(
        dates, initial_train_days=200, test_days=30, step_days=30
    )
    assert with_default == explicit
