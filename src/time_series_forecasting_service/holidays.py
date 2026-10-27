"""A small, fixed US holiday calendar.

This is shared by two very different consumers:

1. ``data_generation`` uses it to inject realistic holiday demand shocks
   (e.g. a gift product spiking around Christmas / Black Friday).
2. ``features`` uses it to derive an ``is_holiday`` feature for the models.

Computing holidays from simple date rules (rather than depending on the
third-party ``holidays`` package) keeps the dependency list short for what
is, in the end, six fixed dates a year.
"""

from __future__ import annotations

from datetime import date, timedelta

# Monday=0 ... Sunday=6, matching ``datetime.date.weekday()``.
_MONDAY = 0
_THURSDAY = 3


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) occurrence of ``weekday`` in ``month``/``year``."""
    first_of_month = date(year, month, 1)
    offset = (weekday - first_of_month.weekday()) % 7
    first_occurrence = first_of_month + timedelta(days=offset)
    return first_occurrence + timedelta(weeks=n - 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """The last occurrence of ``weekday`` in ``month``/``year``."""
    if month == 12:
        first_of_next_month = date(year + 1, 1, 1)
    else:
        first_of_next_month = date(year, month + 1, 1)
    last_day = first_of_next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def us_holidays_for_year(year: int) -> dict[date, str]:
    """Return ``{date: holiday_name}`` for a fixed set of US holidays."""
    thanksgiving = _nth_weekday_of_month(year, 11, _THURSDAY, 4)
    holidays = {
        date(year, 1, 1): "new_years_day",
        _last_weekday_of_month(year, 5, _MONDAY): "memorial_day",
        date(year, 7, 4): "independence_day",
        _nth_weekday_of_month(year, 9, _MONDAY, 1): "labor_day",
        thanksgiving: "thanksgiving",
        thanksgiving + timedelta(days=1): "black_friday",
        date(year, 12, 25): "christmas",
    }
    return holidays


def holiday_name(d: date) -> str | None:
    """Name of the holiday on ``d``, or ``None`` if it is not one."""
    return us_holidays_for_year(d.year).get(d)


def is_holiday(d: date) -> bool:
    return holiday_name(d) is not None


def all_holidays(start: date, end: date) -> dict[date, str]:
    """All ``{date: holiday_name}`` entries with ``start <= date <= end``."""
    combined: dict[date, str] = {}
    for year in range(start.year, end.year + 1):
        combined.update(us_holidays_for_year(year))
    return {d: name for d, name in combined.items() if start <= d <= end}
