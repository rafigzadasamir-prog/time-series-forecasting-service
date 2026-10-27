"""Deterministic synthetic daily demand/sales generator.

Produces ~3 years of daily sales for a handful of synthetic products with
a trend component, weekly seasonality, yearly seasonality, holiday effects,
a few one-off "shock" events (promotions, stockouts, a product launch), and
multiplicative noise. Everything is driven by a single seeded
``numpy.random.Generator`` so the exact same CSV is produced every time the
script is re-run - there is no network access or external download involved,
which is why the resulting CSV is committed to the repo instead of being
treated as a build artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from time_series_forecasting_service.holidays import holiday_name


@dataclass(frozen=True)
class Shock:
    """A one-off demand shock applied to a contiguous block of days.

    ``start_day`` is the 0-indexed offset from the series start date.
    """

    start_day: int
    duration_days: int
    multiplier: float
    name: str


@dataclass(frozen=True)
class ProductProfile:
    """Deterministic demand model parameters for one synthetic product."""

    product_id: str
    base_level: float
    trend_per_day: float
    # Multiplicative day-of-week effect, Monday-first (index 0 = Monday).
    weekly_multipliers: tuple[float, float, float, float, float, float, float]
    yearly_amplitude: float
    # Day-of-year (1-366) on which the yearly sinusoid peaks.
    yearly_peak_doy: int
    noise_frac: float
    # Multiplicative effect applied on specific named holidays.
    holiday_multipliers: dict[str, float] = field(default_factory=dict)
    shocks: tuple[Shock, ...] = field(default_factory=tuple)


PRODUCT_PROFILES: tuple[ProductProfile, ...] = (
    ProductProfile(
        product_id="product_A_grocery_staple",
        base_level=150.0,
        trend_per_day=0.015,
        weekly_multipliers=(0.95, 0.93, 0.95, 1.00, 1.05, 1.25, 1.20),
        yearly_amplitude=0.06,
        yearly_peak_doy=319,  # mid-November: holiday cooking build-up
        noise_frac=0.08,
        holiday_multipliers={"thanksgiving": 1.6, "christmas": 1.4},
        shocks=(Shock(start_day=430, duration_days=5, multiplier=1.45, name="promo_week"),),
    ),
    ProductProfile(
        product_id="product_B_ice_cream_seasonal",
        base_level=80.0,
        trend_per_day=0.006,
        weekly_multipliers=(0.90, 0.90, 0.95, 1.00, 1.05, 1.30, 1.25),
        yearly_amplitude=0.60,
        yearly_peak_doy=196,  # mid-July: peak summer demand
        noise_frac=0.12,
        holiday_multipliers={"independence_day": 1.5, "memorial_day": 1.3},
        shocks=(Shock(start_day=910, duration_days=4, multiplier=1.5, name="heatwave"),),
    ),
    ProductProfile(
        product_id="product_C_growth_gadget",
        base_level=40.0,
        trend_per_day=0.075,
        weekly_multipliers=(1.00, 1.00, 1.00, 1.00, 1.05, 1.10, 1.05),
        yearly_amplitude=0.18,
        yearly_peak_doy=335,  # early December: holiday gifting
        noise_frac=0.15,
        holiday_multipliers={"black_friday": 2.2, "christmas": 1.3},
        shocks=(
            Shock(start_day=250, duration_days=10, multiplier=1.8, name="viral_marketing_campaign"),
        ),
    ),
    ProductProfile(
        product_id="product_D_declining_legacy",
        base_level=200.0,
        trend_per_day=-0.05,
        weekly_multipliers=(1.00, 1.00, 1.00, 1.00, 1.00, 1.05, 1.00),
        yearly_amplitude=0.05,
        yearly_peak_doy=15,
        noise_frac=0.10,
        holiday_multipliers={},
        shocks=(
            Shock(start_day=600, duration_days=7, multiplier=0.4, name="supply_chain_disruption"),
        ),
    ),
    ProductProfile(
        product_id="product_E_holiday_gift",
        base_level=30.0,
        trend_per_day=0.0,
        weekly_multipliers=(0.90, 0.90, 0.95, 1.00, 1.10, 1.20, 1.10),
        yearly_amplitude=0.10,
        yearly_peak_doy=350,
        noise_frac=0.20,
        holiday_multipliers={
            "thanksgiving": 3.0,
            "black_friday": 6.0,
            "christmas": 4.5,
            "new_years_day": 0.3,
        },
        shocks=(),
    ),
)


def _expected_demand(profile: ProductProfile, dates: pd.DatetimeIndex) -> np.ndarray:
    """Deterministic (noise-free) expected demand for every date."""
    day_idx = np.arange(len(dates))
    trend = profile.base_level + profile.trend_per_day * day_idx

    weekday = dates.day_of_week.to_numpy()  # Monday=0
    weekly_mult = np.asarray(profile.weekly_multipliers)[weekday]

    doy = dates.day_of_year.to_numpy()
    yearly_mult = 1.0 + profile.yearly_amplitude * np.cos(
        2 * np.pi * (doy - profile.yearly_peak_doy) / 365.25
    )

    holiday_mult = np.ones(len(dates))
    if profile.holiday_multipliers:
        names = [holiday_name(d.date()) for d in dates]
        holiday_mult = np.array(
            [profile.holiday_multipliers.get(name, 1.0) if name else 1.0 for name in names]
        )

    shock_mult = np.ones(len(dates))
    for shock in profile.shocks:
        shock_mult[shock.start_day : shock.start_day + shock.duration_days] = shock.multiplier

    expected = trend * weekly_mult * yearly_mult * holiday_mult * shock_mult
    return np.clip(expected, a_min=1.0, a_max=None)


def generate_synthetic_sales(
    start_date: str = "2023-01-01",
    n_days: int = 1096,
    seed: int = 42,
    profiles: tuple[ProductProfile, ...] = PRODUCT_PROFILES,
) -> pd.DataFrame:
    """Generate the full synthetic daily sales dataset.

    Deterministic for a fixed ``(start_date, n_days, seed, profiles)``: the
    same arguments always produce byte-identical output, which is what lets
    the generated CSV be committed to the repo and trusted as a fixture.

    Returns a long-format DataFrame with columns ``date``, ``product_id``,
    ``sales`` (non-negative integers), sorted by ``product_id`` then ``date``.
    """
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)

    frames = []
    for profile in profiles:
        expected = _expected_demand(profile, dates)
        noise = rng.normal(loc=0.0, scale=profile.noise_frac * expected, size=len(dates))
        sales = np.clip(np.round(expected + noise), a_min=0, a_max=None).astype(int)
        frames.append(
            pd.DataFrame({"date": dates, "product_id": profile.product_id, "sales": sales})
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def product_ids(profiles: tuple[ProductProfile, ...] = PRODUCT_PROFILES) -> list[str]:
    return [p.product_id for p in profiles]
