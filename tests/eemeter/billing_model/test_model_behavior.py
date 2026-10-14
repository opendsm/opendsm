#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter import BillingModel



BALANCE_POINT = 60.0
TEMPERATURE_SLOPE = 1.2
NOISE_SIGMA = 2.0


def _seasonal_temperature(year=2020, seed=1):
    """Hourly seasonal temperature swinging ~25..85 F, padded a month past year-end so
    the billing frame's closing read still has a temperature."""
    hours = pd.date_range(f"{year}-01-01", periods=396 * 24, freq="h", tz="America/Chicago")
    doy = hours.dayofyear.values
    rng = np.random.default_rng(seed)
    values = 55.0 + 30.0 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 2, len(hours))

    return pd.Series(values, index=hours)


def _heating_load(daily_temp, seed):
    """Daily load with a genuine heating response: 80 + 1.2*HDD(60F) + noise."""
    rng = np.random.default_rng(seed)
    base = 80.0 + TEMPERATURE_SLOPE * np.maximum(BALANCE_POINT - daily_temp.values, 0.0)

    return base + rng.normal(0, NOISE_SIGMA, len(daily_temp))


def _billing_frame(daily_load, temperature):
    """Aggregate a daily generative load into monthly billing reads.

    Each read, placed at a month start, carries the summed load for the period running to
    the next read; a closing read one month past the data marks the final period's end with
    no value, matching the billing data convention.
    """
    days = daily_load.index
    daily_temp = temperature.resample("D").mean().reindex(days)
    period_labels = days.tz_localize(None).to_period("M")
    period_sums = daily_load.groupby(period_labels).sum()
    read_dates = pd.DatetimeIndex(
        [period.start_time.tz_localize(days.tz) for period in period_sums.index]
    )
    closing_date = read_dates[-1] + pd.DateOffset(months=1)

    daily_idx = days.union([closing_date])
    observed = pd.Series(np.nan, index=daily_idx)
    observed.loc[read_dates] = period_sums.values

    frame = pd.DataFrame(
        {"observed": observed, "temperature": daily_temp.reindex(daily_idx)},
        index=daily_idx,
    )

    return frame


@pytest.fixture(scope="module")
def temperature():
    return _seasonal_temperature()


@pytest.fixture(scope="module")
def heating_baseline(temperature):
    """BillingModel fit once on a year of heating-responsive daily load, billed monthly."""
    days = pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")
    daily_temp = temperature.resample("D").mean().reindex(days)
    daily_load = pd.Series(_heating_load(daily_temp, seed=2), index=days)
    df = _billing_frame(daily_load, temperature)
    model = BillingModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    return model, df


def _slope_from_predictions(result):
    """OLS slope of predicted load on HDD(60F) at native (daily-spread) resolution."""
    hdd = np.maximum(BALANCE_POINT - result["temperature"], 0.0)
    x = hdd - hdd.mean()
    y = result["predicted"] - result["predicted"].mean()

    return float((x * y).sum() / (x * x).sum())


@pytest.mark.slow
def test_fit_recovers_generative_temperature_slope(heating_baseline):
    """The fit reproduces the generative heating slope of 1.2/HDD.

    Billing regresses a period-constant response on each day's own temperature, which
    pulls the recovered slope to about 1.11; the tolerance is roughly twice that gap.
    """
    model, df = heating_baseline

    result = model.predict(df, aggregation=None)
    slope = _slope_from_predictions(result)

    assert slope == pytest.approx(TEMPERATURE_SLOPE, rel=0.15), (
        f"recovered slope {slope:.3f} vs true {TEMPERATURE_SLOPE}"
    )


@pytest.mark.slow
def test_known_reduction_recovers_as_savings(heating_baseline, temperature):
    """A known per-day reduction injected into an independent load is recovered as savings.

    Reporting observed is a fresh draw of the generative heating load (not the model's own
    prediction) minus a fixed per-day reduction, billed monthly like the baseline, so
    recovery exercises the model's counterfactual rather than reducing to an identity.
    """
    model, _ = heating_baseline
    days = pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")
    daily_temp = temperature.resample("D").mean().reindex(days)
    reduction_per_day = 5.0
    reporting_load = pd.Series(
        _heating_load(daily_temp, seed=4) - reduction_per_day, index=days
    )
    reporting_df = _billing_frame(reporting_load, temperature)

    result = model.predict(reporting_df)
    recovered = (result["predicted"] - result["observed"]).sum()
    expected = reduction_per_day * len(days)

    assert recovered == pytest.approx(expected, rel=0.1)
