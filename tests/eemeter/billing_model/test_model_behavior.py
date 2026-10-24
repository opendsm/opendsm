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

from ..helpers import (
    HEATING_BALANCE_POINT,
    HEATING_SLOPE,
    heating_load,
    seasonal_temperature,
)



def _billing_frame(daily_load, daily_temperature):
    """Aggregate a daily generative load into monthly billing reads.

    The frame's index is the load's days. Each read, placed at a month start, carries the
    summed load for the period running to the next read; the final period runs through the
    frame's last row, which carries no read.
    """
    days = daily_load.index
    period_labels = days.tz_localize(None).to_period("M")
    period_sums = daily_load.groupby(period_labels).sum()
    read_dates = pd.DatetimeIndex(
        [period.start_time.tz_localize(days.tz) for period in period_sums.index]
    )

    observed = pd.Series(np.nan, index=days)
    observed.loc[read_dates] = period_sums.values

    frame = pd.DataFrame({"observed": observed, "temperature": daily_temperature}, index=days)

    return frame


@pytest.fixture(scope="module")
def days():
    return pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")


@pytest.fixture(scope="module")
def daily_temperature(days):
    return seasonal_temperature().resample("D").mean().reindex(days)


@pytest.fixture(scope="module")
def heating_baseline(daily_temperature):
    """BillingModel fit once on a year of heating-responsive daily load, billed monthly."""
    daily_load = heating_load(daily_temperature, seed=2)
    df = _billing_frame(daily_load, daily_temperature)
    model = BillingModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    return model, df


def _slope_from_predictions(result):
    """OLS slope of predicted load on HDD below HEATING_BALANCE_POINT, over a daily-spread
    prediction.
    """
    hdd = np.maximum(HEATING_BALANCE_POINT - result["temperature"], 0.0)
    x = hdd - hdd.mean()
    y = result["predicted"] - result["predicted"].mean()

    return float((x * y).sum() / (x * x).sum())


@pytest.mark.slow
def test_fit_recovers_generative_temperature_slope(heating_baseline):
    """The fit reproduces the generative heating slope of 1.2/HDD.

    Billing regresses a period-constant response on each day's own temperature, which
    pulls the recovered slope a few percent low.
    """
    model, df = heating_baseline

    result = model.predict(df)
    slope = _slope_from_predictions(result)

    assert slope == pytest.approx(HEATING_SLOPE, rel=0.15), (
        f"recovered slope {slope:.3f} vs true {HEATING_SLOPE}"
    )


@pytest.mark.slow
def test_known_reduction_recovers_as_savings(heating_baseline, days, daily_temperature):
    """A known per-day reduction injected into an independent load is recovered as savings.

    Reporting observed is a fresh draw of the load rather than the model's prediction, so the
    recovery tests the counterfactual instead of an identity.
    """
    model, _ = heating_baseline
    reduction_per_day = 5.0
    reporting_load = heating_load(daily_temperature, seed=4) - reduction_per_day
    reporting_df = _billing_frame(reporting_load, daily_temperature)

    result = model.predict(reporting_df)
    recovered = (result["predicted"] - result["observed"]).sum()
    expected = reduction_per_day * len(days)

    assert recovered == pytest.approx(expected, rel=0.1)
