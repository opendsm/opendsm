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

from opendsm.eemeter import HourlyModel, HourlyNonSolarSettings



BALANCE_POINT = 65.0
TEMPERATURE_SLOPE = 0.15
NOISE_SIGMA = 0.05


def _seasonal_temperature(year=2020, seed=1):
    """Hourly temperature with a seasonal swing and a daily cycle, roughly 20..90 F."""
    hours = pd.date_range(
        f"{year}-01-01", f"{year}-12-31 23:00", freq="h", tz="America/Chicago"
    )
    doy = hours.dayofyear.values
    hour_of_day = hours.hour.values
    rng = np.random.default_rng(seed)
    values = (
        55.0
        + 28.0 * np.sin(2 * np.pi * (doy - 110) / 365)
        + 8.0 * np.sin(2 * np.pi * (hour_of_day - 15) / 24)
        + rng.normal(0, 2.0, len(hours))
    )

    return pd.Series(values, index=hours)


def _cooling_load(temperature, seed):
    """Hour-of-week profile plus a linear response above the balance point, with noise."""
    hour_of_day = temperature.index.hour.values
    is_weekend = (temperature.index.dayofweek.values >= 5).astype(float)
    rng = np.random.default_rng(seed)
    profile = 1.2 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24) + 0.3 * is_weekend
    response = TEMPERATURE_SLOPE * np.maximum(temperature.values - BALANCE_POINT, 0.0)
    values = profile + response + rng.normal(0, NOISE_SIGMA, len(temperature))

    return pd.Series(values, index=temperature.index)


def _hourly_frame(meter, temperature):
    """Combine an hourly meter series and an hourly temperature series into a model input frame."""
    frame = pd.DataFrame({"observed": meter, "temperature": temperature})

    return frame


def _slope_above_balance_point(result):
    """OLS slope of predicted load on temperature over hours above the balance point.

    Hour-of-week means are removed from both series first, so the daily
    temperature cycle cannot leak the load profile into the slope estimate.
    """
    above = result[result["temperature"] > BALANCE_POINT]
    hour_of_week = above.index.dayofweek.values * 24 + above.index.hour.values
    x = above["temperature"] - above.groupby(hour_of_week)["temperature"].transform("mean")
    y = above["predicted"] - above.groupby(hour_of_week)["predicted"].transform("mean")
    slope = float((x * y).sum() / (x * x).sum())

    return slope


@pytest.fixture(scope="module")
def temperature():
    return _seasonal_temperature()


@pytest.fixture(scope="module")
def cooling_baseline(temperature):
    """HourlyModel fit once on a load with a real cooling response, with its data."""
    df = _hourly_frame(_cooling_load(temperature, seed=2), temperature)
    settings = HourlyNonSolarSettings(seed=42)
    model = HourlyModel(settings=settings).fit(
        df, is_electricity_data=True, enforce_compliance=False
    )

    return model, df


@pytest.mark.slow
@pytest.mark.hourly_model
def test_fit_recovers_generative_temperature_slope(cooling_baseline):
    """The fit reproduces the generative response above the balance point.

    The hourly model exposes no single temperature coefficient, so the slope is
    recovered by regressing predicted load on temperature over the hours above
    the balance point. Generator noise (sigma 0.05 on a 0.15/F response over
    thousands of hours) puts the sampling error on that regression well under a
    percent; the tolerance is set an order of magnitude wider to cover the
    smoothing the clustered hour-of-week fit applies to the response.
    """
    model, df = cooling_baseline

    result = model.predict(df)
    slope = _slope_above_balance_point(result)

    assert slope == pytest.approx(TEMPERATURE_SLOPE, rel=0.25)


@pytest.mark.slow
@pytest.mark.hourly_model
def test_known_reduction_recovers_as_savings(cooling_baseline, temperature):
    """A known per-hour reduction injected into an independent load is recovered as savings.

    Reporting observed is a fresh draw of the generative load (not the model's
    own prediction) minus a fixed per-hour reduction, so recovery exercises the
    model's counterfactual rather than reducing to an identity.
    """
    model, _ = cooling_baseline
    reduction_per_hour = 0.5
    reporting_meter = _cooling_load(temperature, seed=4) - reduction_per_hour
    reporting_df = _hourly_frame(reporting_meter, temperature)

    result = model.predict(reporting_df)
    recovered = (result["predicted"] - result["observed"]).sum()
    expected = reduction_per_hour * result["observed"].notna().sum()

    assert recovered == pytest.approx(expected, rel=0.1)
