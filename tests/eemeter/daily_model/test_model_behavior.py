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

from opendsm.eemeter import DailyBaselineData, DailyModel, DailyReportingData



def _seasonal_temperature(year=2020, seed=1):
    """Hourly seasonal temperature swinging ~25..85 F across the year."""
    hours = pd.date_range(f"{year}-01-01", periods=365 * 24, freq="h", tz="America/Chicago")
    doy = hours.dayofyear.values
    rng = np.random.default_rng(seed)
    values = 55.0 + 30.0 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 2, len(hours))

    return pd.Series(values, index=hours)


@pytest.fixture(scope="module")
def temperature():
    return _seasonal_temperature()


def _heating_load(daily_temp, seed):
    """Daily load with a genuine heating response: 80 + 1.2*HDD(60F) + noise."""
    rng = np.random.default_rng(seed)
    base = 80.0 + 1.2 * np.maximum(60.0 - daily_temp.values, 0.0)

    return base + rng.normal(0, 2.0, len(daily_temp))


@pytest.fixture(scope="module")
def heating_baseline(temperature):
    """DailyModel fit once on a load with a real heating response, with its data."""
    days = pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")
    daily_temp = temperature.resample("D").mean().reindex(days)
    meter = pd.Series(_heating_load(daily_temp, seed=2), index=days)
    data = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)
    model = DailyModel().fit(data, ignore_disqualification=True)

    return model, data


@pytest.mark.slow
def test_flat_load_does_not_fabricate_temperature_response(temperature):
    """A temperature-insensitive load yields negligible HDD/CDD slopes.

    The selector may still pick a fuller model_type, but the fitted heating and
    cooling betas must be tiny relative to the intercept (no invented response),
    and the prediction must track the mean rather than the temperature swing.
    """
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")
    meter = pd.Series(40.0 + rng.normal(0, 1.0, 365), index=days)
    data = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    model = DailyModel().fit(data, ignore_disqualification=True)

    for submodel in model.params.submodels.values():
        coef = submodel.coefficients
        for beta in (coef.hdd_beta, coef.cdd_beta):
            if beta is not None:
                assert abs(beta) < 0.02 * abs(coef.intercept)

    predicted = model.predict(data)["predicted"]
    assert predicted.std() < 0.1 * predicted.mean()


@pytest.mark.slow
def test_no_load_change_gives_near_zero_savings(heating_baseline):
    """Predicting on the baseline reproduces it, so modeled savings are ~0.

    With no load change, summed predicted ≈ summed observed; the savings
    fraction (1 - predicted/observed) sits near zero rather than drifting.
    """
    model, data = heating_baseline

    result = model.predict(data)
    savings_fraction = 1.0 - result["predicted"].sum() / result["observed"].sum()

    assert abs(savings_fraction) < 0.05


@pytest.mark.slow
def test_known_reduction_recovers_as_savings(heating_baseline, temperature):
    """A known reduction injected into an independent load is recovered as savings.

    Reporting observed is the generative heating load (not the model's own
    prediction) minus a fixed per-day reduction, so recovery exercises the
    model's counterfactual rather than reducing to an identity.
    """
    model, _ = heating_baseline
    days = pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")
    daily_temp = temperature.resample("D").mean().reindex(days)
    reduction_per_day = 5.0
    reporting_meter = pd.Series(
        _heating_load(daily_temp, seed=4) - reduction_per_day, index=days
    )
    reporting = DailyReportingData.from_series(
        reporting_meter, temperature, is_electricity_data=True
    )

    result = model.predict(reporting)
    recovered = (result["predicted"] - result["observed"]).sum()
    expected = reduction_per_day * result["observed"].notna().sum()

    assert recovered == pytest.approx(expected, rel=0.1)
