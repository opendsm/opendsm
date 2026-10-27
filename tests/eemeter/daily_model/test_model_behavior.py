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

from opendsm.eemeter import DailyModel

from ..helpers import (
    heating_load,
    meter_temperature_frame,
    seasonal_temperature,
)



@pytest.fixture(scope="module")
def temperature():
    return seasonal_temperature()


@pytest.fixture(scope="module")
def days():
    return pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Chicago")


@pytest.fixture(scope="module")
def daily_temperature(temperature, days):
    return temperature.resample("D").mean().reindex(days)


@pytest.fixture(scope="module")
def heating_baseline(temperature, daily_temperature):
    """DailyModel fit once on a load with a real heating response, with its data."""
    meter = heating_load(daily_temperature, seed=2)
    df = meter_temperature_frame(meter, temperature)
    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    return model, df


@pytest.mark.slow
def test_flat_load_does_not_fabricate_temperature_response(temperature, days):
    """A temperature-insensitive load yields negligible HDD/CDD slopes.

    The selector may still pick a fuller model_type, but the fitted heating and
    cooling betas must be tiny relative to the intercept (no invented response),
    and the prediction must track the mean rather than the temperature swing.
    """
    rng = np.random.default_rng(0)
    meter = pd.Series(40.0 + rng.normal(0, 1.0, 365), index=days)
    df = meter_temperature_frame(meter, temperature)

    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    for submodel in model.params.submodels.values():
        coef = submodel.coefficients
        for beta in (coef.hdd_beta, coef.cdd_beta):
            if beta is not None:
                assert abs(beta) < 0.02 * abs(coef.intercept)

    predicted = model.predict(df)["predicted"]
    assert predicted.std() < 0.1 * predicted.mean()


@pytest.mark.slow
def test_no_load_change_gives_near_zero_savings(heating_baseline):
    """Predicting on the baseline reproduces it, so modeled savings are ~0.

    With no load change, summed predicted ≈ summed observed; the savings
    fraction (1 - predicted/observed) sits near zero rather than drifting.
    """
    model, df = heating_baseline

    result = model.predict(df)
    savings_fraction = 1.0 - result["predicted"].sum() / result["observed"].sum()

    assert abs(savings_fraction) < 0.05


@pytest.mark.slow
def test_known_reduction_recovers_as_savings(heating_baseline, temperature, daily_temperature):
    """A known reduction injected into an independent load is recovered as savings.

    Reporting observed is the generative heating load (not the model's own
    prediction) minus a fixed per-day reduction, so recovery exercises the
    model's counterfactual rather than reducing to an identity.
    """
    model, _ = heating_baseline
    reduction_per_day = 5.0
    reporting_meter = heating_load(daily_temperature, seed=4) - reduction_per_day
    reporting_df = meter_temperature_frame(reporting_meter, temperature)

    result = model.predict(reporting_df)
    recovered = (result["predicted"] - result["observed"]).sum()
    expected = reduction_per_day * result["observed"].notna().sum()

    assert recovered == pytest.approx(expected, rel=0.1)
