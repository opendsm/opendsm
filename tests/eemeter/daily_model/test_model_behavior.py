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
    doy = hours.day_of_year.values
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


def _daily_frame(start, temp, observed):
    days = pd.date_range(start, periods=len(temp), freq="D", tz="America/Chicago")
    df = pd.DataFrame({"observed": observed, "temperature": temp}, index=days)

    return df


def _warm_cooling_only():
    """56 days, every one above a cooling balance point at 80 F."""
    rng = np.random.default_rng(0)
    temp = 81.5 + 11.5 * rng.random(56)
    observed = 2500 + 120 * (temp - 80) + rng.normal(0, 60, 56)

    return _daily_frame("2025-07-01", temp, observed)


def _cold_heating_only():
    """56 days, every one below a heating balance point at 40 F."""
    rng = np.random.default_rng(0)
    temp = 20 + 15 * rng.random(56)
    observed = 2500 + 120 * (40 - temp) + rng.normal(0, 60, 56)

    return _daily_frame("2025-01-01", temp, observed)


def _year(seed):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=365, freq="D", tz="America/Chicago")
    temp = 67 - 20 * np.cos(2 * np.pi * (days.day_of_year.values - 15) / 365)
    temp = temp + rng.normal(0, 4, 365)

    return rng, days, temp


def _year_with_summer_step():
    """A year of cooling above 60 F with a summer load step, so summer fits on its own."""
    rng, days, temp = _year(seed=1)
    summer = days.month.isin([6, 7, 8, 9])
    observed = 2500 + 120 * np.clip(temp - 60, 0, None) + 600 * summer + rng.normal(0, 60, 365)

    return _daily_frame("2025-01-01", temp, observed)


def _year_heating_and_cooling():
    """A year with heating below 55 F and cooling above 68 F."""
    rng, _, temp = _year(seed=2)
    observed = (
        2500
        + 60 * np.clip(55 - temp, 0, None)
        + 90 * np.clip(temp - 68, 0, None)
        + rng.normal(0, 60, 365)
    )

    return _daily_frame("2025-01-01", temp, observed)


_SCENARIOS = {
    "warm_cooling_only": _warm_cooling_only,
    "cold_heating_only": _cold_heating_only,
    "year_with_summer_step": _year_with_summer_step,
    "year_heating_and_cooling": _year_heating_and_cooling,
}


@pytest.fixture(scope="module")
def fitted():
    """Returns `fit(scenario, base_settings)`, which fits each pair once for the module and
    returns (model, baseline data)."""
    fits = {}

    def fit(scenario, base_settings):
        if (scenario, base_settings) not in fits:
            data = DailyBaselineData(_SCENARIOS[scenario](), is_electricity_data=True)
            model = DailyModel(model=base_settings).fit(data, ignore_disqualification=True)
            fits[(scenario, base_settings)] = (model, data)

        return fits[(scenario, base_settings)]

    return fit


@pytest.mark.slow
@pytest.mark.parametrize("base_settings", ["current", "legacy"])
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_baseline_prediction_reproduces_fitted_values(fitted, scenario, base_settings):
    """Predicting the baseline returns, day for day, each submodel's fitted values.

    Coefficients are reduced and constrained after the optimizer finishes; none of that may
    move the fitted line.
    """
    model, data = fitted(scenario, base_settings)
    predicted = model.predict(data, ignore_disqualification=True)["predicted"]

    for component, submodel in model.model.items():
        days = model._meter_segment(component).index
        np.testing.assert_allclose(
            predicted.loc[days].values,
            submodel.model,
            rtol=1e-9,
            atol=1e-6,
            err_msg=f"{component} ({submodel.model_name}) predicts its baseline off its fit",
        )


@pytest.mark.slow
@pytest.mark.parametrize("base_settings", ["current", "legacy"])
@pytest.mark.parametrize("scenario", list(_SCENARIOS))
def test_baseline_prediction_sums_to_observed(fitted, scenario, base_settings):
    """Summed baseline predictions match summed observations, including baselines whose days
    all lie on one side of the balance point."""
    model, data = fitted(scenario, base_settings)
    result = model.predict(data, ignore_disqualification=True)

    shortfall = 1.0 - result["predicted"].sum() / result["observed"].sum()

    assert abs(shortfall) < 0.005, (
        f"baseline predicted {shortfall:+.2%} short of its observed total"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scenario", "base_settings", "component"),
    [
        ("warm_cooling_only", "current", "fw-su_sh_wi"),
        ("warm_cooling_only", "legacy", "fw-su_sh_wi"),
        ("cold_heating_only", "legacy", "fw-su_sh_wi"),
        ("year_with_summer_step", "current", "fw-su"),
    ],
)
def test_single_slope_fit_keeps_balance_point_at_data_edge(
    fitted, scenario, base_settings, component
):
    """A one-sided component is stored as a single slope from its coldest (cooling) or
    warmest (heating) day."""
    model, _ = fitted(scenario, base_settings)
    submodel = model.model[component]
    bp, beta = submodel.x[0], submodel.x[1]
    edge = submodel.T_min if beta >= 0 else submodel.T_max

    assert submodel.model_key == "c_hdd_tidd", f"{component} fit {submodel.model_name}"
    assert bp == edge, f"{component} balance point {bp}, expected data edge {edge}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("scenario", "base_settings", "offsets"),
    [
        ("warm_cooling_only", "current", [0.0, -5.0, -15.0, -30.0]),
        ("warm_cooling_only", "legacy", [0.0, -5.0, -15.0, -30.0]),
        ("cold_heating_only", "legacy", [0.0, 5.0, 15.0, 30.0]),
    ],
)
def test_single_slope_prediction_is_flat_beyond_balance_point_edge(
    fitted, scenario, base_settings, offsets
):
    """Reporting days past a single slope's balance point edge predict the value at that edge."""
    model, data = fitted(scenario, base_settings)
    temps = data.df["temperature"]
    edge = temps.min() if offsets[-1] < 0 else temps.max()
    reporting = DailyReportingData(
        _daily_frame("2025-11-01", edge + np.array(offsets), np.zeros(len(offsets))),
        is_electricity_data=True,
    )

    predicted = model.predict(reporting, ignore_disqualification=True)["predicted"].values

    np.testing.assert_allclose(
        predicted, predicted[0], rtol=1e-12, err_msg=f"temperatures {edge + np.array(offsets)}"
    )
