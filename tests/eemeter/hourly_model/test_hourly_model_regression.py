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
"""Regression tests pinning HourlyModel fit & predict outputs."""

import numpy as np
from scipy.stats import spearmanr
import pytest

from opendsm.eemeter.models.hourly.model import HourlyModel
from opendsm.eemeter.models.hourly.settings import (
    HourlyNonSolarSettings,
    HourlySolarSettings,
)

from eemeter.perturbations import FRAME_SCENARIOS
from regression_metrics import TEMP_BIN_EDGES_F, regression_block
from syrupy_extensions import TolerantJSONSnapshotExtension



# Disqualification names each scenario produces against the default meter; the
# negative-observed check runs for gas only, since electricity can net-export.
FRAME_SCENARIO_DISQUALIFICATIONS = {
    ("isolated_gaps", True): set(),
    ("isolated_gaps", False): set(),
    ("multi_day_gap", True): set(),
    ("multi_day_gap", False): set(),
    ("zeros", True): {"eemeter.sufficiency_criteria.missing_monthly_observed_data"},
    ("zeros", False): set(),
    ("negative_export", True): set(),
    ("negative_export", False): {"eemeter.sufficiency_criteria.negative_observed_values"},
    ("dst_week_gap", True): {"eemeter.sufficiency_criteria.missing_monthly_observed_data"},
    ("dst_week_gap", False): {"eemeter.sufficiency_criteria.missing_monthly_observed_data"},
}


IS_ELECTRICITY_BY_FUEL = {"electricity": True, "gas": False}

_LOAD_KEYS = frozenset({"predicted", "residuals", "time_bins", "temp_bins", "ghi_bins"})


def _hourly_snapshot(snapshot, result):
    # An hourly fit moves by about 1e-4 relative across platforms and BLAS builds, and further
    # on cells near zero, so values allow 1e-3 relative and load fields 1e-3 of the mean prediction
    atol = 1e-3 * float(result["predicted"].abs().mean())
    extension = TolerantJSONSnapshotExtension.with_tolerance(
        atol=atol, rtol=1e-3, atol_keys=_LOAD_KEYS, leaf_atol={}
    )

    return snapshot.use_extension(extension)


def _fit_nonsolar(df):
    settings = HourlyNonSolarSettings(seed=42)
    fit = HourlyModel(settings=settings).fit(
        df, is_electricity_data=True, enforce_compliance=False
    )

    return fit


@pytest.fixture(scope="session")
def hourly_baseline_data(comstock_hourly_diverse):
    df_b, _ = comstock_hourly_diverse

    return df_b.reset_index()


@pytest.fixture(scope="session")
def hourly_reporting_data(comstock_hourly_diverse):
    _, df_r = comstock_hourly_diverse

    return df_r.reset_index()


@pytest.fixture(scope="session")
def hourly_nonsolar_fit(hourly_baseline_data):
    return _fit_nonsolar(hourly_baseline_data)


@pytest.fixture(scope="session")
def hourly_solar_fit(hourly_baseline_data):
    settings = HourlySolarSettings(seed=42)

    return HourlyModel(settings=settings).fit(
        hourly_baseline_data, is_electricity_data=True, enforce_compliance=False
    )


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize(
    "fit_fixture_name, data_fixture_name",
    [
        ("hourly_nonsolar_fit", "hourly_baseline_data"),
        ("hourly_nonsolar_fit", "hourly_reporting_data"),
        ("hourly_solar_fit", "hourly_baseline_data"),
        ("hourly_solar_fit", "hourly_reporting_data"),
    ],
    ids=[
        "nonsolar_baseline",
        "nonsolar_reporting",
        "solar_baseline",
        "solar_reporting",
    ],
)
@pytest.mark.usefixtures("hourly_baseline_data")
def test_hourly_predict_regression(fit_fixture_name, data_fixture_name, request, snapshot):
    fit = request.getfixturevalue(fit_fixture_name)
    data = request.getfixturevalue(data_fixture_name)

    result = fit.predict(data, enforce_compliance=False)

    assert regression_block(result, freq="hourly") == _hourly_snapshot(snapshot, result)(
        name="regression"
    )


@pytest.fixture(
    scope="session",
    params=[(scenario, fuel) for scenario in FRAME_SCENARIOS for fuel in IS_ELECTRICITY_BY_FUEL],
    ids="-".join,
)
def hourly_frame_scenario_fit(request, comstock_hourly):
    """Fit the DEFAULT meter's baseline, perturbed by one frame scenario, with model and settings
    compliance enforced; return the expected disqualifications, the model and the perturbed
    baseline.
    """
    scenario_name, fuel = request.param
    is_electricity_data = IS_ELECTRICITY_BY_FUEL[fuel]
    df_b, _ = comstock_hourly
    baseline = FRAME_SCENARIOS[scenario_name](df_b).reset_index()

    settings = HourlyNonSolarSettings(seed=42)
    model = HourlyModel(settings=settings).fit(
        baseline,
        is_electricity_data=is_electricity_data,
        enforce_compliance=["model", "settings"],
    )
    expected = FRAME_SCENARIO_DISQUALIFICATIONS[(scenario_name, is_electricity_data)]

    return expected, model, baseline


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_frame_scenario_disqualifications(hourly_frame_scenario_fit):
    """A frame scenario disqualifies the fit exactly by its table entry."""
    expected, model, _ = hourly_frame_scenario_fit
    actual = {entry.qualified_name for entry in model.compliance.disqualification}

    assert actual == expected
    assert model.compliance.qualified == (not expected)


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_frame_scenario_prediction(hourly_frame_scenario_fit, snapshot):
    """Predictions on the perturbed baseline match the snapshot."""
    _, model, baseline = hourly_frame_scenario_fit
    result = model.predict(baseline, enforce_compliance=["model", "settings"])

    assert regression_block(result, freq="hourly") == _hourly_snapshot(snapshot, result)(
        name="regression"
    )


@pytest.fixture(scope="session")
def hourly_default_nonsolar_fit(comstock_hourly):
    df_b, _ = comstock_hourly

    return _fit_nonsolar(df_b.reset_index())


@pytest.fixture(scope="session", params=[25, -25], ids=["plus25", "minus25"])
def temperature_shift(request):
    return request.param


@pytest.fixture(scope="session")
def hourly_shifted_prediction(
    temperature_shift, hourly_default_nonsolar_fit, comstock_hourly
):
    """Predict on the reporting frame with temperature shifted +/-25 F outside its observed range."""
    _, df_r = comstock_hourly
    shifted = df_r.reset_index()
    shifted["temperature"] = shifted["temperature"] + temperature_shift
    result = hourly_default_nonsolar_fit.predict(shifted, enforce_compliance=False)

    return result


def _assert_edge_bin_trend(result, temperature_shift):
    """Assert extrapolated load trends with temperature in each edge bin the data reaches.

    The side the shift moves into (the cooling edge bin above `TEMP_BIN_EDGES_F[-2]` for a
    positive shift, the heating edge bin below `TEMP_BIN_EDGES_F[1]` for a negative one) must
    hold at least 3 distinct whole degrees. The opposite side
    is checked too whenever it holds at least 3. Cooling load must trend up and heating
    load down.

    The check is a rank correlation on per-whole-degree means, not step-by-step
    monotonicity: the hourly model carries a separate coefficient per hour of week,
    so neighbouring degree means reverse locally even where the trend is clear. A
    sign flip, which is what a broken edge-bin rate produces, still fails this.
    """
    if temperature_shift == 0:
        raise ValueError("temperature_shift must be nonzero to reach an edge bin")

    temperature = result["temperature"]
    predicted = result["predicted"]
    sides = [
        (temperature > TEMP_BIN_EDGES_F[-2], 1.0, f"cooling side (> {TEMP_BIN_EDGES_F[-2]}F)"),
        (temperature < TEMP_BIN_EDGES_F[1], -1.0, f"heating side (< {TEMP_BIN_EDGES_F[1]}F)"),
    ]

    for index, (mask, expected_sign, label) in enumerate(sides):
        degree = temperature[mask].round()
        means = predicted[mask].groupby(degree).mean().sort_index()
        is_shifted_side = (index == 0) == (temperature_shift > 0)

        if is_shifted_side:
            assert len(means) >= 3, (
                f"the {label} holds {len(means)} distinct whole degrees, expected at least 3"
            )
        elif len(means) < 3:
            continue

        rho = spearmanr(means.index.to_numpy(), means.to_numpy()).statistic

        assert rho * expected_sign > 0.5, (
            f"extrapolated load does not trend with temperature on the {label}: "
            f"spearman rho={rho:.3f}, expected sign {expected_sign:+.0f}"
        )


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_predict_extrapolation_regression(hourly_shifted_prediction, snapshot):
    """Predictions on reporting temperature shifted +/-25 F beyond the observed range match
    their snapshot.

    No baseline data exists at these temperatures, so this snapshot does not pin
    correctness of the extrapolated values; it exists only to catch a change in how
    the model extrapolates beyond its training temperature range.
    """
    assert regression_block(hourly_shifted_prediction, freq="hourly") == snapshot(name="regression")


@pytest.mark.slow
def test_hourly_predict_extrapolation_is_finite(hourly_shifted_prediction):
    """Predictions stay finite when reporting temperature lies beyond the observed range."""
    predicted = hourly_shifted_prediction["predicted"]

    assert np.isfinite(predicted.to_numpy()).all()


@pytest.mark.slow
def test_hourly_predict_extrapolation_trends_with_temperature(
    hourly_shifted_prediction, temperature_shift
):
    """Extrapolated load trends with temperature in each edge bin the shifted reporting data
    reaches."""
    _assert_edge_bin_trend(hourly_shifted_prediction, temperature_shift)
