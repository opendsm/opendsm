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

from eemeter.perturbations import (
    dst_week_gap,
    multi_day_gap,
    negative_export,
    single_hour_gaps,
    zeros,
)
from regression_metrics import TEMP_BIN_EDGES_F, regression_block



FRAME_SCENARIOS = {
    "single_hour_gaps": single_hour_gaps,
    "multi_day_gap": multi_day_gap,
    "zeros": zeros,
    "negative_export": negative_export,
    "dst_week_gap": dst_week_gap,
}

# Disqualification names each scenario produces against the default meter,
# read from an actual fit. Single-hour gaps and the 3-day gap stay under the
# coverage thresholds in both fuel modes; the negative-observed check only
# runs for gas, since electricity can legitimately net-export.
FRAME_SCENARIO_DISQUALIFICATIONS = {
    ("single_hour_gaps", True): set(),
    ("single_hour_gaps", False): set(),
    ("multi_day_gap", True): set(),
    ("multi_day_gap", False): set(),
    ("zeros", True): {"eemeter.sufficiency_criteria.missing_monthly_observed_data"},
    ("zeros", False): set(),
    ("negative_export", True): set(),
    ("negative_export", False): {"eemeter.sufficiency_criteria.negative_observed_values"},
    ("dst_week_gap", True): {"eemeter.sufficiency_criteria.missing_monthly_observed_data"},
    ("dst_week_gap", False): {"eemeter.sufficiency_criteria.missing_monthly_observed_data"},
}


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
    settings = HourlyNonSolarSettings(seed=42)

    return HourlyModel(settings=settings).fit(
        hourly_baseline_data, is_electricity_data=True, enforce_compliance=False
    )


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
def test_hourly_predict_regression(
    fit_fixture_name, data_fixture_name, hourly_baseline_data, request, snapshot
):
    fit = request.getfixturevalue(fit_fixture_name)
    data = request.getfixturevalue(data_fixture_name)

    result = fit.predict(data, enforce_compliance=False)

    assert regression_block(result, freq="hourly") == snapshot(name="regression")


@pytest.fixture(scope="session")
def hourly_default_data(comstock_hourly):
    return comstock_hourly


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("scenario_name", list(FRAME_SCENARIOS))
@pytest.mark.parametrize("is_electricity_data", [True, False], ids=["electricity", "gas"])
def test_hourly_frame_scenario_regression(
    scenario_name, is_electricity_data, hourly_default_data, snapshot
):
    df_b, df_r = hourly_default_data
    perturbed_baseline, _ = FRAME_SCENARIOS[scenario_name](df_b, df_r)
    perturbed_baseline = perturbed_baseline.reset_index()

    settings = HourlyNonSolarSettings(seed=42)
    fit = HourlyModel(settings=settings).fit(
        perturbed_baseline,
        is_electricity_data=is_electricity_data,
        enforce_compliance=["model", "settings"],
    )

    expected = FRAME_SCENARIO_DISQUALIFICATIONS[(scenario_name, is_electricity_data)]
    actual = {entry.qualified_name for entry in fit.compliance.disqualification}
    assert actual == expected
    assert fit.compliance.qualified == (not expected)

    result = fit.predict(perturbed_baseline, enforce_compliance=["model", "settings"])

    assert regression_block(result, freq="hourly") == snapshot(name="regression")


@pytest.fixture(scope="session")
def hourly_default_nonsolar_fit(hourly_default_data):
    df_b, _ = hourly_default_data
    settings = HourlyNonSolarSettings(seed=42)
    fit = HourlyModel(settings=settings).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    return fit


@pytest.fixture(scope="session", params=[25, -25], ids=["plus25", "minus25"])
def hourly_shifted_prediction(request, hourly_default_nonsolar_fit, hourly_default_data):
    """Predict on the reporting frame with temperature shifted +/-25 F outside its observed range."""
    _, df_r = hourly_default_data
    shifted = df_r.reset_index()
    shifted["temperature"] = shifted["temperature"] + request.param
    result = hourly_default_nonsolar_fit.predict(shifted, enforce_compliance=False)

    return result


def _assert_edge_bin_trend(result):
    """Assert extrapolated load trends downward with temperature below 30F and
    upward above 90F.

    The check is a rank correlation on per-whole-degree means, not step-by-step
    monotonicity: the hourly model carries a separate coefficient per hour of week,
    so neighbouring degree means reverse locally even where the trend is clear. A
    sign flip, which is what a broken edge-bin rate produces, still fails this.
    """
    temperature = result["temperature"]
    predicted = result["predicted"]
    cold_bin = temperature < TEMP_BIN_EDGES_F[1]
    hot_bin = temperature > TEMP_BIN_EDGES_F[-2]

    for mask, expected_sign, label in [
        (cold_bin, -1.0, "heating side (< 30F)"),
        (hot_bin, 1.0, "cooling side (> 90F)"),
    ]:
        degree = temperature[mask].round()
        means = predicted[mask].groupby(degree).mean().sort_index()
        if len(means) < 3:
            continue

        rho = spearmanr(means.index.to_numpy(), means.to_numpy()).statistic

        assert rho * expected_sign > 0.5, (
            f"extrapolated load does not trend with temperature on the {label}: "
            f"spearman rho={rho:.3f}, expected sign {expected_sign:+.0f}"
        )


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_predict_extrapolation_regression(hourly_shifted_prediction, snapshot):
    """Pin predictions when reporting temperature is shifted +/-25 F beyond the observed range.

    No baseline data exists at these temperatures, so this snapshot does not pin
    correctness of the extrapolated values; it exists only to catch a change in how
    the model extrapolates beyond its training temperature range.
    """
    assert regression_block(hourly_shifted_prediction, freq="hourly") == snapshot(name="regression")


@pytest.mark.slow
def test_hourly_predict_extrapolation_is_finite_and_trends_with_temperature(hourly_shifted_prediction):
    predicted = hourly_shifted_prediction["predicted"]
    assert np.isfinite(predicted.to_numpy()).all()

    _assert_edge_bin_trend(hourly_shifted_prediction)
