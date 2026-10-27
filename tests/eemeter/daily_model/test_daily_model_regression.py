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
"""Regression tests pinning DailyModel fit & predict outputs."""

import sys

import pytest

from opendsm.eemeter.models.daily.model import DailyModel

from eemeter.perturbations import FRAME_SCENARIOS
from regression_metrics import regression_block



# DailyModel's nlopt SBPLX optimizer lands at a different local minimum on
# Windows. Aggregate sum/mean stay within ~0.05%, but per-bin residual means
# diverge by tens of percent. Pin a Windows-specific snapshot so each platform
# keeps full cell-level regression coverage against itself.
SNAP_SUFFIX = "_win" if sys.platform == "win32" else ""


@pytest.fixture(scope="session")
def daily_baseline_data(comstock_daily_diverse):
    df_b, _ = comstock_daily_diverse

    return df_b.reset_index()


@pytest.fixture(scope="session")
def daily_reporting_data(comstock_daily_diverse):
    _, df_r = comstock_daily_diverse

    return df_r.reset_index()


@pytest.fixture(scope="session")
def daily_model_fit(daily_baseline_data):
    model = DailyModel().fit(
        daily_baseline_data, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.mark.slow
@pytest.mark.regression
def test_daily_baseline_predict_regression(
    daily_model_fit, daily_baseline_data, snapshot
):
    """Fit on baseline -> predict on same data. Catches any change to fit + predict."""
    result = daily_model_fit.predict(daily_baseline_data)

    assert regression_block(result, freq="daily") == snapshot(
        name=f"regression{SNAP_SUFFIX}"
    )


@pytest.mark.slow
@pytest.mark.regression
def test_daily_reporting_predict_regression(
    daily_model_fit, daily_reporting_data, snapshot
):
    """Fit on baseline -> predict on reporting. Catches any change that affects out-of-sample predict."""
    result = daily_model_fit.predict(daily_reporting_data)

    assert regression_block(result, freq="daily") == snapshot(
        name=f"regression{SNAP_SUFFIX}"
    )


# (scenario, is_electricity_data) -> disqualifications after fitting the perturbed DEFAULT
# meter; negative readings disqualify gas only, since net export makes them ordinary for
# electricity.
FRAME_SCENARIO_DISQUALIFICATIONS = {
    ("isolated_gaps", True): set(),
    ("isolated_gaps", False): set(),
    ("multi_day_gap", True): set(),
    ("multi_day_gap", False): set(),
    ("zeros", True): set(),
    ("zeros", False): set(),
    ("negative_export", True): set(),
    ("negative_export", False): {"eemeter.sufficiency_criteria.negative_observed_values"},
    ("dst_week_gap", True): set(),
    ("dst_week_gap", False): set(),
}


IS_ELECTRICITY_BY_FUEL = {"electricity": True, "gas": False}


@pytest.fixture(
    scope="session",
    params=[(scenario, fuel) for scenario in FRAME_SCENARIOS for fuel in IS_ELECTRICITY_BY_FUEL],
    ids="-".join,
)
def daily_frame_scenario_fit(request, comstock_daily):
    """Fit the DEFAULT meter's baseline, perturbed by one frame scenario, with model and settings
    compliance enforced; return the expected disqualifications, the model and the perturbed
    baseline.

    Only `comstock_daily` (the single DEFAULT meter) is used, not the diverse fixture, so the
    scenario grid doesn't multiply by five.
    """
    scenario_name, fuel = request.param
    is_electricity_data = IS_ELECTRICITY_BY_FUEL[fuel]
    df_b, _ = comstock_daily
    baseline = FRAME_SCENARIOS[scenario_name](df_b).reset_index()

    model = DailyModel().fit(
        baseline,
        is_electricity_data=is_electricity_data,
        enforce_compliance=["model", "settings"],
    )
    expected = FRAME_SCENARIO_DISQUALIFICATIONS[(scenario_name, is_electricity_data)]

    return expected, model, baseline


@pytest.mark.slow
@pytest.mark.regression
def test_daily_frame_scenario_disqualifications(daily_frame_scenario_fit):
    """A frame scenario disqualifies the fit exactly by its table entry."""
    expected, model, _ = daily_frame_scenario_fit
    actual = {entry.qualified_name for entry in model.compliance.disqualification}

    assert actual == expected
    assert model.compliance.qualified == (not expected)


@pytest.mark.slow
@pytest.mark.regression
def test_daily_frame_scenario_prediction(daily_frame_scenario_fit, snapshot):
    """The daily regression block of the DEFAULT meter, fit and predicted on a baseline
    perturbed by one frame scenario, matches its snapshot.
    """
    _, model, baseline = daily_frame_scenario_fit
    result = model.predict(baseline, enforce_compliance=["model", "settings"])

    assert regression_block(result, freq="daily") == snapshot(
        name=f"regression{SNAP_SUFFIX}"
    )
