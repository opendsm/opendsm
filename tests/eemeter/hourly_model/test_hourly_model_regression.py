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

import pytest

from opendsm.eemeter.models.hourly.data import HourlyBaselineData, HourlyReportingData
from opendsm.eemeter.models.hourly.model import HourlyModel
from opendsm.eemeter.models.hourly.settings import (
    HourlyNonSolarSettings,
    HourlySolarSettings,
)

from regression_metrics import regression_block


@pytest.fixture(scope="session")
def hourly_baseline_data(comstock_hourly):
    df_b, _ = comstock_hourly

    return HourlyBaselineData(df=df_b.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def hourly_reporting_data(comstock_hourly):
    _, df_r = comstock_hourly

    return HourlyReportingData(df=df_r.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def hourly_nonsolar_fit(hourly_baseline_data):
    settings = HourlyNonSolarSettings(seed=42)

    return HourlyModel(settings=settings).fit(
        hourly_baseline_data, ignore_disqualification=True
    )


@pytest.fixture(scope="session")
def hourly_solar_fit(hourly_baseline_data):
    settings = HourlySolarSettings(seed=42)

    return HourlyModel(settings=settings).fit(
        hourly_baseline_data, ignore_disqualification=True
    )


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_nonsolar_baseline_predict_regression(
    hourly_nonsolar_fit, hourly_baseline_data, snapshot
):
    result = hourly_nonsolar_fit.predict(
        hourly_baseline_data, ignore_disqualification=True
    )

    assert regression_block(result, freq="hourly") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_nonsolar_reporting_predict_regression(
    hourly_nonsolar_fit, hourly_reporting_data, snapshot
):
    result = hourly_nonsolar_fit.predict(
        hourly_reporting_data, ignore_disqualification=True
    )

    assert regression_block(result, freq="hourly") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_solar_baseline_predict_regression(
    hourly_solar_fit, hourly_baseline_data, snapshot
):
    result = hourly_solar_fit.predict(
        hourly_baseline_data, ignore_disqualification=True
    )

    assert regression_block(result, freq="hourly") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
def test_hourly_solar_reporting_predict_regression(
    hourly_solar_fit, hourly_reporting_data, snapshot
):
    result = hourly_solar_fit.predict(
        hourly_reporting_data, ignore_disqualification=True
    )

    assert regression_block(result, freq="hourly") == snapshot(name="regression")
