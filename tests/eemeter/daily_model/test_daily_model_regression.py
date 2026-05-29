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

import pytest

from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData

from regression_metrics import regression_block


@pytest.fixture(scope="session")
def daily_baseline_data(comstock_daily):
    df_b, _ = comstock_daily

    return DailyBaselineData(df=df_b.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def daily_reporting_data(comstock_daily):
    _, df_r = comstock_daily

    return DailyReportingData(df=df_r.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def daily_model_fit(daily_baseline_data):
    return DailyModel().fit(daily_baseline_data, ignore_disqualification=True)


@pytest.mark.slow
@pytest.mark.regression
def test_daily_baseline_predict_regression(
    daily_model_fit, daily_baseline_data, snapshot
):
    """Fit on baseline -> predict on same data. Catches any change to fit + predict."""
    result = daily_model_fit.predict(daily_baseline_data)

    assert regression_block(result, freq="daily") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
def test_daily_reporting_predict_regression(
    daily_model_fit, daily_reporting_data, snapshot
):
    """Fit on baseline -> predict on reporting. Catches any change that affects out-of-sample predict."""
    result = daily_model_fit.predict(daily_reporting_data)

    assert regression_block(result, freq="daily") == snapshot(name="regression")
