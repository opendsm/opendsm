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
"""Regression tests pinning BillingModel fit & predict outputs."""

import pytest

from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)


@pytest.fixture(scope="session")
def billing_baseline_data(comstock_monthly):
    df_b, _ = comstock_monthly

    return BillingBaselineData(df=df_b.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def billing_reporting_data(comstock_monthly):
    _, df_r = comstock_monthly

    return BillingReportingData(df=df_r.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def billing_model_fit(billing_baseline_data):
    return BillingModel().fit(billing_baseline_data, ignore_disqualification=True)


def _summary(series):
    return {
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "n": int(series.shape[0]),
    }


@pytest.mark.slow
@pytest.mark.regression
def test_billing_baseline_predict_regression(
    billing_model_fit, billing_baseline_data, snapshot
):
    results = billing_model_fit.predict(billing_baseline_data)

    assert _summary(results["predicted"]) == snapshot(name="predicted_summary")
    assert results["predicted"].values.tolist() == snapshot(name="predicted_values")


@pytest.mark.slow
@pytest.mark.regression
def test_billing_reporting_predict_regression(
    billing_model_fit, billing_reporting_data, snapshot
):
    results = billing_model_fit.predict(billing_reporting_data)

    assert _summary(results["predicted"]) == snapshot(name="predicted_summary")
    assert results["predicted"].values.tolist() == snapshot(name="predicted_values")
