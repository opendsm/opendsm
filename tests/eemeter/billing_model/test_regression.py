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

from eemeter.perturbations import bimonthly_reads, estimated_read, offcycle_read
from regression_metrics import regression_block



# Each scenario's expected disqualification qualified_names by is_electricity_data,
# read from an actual fit against the DEFAULT meter's monthly frame. An empty list
# means the scenario is expected to stay qualified in that fuel mode.
BILLING_SCHEDULE_SCENARIOS = {
    "offcycle_read": (
        offcycle_read,
        {
            True: ["eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data"],
            False: ["eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data"],
        },
    ),
    "bimonthly_reads": (
        bimonthly_reads,
        {
            True: [
                "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
            ],
            False: [
                "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
            ],
        },
    ),
    "estimated_read": (
        estimated_read,
        {
            True: [
                "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
            ],
            False: [],
        },
    ),
}


@pytest.fixture(scope="session")
def billing_baseline_df(comstock_monthly_diverse):
    df_b, _ = comstock_monthly_diverse

    return df_b.reset_index()


@pytest.fixture(scope="session")
def billing_reporting_df(comstock_monthly_diverse):
    _, df_r = comstock_monthly_diverse

    return df_r.reset_index()


@pytest.fixture(scope="session")
def billing_model_fit(billing_baseline_df):
    model = BillingModel().fit(
        billing_baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.mark.slow
@pytest.mark.regression
def test_billing_baseline_predict_regression(
    billing_model_fit, billing_baseline_df, snapshot
):
    result = billing_model_fit.predict(billing_baseline_df)

    assert regression_block(result, freq="daily") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
def test_billing_reporting_predict_regression(
    billing_model_fit, billing_reporting_df, snapshot
):
    result = billing_model_fit.predict(billing_reporting_df)

    assert regression_block(result, freq="daily") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("is_electricity_data", [True, False], ids=["electricity", "gas"])
@pytest.mark.parametrize("scenario_name", sorted(BILLING_SCHEDULE_SCENARIOS))
def test_billing_schedule_perturbation_regression(
    scenario_name, is_electricity_data, comstock_monthly, snapshot
):
    scenario, expected_by_fuel = BILLING_SCHEDULE_SCENARIOS[scenario_name]
    baseline, reporting = comstock_monthly
    perturbed_baseline, perturbed_reporting = scenario(baseline, reporting)

    model = BillingModel().fit(
        perturbed_baseline.reset_index(),
        is_electricity_data=is_electricity_data,
        enforce_compliance=["model", "settings"],
    )

    expected_names = expected_by_fuel[is_electricity_data]
    if expected_names:
        actual_names = sorted(entry.qualified_name for entry in model.compliance.disqualification)
        assert actual_names == sorted(expected_names)
    else:
        assert model.compliance.qualified
        assert model.compliance.disqualification == []

    result = model.predict(
        perturbed_reporting.reset_index(), enforce_compliance=["model", "settings"]
    )

    assert regression_block(result, freq="daily") == snapshot(name="regression")
